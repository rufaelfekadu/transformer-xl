import os, sys
import glob
from typing import Optional, Iterator, Union, Iterable, List
import math

from collections import Counter, OrderedDict
import numpy as np
import torch
from torch.utils.data import Dataset, DistributedSampler, BatchSampler, Sampler, DataLoader
from torch.utils.data.distributed import T_co
import torch.distributed as dist

from utils.vocabulary import Vocab


class CustomDataset(Dataset):
    def __init__(self, data, bptt=512) -> None:
        super().__init__()
        self.data = data
        self.bptt = bptt
        
        
    def __getitem__(self, index):
        start_idx = index*self.bptt
        end_idx = (index+1)*self.bptt
        return self.data[start_idx:end_idx], self.data[start_idx+1:end_idx+1] # data, target
    
    def __len__(self):
        return self.data.shape[0]//self.bptt

class CustomDistributedSampler(DistributedSampler):
    def __init__(self, dataset: Dataset, batch_size:int, num_replicas: Optional[int] = None, rank: Optional[int] = None, shuffle: bool = True, seed: int = 0, drop_last: bool = False) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        
        self.batch_size = batch_size
    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size
        world_size = dist.get_world_size()
        indices_per_rank = len(indices) // world_size

        # subsample
        rank_start = self.rank * indices_per_rank
        rank_end = rank_start + indices_per_rank
        rank_indices = indices[rank_start:rank_end]
        assert len(rank_indices) == self.num_samples
        
        # Rearrange the indices to ensure that for a batch size, b, the i-th axis of successive
        # batches loaded are contiguous in the original data. This is to make it possible to take
        # advantage of the memory component. For example, if the following are indices to loaded
        # by a single device using batch size 4,
        #  
        # 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
        # 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
        # 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
        # 
        # We will need to transform it as shown below 
        # 
        # [ 0, 12, 24, 36,  1, 13, 25, 37,  2, 14, 26, 38,  3, 15, 27, 39,  4, 16,
        # 28, 40,  5, 17, 29, 41,  6, 18, 30, 42,  7, 19, 31, 43,  8, 20, 32, 44,
        #  9, 21, 33, 45, 10, 22, 34, 46, 11, 23, 35, 47]
        # 
        # this is to ensure that when the first three batches
        # are processed, they will be in this form:
        # [0,12,24,36], [1,13,25,37], [2,14,26,38], which shows that the batches first axes data are 
        # 1,2,3 (i.e, they are contiguous)
        
        
        # Remove excess padding to make it divisible by the batch size
        if not self.drop_last:
            padding_size = -len(rank_indices)%self.batch_size
            rank_indices += rank_indices[:padding_size]
        else:
            excess_size = len(rank_indices)%self.batch_size
            rank_indices = rank_indices[:len(rank_indices)-excess_size]
        
        rank_indices = np.array(rank_indices).reshape(self.batch_size, -1).T.flatten()

        return iter(rank_indices)

class CustomBatchSampler(BatchSampler):
    def __init__(self, sampler: Union[Sampler[int], Iterable[int]], batch_size: int, drop_last: bool) -> None:
        super().__init__(sampler, batch_size, drop_last)
    def __iter__(self) -> Iterator[List[int]]:
        return super().__iter__()
        

class LMOrderedIterator(object):
    def __init__(self, data, bsz, bptt, device='cpu', ext_len=None):
        """
            data -- LongTensor -- the LongTensor is strictly ordered
        """
        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0

        self.device = device

        # Work out how cleanly we can divide the dataset into bsz parts.
        self.n_step = data.size(0) // bsz

        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, self.n_step * bsz)
        # print(data.shape)

        # Evenly divide the data across the bsz batches.
        self.data = data.view(bsz, -1).t().contiguous().to(device)

        # Number of mini-batches
        self.n_batch = (self.n_step + self.bptt - 1) // self.bptt

    def get_batch(self, i, bptt=None):
        # print("bptt", bptt)
        if bptt is None: bptt = self.bptt
        seq_len = min(bptt, self.data.size(0) - 1 - i)

        end_idx = i + seq_len
        beg_idx = max(0, i - self.ext_len)
        # print(beg_idx, self.ext_len, self.data.shape, self.bsz)

        data = self.data[beg_idx:end_idx]
        target = self.data[i+1:i+1+seq_len]

        return data, target, seq_len

    def get_fixlen_iter(self, start=0):
        for i in range(start, self.data.size(0) - 1, self.bptt):
            yield self.get_batch(i)

    def get_varlen_iter(self, start=0, std=5, min_len=5, max_deviation=3):
        max_len = self.bptt + max_deviation * std
        i = start
        while True:
            bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
            bptt = min(max_len, max(min_len, int(np.random.normal(bptt, std))))
            data, target, seq_len = self.get_batch(i, bptt)
            i += seq_len
            yield data, target, seq_len
            if i >= self.data.size(0) - 2:
                break

    def __iter__(self):
        return self.get_fixlen_iter()


class LMShuffledIterator(object):
    def __init__(self, data, bsz, bptt, device='cpu', ext_len=None, shuffle=False):
        """
            data -- list[LongTensor] -- there is no order among the LongTensors
        """
        self.data = data

        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0

        self.device = device
        self.shuffle = shuffle

    def get_sent_stream(self):
        # index iterator
        epoch_indices = np.random.permutation(len(self.data)) if self.shuffle \
            else np.array(range(len(self.data)))

        # sentence iterator
        for idx in epoch_indices:
            yield self.data[idx]

    def stream_iterator(self, sent_stream):
        # streams for each data in the batch
        streams = [None] * self.bsz

        data = torch.LongTensor(self.bptt, self.bsz)
        target = torch.LongTensor(self.bptt, self.bsz)

        n_retain = 0

        while True:
            # data   : [n_retain+bptt x bsz]
            # target : [bptt x bsz]
            data[n_retain:].fill_(-1)
            target.fill_(-1)

            valid_batch = True

            for i in range(self.bsz):
                n_filled = 0
                try:
                    while n_filled < self.bptt:
                        if streams[i] is None or len(streams[i]) <= 1:
                            streams[i] = next(sent_stream)
                        # number of new tokens to fill in
                        n_new = min(len(streams[i]) - 1, self.bptt - n_filled)
                        # first n_retain tokens are retained from last batch
                        data[n_retain+n_filled:n_retain+n_filled+n_new, i] = \
                            streams[i][:n_new]
                        target[n_filled:n_filled+n_new, i] = \
                            streams[i][1:n_new+1]
                        streams[i] = streams[i][n_new:]
                        n_filled += n_new
                except StopIteration:
                    valid_batch = False
                    break

            if not valid_batch:
                return

            data = data.to(self.device)
            target = target.to(self.device)

            yield data, target, self.bptt

            n_retain = min(data.size(0), self.ext_len)
            if n_retain > 0:
                data[:n_retain] = data[-n_retain:]
            data.resize_(n_retain + self.bptt, data.size(1))

    def __iter__(self):
        # sent_stream is an iterator
        sent_stream = self.get_sent_stream()

        for batch in self.stream_iterator(sent_stream):
            yield batch


class LMMultiFileIterator(LMShuffledIterator):
    def __init__(self, paths, vocab, bsz, bptt, device='cpu', ext_len=None,
        shuffle=False):

        self.paths = paths
        self.vocab = vocab

        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0

        self.device = device
        self.shuffle = shuffle

    def get_sent_stream(self, path):
        sents = self.vocab.encode_file(path, add_double_eos=True)
        if self.shuffle:
            np.random.shuffle(sents)
        sent_stream = iter(sents)

        return sent_stream

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.paths)

        for path in self.paths:
            # sent_stream is an iterator
            sent_stream = self.get_sent_stream(path)
            for batch in self.stream_iterator(sent_stream):
                yield batch


class Corpus(object):
    def __init__(self, path, dataset, *args, **kwargs):
        self.dataset = dataset
        self.vocab = Vocab(*args, **kwargs)

        if self.dataset in ['ptb', 'wt2', 'enwik8', 'text8']:
            self.vocab.count_file(os.path.join(path, 'train.txt'))
            self.vocab.count_file(os.path.join(path, 'valid.txt'))
            self.vocab.count_file(os.path.join(path, 'test.txt'))
        elif self.dataset == 'wt103':
            self.vocab.count_file(os.path.join(path, 'train.txt'))
        elif self.dataset == 'lm1b':
            train_path_pattern = os.path.join(
                path, '1-billion-word-language-modeling-benchmark-r13output',
                'training-monolingual.tokenized.shuffled', 'news.en-*')
            train_paths = glob.glob(train_path_pattern)
            # the vocab will load from file when build_vocab() is called

        self.vocab.build_vocab()

        if self.dataset in ['ptb', 'wt2', 'wt103']:
            self.train = self.vocab.encode_file(
                os.path.join(path, 'train.txt'), ordered=True)
            self.valid = self.vocab.encode_file(
                os.path.join(path, 'valid.txt'), ordered=True)
            self.test  = self.vocab.encode_file(
                os.path.join(path, 'test.txt'), ordered=True)
        elif self.dataset in ['enwik8', 'text8']:
            self.train = self.vocab.encode_file(
                os.path.join(path, 'train.txt'), ordered=True, add_eos=False)
            self.valid = self.vocab.encode_file(
                os.path.join(path, 'valid.txt'), ordered=True, add_eos=False)
            self.test  = self.vocab.encode_file(
                os.path.join(path, 'test.txt'), ordered=True, add_eos=False)
        elif self.dataset == 'lm1b':
            self.train = train_paths
            self.valid = self.vocab.encode_file(
                os.path.join(path, 'valid.txt'), ordered=False, add_double_eos=True)
            self.test  = self.vocab.encode_file(
                os.path.join(path, 'test.txt'), ordered=False, add_double_eos=True)

    def get_iterator(self, split, *args, **kwargs):
        if split == 'train':
            if self.dataset in ['ptb', 'wt2', 'wt103', 'enwik8', 'text8']:
                data_iter = LMOrderedIterator(self.train, *args, **kwargs)
            elif self.dataset == 'lm1b':
                kwargs['shuffle'] = True
                data_iter = LMMultiFileIterator(self.train, self.vocab, *args, **kwargs)
        elif split in ['valid', 'test']:
            data = self.valid if split == 'valid' else self.test
            if self.dataset in ['ptb', 'wt2', 'wt103', 'enwik8', 'text8']:
                data_iter = LMOrderedIterator(data, *args, **kwargs)
            elif self.dataset == 'lm1b':
                data_iter = LMShuffledIterator(data, *args, **kwargs)

        return data_iter
    
    def get_dataset(self, split, *args, **kwargs):
        if split=='train':
            return CustomDataset(self.train, *args, **kwargs)
        elif split in ["valid", "test"]:
            data = self.valid if split == 'valid' else self.test
            return CustomDataset(data, *args, **kwargs)


def get_lm_corpus(datadir, dataset):
    fn = os.path.join(datadir, 'cache.pt')
    if os.path.exists(fn):
        print('Loading cached dataset...')
        corpus = torch.load(fn)
    else:
        print('Producing dataset {}...'.format(dataset))
        kwargs = {}
        if dataset in ['wt103', 'wt2']:
            kwargs['special'] = ['<eos>']
            kwargs['lower_case'] = False
        elif dataset == 'ptb':
            kwargs['special'] = ['<eos>']
            kwargs['lower_case'] = True
        elif dataset == 'lm1b':
            kwargs['special'] = []
            kwargs['lower_case'] = False
            kwargs['vocab_file'] = os.path.join(datadir, '1b_word_vocab.txt')
        elif dataset in ['enwik8', 'text8']:
            pass

        corpus = Corpus(datadir, dataset, **kwargs)
        torch.save(corpus, fn)

    return corpus

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='unit test')
    parser.add_argument('--datadir', type=str, default='../data/enwik8',
                        help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='enwik8',
                        choices=['ptb', 'wt2', 'wt103', 'lm1b', 'enwik8', 'text8'],
                        help='dataset name')
    args = parser.parse_args()
    import pathlib
    print(pathlib.Path(args.datadir).exists())
    corpus = get_lm_corpus(args.datadir, args.dataset)
    print('Vocab size : {}'.format(len(corpus.vocab.idx2sym)))
    iterator = corpus.get_iterator("train", 22, 512, device=torch.device("cuda"), ext_len=0)
    # for i, (data, target, seq_len) in enumerate(iterator):
    #     print(data[1, :3], target[:5, :3], sep='\n')
        
        # if i==0: break
    
    dataset = corpus.get_dataset("train")
    print("Length of dataset:", len(dataset))
    print("Length of data:", dataset.data.shape)
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    
    # # for i in range(4):
    # #     print("Rank:", i)
    dist.init_process_group("nccl", rank=0, world_size=1)
        # print("Rank done", i)
    print("creating sampler instance")
    sampler = CustomDistributedSampler(dataset, shuffle=False, drop_last=True, batch_size=22)
    d1, t1, _ = next(iter(iterator))
    train_kwargs = {"batch_size": 22, "drop_last":True}
    dataloader = DataLoader(dataset,**train_kwargs, shuffle=False)
    
    print("Number of sequences from custom", len(sampler))
    
    i=0
    for i, ((d2, t2), (d1, t1, _)) in enumerate(zip(dataloader, iterator)) :
        # for d1, t1, _ in iterator:
        # print(d2.T.sum(dim=0), d1.sum(dim=0), sep="\n", end="\n###\n")
        if i >10:
            break
        
    dist.destroy_process_group()
    # print(all(dataset[0][0]==data.cpu()[:,0]))
