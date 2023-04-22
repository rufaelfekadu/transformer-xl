import os, sys
import math
import time
import argparse
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import itertools
import multiprocessing as python_multiprocessing

from torch.optim.lr_scheduler import StepLR
import torch.multiprocessing as multiprocessing

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
   apply_activation_checkpointing,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
    transformer_auto_wrap_policy,
)

from mem_transformer import MemTransformerLM, RelPartialLearnableDecoderLayer
from custom_parser import arguments
from data_utils import get_lm_corpus, CustomDistributedSampler
from utils.weight_init import  weights_init
from utils.exp_utils import create_exp_dir, init_wandb

lead_device=0

eval_batch_size = 10
best_val_loss = float("inf")
def init_model(args):
    cutoffs, tie_projs = [], [False]

    model = MemTransformerLM(args.n_token, args.n_layer, args.n_head, args.d_model,
            args.d_head, args.d_inner, args.dropout, args.dropatt,
            tie_weight=args.tied, d_embed=args.d_embed, div_val=args.div_val,
            tie_projs=tie_projs, pre_lnorm=args.pre_lnorm, tgt_len=args.tgt_len,
            ext_len=args.ext_len, mem_len=args.mem_len, cutoffs=cutoffs,
            same_length=args.same_length, attn_type=args.attn_type,
            clamp_len=args.clamp_len, sample_softmax=args.sample_softmax)

    model.apply(weights_init)
    model.word_emb.apply(weights_init)
    return model

def setup(rank, world_size, port=None):
    import random
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port or "12356"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    
    
def train(args, model: MemTransformerLM, rank, world_size, train_loader, valid_loader, optimizer, scheduler, epoch, wandb, sampler=None):
    global train_step_, logging
    model.train()
    ddp_loss = torch.zeros(1).to(rank)
    if sampler:
        sampler.set_epoch(epoch)

    mems = tuple()

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)
    # barrier_end = torch.cuda.Event(True)
    
    init_start_event.record()
    cur_loss = prev_loss = 0
    start = time.time()
    data: torch.Tensor
    for batch, (data, target) in enumerate(train_loader):
        data, target = data.T.to(rank, non_blocking=True), target.T.to(rank, non_blocking=True)
        # data.bfloat16()
        # Ensure that the batch axis of data matches that of mem
        # st = time.time()
        mems = tuple(mem[:, :data.size(1), :].to(rank, non_blocking=True) for mem in mems)
        model.zero_grad()
        
        ret = model(data, target, *mems)
        loss, mems = ret[0], ret[1:]
        if args.mem_offload:
            mems = tuple(mem.to("cpu", non_blocking=True).pin_memory() for i, mem in enumerate(mems))
        # print(torch.cuda.memory_allocated()/2**30)
        loss = loss.float().mean().type_as(loss)
        loss.backward()
        # print("Total duration:", time.time()-st, torch.cuda.memory_allocated()/2**30, torch.cuda.max_memory_allocated()/2**30)
            
        ddp_loss[0] += loss.item()


        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()

        train_step = (epoch-1) * len(train_loader) + batch + 1
        train_step_ = train_step
        # step-wise learning rate annealing
        if train_step < args.warmup_step:
            curr_lr = args.lr * train_step / args.warmup_step
            optimizer.param_groups[0]['lr'] = curr_lr
        else:
            if args.scheduler == 'cosine':
                scheduler.step()

        
        if train_step % args.log_interval == 0:
            dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
            cur_loss = ddp_loss[0] / args.log_interval / world_size
            
        init_end_event.record()
        dist.barrier()
        if rank==lead_device and train_step % args.log_interval == 0:
            logging = create_exp_dir(args.work_dir)
            elapsed = init_start_event.elapsed_time(init_end_event)
            log_str = '| epoch {:3d} step {:>8d} | {:>6d} batches | lr {:.3g} ' \
                        '| ms/batch {:5.2f} | loss {:5.2f}'.format(
                epoch, train_step, batch+1, optimizer.param_groups[0]['lr'],
                elapsed / args.log_interval, cur_loss)
            log_str += ' | bpc {:9.5f}'.format(cur_loss / math.log(2))
            logging(log_str)
            log_obj = {
                "train_loss":cur_loss, 
                "train_step":train_step, 
                "iter_duration":(elapsed/1000)/args.log_interval,
                "throughput": (world_size * train_loader.batch_size)/(elapsed/ 1000 / args.log_interval),
                "stat_eff": abs(prev_loss - cur_loss)/args.log_interval,
                "samples_processed": train_step * train_loader.batch_size * world_size}
            log_obj.update({"goodput": log_obj["throughput"] * log_obj["stat_eff"]})
            wandb.log(log_obj)
            prev_loss = cur_loss
            
        if train_step % args.eval_interval == 0:
            evaluate(model, rank, world_size, valid_loader, train_step, optimizer, args)
            # Switch back to the training mode
            model.reset_length(args.tgt_len, args.ext_len, args.mem_len)
            model.train()
            
        if train_step % args.log_interval == 0:
            ddp_loss = torch.zeros(1).to(rank)
            init_start_event.record()
            
        if (cur_loss <= 1.0 or train_step >= args.max_step) and cur_loss!=0:
            sys.exit()
            
        print("Max mem:", torch.cuda.max_memory_allocated()/2**30)
        torch.cuda.empty_cache()


def evaluate(model, rank, world_size, test_loader, train_step, optimizer, args):
    global best_val_loss
    model.eval()
    ddp_loss = torch.zeros(2).to(rank)

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)
    
    init_start_event.record()
    if args.mem_len == 0:
        model.reset_length(args.eval_tgt_len,
            args.ext_len+args.tgt_len-args.eval_tgt_len, args.mem_len)
    else:
        model.reset_length(args.eval_tgt_len,
            args.ext_len, args.mem_len+args.tgt_len-args.eval_tgt_len)

    with torch.no_grad():
        mems = tuple()
        for i, (data, target) in enumerate(test_loader):
            data, target = data.T.to(rank, non_blocking=True), target.T.to(rank, non_blocking=True)
            if args.max_eval_steps > 0 and i >= args.max_eval_steps:
                break
            ret = model(data, target, *mems)
            loss, mems = ret[0], ret[1:]
            ddp_loss[0] += data.shape[0] * loss.mean().float().item()
            ddp_loss[1] += data.shape[0]

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM) 

    init_end_event.record()
    
    dist.barrier()
    val_loss = ddp_loss[0] / ddp_loss[1]
    if rank == lead_device:
        logging = create_exp_dir(args.work_dir)
        logging('-' * 100)
        elapsed = init_start_event.elapsed_time(init_end_event) / 1000
        log_str = '| Eval {:3d} at step {:>8d} | time: {:5.2f}s ' \
                    '| valid loss {:5.2f}'.format(
            train_step // args.eval_interval, train_step,
            elapsed, val_loss)

        log_str += ' | bpc {:9.5f}'.format(val_loss / math.log(2))
        logging(log_str)
        logging('-' * 100)
        
    # Checkpointing
    dist.barrier()
    if not args.debug and rank==lead_device and \
        not (best_val_loss or val_loss < best_val_loss):
        logging(f"Checkpointing model and optimizer to {args.work_dir}...")
        with open(os.path.join(args.work_dir, 'model.pt'), 'wb') as f:
            torch.save(model.state_dict(), f)
        with open(os.path.join(args.work_dir, 'optimizer.pt'), 'wb') as f:
            torch.save(optimizer.state_dict(), f)
        best_val_loss = val_loss

def fsdp_main(rank, world_size, args, corpus):
    wandb=None
    setup(rank, world_size, args.port)
    run_name=f"{args.strategy}_bs-{args.batch_size}_ws-{world_size}_bfp-{args.fp16}" \
            f"_wrap-{args.wrap}_chkpt-{args.chkpt}_offload-{args.mem_offload}".replace("True","Yes").replace("False","No")
    if rank==lead_device:
        wandb = init_wandb(
            run_name=run_name,
            run_cfg=args
        )
    
    train_dataset = corpus.get_dataset("train")
    valid_dataset = corpus.get_dataset("valid")
    test_dataset = corpus.get_dataset("test")

    per_dev_batch_size = args.batch_size//world_size
    per_dev_eval_batch_size = eval_batch_size//world_size
    
    train_sampler = CustomDistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=False, batch_size=per_dev_batch_size)
    valid_sampler = CustomDistributedSampler(valid_dataset, rank=rank, num_replicas=world_size, shuffle=False, batch_size=per_dev_eval_batch_size)
    test_sampler = CustomDistributedSampler(valid_dataset, rank=rank, num_replicas=world_size, shuffle=False, batch_size=per_dev_eval_batch_size)

    train_kwargs = {'batch_size': per_dev_batch_size, 'sampler': train_sampler}
    valid_kwargs = {'batch_size': per_dev_eval_batch_size, 'sampler': valid_sampler}
    
    cuda_kwargs = {'num_workers': 2,
                    'pin_memory': True,
                    'shuffle': False}
    train_kwargs.update(cuda_kwargs)
    valid_kwargs.update(cuda_kwargs)

    train_loader = DataLoader(train_dataset,**train_kwargs)
    valid_loader = DataLoader(valid_dataset, **valid_kwargs)
    
    torch.cuda.set_device(rank)


    model = init_model(args).to(rank)

    model = FSDP(
        model,
        device_id=torch.cuda.current_device(),
        sharding_strategy=args.strategy_obj,
        auto_wrap_policy=args.wrap_obj,
        mixed_precision=args.fp16_obj
    )
    
    if args.chkpt:
        non_reentrant_wrapper = functools.partial(
            checkpoint_wrapper,
            offload_to_cpu=False,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        check_fn = lambda submodule: isinstance(submodule, RelPartialLearnableDecoderLayer)
        apply_activation_checkpointing(
            model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
        )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
        args.max_step, eta_min=args.eta_min)
    
    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)
    
    for epoch in itertools.count(start=1):
        init_start_event.record()
        try:
            train(args, model, rank, world_size, train_loader, valid_loader, optimizer, scheduler, epoch, wandb, train_sampler)
        except SystemExit:
            print("Ending training...")
        dist.barrier()
        init_end_event.record()
        if rank==lead_device:
            wandb.log({"epoch": epoch, "epoch_duration": init_start_event.elapsed_time(init_end_event)/1000})
        if train_step_ >= args.max_step:
            logging = create_exp_dir(args.work_dir)
            logging('-' * 100)
            logging('End of training')
            break

    init_end_event.record()

    if rank == lead_device:
        logging = create_exp_dir(args.work_dir)
        logging(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")

    cleanup()
    
    
if __name__ == '__main__':
    args = arguments()
    logging = create_exp_dir(args.work_dir, scripts_to_save=['train.py', 'mem_transformer.py'], debug=args.debug)
    
    corpus = get_lm_corpus(args.data, args.dataset)
    ntokens = len(corpus.vocab)
    args.n_token = ntokens

    sharding_strategy_map = {
        "full_shard":ShardingStrategy.FULL_SHARD,
        "grad_op":ShardingStrategy.SHARD_GRAD_OP,
        "no_shard":ShardingStrategy.NO_SHARD,
        "hybrid":ShardingStrategy.HYBRID_SHARD,
        "hybrid_grad_op":ShardingStrategy._HYBRID_SHARD_ZERO2}
    
    args.strategy_obj = sharding_strategy_map[args.strategy]
    
    bfSixteen = MixedPrecision(
        # param_dtype=torch.bfloat16,
        # Gradient communication precision.
        reduce_dtype=torch.bfloat16,
        # Buffer precision.
        buffer_dtype=torch.bfloat16,
    )
    args.fp16_obj = bfSixteen if args.fp16 else None
    tx_xl_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={RelPartialLearnableDecoderLayer}
    )
    args.wrap_obj = tx_xl_auto_wrap_policy if args.wrap else None
    # Training settings
    torch.cuda.manual_seed_all(args.seed)
    train_step_ = 0
    # best_val_loss = None
    WORLD_SIZE = torch.cuda.device_count() if args.multi_gpu else 1

    try:
        mp.spawn(fsdp_main,
            args=(WORLD_SIZE, args, corpus),
            nprocs=WORLD_SIZE,
            join=True)
    except KeyboardInterrupt:
        logging('-' * 100)
        logging('Exiting from training early')
    