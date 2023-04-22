import sys
import math
import functools

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.proj_adaptive_softmax import ProjectedAdaptiveLogSoftmax
from utils.log_uniform_sampler import LogUniformSampler, sample_logits


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb  #  size of the embedding vector

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))  # frequency of something
        self.register_buffer('inv_freq', inv_freq)  # what is this function?

    def forward(self, pos_seq, bsz=None):  # standard forward
        # pos_seq position of the token in the sequence
        # bsz - batch size
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)  # sinusoidal position embeddings
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)  # same shape for sequences in batch
        else:
            return pos_emb[:, None, :]


class PositionwiseFF(nn.Module):
    # The output of the Position-wise Feedforward Network for each position is then added to the output
    # of the multi-head attention module in a residual connection.
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False, **kwargs):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner, device=kwargs['device']),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model, device=kwargs['device']),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)

        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        # 1	To mitigate the internal covariate shift problem: The internal covariate shift occurs when the
        # distribution of the input to each layer changes during training. This can make it difficult for the model to
        # converge as each layer may be forced to adapt to the changing distribution of the input. Layer normalization
        # helps mitigate this problem by normalizing the activations of each layer to have zero mean and unit
        # variance, which reduces the dependence of each layer on the distribution of the input.
        #
        # 2	To improve generalization: Layer normalization has been shown to improve the generalization performance of
        # neural networks. By normalizing the activations of each layer, layer normalization reduces the sensitivity of
        # the model to the scale of the weights, which can improve the generalization performance of the model.

        if self.pre_lnorm:
            ##### layer normalization + positionwise feed-forward
            core_out = self.CoreNet(self.layer_norm(inp))

            ##### residual connection
            output = core_out + inp
        else:
            ##### positionwise feed-forward
            core_out = self.CoreNet(inp)
            core_out = core_out.to(inp.get_device())

            ##### residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output


class MultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 pre_lnorm=False):
        super(MultiHeadAttn, self).__init__()

        self.n_head = n_head  # number of heads of attention
        self.d_model = d_model  # dimensionality of the input embeddings
        self.d_head = d_head  # dimensionality of each attention head
        self.dropout = dropout  # probability of drop out

        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)  # all values of query
        # randomly initialized parameters?
        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)  # all values of key, value pair
        # randomly initialized parameters?

        self.drop = nn.Dropout(dropout)  # drop for feedforward networks
        self.dropatt = nn.Dropout(dropatt)  # drop for attention layer
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)
        # output net, outputs of different heads of attention

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)   # heuristic from the Attention is All you need paper

        self.pre_lnorm = pre_lnorm  # user flag for applying normalization before or after in position feed forward

    def forward(self, h, attn_mask=None, mems=None):
        # In the Transformer-XL architecture, "mem" refers to the memory mechanism that enables the model
        # to maintain a longer context than the input sequence length.

        ##### multihead attention
        # [hlen x bsz x n_head x d_head]

        if mems is not None:
            c = torch.cat([mems, h], 0)  # memory footprint
        else:
            c = h

        if self.pre_lnorm:
            ##### layer normalization
            c = self.layer_norm(c)

        head_q = self.q_net(h)  # weights for query
        head_k, head_v = torch.chunk(self.kv_net(c), 2, -1)  # weight for key and value
        # Attempts to split a tensor into the specified number of chunks. Each chunk is a view of the input tensor.

        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_k = head_k.view(c.size(0), c.size(1), self.n_head, self.d_head)
        head_v = head_v.view(c.size(0), c.size(1), self.n_head, self.d_head)

        # [qlen x klen x bsz x n_head]
        attn_score = torch.einsum('ibnd,jbnd->ijbn', (head_q, head_k))
        attn_score.mul_(self.scale)
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None, :, :, None], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:, :, :, None], -float('inf'))

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        # [qlen x klen x bsz x n_head] + [klen x bsz x n_head x d_head] -> [qlen x bsz x n_head x d_head]
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, head_v))
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = h + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        return output


class RelMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 tgt_len=None, ext_len=None, mem_len=None, pre_lnorm=False):
        super(RelMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def _parallelogram_mask(self, h, w, left=False):
        mask = torch.ones((h, w)).byte()
        m = min(h, w)
        mask[:m,:m] = torch.triu(mask[:m,:m])
        mask[-m:,-m:] = torch.tril(mask[-m:,-m:])

        if left:
            return mask
        else:
            return mask.flip(0)

    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen-1, x.size(2), x.size(3)),
                                    device=x.device, dtype=x.dtype)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:,:,None,None]) \
                    .view(qlen, klen, x.size(2), x.size(3))

        return x

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:,:,None,None]

        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError


class RelPartialLearnableMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__()
        
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False, device=kwargs['device'])

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False, device=kwargs['device'])

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(kwargs.get('dropatt', 0))
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False, device=kwargs['device'])

        self.layer_norm = nn.LayerNorm(d_model, device=kwargs['device'])

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = kwargs.get('pre_lnorm', False)

    def _parallelogram_mask(self, h, w, left=False):
        mask = torch.ones((h, w)).byte()
        m = min(h, w)
        mask[:m, :m] = torch.triu(mask[:m, :m])
        mask[-m:, -m:] = torch.tril(mask[-m:, -m:])

        if left:
            return mask
        else:
            return mask.flip(0)

    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen - 1, x.size(2), x.size(3)),
                                   device=x.device, dtype=x.dtype)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:, :, None, None]) \
            .view(qlen, klen, x.size(2), x.size(3))

        return x

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:, :, None, None]

        return x

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            w_heads = w_heads.to(w.get_device())
            r_head_k = self.r_net(r)
            r_head_k = r_head_k.to(w.get_device())

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)

            w_heads = w_heads.to(w.get_device())
            r_head_k = self.r_net(r)
            r_head_k = r_head_k.to(w.get_device())

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)  # qlen x n_head x d_head

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias  # qlen x bsz x n_head x d_head
        # print(rw_head_q.shape, w_head_k.shape)
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))  # qlen x klen x bsz x n_head
        # print(rw_head_q.shape, w_head_k.shape)

        rr_head_q = w_head_q + r_r_bias
        # print(rr_head_q.shape, r_head_k.shape)
        BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))  # qlen x klen x bsz x n_head
        BD = self._rel_shift(BD)
        # print(rr_head_q.shape, r_head_k.shape)

        # [qlen x klen x bsz x n_head]
        # print(AC.shape)
        # print(BD.shape)
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[None, :, :, None], -float('inf')).type_as(attn_score)
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[:, :, :, None], -float('inf')).type_as(attn_score)

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)
        attn_out = attn_out.to(w.get_device())

        if self.pre_lnorm:
            ##### residual connection
            output = w + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(w + attn_out)
        output = output.to(w.get_device())

        return output


class RelLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

    def forward(self, w, r_emb, r_w_bias, r_bias, attn_mask=None, mems=None):
        # r_emb: [klen, n_head, d_head], used for term B
        # r_w_bias: [n_head, d_head], used for term C
        # r_bias: [klen, n_head], used for term D

        qlen, bsz = w.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)

        if klen > r_emb.size(0):
            r_emb_pad = r_emb[0:1].expand(klen - r_emb.size(0), -1, -1)
            r_emb = torch.cat([r_emb_pad, r_emb], 0)
            r_bias_pad = r_bias[0:1].expand(klen - r_bias.size(0), -1)
            r_bias = torch.cat([r_bias_pad, r_bias], 0)
        else:
            r_emb = r_emb[-klen:]
            r_bias = r_bias[-klen:]

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias[None]  # qlen x bsz x n_head x d_head

        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))  # qlen x klen x bsz x n_head
        B_ = torch.einsum('ibnd,jnd->ijbn', (w_head_q, r_emb))  # qlen x klen x bsz x n_head
        D_ = r_bias[None, :, None]  # 1    x klen x 1   x n_head
        BD = self._rel_shift(B_ + D_)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None, :, :, None], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:, :, :, None], -float('inf'))

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = w + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output


class DecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super(DecoderLayer, self).__init__()

        self.dec_attn = MultiHeadAttn(n_head, d_model, d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout,
                                     pre_lnorm=kwargs.get('pre_lnorm'), **kwargs)

    def forward(self, dec_inp, dec_attn_mask=None, mems=None):
        output = self.dec_attn(dec_inp, attn_mask=dec_attn_mask,
                               mems=mems)
        output = self.pos_ff(output)

        return output


class RelLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout,
                 **kwargs):
        super(RelLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelLearnableMultiHeadAttn(n_head, d_model, d_head, dropout,
                                                  **kwargs).to(kwargs['device'])
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout,
                                     pre_lnorm=kwargs.get('pre_lnorm'), **kwargs)

    def forward(self, dec_inp, r_emb, r_w_bias, r_bias, dec_attn_mask=None, mems=None):

        output = self.dec_attn(dec_inp, r_emb, r_w_bias, r_bias,
                               attn_mask=dec_attn_mask,
                               mems=mems)
        output = self.pos_ff(output)

        return output


class RelPartialLearnableDecoderLayerFirst(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, index,
                cuda, **kwargs):
        super().__init__()
        self.dec_attn = RelPartialLearnableMultiHeadAttn(n_head, d_model,
                                                         d_head, dropout, **kwargs).to(cuda)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout,
                                      **kwargs)
        self.index = index

        self.cuda = cuda

    def initialize_additional_vars(self, pos_emb, dec_attn_mask, r_w_bias, r_r_bias, mem):
        self.pos_emb = pos_emb.to(self.cuda)
        self.dec_attn_mask = dec_attn_mask.to(self.cuda)
        self.r_w_bias = r_w_bias.to(self.cuda)
        self.r_r_bias = r_r_bias.to(self.cuda)
        self.mem = mem.to(self.cuda)

    def forward(self, data):
        output = self.dec_attn(data, 
            self.pos_emb, 
            self.r_w_bias,
            self.r_r_bias,
            self.dec_attn_mask,
            self.mem,
        )
        output = self.pos_ff(output)
        
        # print(output.shape)
        return output, torch.clone(output)


class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, index,
                cuda,
                 **kwargs):
        super().__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(n_head, d_model,
                                                         d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout,
                                     **kwargs)
        self.index = index

        

        self.cuda = cuda

    def initialize_additional_vars(self, pos_emb, dec_attn_mask, r_w_bias, r_r_bias, mem):
        self.pos_emb = pos_emb.to(self.cuda)
        self.dec_attn_mask = dec_attn_mask.to(self.cuda)
        self.r_w_bias = r_w_bias.to(self.cuda)
        self.r_r_bias = r_r_bias.to(self.cuda)
        self.mem = mem.to(self.cuda)

    def forward(self, data_and_skip):
        data, skip = data_and_skip
        # print(self.index)
        # print(
        #     data.get_device(), 
        #     self.pos_emb.get_device(),
        #     self.r_w_bias.get_device(),
        #     self.r_r_bias.get_device(),
        #     self.dec_attn_mask.get_device(),
        #     self.mem,)

        output = self.dec_attn(
            data, 
            self.pos_emb, 
            self.r_w_bias,
            self.r_r_bias,
            self.dec_attn_mask,
            self.mem,
        )
        # print(self.index)
        output = self.pos_ff(output)
        output = output.to(data.get_device())
        # print(output.shape)
        skip = torch.cat((skip, output), 0)
        # print('skip connection')
        # print(self.index)

        return output, skip


class AdaptiveEmbedding(nn.Module):
    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1, 
                 sample_softmax=False):
        super(AdaptiveEmbedding, self).__init__()

        self.n_token = n_token
        self.d_embed = d_embed

        self.cutoffs = cutoffs + [n_token]
        self.div_val = div_val
        self.d_proj = d_proj

        self.emb_scale = d_proj ** 0.5

        self.cutoff_ends = [0] + self.cutoffs

        self.emb_layers = nn.ModuleList()
        self.emb_projs = nn.ParameterList()
        # print('###############')
        # print(n_token)
        # print(d_embed)
        # print(sample_softmax)
        # print(d_proj != d_embed)
        # print(d_proj)
        # print(d_embed)
        if div_val == 1:
            self.emb_layers.append(
                nn.Embedding(n_token, d_embed, sparse=sample_softmax>0)
            )
            # print(self.params)
            if d_proj != d_embed:
                self.emb_projs.append(nn.Parameter(torch.Tensor(d_proj, d_embed)))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i+1]
                d_emb_i = d_embed // (div_val ** i)
                self.emb_layers.append(nn.Embedding(r_idx-l_idx, d_emb_i))
                self.emb_projs.append(nn.Parameter(torch.Tensor(d_proj, d_emb_i)))

    def forward(self, inp):
        # print(list(self.emb_layers[0].parameters()))
        # print(self.params)
        
        if self.div_val == 1:
            embed = self.emb_layers[0](inp)
            if self.d_proj != self.d_embed:
                embed  = F.linear(embed, self.emb_projs[0])
        else:
            param = next(self.parameters())
            inp_flat = inp.view(-1)
            emb_flat = torch.zeros([inp_flat.size(0), self.d_proj], 
                dtype=param.dtype, device=param.device)
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]

                mask_i = (inp_flat >= l_idx) & (inp_flat < r_idx)
                indices_i = mask_i.nonzero().squeeze()

                if indices_i.numel() == 0:
                    continue

                inp_i = inp_flat.index_select(0, indices_i) - l_idx
                emb_i = self.emb_layers[i](inp_i)
                emb_i = F.linear(emb_i, self.emb_projs[i])

                emb_flat.index_copy_(0, indices_i, emb_i)

            embed = emb_flat.view(*inp.size(), self.d_proj)

        # print(embed)
        # a = 0 - None
        embed.mul_(self.emb_scale)

        return embed
    


class MemTransformerLM(nn.Module):
    def __init__(self, n_token, n_layer, n_head, d_model, d_head, d_inner,
                 dropout, dropatt, tie_weight=True, d_embed=None,
                 div_val=1, tie_projs=[False], pre_lnorm=False,
                 tgt_len=None, ext_len=None, mem_len=None,
                 cutoffs=[], adapt_inp=False,
                 same_length=False, attn_type=0, clamp_len=-1,
                 sample_softmax=-1, pipeline_devices=4):
        super(MemTransformerLM, self).__init__()
        self.n_token = n_token

        d_embed = d_model if d_embed is None else d_embed  # user specified embedding size
        self.d_embed = d_embed
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.dropout = dropout
        self.d_inner = d_inner
        self.dropatt = dropatt
        self.pre_lnorm = pre_lnorm

        self.word_emb = AdaptiveEmbedding(n_token, d_embed, d_model, cutoffs, 
                                          div_val=div_val)

        self.drop = nn.Dropout(dropout)
        

        self.n_layer = n_layer

        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len
        self.max_klen = tgt_len + ext_len + mem_len

        self.attn_type = attn_type

        self.pipeline_devices = pipeline_devices
        self.layers = []
        self._create_params()
        if self.attn_type == 0:  # the default attention
            # print('0 layer')
            self.layers.append(RelPartialLearnableDecoderLayerFirst(
                        self.n_head, self.d_model, self.d_head, self.d_inner, self.dropout,
                        tgt_len=self.tgt_len, ext_len=self.ext_len, mem_len=self.mem_len,
                        dropatt=self.dropatt, pre_lnorm=self.pre_lnorm, index=0,
                        cuda='cuda:0', device='cuda:0'
                        ).to('cuda:0')
                    )
            for i in range(1, self.n_layer):
                if self.pipeline_devices == 2:
                    #print(f'{i} layer')
                    temp_cuda = 'cuda:0'
                    device2 = None
                    if i == self.n_layer // 2 - 1:
                        device2 = 'cuda:1'
                    else:
                        device2 = None

                    if i >= self.n_layer // 2:
                        temp_cuda = 'cuda:1'
                        self.layers.append(
                            RelPartialLearnableDecoderLayer(
                                self.n_head, self.d_model, self.d_head, self.d_inner, self.dropout,
                                tgt_len=self.tgt_len, ext_len=self.ext_len, mem_len=self.mem_len,
                                dropatt=self.dropatt, pre_lnorm=self.pre_lnorm, index=i, 
                                cuda=temp_cuda, device=temp_cuda
                            ).to(temp_cuda)
                        )
                elif self.pipeline_devices == 3:
                    temp_cuda1 = 'cuda:0'
                    temp_cuda2 = 'cuda:1'
                    temp_cuda3 = 'cuda:2'
                    if i < self.n_layer // 3:

                        self.layers.append(
                            RelPartialLearnableDecoderLayer(
                                self.n_head, self.d_model, self.d_head, self.d_inner, self.dropout,
                                tgt_len=self.tgt_len, ext_len=self.ext_len, mem_len=self.mem_len,
                                dropatt=self.dropatt, pre_lnorm=self.pre_lnorm, index=i,
                                cuda=temp_cuda1, device=temp_cuda1
                            ).to(temp_cuda1)
                        )
                    elif self.n_layer // 3 <= i < 2 * self.n_layer // 3:
                        self.layers.append(
                            RelPartialLearnableDecoderLayer(
                                self.n_head, self.d_model, self.d_head, self.d_inner, self.dropout,
                                tgt_len=self.tgt_len, ext_len=self.ext_len, mem_len=self.mem_len,
                                dropatt=self.dropatt, pre_lnorm=self.pre_lnorm, index=i,
                                cuda=temp_cuda2, device=temp_cuda2
                            ).to(temp_cuda2)
                        )
                    else:
                        self.layers.append(
                            RelPartialLearnableDecoderLayer(
                                self.n_head, self.d_model, self.d_head, self.d_inner, self.dropout,
                                tgt_len=self.tgt_len, ext_len=self.ext_len, mem_len=self.mem_len,
                                dropatt=self.dropatt, pre_lnorm=self.pre_lnorm, index=i,
                                cuda=temp_cuda3, device=temp_cuda3
                            ).to(temp_cuda3)
                        )
                else:
                    temp_cuda1 = 'cuda:0'
                    temp_cuda2 = 'cuda:1'
                    temp_cuda3 = 'cuda:2'
                    temp_cuda4 = 'cuda:3'
                    if i < self.n_layer // 4:

                        self.layers.append(
                            RelPartialLearnableDecoderLayer(
                                self.n_head, self.d_model, self.d_head, self.d_inner, self.dropout,
                                tgt_len=self.tgt_len, ext_len=self.ext_len, mem_len=self.mem_len,
                                dropatt=self.dropatt, pre_lnorm=self.pre_lnorm, index=i,
                                cuda=temp_cuda1, device=temp_cuda1
                            ).to(temp_cuda1)
                        )
                    elif self.n_layer // 4 <= i < 2 * self.n_layer // 4:
                        self.layers.append(
                            RelPartialLearnableDecoderLayer(
                                self.n_head, self.d_model, self.d_head, self.d_inner, self.dropout,
                                tgt_len=self.tgt_len, ext_len=self.ext_len, mem_len=self.mem_len,
                                dropatt=self.dropatt, pre_lnorm=self.pre_lnorm, index=i,
                                cuda=temp_cuda2, device=temp_cuda2
                            ).to(temp_cuda2)
                        )
                    elif 2 * self.n_layer // 4 <= i < 3 * self.n_layer // 4:
                        self.layers.append(
                            RelPartialLearnableDecoderLayer(
                                self.n_head, self.d_model, self.d_head, self.d_inner, self.dropout,
                                tgt_len=self.tgt_len, ext_len=self.ext_len, mem_len=self.mem_len,
                                dropatt=self.dropatt, pre_lnorm=self.pre_lnorm, index=i,
                                cuda=temp_cuda3, device=temp_cuda3
                            ).to(temp_cuda3)
                        )
                    else:
                        self.layers.append(
                            RelPartialLearnableDecoderLayer(
                                self.n_head, self.d_model, self.d_head, self.d_inner, self.dropout,
                                tgt_len=self.tgt_len, ext_len=self.ext_len, mem_len=self.mem_len,
                                dropatt=self.dropatt, pre_lnorm=self.pre_lnorm, index=i,
                                cuda=temp_cuda4, device=temp_cuda4
                            ).to(temp_cuda4)
                        )

        elif self.attn_type == 1:  # learnable embeddings
            for i in range(self.n_layer):
                self.layers.append(
                    RelLearnableDecoderLayer(
                        self.n_head, self.d_model, self.d_head, self.d_inner, self.dropout,
                        tgt_len=self.tgt_len, ext_len=self.ext_len, mem_len=self.mem_len,
                        dropatt=self.dropatt, pre_lnorm=self.pre_lnorm,
                    )
                )
        elif self.attn_type in [2, 3]:  # absolute embeddings
            for i in range(self.n_layer):
                self.layers.append(
                    DecoderLayer(
                        self.n_head, self.d_model, self.d_head, self.d_inner, self.dropout,
                        dropatt=self.dropatt, pre_lnorm=self.pre_lnorm,
                    )
                )

        
        from torchgpipe import GPipe
        self.nn_sequential_model = nn.Sequential(*self.layers)

        if self.pipeline_devices == 2:
            self.gpipe_model = GPipe(
                self.nn_sequential_model,
                balance=[self.n_layer // 2, self.n_layer - self.n_layer//2],
                devices=[1, 2],
                checkpoint='never',
                chunks=1
            )
        elif self.pipeline_devices == 3:
            self.gpipe_model = GPipe(
                self.nn_sequential_model,
                balance=[self.n_layer // 3, self.n_layer // 3, self.n_layer - 2 * (self.n_layer // 3)],
                devices=[0, 1, 2],
                checkpoint='never',
                chunks=1
            )
        else:
            self.gpipe_model = GPipe(
                self.nn_sequential_model,
                balance=[self.n_layer // 4, self.n_layer // 4, self.n_layer // 4, self.n_layer - 3 * (self.n_layer // 4)],
                devices=[0, 1, 2, 3],
                checkpoint='never',
                chunks=1
            )
        #####

        self.sample_softmax = sample_softmax
        # use sampled softmax
        if sample_softmax > 0:
            self.out_layer = nn.Linear(d_model, n_token)
            if tie_weight:
                self.out_layer.weight = self.word_emb.weight
            self.tie_weight = tie_weight
            self.sampler = LogUniformSampler(n_token, sample_softmax)

        # use adaptive softmax (including standard softmax)
        else:
            self.crit = ProjectedAdaptiveLogSoftmax(n_token, d_embed, d_model,
                                                    cutoffs, div_val=div_val)

            if tie_weight:
                for i in range(len(self.crit.out_layers)):
                    self.crit.out_layers[i].weight = self.word_emb.emb_layers[i].weight

            if tie_projs:
                for i, tie_proj in enumerate(tie_projs):
                    if tie_proj and div_val == 1 and d_model != d_embed:
                        self.crit.out_projs[i] = self.word_emb.emb_projs[0]
                    elif tie_proj and div_val != 1:
                        self.crit.out_projs[i] = self.word_emb.emb_projs[i]

        self.same_length = same_length
        self.clamp_len = clamp_len


    def backward_compatible(self):
        self.sample_softmax = -1

    def _create_params(self):
        if self.attn_type == 0:  # default attention
            self.pos_emb = PositionalEmbedding(self.d_model)
            self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
            self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        elif self.attn_type == 1:  # learnable
            self.r_emb = nn.Parameter(torch.Tensor(
                self.n_layer, self.max_klen, self.n_head, self.d_head))
            self.r_w_bias = nn.Parameter(torch.Tensor(
                self.n_layer, self.n_head, self.d_head))
            self.r_bias = nn.Parameter(torch.Tensor(
                self.n_layer, self.max_klen, self.n_head))
        elif self.attn_type == 2:  # absolute standard
            self.pos_emb = PositionalEmbedding(self.d_model)
        elif self.attn_type == 3:  # absolute deeper SA
            self.r_emb = nn.Parameter(torch.Tensor(
                self.n_layer, self.max_klen, self.n_head, self.d_head))

    def reset_length(self, tgt_len, ext_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

    def init_mems(self):
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for i in range(self.n_layer + 1):
                empty = torch.empty(0, dtype=param.dtype, device=param.device)
                mems.append(empty)

            return mems
        else:
            return None

    def _update_mems(self, hids, mems, qlen, mlen):
        # does not deal with None
        if mems is None: return None

        # mems is not None
        # print(len(hids))
        # print(len(mems))
        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen - 0 - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):
                # print(mems[i])
                # print(hids[i])
                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())

        return new_mems

    def _forward(self, dec_inp, mems=None):
        qlen, bsz = dec_inp.size()

        word_emb = self.word_emb(dec_inp)

        mlen = mems[0].size(0) if mems is not None else 0
        # print(mlen)
        klen = mlen + qlen
        if self.same_length:
            all_ones = word_emb.new_ones(qlen, klen)
            mask_len = klen - self.mem_len
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (torch.triu(all_ones, 1 + mlen)
                             + torch.tril(all_ones, -mask_shift_len)).byte()[:, :, None]  # -1
        else:
            dec_attn_mask = torch.triu(
                word_emb.new_ones(qlen, klen), diagonal=1 + mlen).byte()[:, :, None]
        hids = []
        if self.attn_type == 0:  # default
            pos_seq = torch.arange(klen - 1, -1, -1.0, device=word_emb.device,
                                   dtype=word_emb.dtype)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)  # positional encoding

            core_out = self.drop(word_emb)
            pos_emb = self.drop(pos_emb)  # drop out

            #########################################################################################################
            #########################################################################################################
            hids.append(core_out)

            for i, layer in enumerate(self.layers):
                layer.initialize_additional_vars(
                    torch.clone(pos_emb), 
                    torch.clone(dec_attn_mask), 
                    torch.clone(self.r_w_bias), 
                    torch.clone(self.r_r_bias), 
                    torch.clone(mems[i])
                )
            

            # RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:1 and cuda:2! 
            # (when checking argument for argument mat2 in method wrapper_CUDA_mm


            # in the above error i have concluded that the model tries to access data that is located in the gpu 1
            # but model is the gpu2. 
            #########################################################################################################
            #########################################################################################################


            first_core_out = core_out
            # for i, layer in enumerate(self.layers):
            #     mems_i = None if mems is None else mems[i]
            #     core_outs.append(core_out)
            #     core_out = layer(core_out, pos_emb, self.r_w_bias,
            #                      self.r_r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
            #     hids.append(core_out)

            #########################################################################################################
            # input_ = first_core_out.to(in_device, non_blocking=True)
            core_out, hidden_states = self.gpipe_model(core_out)
            hidden_states = hidden_states.view(self.n_layer, *first_core_out.shape)
            hids = hids + [hid_state.to('cuda:0') for hid_state in hidden_states]

            #########################################################################################################

        elif self.attn_type == 1:  # learnable
            core_out = self.drop(word_emb)
            hids.append(core_out)
            for i, layer in enumerate(layers):
                if self.clamp_len > 0:
                    r_emb = self.r_emb[i][-self.clamp_len:]
                    r_bias = self.r_bias[i][-self.clamp_len:]
                else:
                    r_emb, r_bias = self.r_emb[i], self.r_bias[i]

                mems_i = None if mems is None else mems[i]
                core_out = layer(core_out, r_emb, self.r_w_bias[i],
                                 r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
                hids.append(core_out)
        elif self.attn_type == 2:  # absolute
            pos_seq = torch.arange(klen - 1, -1, -1.0, device=word_emb.device,
                                   dtype=word_emb.dtype)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)

            core_out = self.drop(word_emb + pos_emb[-qlen:])

            hids.append(core_out)
            for i, layer in enumerate(layers):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and i == 0:
                    mems_i += pos_emb[:mlen]
                core_out = layer(core_out, dec_attn_mask=dec_attn_mask,
                                 mems=mems_i)
                hids.append(core_out)
        elif self.attn_type == 3:
            core_out = self.drop(word_emb)

            hids.append(core_out)
            for i, layer in enumerate(layers):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and mlen > 0:
                    cur_emb = self.r_emb[i][:-qlen]
                    cur_size = cur_emb.size(0)
                    if cur_size < mlen:
                        cur_emb_pad = cur_emb[0:1].expand(mlen - cur_size, -1, -1)
                        cur_emb = torch.cat([cur_emb_pad, cur_emb], 0)
                    else:
                        cur_emb = cur_emb[-mlen:]
                    mems_i += cur_emb.view(mlen, 1, -1)
                core_out += self.r_emb[i][-qlen:].view(qlen, 1, -1)

                core_out = layer(core_out, dec_attn_mask=dec_attn_mask,
                                 mems=mems_i)
                hids.append(core_out)

        core_out = self.drop(core_out)

        new_mems = self._update_mems(hids, mems, mlen, qlen)

        return core_out, new_mems

    def forward(self, data, target, *mems):
        # nn.DataParallel does not allow size(0) tensors to be broadcasted.
        # So, have to initialize size(0) mems inside the model forward.
        # Moreover, have to return new_mems to allow nn.DataParallel to piece
        # them together.
        if not mems: mems = self.init_mems()

        tgt_len = target.size(0)
        hidden, new_mems = self._forward(data, mems=mems)
        hidden = hidden.to('cuda:0')

        pred_hid = hidden[-tgt_len:]
        if self.sample_softmax > 0 and self.training:
            assert self.tie_weight
            logit = sample_logits(self.word_emb,
                                  self.out_layer.bias, target, pred_hid, self.sampler)
            loss = -F.log_softmax(logit, -1)[:, :, 0]
        else:
            loss = self.crit(pred_hid.view(-1, pred_hid.size(-1)), target.view(-1))
            loss = loss.view(tgt_len, -1)

        if new_mems is None:
            return [loss]
        else:
            return [loss] + new_mems


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='unit test')

    parser.add_argument('--n_layer', type=int, default=4, help='')
    parser.add_argument('--n_rel_layer', type=int, default=4, help='')
    parser.add_argument('--n_head', type=int, default=2, help='')
    parser.add_argument('--d_head', type=int, default=2, help='')
    parser.add_argument('--d_model', type=int, default=200, help='')
    parser.add_argument('--d_embed', type=int, default=200, help='')
    parser.add_argument('--d_inner', type=int, default=200, help='')
    parser.add_argument('--dropout', type=float, default=0.0, help='')
    parser.add_argument('--cuda', action='store_true', help='')
    parser.add_argument('--seed', type=int, default=1111, help='')
    parser.add_argument('--multi_gpu', action='store_true', help='')

    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")

    B = 4
    tgt_len, mem_len, ext_len = 36, 36, 0
    data_len = tgt_len * 20
    args.n_token = 10000

    import data_utils

    data = torch.LongTensor(data_len * B).random_(0, args.n_token).to(device)  # generates random 4 * 20 * 36 elements
    # random_(0, args.n_token) - ranging from 0 to 1000
    diter = data_utils.LMOrderedIterator(data, B, tgt_len, device=device, ext_len=ext_len)

    cutoffs = [args.n_token // 2]
    tie_projs = [False] + [True] * len(cutoffs)

    for div_val in [1, 2]:
        for d_embed in [200, 100]:
            model = MemTransformerLM(args.n_token, args.n_layer, args.n_head,
                                     args.d_model, args.d_head, args.d_inner, args.dropout,
                                     dropatt=args.dropout, tie_weight=True,
                                     d_embed=d_embed, div_val=div_val,
                                     tie_projs=tie_projs, pre_lnorm=True,
                                     tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                                     cutoffs=cutoffs, attn_type=0).to(device)

            print(sum(p.numel() for p in model.parameters()))

            mems = tuple()
            for idx, (inp, tgt, seqlen) in enumerate(diter):
                print('batch {}'.format(idx))
                out = model(inp, tgt, *mems)
                mems = out[1:]
