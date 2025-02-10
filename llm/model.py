import math

import torch
import torch.nn as nn
from llm.utils import precompute_freqs_cis, apply_rotary_emb, repeat_kv
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super(RMSNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)


class MultiheadAttention(nn.Module):
    def __init__(self, config):
        super(MultiheadAttention, self).__init__()
        assert config.embed_dim % config.n_head == 0, "hidden_dim must be divisible by n_head"
        self.hidden_dim = config.embed_dim
        self.n_head = config.n_head
        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=config.att_bias)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=config.att_bias)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=config.att_bias)
        self.output = nn.Linear(self.hidden_dim, self.hidden_dim, bias=config.att_bias)
        self.dropout = nn.Dropout(p=config.att_dropout)

        self.register_buffer('kv_cache', None)
        self.register_buffer('freqs_cis', precompute_freqs_cis(self.hidden_dim, config.max_seq_len * 2))

    def reset_cache(self):
        self.kv_cache = None

    def forward(self, x, mask=None, use_cache=False):
        bsz = x.size(0)
        if use_cache and self.kv_cache is not None:
            x = x[:, -1:, :]

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if use_cache and self.kv_cache is not None:
            start_pos = self.kv_cache.size(1)
            end_pos = start_pos + q.size(1)
        else:
            start_pos = 0
            end_pos = q.size(1)
        freqs_cis = self.freqs_cis[start_pos: end_pos].to(x.device)
        q, k = apply_rotary_emb(q, k, freqs_cis)

        if use_cache:
            if self.kv_cache is not None:
                k_cache, v_cache = torch.chunk(self.kv_cache, 2)
                k = torch.cat((k_cache, k), dim=1)
                v = torch.cat((v_cache, v), dim=1)
            self.kv_cache = torch.cat((k, v))

        q = q.view(bsz, -1, self.n_head, self.hidden_dim // self.n_head).transpose(1, 2)
        k = k.view(bsz, -1, self.n_head, self.hidden_dim // self.n_head).transpose(1, 2)
        v = v.view(bsz, -1, self.n_head, self.hidden_dim // self.n_head).transpose(1, 2)

        scores = q @ k.transpose(2, 3) / math.sqrt(self.hidden_dim)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))

        scores = torch.softmax(scores, dim=-1)
        v = scores @ v
        v = v.transpose(1, 2).contiguous().view(bsz, -1, self.hidden_dim)
        return self.dropout(self.output(v))


class GroupQueryAttention(nn.Module):
    def __init__(self, config):
        super(GroupQueryAttention, self).__init__()
        self.hidden_dim = config.embed_dim
        self.q_head = config.n_head
        self.n_group = config.n_group
        self.kv_head = self.q_head // self.n_group
        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=config.att_bias)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim // self.n_group, bias=config.att_bias)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim // self.n_group, bias=config.att_bias)
        self.output = nn.Linear(self.hidden_dim, self.hidden_dim, bias=config.att_bias)
        self.dropout = nn.Dropout(p=config.att_dropout)

        self.register_buffer('kv_cache', None)
        self.register_buffer('freqs_cis', precompute_freqs_cis(self.hidden_dim, config.max_seq_len * 2))

    def forward(self, x, mask=None, use_cache=False):
        bsz = x.size(0)
        if use_cache and self.kv_cache is not None:
            x = x[:, -1:, :]

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if use_cache:
            if self.kv_cache is not None:
                k_cache, v_cache = torch.chunk(self.kv_cache, 2)
                k = torch.cat((k_cache, k), dim=1)
                v = torch.cat((v_cache, v), dim=1)
            self.kv_cache = torch.cat((k, v))

        k = repeat_kv(k.reshape(k.size(0), k.size(1), self.kv_head, k.size(2) // self.kv_head), self.n_group)
        v = repeat_kv(v.reshape(v.size(0), v.size(1), self.kv_head, v.size(2) // self.kv_head), self.n_group)

        if use_cache and self.kv_cache is not None:
            start_pos = self.kv_cache.size(1)
            end_pos = start_pos + q.size(1)
        else:
            start_pos = 0
            end_pos = q.size(1)

        freqs_cis = self.freqs_cis[start_pos: end_pos].to(x.device)
        q, k = apply_rotary_emb(q, k, freqs_cis)

        q = q.view(bsz, -1, self.q_head, self.hidden_dim // self.q_head).transpose(1, 2)
        k = k.view(bsz, -1, self.q_head, self.hidden_dim // self.q_head).transpose(1, 2)
        v = v.view(bsz, -1, self.q_head, self.hidden_dim // self.q_head).transpose(1, 2)

        scores = q @ k.transpose(2, 3) / math.sqrt(self.hidden_dim)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))

        scores = torch.softmax(scores, dim=-1)
        v = scores @ v
        v = v.transpose(1, 2).contiguous().view(bsz, -1, self.hidden_dim)
        return self.dropout(self.output(v))

class GroupLatentAttention(nn.Module):
    def __init__(self):
        super(GroupLatentAttention, self).__init__()

    def forward(self, x):
        return x


class FeedForward(nn.Module):
    def __init__(self, config):
        super(FeedForward, self).__init__()
        self.in_feature = config.embed_dim
        self.hidden_dim = config.ffn_hidden
        self.up_proj1 = nn.Linear(self.in_feature, self.hidden_dim)
        self.up_proj2 = nn.Linear(self.in_feature, self.hidden_dim)
        self.down_proj = nn.Linear(self.hidden_dim, self.in_feature)
        self.dropout = nn.Dropout(p=config.ffn_dropout)

    def forward(self, x):
        x = self.down_proj(F.silu(self.up_proj1(x)) * self.up_proj2(x))
        return self.dropout(x)

class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.att_layer = config.att_layer(config)
        self.dense_layer = config.ffn_layer(config)
        self.att_norm = RMSNorm(dim=config.embed_dim)
        self.dense_norm = RMSNorm(dim=config.embed_dim)

    def forward(self, x, mask=None, use_cache=False):
        x = x + self.att_layer(self.att_norm(x), mask, use_cache)
        return x + self.dense_layer(self.dense_norm(x))

class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.word_embedding = nn.Embedding(num_embeddings=config.n_vocabs, embedding_dim=config.embed_dim)
        self.layers = nn.ModuleList()
        for _ in range(config.n_layers):
            self.layers.append(Block(config))
        self.output_layer = nn.Linear(in_features=config.embed_dim, out_features=config.n_vocabs)

    def forward(self, x, mask=None, use_cache=False, y=None):
        x = self.word_embedding(x)
        for layer in self.layers:
            x = layer(x, mask=mask, use_cache=use_cache)
        logits = self.output_layer(x)
        if y is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),  # 展平前两维
                y.view(-1),  # 展平成1D向量
                #ignore_index=-100  # 可选：忽略padding位置（若标签用-100填充）
            )
        else:
            loss = None

        return logits, loss