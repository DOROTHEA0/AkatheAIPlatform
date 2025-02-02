import math

import torch
import torch.nn as nn
from utils import precompute_freqs_cis, apply_rotary_emb, generate_att_mask, repeat_kv


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class MultiheadAttention(nn.Module):
    def __init__(self, hidden_dim, n_head, bias=False, dropout=0.1, max_seq_len=1024):
        super(MultiheadAttention, self).__init__()
        assert hidden_dim % n_head == 0, "hidden_dim must be divisible by n_head"
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.output = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('kv_cache', None)
        self.register_buffer('freqs_cis', precompute_freqs_cis(hidden_dim, max_seq_len * 2))

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
        scores = self.dropout(scores)
        v = scores @ v
        v = v.transpose(1, 2).contiguous().view(bsz, -1, self.hidden_dim)
        return self.output(v)


class GroupQueryAttention(nn.Module):
    def __init__(self, hidden_dim, q_head, n_group, bias=False, dropout=0.1, max_seq_len=1024):
        super(GroupQueryAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.q_head = q_head
        self.n_group = n_group
        self.kv_head = q_head // n_group
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim // n_group, bias=bias)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim // n_group, bias=bias)
        self.output = nn.Linear(hidden_dim, hidden_dim, bias=bias)

        self.register_buffer('kv_cache', None)
        self.register_buffer('freqs_cis', precompute_freqs_cis(hidden_dim, max_seq_len * 2))

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

        print(q.shape, k.shape, v.shape)

        q = q.view(bsz, -1, self.q_head, self.hidden_dim // self.q_head).transpose(1, 2)
        k = k.view(bsz, -1, self.q_head, self.hidden_dim // self.q_head).transpose(1, 2)
        v = v.view(bsz, -1, self.q_head, self.hidden_dim // self.q_head).transpose(1, 2)

        scores = q @ k.transpose(2, 3) / math.sqrt(self.hidden_dim)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))

        scores = torch.softmax(scores, dim=-1)
        #scores = self.dropout(scores)
        v = scores @ v
        v = v.transpose(1, 2).contiguous().view(bsz, -1, self.hidden_dim)
        return self.output(v)

class GroupLatentAttention(nn.Module):
    def __init__(self):
        super(GroupLatentAttention, self).__init__()

    def forward(self, x):
        return x



x = torch.randn(1, 4, 32)
m = GroupQueryAttention(hidden_dim=32, q_head=8, n_group=2)
print(m(x, mask=generate_att_mask(4, x.device)))
