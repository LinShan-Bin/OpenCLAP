from functools import wraps

import numpy as np

import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, drop_path_rate = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        return self.drop_path(self.net(x))

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, drop_path_rate = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.drop_path(self.to_out(out))


class TimestepEmbed(nn.Module):
    def __init__(self, hidden_dim=64, dim=128):
        super().__init__()
        
        self.embedding_dim = hidden_dim
        e = torch.pow(2, torch.arange(self.embedding_dim // 2)).float() * np.pi
        self.register_buffer('basis', e)
        
        self.mlp = nn.Linear(self.embedding_dim + 1, dim)  # +1 for the original timestep value
        
    @staticmethod
    def embed(timesteps, basis):
        # timesteps: B x N
        # basis: embedding_dim // 2
        projections = timesteps[:, :, None] * basis
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings
    
    def forward(self, timesteps):
        # timesteps: B x N
        embed = self.embed(timesteps, self.basis)  # B x N x embedding_dim
        output = self.mlp(torch.cat([embed, timesteps.unsqueeze(-1)], dim=2))  # B x N x dim
        return output


class ActionEmbed(nn.Module):
    def __init__(self, max_action_dim=14, hidden_dim=64, dim=128):
        super().__init__()

        assert max_action_dim == 14

        self.embedding_dim = hidden_dim
        e = torch.pow(2, torch.arange(self.embedding_dim // 16)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(self.embedding_dim // 16 * 7)]),
            torch.cat([torch.zeros(self.embedding_dim // 16 * 1), e, torch.zeros(self.embedding_dim // 16 * 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 16 * 2), e, torch.zeros(self.embedding_dim // 16 * 5)]),
            torch.cat([torch.zeros(self.embedding_dim // 16 * 3), e, torch.zeros(self.embedding_dim // 16 * 4)]),
            torch.cat([torch.zeros(self.embedding_dim // 16 * 4), e, torch.zeros(self.embedding_dim // 16 * 3)]),
            torch.cat([torch.zeros(self.embedding_dim // 16 * 5), e, torch.zeros(self.embedding_dim // 16 * 2)]),
            torch.cat([torch.zeros(self.embedding_dim // 16 * 6), e, torch.zeros(self.embedding_dim // 16 * 1)]),
            torch.cat([torch.zeros(self.embedding_dim // 16 * 7), e]),
        ])
        self.register_buffer('basis', e)

        self.mlp = nn.Linear(self.embedding_dim + max_action_dim // 2, dim // 2)

    @staticmethod
    def embed(input, basis, timesteps):
        x = torch.cat([input, timesteps.unsqueeze(-1)], dim=2)
        projections = torch.einsum(
            'bnd,de->bne', x, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings
    
    def forward(self, input, timesteps):
        # input: B x N x max_action_dim
        l_input, r_input = input.chunk(2, dim=2)  # Dual-arm
        embed = torch.cat([
            self.mlp(torch.cat([self.embed(l_input, self.basis, timesteps), l_input], dim=2)),
            self.mlp(torch.cat([self.embed(r_input, self.basis, timesteps), r_input], dim=2)),
        ], dim=2) # B x N x C
        return embed


class ActionEncoder(nn.Module):
    def __init__(
        self,
        dim=128,
    ):
        super().__init__()

        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(dim, Attention(dim, dim, heads=1, dim_head=dim), context_dim=dim),
            PreNorm(dim, FeedForward(dim))
        ])
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, query, key):
        # query: B x num_query x dim
        # key: B x num_latents x dim

        cross_attn, cross_ff = self.cross_attend_blocks
        mask = None

        x = cross_attn(query, context=key, mask=mask) + query
        x = cross_ff(x) + x

        return self.out_norm(x)


class ActionDecoder(nn.Module):
    def __init__(
        self,
        depth=8,
        dim=128,
        output_dim=14,
        heads=4,
        dim_head=32,
    ):
        super().__init__()

        self.depth = depth

        get_latent_attn = lambda: PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, drop_path_rate=0))
        get_latent_ff = lambda: PreNorm(dim, FeedForward(dim, drop_path_rate=0))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {'_cache': True}

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

        self.decoder_cross_attn = PreNorm(dim, Attention(dim, dim, heads=1, dim_head=dim), context_dim=dim)
        self.decoder_ff = PreNorm(dim, FeedForward(dim))

        self.to_outputs = nn.Linear(dim, output_dim)

    def forward(self, x, queries):

        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        # cross attend from decoder queries to latents
        latents = self.decoder_cross_attn(queries, context=x)
        latents = latents + self.decoder_ff(latents)

        return self.to_outputs(latents)
