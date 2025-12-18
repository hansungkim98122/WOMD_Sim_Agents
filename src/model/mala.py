import math
import torch
import torch.nn as nn
from timm.models.layers import DropPath
from einops import rearrange
# Import MessagePassing to satisfy inheritance request, 
# though we won't use its propagate mechanism for MALA.
from torch_geometric.nn.conv import MessagePassing 

# ... (Swish, RoPE1D, MultiHeadMALAAttention, MALA_FFN, MALA_Block remain mostly same) ...

# Copying helper classes for completeness
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(nn.Module):
    def forward(self, x): return SwishImplementation.apply(x)

def rotate_every_two(x):
    x1 = x[..., ::2]; x2 = x[..., 1::2]
    x = torch.stack([-x2, x1], dim=-1)
    return x.flatten(-2)

def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)

class RoPE1D(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)
    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.sin()[None, None, :, :], emb.cos()[None, None, :, :]

class MultiHeadMALAAttention(nn.Module):
    def __init__(self, dim, num_heads, head_dim=None, dropout=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.internal_dim = self.head_dim * self.num_heads
        
        self.qkvo = nn.Linear(dim, self.internal_dim * 4)
        self.lepe = nn.Conv1d(self.internal_dim, self.internal_dim, 3, 1, 1, groups=self.internal_dim)
        self.proj = nn.Linear(self.internal_dim, dim)
        self.elu = nn.ELU()
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x, sin, cos):
        B, N, C = x.shape
        qkvo = self.qkvo(x)
        q, k, v, o = qkvo.chunk(4, dim=-1)
        
        v_t = v.transpose(1, 2)
        lepe_out = self.lepe(v_t).transpose(1, 2)
        
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        
        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0
        
        z = q @ k.mean(dim=2, keepdim=True).transpose(-2, -1) * self.scale
        q = theta_shift(q, sin, cos)
        k = theta_shift(k, sin, cos)
        
        kv_state = (k.transpose(-2, -1) * (self.scale / N)**0.5) @ (v * (self.scale / N)**0.5)
        attn_out = q @ kv_state
        
        res = attn_out * (1 + 1/(z + 1e-6)) - z * v.mean(dim=2, keepdim=True)
        res = rearrange(res, 'b h n d -> b n (h d)')
        res = res + lepe_out
        res = self.attn_drop(res)
        return self.proj(res * o)

class MALA_FFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim), nn.Dropout(dropout)
        )
    def forward(self, x): return self.net(x)

class MALA_Block(nn.Module):
    def __init__(self, dim, num_heads, head_dim=None, mlp_ratio=4., drop_path=0., dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadMALAAttention(dim, num_heads, head_dim=head_dim, dropout=dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = MALA_FFN(dim, int(dim * mlp_ratio), dropout=dropout)
        self.pos_block = nn.Conv1d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x, sin, cos):
        x = x + self.pos_block(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.drop_path(self.attn(self.norm1(x), sin, cos))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x

# src/model/mala.py
import torch
import torch.nn as nn

try:
    from torch_scatter import scatter_add
except ImportError:
    scatter_add = None

def _phi(x: torch.Tensor) -> torch.Tensor:
    # Positive feature map for linear attention
    return torch.nn.functional.elu(x) + 1.0

class MotionMALA(nn.Module):
    """
    Graph-aware linear attention drop-in replacement for AttentionLayer.

    Signature matches:
      forward(x, r, edge_index) where
        x: Tensor [N, D] or tuple (x_src [Ns,D], x_dst [Nd,D]) if bipartite
        r: Tensor [E, D] (already embedded to hidden_dim in your code)
        edge_index: LongTensor [2, E] with (src, dst)

    Respects edge_index (so temporal causality from edge filtering is preserved).
    Uses r by adding projections into K/V per edge (like SMART's AttentionLayer).
    """
    def __init__(self,
                 hidden_dim: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 bipartite: bool = False,
                 has_pos_emb: bool = True,
                 eps: float = 1e-6,
                 **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = nn.Dropout(dropout)
        self.bipartite = bipartite
        self.has_pos_emb = has_pos_emb
        self.eps = eps

        self.attn_prenorm_x_src = nn.LayerNorm(hidden_dim)
        self.attn_prenorm_x_dst = nn.LayerNorm(hidden_dim) if bipartite else self.attn_prenorm_x_src
        self.attn_prenorm_r = nn.LayerNorm(hidden_dim) if has_pos_emb else None

        self.to_q = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.to_k = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.to_v = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)

        if has_pos_emb:
            self.to_k_r = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
            self.to_v_r = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)

        self.to_out = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)

        # same gating style as AttentionLayer
        self.to_g = nn.Linear(2 * hidden_dim, hidden_dim)
        self.to_s = nn.Linear(hidden_dim, hidden_dim)

        self.attn_postnorm = nn.LayerNorm(hidden_dim)
        self.ff_prenorm = nn.LayerNorm(hidden_dim)
        self.ff_postnorm = nn.LayerNorm(hidden_dim)

        self.ff_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, r, edge_index):
        if isinstance(x, torch.Tensor):
            x_src = x_dst = self.attn_prenorm_x_src(x)
            x_res = x
        else:
            x_src, x_dst = x
            x_src = self.attn_prenorm_x_src(x_src)
            x_dst = self.attn_prenorm_x_dst(x_dst)
            x_res = x[1]  # residual is dst

        if self.has_pos_emb and r is not None:
            r = self.attn_prenorm_r(r)

        attn_out = self._attn_block(x_src, x_dst, r, edge_index)  # [Nd, D]
        x_mid = x_res + self.attn_postnorm(attn_out)
        x_out = x_mid + self.ff_postnorm(self.ff_mlp(self.ff_prenorm(x_mid)))
        return x_out

    def _attn_block(self, x_src, x_dst, r, edge_index):
        if scatter_add is None:
            raise ImportError("torch_scatter is required for MotionMALA graph attention")

        src, dst = edge_index[0], edge_index[1]
        Nd = x_dst.size(0)

        q = self.to_q(x_dst).view(Nd, self.num_heads, self.head_dim)          # [Nd,H,Dh]
        k = self.to_k(x_src).view(-1, self.num_heads, self.head_dim)          # [Ns,H,Dh]
        v = self.to_v(x_src).view(-1, self.num_heads, self.head_dim)          # [Ns,H,Dh]

        k_e = k[src]  # [E,H,Dh]
        v_e = v[src]  # [E,H,Dh]

        if self.has_pos_emb and r is not None:
            k_e = k_e + self.to_k_r(r).view(-1, self.num_heads, self.head_dim)
            v_e = v_e + self.to_v_r(r).view(-1, self.num_heads, self.head_dim)

        # linear attention
        phi_q = _phi(q)     # [Nd,H,Dh]
        phi_k = _phi(k_e)   # [E,H,Dh]

        # S_i = sum_j phi(k_j) v_j^T  -> [Nd,H,Dh,Dh]
        kv = phi_k.unsqueeze(-1) * v_e.unsqueeze(-2)  # [E,H,Dh,Dh]
        S = scatter_add(kv, dst, dim=0, dim_size=Nd)  # [Nd,H,Dh,Dh]

        # Z_i = sum_j phi(k_j) -> [Nd,H,Dh]
        Z = scatter_add(phi_k, dst, dim=0, dim_size=Nd)  # [Nd,H,Dh]

        # out_i = (phi(q_i)^T S_i) / (phi(q_i)^T Z_i)
        num = torch.einsum("nhd,nhdv->nhv", phi_q, S)  # [Nd,H,Dh]
        den = (phi_q * Z).sum(-1, keepdim=True).clamp_min(self.eps)  # [Nd,H,1]
        out = num / den  # [Nd,H,Dh]

        out = out.reshape(Nd, self.num_heads * self.head_dim)
        out = self.to_out(self.dropout(out))  # [Nd,D]

        # gating (same spirit as AttentionLayer.update)
        g = torch.sigmoid(self.to_g(torch.cat([out, x_dst], dim=-1)))
        out = out + g * (self.to_s(x_dst) - out)
        return out
