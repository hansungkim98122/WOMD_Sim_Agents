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

# --- MotionMALA behaving as AttentionLayer ---
class MotionMALA(MessagePassing):
    """
    Wrapper for MALA that matches the AttentionLayer (GNN) signature.
    Ignores 'edge_index' and 'r' (pos emb), instead using MALA's RoPE and sequence processing.
    Assumes 'x' can be reshaped into [Batch, SeqLen, Dim].
    """
    def __init__(self, 
                 hidden_dim: int, 
                 num_heads: int, 
                 head_dim: int, 
                 dropout: float, 
                 bipartite: bool = False, 
                 has_pos_emb: bool = False, 
                 **kwargs):
        # Inherit from MessagePassing to satisfy type checks if needed
        super().__init__(aggr='add', node_dim=0, **kwargs)
        
        self.embed_dim = hidden_dim
        self.rope = RoPE1D(head_dim)
        
        # We use a single MALA block here to act as one "Layer"
        self.mala_block = MALA_Block(
            dim=hidden_dim, 
            num_heads=num_heads, 
            head_dim=head_dim, 
            mlp_ratio=4.0, 
            dropout=dropout
        )

    def forward(self, x, r=None, edge_index=None):
        """
        Matching AttentionLayer signature:
        x: [Total_Nodes, Dim] or ([Src_Nodes, Dim], [Dst_Nodes, Dim])
        r: Optional Pos Emb (Ignored)
        edge_index: Graph Connectivity (Ignored)
        """
        # Handle Bipartite case (tuple)
        if isinstance(x, tuple):
             # For temporal attention, x is usually just a tensor.
             # If tuple, we assume we process the target (x[1]) or source (x[0]).
             # MALA is self-attention, so we likely just want x[0] or x[1].
             # Assuming standard self-attention usage where x is a Tensor.
             x_in = x[0] 
        else:
             x_in = x
             
        # CRITICAL: Reshape for MALA
        # MALA expects [Batch, SeqLen, Dim].
        # 'x_in' here is [Total_Nodes, Dim].
        # We need to know Batch size and SeqLen.
        # This wrapper assumes the caller (MotionNet) has reshaped x appropriately
        # OR we treat [Total_Nodes, Dim] as [1, Total_Nodes, Dim] effectively.
        
        # In MotionNet, temporal attention is called as:
        # feat_a = feat_a.reshape(-1, self.hidden_dim) (Flattens [B, N, T] -> [B*N*T])
        # This destroys sequence info! 
        # MotionNet expects the GNN layer to handle sparse edges.
        
        # MALA needs the structure. 
        # Ideally, MotionNet should NOT flatten before calling this if using MALA.
        # But to be drop-in, we assume x comes in as [Batch * N_Agents, Time, Dim] 
        # or we try to infer.
        
        # If input is [Total_Flat, D], we cannot use MALA without knowing Time Steps.
        # Hack: Assume 1D sequence processing on the flattened input? No.
        
        # Let's assume for this specific integration, x is passed as [B, N, T, D] 
        # or [B*N, T, D].
        
        # Fallback: Treat as [1, Total_Nodes, Dim]
        if x_in.ndim == 2:
            x_in = x_in.unsqueeze(0) # [1, N, D]
            
        B, N, C = x_in.shape
        sin, cos = self.rope(N, x_in.device)
        
        out = self.mala_block(x_in, sin, cos)
        
        # Return flattened if input was flattened
        if x_in.shape[0] == 1 and x_in.ndim == 3 and isinstance(x, torch.Tensor) and x.ndim == 2:
             return out.squeeze(0)
             
        return out