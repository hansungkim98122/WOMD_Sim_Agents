import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

import math

# Implementation of Multi Head Latent Attention from DeepSeek-V2 
# [Source] https://huggingface.co/bird-of-paradise/deepseek-mla

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Validate input dimensions
    assert xq.shape[-1] == xk.shape[-1], "Query and Key must have same embedding dimension"
    assert xq.shape[-1] % 2 == 0, "Embedding dimension must be even"

    # Get sequence lengths
    q_len = xq.shape[1]
    k_len = xk.shape[1]
    
    # Use appropriate part of freqs_cis for each sequence
    q_freqs = freqs_cis[:q_len]
    k_freqs = freqs_cis[:k_len]
    
    # Apply rotary embeddings separately
    # split last dimention to [xq.shape[:-1]/2, 2]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
 
    # Reshape freqs for each
    q_freqs = reshape_for_broadcast(q_freqs, xq_)
    k_freqs = reshape_for_broadcast(k_freqs, xk_)
    
    # Works for both [bsz, seqlen, n_heads*head_dim] and [bsz, seqlen, n_heads, head_dim]
    xq_out = torch.view_as_real(xq_ * q_freqs).flatten(xq.ndim-1) 
    xk_out = torch.view_as_real(xk_ * k_freqs).flatten(xk.ndim-1)

    return xq_out.type_as(xq), xk_out.type_as(xk)

class MultiHeadLatentAttention(nn.Module):
    """
        Multi-Head Latent Attention(MLA) Module As in DeepSeek_V2 paper
        Key innovation from standard MHA:
             1. Low-Rank Key-Value Joint Compression 
             2. Decoupled Rotary Position Embedding
             
    Args:
        d_model:  Total dimension of the model.
        num_head: Number of attention heads.
        d_embed:  Embedding dimension
        d_c:      K/V compression dimension
        d_c1:     Q compression dimension
        d_rotate: Dimension for Rotary Position Embedding
        dropout:  Dropout rate for attention scores.
        bias:     Whether to include bias in linear projections.
        d_head:   Inferred from d_model//num_head
    Inputs:
        sequence: input sequence for self-attention and the query for cross-attention
        key_value_state: input for the key, values for cross-attention
    """
    def __init__(
        self, 
        d_model,             # Infer d_head from d_model
        num_head, 
        d_embed, 
        d_c, 
        d_c1, 
        d_rotate, 
        dropout=0.1, 
        bias=True,
        max_batch_size=32,   # For KV cache sizing
        max_seq_len=2048     # For KV cache sizing 
        ):
        super().__init__()
        
        assert d_model % num_head == 0, "d_model must be divisible by num_head"
        assert d_c < d_embed, "Compression dim should be smaller than embedding dim"
        assert d_c1 < d_embed, "Query compression dim should be smaller than embedding dim"
        
        self.d_model = d_model
        self.num_head = num_head
        # Verify dimensions match up
        assert d_model % num_head == 0, f"d_model ({d_model}) must be divisible by num_head ({num_head})"
        self.d_head=d_model//num_head
        self.d_embed = d_embed
        self.d_c = d_c
        self.d_c1 = d_c1
        self.d_rotate = d_rotate
        self.dropout_rate = dropout  # Store dropout rate separately

        # Linear down-projection(compression) transformations
        self.DKV_proj = nn.Linear(d_embed, d_c, bias=bias)
        self.DQ_proj = nn.Linear(d_embed, d_c1, bias=bias)
        
        # linear up-projection transformations
        self.UQ_proj = nn.Linear(d_c1, d_model, bias=bias)
        self.UK_proj = nn.Linear(d_c, d_model, bias=bias)
        self.UV_proj = nn.Linear(d_c, d_model, bias=bias)

        # Linear RoPE-projection
        self.RQ_proj = nn.Linear(d_c1, num_head*d_rotate, bias=bias)
        self.RK_proj = nn.Linear(d_embed, d_rotate, bias=bias)
        
        # linear output transformations
        self.output_proj = nn.Linear( d_model, d_model, bias=bias)

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

        # Initiialize scaler
        self.scaler = float(1.0 / math.sqrt(self.d_head + d_rotate)) # Store as float in initialization

        # Initialize C_KV and R_K cache for inference
        self.cache_kv = torch.zeros(
            (max_batch_size, max_seq_len, d_c)
        )
        self.cache_rk = torch.zeros(
            (max_batch_size, max_seq_len, d_rotate)
        )

        # Initialize freqs_cis for RoPE
        self.freqs_cis = precompute_freqs_cis(
            d_rotate, max_seq_len * 2
        )

    def forward(
        self, 
        sequence, 
        key_value_states = None, 
        att_mask=None,
        use_cache=False,
        start_pos: int = 0
    ):

        """
        Forward pass supporting both standard attention and cached inference
        Input shape: [batch_size, seq_len, d_model=num_head * d_head]
        Args:
            sequence: Input sequence [batch_size, seq_len, d_model]
            key_value_states: Optional states for cross-attention
            att_mask: Optional attention mask
            use_cache: Whether to use KV caching (for inference)
            start_pos: Position in sequence when using KV cache
        """
        batch_size, seq_len, model_dim = sequence.size()
        # prepare for RoPE
        self.freqs_cis = self.freqs_cis.to(sequence.device)
        freqs_cis = self.freqs_cis[start_pos : ]

        # Check only critical input dimensions
        assert model_dim == self.d_model, f"Input dimension {model_dim} doesn't match model dimension {self.d_model}"
        if key_value_states is not None:
            assert key_value_states.size(-1) == self.d_model, \
            f"Cross attention key/value dimension {key_value_states.size(-1)} doesn't match model dimension {self.d_model}"

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        # Determine kv_seq_len early
        kv_seq_len = key_value_states.size(1) if is_cross_attention else seq_len
        
        # Linear projections and reshape for multi-head, in the order of Q, K/V
        # Down and up projection for query
        C_Q = self.DQ_proj(sequence)     #[batch_size, seq_len, d_c1]
        Q_state = self.UQ_proj(C_Q)      #[batch_size, seq_len, d_model]
        # Linear projection for query RoPE pathway
        Q_rotate = self.RQ_proj(C_Q)      #[batch_size, seq_len, num_head*d_rotate]


        if use_cache:
            #Equation (41) in DeepSeek-v2 paper: cache c^{KV}_t
            self.cache_kv = self.cache_kv.to(sequence.device)

            # Get current compressed KV states
            current_kv = self.DKV_proj(key_value_states if is_cross_attention else sequence) #[batch_size, kv_seq_len, d_c]
            # Update cache using kv_seq_len instead of seq_len
            self.cache_kv[:batch_size, start_pos:start_pos + kv_seq_len] = current_kv
            # Use cached compressed KV up to current position
            C_KV = self.cache_kv[:batch_size, :start_pos + kv_seq_len]

            #Equation (43) in DeepSeek-v2 paper: cache the RoPE pathwway for shared key k^R_t
            assert self.cache_rk.size(-1) == self.d_rotate, "RoPE cache dimension mismatch"
            self.cache_rk = self.cache_rk.to(sequence.device)
            # Get current RoPE key
            current_K_rotate = self.RK_proj(key_value_states if is_cross_attention else sequence) #[batch_size, kv_seq_len, d_rotate]
            # Update cache using kv_seq_len instead of seq_len
            self.cache_rk[:batch_size, start_pos:start_pos + kv_seq_len] = current_K_rotate
            # Use cached RoPE key up to current position
            K_rotate = self.cache_rk[:batch_size, :start_pos + kv_seq_len] #[batch_size, cached_len, d_rotate]
            
            
            """handling attention mask"""
            if att_mask is not None:
                # Get the original mask shape
                mask_size = att_mask.size(-1)
                cached_len = start_pos + kv_seq_len        # cached key_len, including previous key
                assert C_KV.size(1) == cached_len, \
            f"Cached key/value length {C_KV.size(1)} doesn't match theoretical length {cached_len}"
                
                # Create new mask matching attention matrix shape
                extended_mask = torch.zeros(
                    (batch_size, 1, seq_len, cached_len),  # [batch, head, query_len, key_len]
                    device=att_mask.device,
                    dtype=att_mask.dtype
                )
                
                # Fill in the mask appropriately - we need to be careful about the causality here
                # For each query position, it should only attend to cached positions up to that point
                for i in range(seq_len):
                    extended_mask[:, :, i, :(start_pos + i + 1)] = 0  # Can attend
                    extended_mask[:, :, i, (start_pos + i + 1):] = float('-inf')  # Cannot attend
                    
                att_mask = extended_mask
        else:
            # Compression projection for C_KV
            C_KV = self.DKV_proj(key_value_states if is_cross_attention else sequence) #[batch_size, kv_seq_len, d_c]\
            # RoPE pathway for *shared* key
            K_rotate = self.RK_proj(key_value_states if is_cross_attention else sequence)
            

        # Up projection for key and value
        K_state = self.UK_proj(C_KV)               #[batch_size, kv_seq_len/cached_len, d_model]
        V_state = self.UV_proj(C_KV)               #[batch_size, kv_seq_len/cached_len, d_model]

        
        Q_state = Q_state.view(batch_size, seq_len, self.num_head, self.d_head)

        # After getting K_state from projection, get its actual sequence length
        actual_kv_len = K_state.size(1)    # kv_seq_len or start_pos + kv_seq_len
        # in cross-attention, key/value sequence length might be different from query sequence length
        # Use actual_kv_len instead of kv_seq_len for reshaping
        K_state = K_state.view(batch_size, actual_kv_len, self.num_head, self.d_head) 
        V_state = V_state.view(batch_size, actual_kv_len, self.num_head, self.d_head)


        #Apply RoPE to query and shared key
        Q_rotate = Q_rotate.view(batch_size, seq_len, self.num_head, self.d_rotate)
        K_rotate = K_rotate.unsqueeze(2).expand(-1, -1, self.num_head, -1)  # [batch, cached_len, num_head, d_rotate]
        Q_rotate, K_rotate = apply_rotary_emb(Q_rotate, K_rotate, freqs_cis=freqs_cis)


        # Concatenate along head dimension
        Q_state = torch.cat([Q_state, Q_rotate], dim=-1)  # [batch_size, seq_len, num_head, d_head + d_rotate]
        K_state = torch.cat([K_state, K_rotate], dim=-1)  # [batch_size, actual_kv_len, num_head, d_head + d_rotate]


        # Scale Q by 1/sqrt(d_k)
        Q_state = Q_state * self.scaler
        Q_state = Q_state.transpose(1, 2)  # [batch_size, num_head, seq_len, head_dim]
        K_state = K_state.transpose(1, 2)  # [batch_size, num_head, actual_kv_len, head_dim]
        V_state = V_state.transpose(1, 2)  # [batch_size, num_head, actual_kv_len, head_dim]

    
        # Compute attention matrix: QK^T
        self.att_matrix = torch.matmul(Q_state, K_state.transpose(-1,-2)) 
    
        # apply attention mask to attention matrix
        if att_mask is not None and not isinstance(att_mask, torch.Tensor):
            raise TypeError("att_mask must be a torch.Tensor")

        if att_mask is not None:
            self.att_matrix = self.att_matrix + att_mask
        
        # apply softmax to the last dimension to get the attention score: softmax(QK^T)
        att_score = F.softmax(self.att_matrix, dim = -1)
    
        # apply drop out to attention score
        att_score = self.dropout(att_score)
    
        # get final output: softmax(QK^T)V
        att_output = torch.matmul(att_score, V_state)
        assert att_output.size(0) == batch_size, "Batch size mismatch"
        assert att_output.size(2) == seq_len, "Output sequence length should match query sequence length"
        
            
        # concatinate all attention heads
        att_output = att_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_head*self.d_head) 


        # final linear transformation to the concatenated output
        att_output = self.output_proj(att_output)

        assert att_output.size() == (batch_size, seq_len, self.d_model), \
        f"Final output shape {att_output.size()} incorrect"

        return att_output

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RPEMultiheadSelfAttention(nn.Module):
    """
    Multi-head self-attention where:
      Q comes from token embeddings r,
      K and V come from concatenating r and extra RPE features.

    r:   [B, N, D_r]
    rpe: [B, N, D_rpe]   (could be pos/dir/type or some learned feature)
    mask: [B, N] bool    (True = valid, False = pad)
    """
    def __init__(
        self,
        d_r: int,          # dim of token embedding r
        d_rpe: int,        # dim of RPE / extra feature per token
        d_model: int,      # model dimension (per token)
        n_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_r = d_r
        self.d_rpe = d_rpe
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Q from r
        self.q_proj = nn.Linear(d_r, d_model)

        # K and V from [r ; rpe]
        self.k_proj = nn.Linear(d_r + d_rpe, d_model)
        self.v_proj = nn.Linear(d_r + d_rpe, d_model)

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        r: torch.Tensor,               # [B, N, D_r]
        rpe: torch.Tensor,             # [B, N, D_rpe]
        mask: Optional[torch.Tensor] = None,  # [B, N] bool
    ) -> torch.Tensor:
        B, N, _ = r.shape
        H = self.n_heads
        dh = self.d_head

        # --- 1) Project Q, K, V ---
        q = self.q_proj(r)                          # [B, N, D]
        kv_in = torch.cat([r, rpe], dim=-1)         # [B, N, D_r + D_rpe]
        k = self.k_proj(kv_in)                      # [B, N, D]
        v = self.v_proj(kv_in)                      # [B, N, D]

        # --- 2) Reshape for multi-head: [B, H, N, d_head] ---
        q = q.view(B, N, H, dh).transpose(1, 2)     # [B, H, N, dh]
        k = k.view(B, N, H, dh).transpose(1, 2)     # [B, H, N, dh]
        v = v.view(B, N, H, dh).transpose(1, 2)     # [B, H, N, dh]

        # Flatten heads into batch for scaled_dot_product_attention
        q = q.reshape(B * H, N, dh)                 # [B*H, N, dh]
        k = k.reshape(B * H, N, dh)
        v = v.reshape(B * H, N, dh)

        # --- 3) Build attention mask ---
        # scaled_dot_product_attention expects attn_mask [B*H, N, N] or [N, N]
        attn_mask = None
        if mask is not None:
            # mask: [B, N] (True = valid, False = pad)
            mask = mask.bool()
            # pair valid if both tokens valid
            pair_valid = mask.unsqueeze(2) & mask.unsqueeze(1)  # [B, N, N]
            # expand over heads
            pair_valid = pair_valid.unsqueeze(1).expand(B, H, N, N)
            pair_valid = pair_valid.reshape(B * H, N, N)        # [B*H, N, N]

            # where invalid, set to -inf
            big_neg = torch.finfo(q.dtype).min
            attn_mask = torch.where(
                pair_valid,
                torch.zeros_like(pair_valid, dtype=q.dtype),
                torch.full_like(pair_valid, big_neg, dtype=q.dtype),
            )  # [B*H, N, N]

        # --- 4) Scaled dot-product attention ---
        # q,k,v: [B*H, N, dh]
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        )  # [B*H, N, dh]

        # --- 5) Merge heads back ---
        attn_out = attn_out.reshape(B, H, N, dh).transpose(1, 2)  # [B, N, H, dh]
        attn_out = attn_out.reshape(B, N, self.d_model)           # [B, N, D]

        out = self.out_proj(attn_out)                             # [B, N, D]
        out = self.dropout(out)
        return out
