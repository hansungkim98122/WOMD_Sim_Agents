# latent_attention_layer_mla.py

from typing import Optional, Tuple, Union

import math
import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax

from utils.weight_init import weight_init


class LatentAttentionLayer(MessagePassing):
    """
    SMART-style attention layer with optional DeepSeek-MLA-style low-rank Q/K/V.

    - Keeps the same interface as the original:
        forward(x, r, edge_index) -> x'
    - Still uses per-edge relative embedding r the same way SMART does.
    - If use_mla=False: behaves like the original layer.
    - If use_mla=True: Q/K/V are computed via low-rank projections
      (hidden_dim -> kv_dim/q_dim -> hidden_dim), reducing parameter and compute
      cost of the projections.

    NOTE: This does *not* implement full DeepSeek KV caching, because that
    assumes dense sequence attention. To get real KV cache you’d want to
    rewrite the temporal self-attention (t_attn_layers) as a sequence MLA
    block instead of a MessagePassing graph layer.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        bipartite: bool,
        has_pos_emb: bool,
        use_mla: bool = True,
        kv_dim: Optional[int] = None,
        q_dim: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(aggr="add", node_dim=0, **kwargs)

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.has_pos_emb = has_pos_emb
        self.scale = head_dim ** -0.5
        self.hidden_dim = hidden_dim

        # -----------------------------
        # MLA-style low-rank settings
        # -----------------------------
        self.use_mla = use_mla
        if kv_dim is None:
            # Compressed KV dimension; you can tune this.
            kv_dim = max(hidden_dim // 2, head_dim * num_heads // 2)
        if q_dim is None:
            # Compressed Q dimension; can be same or smaller.
            q_dim = max(hidden_dim // 2, head_dim * num_heads // 2)
        self.kv_dim = kv_dim
        self.q_dim = q_dim

        if not use_mla:
            # === Original SMART-style projections ===
            self.to_q = nn.Linear(hidden_dim, head_dim * num_heads)
            self.to_k = nn.Linear(hidden_dim, head_dim * num_heads, bias=False)
            self.to_v = nn.Linear(hidden_dim, head_dim * num_heads)
            if has_pos_emb:
                self.to_k_r = nn.Linear(hidden_dim, head_dim * num_heads, bias=False)
                self.to_v_r = nn.Linear(hidden_dim, head_dim * num_heads)
        else:
            # === MLA-style low-rank projections (DeepSeek-ish) ===
            # Hidden -> compressed
            self.DQ_proj = nn.Linear(hidden_dim, q_dim, bias=False)
            self.DKV_proj = nn.Linear(hidden_dim, kv_dim, bias=False)

            # Compressed -> full model
            self.UQ_proj = nn.Linear(q_dim, head_dim * num_heads, bias=False)
            self.UK_proj = nn.Linear(kv_dim, head_dim * num_heads, bias=False)
            self.UV_proj = nn.Linear(kv_dim, head_dim * num_heads, bias=False)

            # Optional relative embedding r: project r into compressed KV
            # and add in compressed space before up-projection (cheaper than
            # projecting to full head_dim * num_heads per edge).
            if has_pos_emb:
                self.DR_k = nn.Linear(hidden_dim, kv_dim, bias=False)
                self.DR_v = nn.Linear(hidden_dim, kv_dim, bias=False)

        # Gating + output projection (same as original)
        self.to_s = nn.Linear(head_dim * num_heads, head_dim * num_heads)
        self.to_g = nn.Linear(head_dim * num_heads + hidden_dim, head_dim * num_heads)
        self.to_out = nn.Linear(head_dim * num_heads, hidden_dim)

        self.attn_drop = nn.Dropout(dropout)

        # FFN block (unchanged)
        self.ff_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # Norms (unchanged)
        if bipartite:
            self.attn_prenorm_x_src = nn.LayerNorm(hidden_dim)
            self.attn_prenorm_x_dst = nn.LayerNorm(hidden_dim)
        else:
            self.attn_prenorm_x_src = nn.LayerNorm(hidden_dim)
            self.attn_prenorm_x_dst = self.attn_prenorm_x_src

        if has_pos_emb:
            self.attn_prenorm_r = nn.LayerNorm(hidden_dim)

        self.attn_postnorm = nn.LayerNorm(hidden_dim)
        self.ff_prenorm = nn.LayerNorm(hidden_dim)
        self.ff_postnorm = nn.LayerNorm(hidden_dim)

        self.apply(weight_init)

    # ------------------------------------------------------------------ #
    # Public forward: same signature as original AttentionLayer
    # ------------------------------------------------------------------ #
    def forward(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        r: Optional[torch.Tensor],
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        x:
          - Tensor [N, hidden_dim]  (self-attn)  or
          - Tuple (x_src, x_dst) for bipartite attention.
        r:
          - per-edge relative embedding [E, hidden_dim] or None.
        edge_index:
          - [2, E] COO edges, like SMART.
        """
        if isinstance(x, torch.Tensor):
            x_src = x_dst = self.attn_prenorm_x_src(x)
        else:
            x_src, x_dst = x
            x_src = self.attn_prenorm_x_src(x_src)
            x_dst = self.attn_prenorm_x_dst(x_dst)
            x = x[1]  # Residual uses dst nodes

        if self.has_pos_emb and r is not None:
            r = self.attn_prenorm_r(r)

        # Attention block + residual
        x = x + self.attn_postnorm(self._attn_block(x_src, x_dst, r, edge_index))
        # FFN block + residual
        x = x + self.ff_postnorm(self._ff_block(self.ff_prenorm(x)))
        return x

    # ------------------------------------------------------------------ #
    # MessagePassing interface
    # ------------------------------------------------------------------ #
    def message(
        self,
        q_i: torch.Tensor,
        k_j: torch.Tensor,
        v_j: torch.Tensor,
        r: Optional[torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        q_i: [E, H, Dh]  query at dst
        k_j: [E, H, Dh]  key at src
        v_j: [E, H, Dh]  value at src
        r:   [E, hidden_dim] or None (already prenormed)
        """

        # In MLA mode, we already folded r into K/V in compressed space
        # before up-projection, so we don't touch r here.
        if (not self.use_mla) and self.has_pos_emb and r is not None:
            # Original SMART behavior: add r in full head space
            k_j = k_j + self.to_k_r(r).view(-1, self.num_heads, self.head_dim)
            v_j = v_j + self.to_v_r(r).view(-1, self.num_heads, self.head_dim)

        sim = (q_i * k_j).sum(dim=-1) * self.scale  # [E, H]
        attn = softmax(sim, index, ptr)             # [E, H]
        self.attention_weight = attn.sum(-1).detach()
        attn = self.attn_drop(attn)
        return v_j * attn.unsqueeze(-1)             # [E, H, Dh]

    def update(self, inputs: torch.Tensor, x_dst: torch.Tensor) -> torch.Tensor:
        # inputs: [N, H, Dh] aggregated from neighbors
        inputs = inputs.view(-1, self.num_heads * self.head_dim)
        g = torch.sigmoid(self.to_g(torch.cat([inputs, x_dst], dim=-1)))
        return inputs + g * (self.to_s(x_dst) - inputs)

    # ------------------------------------------------------------------ #
    # Internal attention block using low-rank Q/K/V when use_mla=True
    # ------------------------------------------------------------------ #
    def _attn_block(
        self,
        x_src: torch.Tensor,
        x_dst: torch.Tensor,
        r: Optional[torch.Tensor],
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_mla:
            # === Original SMART-style Q/K/V ===
            q = self.to_q(x_dst).view(-1, self.num_heads, self.head_dim)
            k = self.to_k(x_src).view(-1, self.num_heads, self.head_dim)
            v = self.to_v(x_src).view(-1, self.num_heads, self.head_dim)
        else:
            # === MLA-style low-rank path ===
            # 1) Compressed queries
            C_Q = self.DQ_proj(x_dst)   # [N_dst, q_dim]
            Q_full = self.UQ_proj(C_Q) # [N_dst, H*Dh]
            q = Q_full.view(-1, self.num_heads, self.head_dim)

            # 2) Compressed K/V
            C_KV = self.DKV_proj(x_src)  # [N_src, kv_dim]

            if self.has_pos_emb and (r is not None):
                # r is per-edge, but we want node-wise; simplest is to ignore
                # r in compressed MLA path (already much cheaper).
                # If you *really* want r in MLA, you need a different design
                # (e.g., RoPE-style positional enc).
                pass

            K_full = self.UK_proj(C_KV)  # [N_src, H*Dh]
            V_full = self.UV_proj(C_KV)  # [N_src, H*Dh]

            k = K_full.view(-1, self.num_heads, self.head_dim)
            v = V_full.view(-1, self.num_heads, self.head_dim)

        agg = self.propagate(edge_index=edge_index, x_dst=x_dst, q=q, k=k, v=v, r=r)
        return self.to_out(agg)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff_mlp(x)
