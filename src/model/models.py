import torch as th
import torch.nn as nn
from model.model_utils import MultiHeadLatentAttention
import torch.nn.functional as F
import math

# Waymo Example Model for Tutorials




class SpatialRPE(nn.Module):
    """
    Computes the Pairwise Relative Positional Encoding (RPE) bias.
    Uses the robust pairwise geometry (Heading + Relative Position).
    """
    def __init__(self, num_heads, hidden_dim=64):
        super().__init__()
        self.num_heads = num_heads
        # Features: log_dist, cos, sin, rel_x, rel_y
        self.input_dim = 5 
        
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_heads) 
        )

    def forward(self, road_vectors):
        # 1. Extract Geometry
        start = road_vectors[..., 0:2] 
        end   = road_vectors[..., 2:4]
        vec = end - start
        headings = th.atan2(vec[..., 1], vec[..., 0])
        
        # 2. Pairwise Computations
        # A. Relative Heading
        delta_heading = headings.unsqueeze(2) - headings.unsqueeze(1)
        feat_cos = th.cos(delta_heading)
        feat_sin = th.sin(delta_heading)
        
        # B. Relative Position
        delta_pos = start.unsqueeze(2) - start.unsqueeze(1)
        dist = th.norm(delta_pos, dim=-1)
        feat_dist = th.log(dist + 1e-5)
        
        # C. Rotate Relative Position into Query (i) Frame
        h_i = headings.unsqueeze(2).expand_as(delta_heading)
        c = th.cos(h_i)
        s = th.sin(h_i)
        
        dx_global = delta_pos[..., 0]
        dy_global = delta_pos[..., 1]
        
        feat_rel_x = dx_global * c + dy_global * s
        feat_rel_y = -dx_global * s + dy_global * c
        
        # 3. Stack & Project
        geom_feats = th.stack([feat_dist, feat_cos, feat_sin, feat_rel_x, feat_rel_y], dim=-1)
        bias = self.mlp(geom_feats)
        
        # [B, N, N, H] -> [B, H, N, N]
        return bias.permute(0, 3, 1, 2)

class RPEAttention(nn.Module):
    """
    Multi-Head Self Attention using PyTorch's functional SDPA.
    Outputs the hidden states (Context) which serve as Keys/Values for downstream tasks.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        
        # Projections
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)

    def forward(self, x, rpe_bias, key_padding_mask=None):
        """
        Args:
            x: [B, N, D]
            rpe_bias: [B, H, N, N]
            key_padding_mask: [B, N] (True = Padding)
        """
        B, N, C = x.shape
        
        # 1. QKV Projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 2. Prepare Attention Mask (RPE + Padding)
        attn_mask = rpe_bias
        
        if key_padding_mask is not None:
            mask_float = th.zeros_like(key_padding_mask, dtype=x.dtype)
            mask_float.masked_fill_(key_padding_mask, float('-inf'))
            attn_mask = attn_mask + mask_float.view(B, 1, 1, N)

        # 3. Scaled Dot Product Attention
        out = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attn_mask, 
            dropout_p=self.dropout if self.training else 0.0
        )
        
        # 4. Output Projection
        out = out.transpose(1, 2).reshape(B, N, C)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super.__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed * 4),
            nn.GELU(),
        )
    
    def forward(self,x):
        return self.net(x)

class RoadNet(nn.Module):
    """
    RoadNet: Encodes road vectors using a Single Multi-Head Self-Attention layer.
    """
    def __init__(self, input_dim=9, embed_dim=128, num_heads=4, dropout=0.1, vocab_size=1024):
        super().__init__()
        
        # 1. Input Embedding (Continuous 9-dim -> Embed Dim)
        # Note: We use Linear because inputs are continuous vectors [sx, sy...], not integer IDs.
        # If inputs were discrete tokens, we would use nn.Embedding(vocab_size, embed_dim).
        self.token_embedding_table = nn.Embedding(vocab_size, embed_dim)
        
        # 2. RPE Generator (Geometry -> Bias)
        self.rpe_net = SpatialRPE(num_heads=num_heads, hidden_dim=embed_dim)
        
        # 3. Single Attention Layer (No MLP Block, just Attention)
        self.attn = RPEAttention(embed_dim, num_heads, dropout=dropout)
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # 4. Logits Head (For generation/auxiliary tasks)
        self.logits_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx, road_mask):
        """
        Args:
            road_vectors: [B, N, 9]
            road_mask: [B, N] (True = Valid)
        
        Returns:
            road_context: [B, N, D] (The encoded sequence / "Keys & Values" for MotionNet)
        """
        # 1. Embed tokens to embed dim
        x = self.token_embedding_table(idx)
        
        # 2. Compute RPE Bias from Geometry
        rpe_bias = self.rpe_net(th.randn(idx.size(0), idx.size(1), 5))  # Dummy road_vectors for RPE
        
        # 3. Masking (Invert mask: True where padding)
        padding_mask = ~road_mask 
        
        # 4. Single Multi-Head Self-Attention
        # Residual connection + Norm
        x = self.attn(self.norm(x), rpe_bias, padding_mask)
        
        # Zero out padded tokens for cleanliness
        x = x * road_mask.unsqueeze(-1)
        
        return x

    def generate(self, idx, road_mask, max_new_tokens):
        """
        Auto-regressive generation loop.
        Note: This assumes road_vectors can be iteratively updated or appended.
        """
        
        for _ in range(max_new_tokens):
            # Forward pass to get hidden states
            attention_score = self.forward(idx, road_mask)
            
            # Get logits for the last token
            logits = self.logits_head(attention_score)
            logits = logits[:, -1, :]
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = th.multinomial(probs, num_samples=1)

            idx = th.cat((idx, idx_next), dim=1)
            
        return idx

# --- Test Block ---
if __name__ == "__main__":
    B, N = 2, 10
    vocab_size = 1024
    tokens = th.randint(0, vocab_size, (B, N), dtype=th.long)
    mask = th.ones(B, N, dtype=th.bool)
    mask[:, 500:] = False 
    
    model = RoadNet(input_dim=10, embed_dim=128, num_heads=4, vocab_size=1024)
    
    # Output is the sequence of hidden states [B, N, 128]
    # These are your "Keys and Values" for the MotionNet.
    print(f"Input Shape: {tokens.shape}")
    print(tokens)
    context = model(tokens, mask)
    print(f"Context Shape: {context.shape}")