import tensorflow as tf
import torch as th
import torch.nn as nn
from model_utils import MultiHeadLatentAttention, RoadNetNetBlock

# Waymo Example Model for Tutorials
class SimpleModel(tf.keras.Model):
  """A simple one-layer regressor."""
  def __init__(self, num_agents_per_scenario, num_states_steps,
               num_future_steps):
    super(SimpleModel, self).__init__()
    self._num_agents_per_scenario = num_agents_per_scenario
    self._num_states_steps = num_states_steps
    self._num_future_steps = num_future_steps
    self.regressor = tf.keras.layers.Dense(num_future_steps * 2)

  def call(self, states):
    states = tf.reshape(states, (-1, self._num_states_steps * 2))
    pred = self.regressor(states)
    pred = tf.reshape(
        pred, [-1, self._num_agents_per_scenario, self._num_future_steps, 2])
    return pred
  
class MLP(nn.Module):
    """
    A custom Multi-Layer Perceptron (MLP) for simple fully connected feedforward neural network
    """
    def __init__(self, name, input_dim, output_dim, hidden_dim, num_layers, activation, dropout, num_states_steps,num_agents_per_scenario,num_future_steps):
        input_dim = input_dim
        output_dim = output_dim
        hidden_dim = hidden_dim
        num_layers = num_layers
        activation = activation
        dropout = dropout

        self._num_states_steps = num_states_steps
        self._num_agents_per_scenario = num_agents_per_scenario
        self._num_future_steps = num_future_steps

        super(MLP,self).__init__()
        self.act = {'relu': nn.ReLU, 'gelu': nn.GELU, 'tanh': nn.Tanh}[activation]
        layers = [nn.Linear(input_dim, hidden_dim), self.act()]
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(self.act())
            if dropout>0:
                layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(hidden_dim,output_dim))
        self.ff = nn.Sequential(*layers)  
    @classmethod
    def from_pretrained(cls, cfg, path):
        print("Loaded from", path)
        return cls(cfg)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: th.tensor):
        x = th.reshape(x,(-1, self._num_states_steps * 2)) #(_, input_dim = num_states_steps * 2)
        pred = self.ff(x)
        return th.reshape(pred, (-1, self._num_agents_per_scenario, self._num_future_steps, 2))

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import math

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
        self.out_proj = nn.Linear(embed_dim, embed_dim)

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
        return self.out_proj(out)

class RoadNet(nn.Module):
    """
    RoadNet: Encodes road vectors using a Single Multi-Head Self-Attention layer.
    """
    def __init__(self, input_dim=9, embed_dim=128, num_heads=4, dropout=0.1, vocab_size=1024):
        super().__init__()
        
        # 1. Input Embedding (Continuous 9-dim -> Embed Dim)
        # Note: We use Linear because inputs are continuous vectors [sx, sy...], not integer IDs.
        # If inputs were discrete tokens, we would use nn.Embedding(vocab_size, embed_dim).
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # 2. RPE Generator (Geometry -> Bias)
        self.rpe_net = SpatialRPE(num_heads=num_heads)
        
        # 3. Single Attention Layer (No MLP Block, just Attention)
        self.attn = RPEAttention(embed_dim, num_heads, dropout=dropout)
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # 4. Logits Head (For generation/auxiliary tasks)
        self.logits_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, road_vectors, road_mask):
        """
        Args:
            road_vectors: [B, N, 9]
            road_mask: [B, N] (True = Valid)
        
        Returns:
            road_context: [B, N, D] (The encoded sequence / "Keys & Values" for MotionNet)
        """
        # 1. Embed inputs
        x = self.input_proj(road_vectors)
        
        # 2. Compute RPE Bias from Geometry
        rpe_bias = self.rpe_net(road_vectors)
        
        # 3. Masking (Invert mask: True where padding)
        padding_mask = ~road_mask 
        
        # 4. Single Multi-Head Self-Attention
        # Residual connection + Norm
        x = x + self.attn(self.norm(x), rpe_bias, padding_mask)
        
        # Zero out padded tokens for cleanliness
        x = x * road_mask.unsqueeze(-1)
        
        return x

    def generate(self, road_vectors, road_mask, max_new_tokens):
        """
        Auto-regressive generation loop.
        Note: This assumes road_vectors can be iteratively updated or appended.
        """
        curr_vectors = road_vectors
        
        for _ in range(max_new_tokens):
            # Forward pass to get hidden states
            context = self.forward(curr_vectors, road_mask)
            
            # Get logits for the last token
            logits = self.logits_head(context[:, -1, :])
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = th.multinomial(probs, num_samples=1)
            
            # Note: To continue generation, you would need to convert idx_next 
            # back into a geometry vector [1, 9] via a codebook (VQ-Decoder)
            # and append it to curr_vectors. 
            # For now, we just return the index.
            return idx_next

# --- Test Block ---
if __name__ == "__main__":
    B, N, D = 2, 1024, 9
    vectors = th.randn(B, N, D)
    mask = th.ones(B, N, dtype=th.bool)
    mask[:, 500:] = False 
    
    model = RoadNet(input_dim=9, embed_dim=128, num_heads=4)
    
    # Output is the sequence of hidden states [B, N, 128]
    # These are your "Keys and Values" for the MotionNet.
    context = model(vectors, mask)
    print(f"Input Shape: {vectors.shape}")
    print(f"Context Shape (Keys/Values): {context.shape}")