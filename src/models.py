import tensorflow as tf
import torch as th
import torch.nn as nn
from model_utils import MultiHeadLatentAttention

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

class Backbone(nn.Module):
    def __init__(self, name, 
                 input_dim, 
                 output_dim, 
                 num_agents_per_scenario, 
                 num_future_steps, 
                 num_states_steps, 
                 MLA: bool = False):
        super(Backbone,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._num_agents_per_scenario = num_agents_per_scenario
        self._num_future_steps = num_future_steps
        self._num_states_steps = num_states_steps
        if MLA:
            self.attention = MultiHeadLatentAttention(
                d_model=512,      # Model dimension
                num_head=8,       # Number of attention heads
                d_embed=512,      # Embedding dimension
                d_c=64,          # KV compression dimension
                d_c1=64,         # Query compression dimension
                d_rotate=32,     # Rotary embedding dimension
            )
        else:
            self.attention = th.nn.MultiheadAttention(
                
            )

    def forward(self, x):
        return x
        
