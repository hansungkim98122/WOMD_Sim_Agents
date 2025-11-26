from typing import Dict
import torch
import torch.nn as nn
from torch_cluster import radius_graph
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData
from torch_geometric.utils import dense_to_sparse, subgraph
from model.attention_layer import AttentionLayer
from model.mlp_layer import MLPLayer
from model.fourier_embedding import FourierEmbedding, MLPEmbedding
from utils.geometry import angle_between_2d_vectors, wrap_angle
from utils.weight_init import weight_init

class RoadNet(nn.Module):
    def __init__(self,
                 dataset: str,
                 input_dim: int,
                 hidden_dim: int,
                 num_historical_steps: int,
                 pl2pl_radius: float,
                 num_freq_bands: int,
                 num_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 map_token) -> None:
        super(RoadNet, self).__init__()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.pl2pl_radius = pl2pl_radius #polyline to polyline radius
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout

        if input_dim == 2:
            input_dim_r_pt2pt = 3
        elif input_dim == 3:
            input_dim_r_pt2pt = 4
        else:
            raise ValueError('{} is not a valid dimension'.format(input_dim))

        self.type_pt_emb = nn.Embedding(17, hidden_dim)
        self.side_pt_emb = nn.Embedding(4, hidden_dim)
        self.polygon_type_emb = nn.Embedding(4, hidden_dim)
        self.light_pl_emb = nn.Embedding(4, hidden_dim)

        self.r_pt2pt_emb = FourierEmbedding(input_dim=input_dim_r_pt2pt, hidden_dim=hidden_dim,
                                            num_freq_bands=num_freq_bands)
        self.pt2pt_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=False, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.token_size = 1024
        self.token_predict_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                           output_dim=self.token_size)
        input_dim_token = 22
        self.token_emb = MLPEmbedding(input_dim=input_dim_token, hidden_dim=hidden_dim)
        self.map_token = map_token
        self.apply(weight_init)
        self.mask_pt = False

    def maybe_autocast(self, dtype=torch.float32):
        return torch.cuda.amp.autocast(dtype=dtype)

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        token_valid_mask = data['pt_token']['pt_valid_mask']
        token_pred_mask = data['pt_token']['pt_pred_mask']
        token_target_mask = data['pt_token']['pt_target_mask']
        node_mask = token_valid_mask

        token_positions = data['pt_token']['position'][:, :self.input_dim].contiguous()
        token_orientations = data['pt_token']['orientation'].contiguous()
        orientation_vectors = torch.stack([token_orientations.cos(), token_orientations.sin()], dim=-1)
        vocab_vectors = self.map_token['traj_src'].to(token_positions.device).to(torch.float)
        vocab_embeddings = self.token_emb(vocab_vectors.view(vocab_vectors.shape[0], -1))
        token_embeddings = vocab_embeddings[data['pt_token']['token_idx']]

        if self.input_dim == 2:
            x_pt = token_embeddings
        elif self.input_dim == 3:
            x_pt = token_embeddings
        else:
            raise ValueError('{} is not a valid dimension'.format(self.input_dim))

        token_to_polygon_edges = data[('pt_token', 'to', 'map_polygon')]['edge_index']
        token_light_type = data['map_polygon']['light_type'][token_to_polygon_edges[1]]
        categorical_embs = [self.type_pt_emb(data['pt_token']['type'].long()),
                            self.polygon_type_emb(data['pt_token']['pl_type'].long()),
                            self.light_pl_emb(token_light_type.long()),]
        x_pt = x_pt + torch.stack(categorical_embs).sum(dim=0)

        token_graph_edges = radius_graph(x=token_positions[:, :2], r=self.pl2pl_radius,
                                        batch=data['pt_token']['batch'] if isinstance(data, Batch) else None,
                                        loop=False, max_num_neighbors=100)
        if self.mask_pt:
            token_graph_edges = subgraph(subset=node_mask, edge_index=token_graph_edges)[0]

        relative_positions = token_positions[token_graph_edges[0]] - token_positions[token_graph_edges[1]]
        relative_orientations = wrap_angle(token_orientations[token_graph_edges[0]] -
                                           token_orientations[token_graph_edges[1]])
        if self.input_dim == 2:
            r_pt2pt = torch.stack(
                [torch.norm(relative_positions[:, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=orientation_vectors[token_graph_edges[1]],
                                          nbr_vector=relative_positions[:, :2]),
                 relative_orientations], dim=-1)
        elif self.input_dim == 3:
            r_pt2pt = torch.stack(
                [torch.norm(relative_positions[:, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=orientation_vectors[token_graph_edges[1]],
                                          nbr_vector=relative_positions[:, :2]),
                 relative_positions[:, -1],
                 relative_orientations], dim=-1)
        else:
            raise ValueError('{} is not a valid dimension'.format(self.input_dim))
        r_pt2pt = self.r_pt2pt_emb(continuous_inputs=r_pt2pt, categorical_embs=None)
        for i in range(self.num_layers):
            x_pt = self.pt2pt_layers[i](x_pt, r_pt2pt, token_graph_edges)

        next_token_prob = self.token_predict_head(x_pt[token_pred_mask])
        next_token_prob_softmax = torch.softmax(next_token_prob, dim=-1)
        _, next_token_idx = torch.topk(next_token_prob_softmax, k=10, dim=-1)
        next_token_index_gt = data['pt_token']['token_idx'][token_target_mask]

        return {
            'x_pt': x_pt,
            'map_next_token_idx': next_token_idx,
            'map_next_token_prob': next_token_prob,
            'map_next_token_idx_gt': next_token_index_gt,
            'map_next_token_eval_mask': token_pred_mask[token_pred_mask]
        }
