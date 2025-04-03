import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv

class GraphAutoencoder(nn.Module):
    def __init__(self, in_dim_dict, hidden_dim, temporal_hidden_dim, out_dim):
        super().__init__()

        # 1. Node Feature Projection
        self.node_proj = nn.ModuleDict({
            k: nn.Linear(sum(in_dim_dict[k]), hidden_dim) for k in in_dim_dict
        })

        # 2. Spatial Encoder (HeteroConv)
        self.spatial_gnn = HeteroConv({
            ('Attk', 'interaction', 'Attk'): GCNConv(hidden_dim, hidden_dim, add_self_loops=False),
            ('Attk', 'interaction', 'Def'): GCNConv(hidden_dim, hidden_dim, add_self_loops=False),
            ('Def', 'interaction', 'Def'): GCNConv(hidden_dim, hidden_dim, add_self_loops=False),
            ('Attk', 'interaction', 'Ball'): GCNConv(hidden_dim, hidden_dim, add_self_loops=False),
            ('Def', 'interaction', 'Ball'): GCNConv(hidden_dim, hidden_dim, add_self_loops=False)
        }, aggr='sum')

        # 3. Temporal Encoder (GRU)
        self.temporal_rnn = nn.GRU(
            input_size=hidden_dim, hidden_size=temporal_hidden_dim, batch_first=True
        )

        # 4. H projection
        self.h_proj = nn.Linear(temporal_hidden_dim * len(in_dim_dict), out_dim)

        # 5. Decoder: H -> z_final
        self.h_to_z = nn.ModuleDict({
            k: nn.Linear(out_dim, temporal_hidden_dim) for k in in_dim_dict
        })

        # 6. Decoder: z_final -> node feature 복원
        self.node_decoder = nn.ModuleDict({
            k: nn.Linear(temporal_hidden_dim, sum(in_dim_dict[k])) for k in in_dim_dict
        })

    def encode_frame(self, data):
        x_dict = {}
        for node_type in data.x_dict:
            features = [data[node_type][key] for key in data[node_type].keys()]
            x_cat = torch.cat(features, dim=-1)
            x_dict[node_type] = self.node_proj[node_type](x_cat)
        z_dict = self.spatial_gnn(x_dict, data.edge_index_dict)
        return z_dict

    def forward(self, graph_seq):
        z_seq_dict = {k: [] for k in graph_seq[0].x_dict.keys()}

        for t in range(len(graph_seq)):
            z_t = self.encode_frame(graph_seq[t])
            for k in z_seq_dict:
                z_seq_dict[k].append(z_t[k])

        pooled = []
        for k in z_seq_dict:
            z_seq = torch.stack(z_seq_dict[k], dim=1)  # [N, T, d]
            _, h_n = self.temporal_rnn(z_seq)         # [1, N, d_t]
            h_last = h_n.squeeze(0).mean(dim=0)       # [d_t]
            pooled.append(h_last)

        H = self.h_proj(torch.cat(pooled, dim=-1))    # [out_dim]
        return H

    def decode_from_H(self, H, data):
        # 1. H -> z_final_hat
        z_final_hat = {
            k: self.h_to_z[k](H).expand(data[k].num_nodes, -1)
            for k in data.x_dict
        }

        # 2. Node reconstruction
        node_loss = 0.0
        for k in z_final_hat:
            x_hat = self.node_decoder[k](z_final_hat[k])
            features = [data[k][key] for key in data[k].keys()]
            x_gt = torch.cat(features, dim=-1)
            node_loss += F.mse_loss(x_hat, x_gt)

        # 3. Edge reconstruction
        edge_loss = 0.0
        for (src, _, dst), edge_index in data.edge_index_dict.items():
            z_src, z_dst = z_final_hat[src], z_final_hat[dst]
            u, v = edge_index
            pos_score = (z_src[u] * z_dst[v]).sum(dim=-1)
            pos_label = torch.ones_like(pos_score)

            u_neg = torch.randint(0, z_src.size(0), (u.size(0),))
            v_neg = torch.randint(0, z_dst.size(0), (v.size(0),))
            neg_score = (z_src[u_neg] * z_dst[v_neg]).sum(dim=-1)
            neg_label = torch.zeros_like(neg_score)

            score = torch.cat([pos_score, neg_score], dim=0)
            label = torch.cat([pos_label, neg_label], dim=0)
            edge_loss += F.binary_cross_entropy_with_logits(score, label)

        return node_loss + edge_loss
