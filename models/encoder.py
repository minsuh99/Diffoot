import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HeteroConv
from torch_geometric.data import HeteroData
from torch_geometric.utils import softmax


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(hidden_dim) * 0.02)

    def forward(self, x, batch):
        scores = (x * self.query).sum(-1)
        weights = softmax(scores, batch)
        out = weights.unsqueeze(-1) * x

        B = int(batch.max().item()) + 1
        C = x.size(-1)

        pool = out.new_zeros((B, C))
        pool.index_add_(0, batch, out)

        return pool


class InteractionGraphEncoder(nn.Module):
    def __init__(self, in_dim, pos_emb_dim=8, hidden_dim=128, out_dim=128):
        super().__init__()
        # normalization and pooling
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.pool = AttentionPooling(hidden_dim)
        self.dropout = nn.Dropout(0.1)

        self.position_emb = nn.Embedding(24, pos_emb_dim)
        self.in_dim = in_dim - 1 + pos_emb_dim

        # define edge types
        edge_types = [
            ('Node', 'attk_and_attk', 'Node'),
            ('Node', 'attk_and_def', 'Node'),
            ('Node', 'def_and_def', 'Node'),
            ('Node', 'attk_and_ball', 'Node'),
            ('Node', 'def_and_ball', 'Node'),
            ('Node', 'temporal', 'Node'),
        ]

        conv1 = { rel: GATConv(
                    in_channels=self.in_dim,
                    out_channels=hidden_dim,
                    heads=1,
                    concat=False,
                    add_self_loops=False,
                    dropout=0.1
                ) for rel in edge_types }

        conv2 = { rel: GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    heads=1,
                    concat=False,
                    add_self_loops=False,
                    dropout=0.1
                ) for rel in edge_types }
        self.het1 = HeteroConv(conv1, aggr='sum')
        self.het2 = HeteroConv(conv2, aggr='sum')
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, graph: HeteroData):
        x_all = graph['Node'].x
        # possession Encoding (with Learnable Parameter)
        pos_idx = 5                  
        cont = torch.cat([x_all[:, :pos_idx], x_all[:, pos_idx+1:]], dim=1)
        pos_raw = x_all[:, pos_idx].long()

        pos_idx_emb = pos_raw.clamp(min=0)
        pos_emb = self.position_emb(pos_idx_emb)

        x = torch.cat([cont, pos_emb], dim=1)

        x_dict = {'Node': x}

        x_dict = self.het1(x_dict, graph.edge_index_dict, edge_attr_dict=graph.edge_attr_dict)
        x = x_dict['Node']
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.norm1(x)

        x_dict = self.het2({'Node': x}, graph.edge_index_dict, edge_attr_dict=graph.edge_attr_dict)
        x = x_dict['Node']
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.norm2(x)

        # pooling
        batch = graph['Node'].batch
        graph_rep = self.pool(x, batch)
        graph_rep = self.dropout(graph_rep)
        return self.proj(graph_rep)


# class TargetTrajectoryEncoder(nn.Module):
#     def __init__(self, input_dim = 46, hidden_dim = 64, num_layers = 1, bidirectional = True, rnn_type = "gru"):
#         super().__init__()
#         self.rnn_type = rnn_type.lower()
#         self.num_directions = 2 if bidirectional else 1
#         self.hidden_dim = hidden_dim
#         rnn_cls = nn.LSTM if self.rnn_type == "lstm" else nn.GRU

#         self.rnn = rnn_cls(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        
#         self.dropout = nn.Dropout(0.1)
#         self.layernorm = nn.LayerNorm(hidden_dim * self.num_directions)
        
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         _, h_n = self.rnn(x)
#         if self.rnn_type == "lstm":
#             h_n = h_n[0]

#         last = h_n.view(self.rnn.num_layers, self.num_directions, x.size(0), self.hidden_dim)[-1]

#         concat = last.permute(1, 0, 2).reshape(x.size(0), -1)  # (B, hidden_dim * num_directions)
#         concat = self.dropout(concat)
#         concat = self.layernorm(concat)
        
#         return concat