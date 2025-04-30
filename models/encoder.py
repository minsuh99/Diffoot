# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GATv2Conv, HeteroConv
# from torch_geometric.data import HeteroData
# from torch_geometric.utils import softmax


# class AttentionPooling(nn.Module):
#     def __init__(self, hidden_dim):
#         super().__init__()
#         self.query = nn.Parameter(torch.randn(hidden_dim) * 0.02)

#     def forward(self, x, batch):
#         # x: (N, C)
#         scores  = (x * self.query).sum(-1)    # (N,)
#         weights = softmax(scores, batch)      # (N,)
#         out     = weights.unsqueeze(-1) * x   # (N, C)

#         B = int(batch.max().item()) + 1
#         C = x.size(-1)

#         graph_rep = out.new_zeros((B, C))
#         graph_rep.index_add_(0, batch, out)

#         return graph_rep

# class InteractionGraphEncoder(nn.Module):
#     """
#     A lightweight encoder using built-in GATv2Conv per relation (no custom message-passing).
#     """
#     def __init__(self, in_dim, hidden_dim=128, out_dim=128, heads=2):
#         super().__init__()
#         # normalization and pooling
#         self.norm1 = nn.LayerNorm(hidden_dim)
#         self.norm2 = nn.LayerNorm(hidden_dim)
#         self.pool  = AttentionPooling(hidden_dim)

#         # define edge types
#         edge_types = [
#             ('Node', 'attk_and_attk', 'Node'),
#             ('Node', 'attk_and_def', 'Node'),
#             ('Node', 'def_and_def', 'Node'),
#             ('Node', 'attk_and_ball', 'Node'),
#             ('Node', 'def_and_ball', 'Node'),
#             ('Node', 'temporal', 'Node'),
#         ]
#         # 1st layer convs
#         conv1 = {
#             rel: GATv2Conv(in_dim, hidden_dim, heads=heads, concat=False)
#             for rel in edge_types
#         }
#         # 2nd layer convs
#         conv2 = {
#             rel: GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False)
#             for rel in edge_types
#         }

#         self.het1 = HeteroConv(conv1, aggr='sum')
#         self.het2 = HeteroConv(conv2, aggr='sum')
#         self.proj = nn.Linear(hidden_dim, out_dim)

#     def forward(self, graph: HeteroData):
#         x = graph['Node'].x  # (N, F)

#         # first heterogeneous attention
#         x = self.het1({'Node': x}, graph.edge_index_dict)
#         x = x['Node']
#         # x = F.relu(x)
#         x = F.softsign(x)
#         x = self.norm1(x)

#         # second heterogeneous attention
#         x = self.het2({'Node': x}, graph.edge_index_dict)
#         x = x['Node']
#         # x = F.relu(x)
#         x = F.softsign(x)
#         x = self.norm2(x)

#         # pooling to graph-level
#         batch = graph['Node'].batch
#         graph_rep = self.pool(x, batch)
#         return self.proj(graph_rep)  # (B, out_dim)


# class TargetTrajectoryEncoder(nn.Module):
#     def __init__(self, input_dim = 22, hidden_dim = 64, num_layers = 1, bidirectional = True, rnn_type = "gru"):
#         super().__init__()
#         self.rnn_type = rnn_type.lower()
#         self.num_directions = 2 if bidirectional else 1
#         self.hidden_dim = hidden_dim
#         rnn_cls = nn.LSTM if self.rnn_type == "lstm" else nn.GRU

#         self.rnn = rnn_cls(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         _, h_n = self.rnn(x)
#         if self.rnn_type == "lstm":
#             h_n = h_n[0]

#         last = h_n.view(self.rnn.num_layers, self.num_directions, x.size(0), self.hidden_dim)[-1]

#         concat = last.permute(1, 0, 2).reshape(x.size(0), -1)  # (B, hidden_dim * num_directions)
#         return concat

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
        # x: (N, C)
        scores  = (x * self.query).sum(-1)    # (N,)
        weights = softmax(scores, batch)      # (N,)
        out     = weights.unsqueeze(-1) * x   # (N, C)

        B = int(batch.max().item()) + 1
        C = x.size(-1)

        graph_rep = out.new_zeros((B, C))
        graph_rep.index_add_(0, batch, out)

        return graph_rep


class InteractionGraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=128, heads=4):
        super().__init__()
        # normalization and pooling
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.pool  = AttentionPooling(hidden_dim)

        # define edge types
        edge_types = [
            ('Node', 'attk_and_attk', 'Node'),
            ('Node', 'attk_and_def', 'Node'),
            ('Node', 'def_and_def', 'Node'),
            ('Node', 'attk_and_ball', 'Node'),
            ('Node', 'def_and_ball', 'Node'),
            ('Node', 'temporal', 'Node'),
        ]
        # 1st layer convs: from in_dim to hidden_dim
        conv1 = { rel: GATConv(
                    in_channels=in_dim,
                    out_channels=hidden_dim // heads,
                    heads=heads,
                    concat=True,
                    dropout=0.1,
                    add_self_loops=False
                ) for rel in edge_types }
        # 2nd layer convs: from hidden_dim to hidden_dim
        conv2 = { rel: GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // heads,
                    heads=heads,
                    concat=True,
                    dropout=0.1,
                    add_self_loops=False
                ) for rel in edge_types }
        self.het1 = HeteroConv(conv1, aggr='sum')
        self.het2 = HeteroConv(conv2, aggr='sum')
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, graph: HeteroData):
        x_dict = {'Node': graph['Node'].x}

        # first heterogeneous attention layer (no edge_weight)
        x_dict = self.het1(x_dict, graph.edge_index_dict)
        x = x_dict['Node']
        x = F.softsign(x)
        x = self.norm1(x)

        # second heterogeneous attention layer (no edge_weight)
        x_dict = self.het2({'Node': x}, graph.edge_index_dict)
        x = x_dict['Node']
        x = F.softsign(x)
        x = self.norm2(x)

        # pooling to graph-level representation
        batch = graph['Node'].batch
        graph_rep = self.pool(x, batch)
        return self.proj(graph_rep)  # (B, out_dim)


class TargetTrajectoryEncoder(nn.Module):
    def __init__(self, input_dim = 22, hidden_dim = 64, num_layers = 1, bidirectional = True, rnn_type = "gru"):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.num_directions = 2 if bidirectional else 1
        self.hidden_dim = hidden_dim
        rnn_cls = nn.LSTM if self.rnn_type == "lstm" else nn.GRU

        self.rnn = rnn_cls(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.rnn(x)
        if self.rnn_type == "lstm":
            h_n = h_n[0]

        last = h_n.view(self.rnn.num_layers, self.num_directions, x.size(0), self.hidden_dim)[-1]

        concat = last.permute(1, 0, 2).reshape(x.size(0), -1)  # (B, hidden_dim * num_directions)
        return concat