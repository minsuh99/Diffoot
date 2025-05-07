# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# from torch_geometric.data import HeteroData
# from torch_geometric.nn import HeteroConv, GATConv
# from torch_geometric.utils import softmax
# from torch.nn import MultiheadAttention

# class AttentionPooling(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.query = nn.Parameter(torch.randn(dim) * 0.02)
        
#     def forward(self, x, batch):
#         scores = (x * self.query).sum(-1)
#         weights = softmax(scores, batch).unsqueeze(-1)
#         weighted = weights * x
#         num_graphs = int(batch.max().item()) + 1
#         pooled = weighted.new_zeros((num_graphs, x.size(-1)))
#         pooled.index_add_(0, batch, weighted)
#         return pooled

# class InteractionGraphEncoder(nn.Module):
#     def __init__(self, in_dim, hidden, heads=4):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(hidden)
#         self.norm2 = nn.LayerNorm(hidden)
#         edge_types = [("Node", et, "Node") for et in [
#             'attk_and_attk','attk_and_def','def_and_def',
#             'attk_and_ball','def_and_ball','temporal'
#         ]]
#         conv1 = {rel: GATConv(in_dim, hidden//heads, heads, concat=True, dropout=0.1,
#                                add_self_loops=False) for rel in edge_types}
#         conv2 = {rel: GATConv(hidden, hidden//heads, heads, concat=True, dropout=0.1,
#                                add_self_loops=False) for rel in edge_types}
#         self.layer1 = HeteroConv(conv1, aggr='sum')
#         self.layer2 = HeteroConv(conv2, aggr='sum')
#         self.pool = AttentionPooling(hidden)
#         self.proj = nn.Linear(hidden, hidden)
        
#     def forward(self, graph: HeteroData):
#         x = {'Node': graph['Node'].x}
#         x = self.layer1(x, graph.edge_index_dict)['Node']
#         x = F.softsign(x); x = self.norm1(x)
#         x = self.layer2({'Node': x}, graph.edge_index_dict)['Node']
#         x = F.softsign(x); x = self.norm2(x)
#         pooled = self.pool(x, graph['Node'].batch)
#         return self.proj(pooled)

# class DiffusionEmbedding(nn.Module):
#     def __init__(self, steps, dim):
#         super().__init__()
#         t = torch.arange(steps).unsqueeze(1)
#         freqs = 10 ** (torch.arange(dim//2) / (dim//2 - 1) * 4)
#         table = torch.cat([torch.sin(t * freqs), torch.cos(t * freqs)], dim=1)
#         self.register_buffer('table', table)
#         self.proj1 = nn.Linear(dim, dim)
#         self.proj2 = nn.Linear(dim, dim)
        
#     def forward(self, timesteps):
#         timesteps = timesteps.to(self.table.device)
#         emb = self.table[timesteps]
#         emb = F.silu(self.proj1(emb))
#         return F.silu(self.proj2(emb))

# class ResidualBlock(nn.Module):
#     def __init__(self, cond_dim, channels, diff_dim, heads):
#         super().__init__()
#         self.diffusion_proj = nn.Linear(diff_dim, channels)
#         # Temporal Conv for local time patterns
#         self.temporal_conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False)
#         # Self-conditioning + cond_info gating
#         self.cond_conv = nn.Conv1d(cond_dim + channels, 2*channels, 1)
#         self.selfcond_conv = nn.Conv1d(cond_dim + 2*channels, 2*channels, 1)
#         self.mid_conv = nn.Conv1d(channels, 2*channels, 1)
#         self.out_conv = nn.Conv1d(channels, 2*channels, 1)
        
#         self.time_mixer = nn.TransformerEncoderLayer(d_model=channels, nhead=heads,
#             dim_feedforward=channels*4, dropout=0.0, activation='gelu')
#         self.feat_mixer = self.time_mixer
        
#         self.cross_attn_time = MultiheadAttention(channels, heads)
#         self.cross_attn_feat = MultiheadAttention(channels, heads)
        
#     def forward(self, input_feat, cond_info, diff_emb, graph_seq, self_cond=None):
#         B, C, K, L = input_feat.shape
#         num_tokens = K * L
#         y = input_feat.reshape(B, C, num_tokens) + self.diffusion_proj(diff_emb).unsqueeze(-1)
        
#         # TCN
#         y_t = y.reshape(B*K, C, L)
#         y_t = F.gelu(self.temporal_conv(y_t))
#         y = y_t.reshape(B, C, num_tokens)
#         # self-conditioning
#         cond_flat = cond_info.reshape(B, cond_info.size(1), num_tokens)
#         if self_cond is not None:
#             sc = self_cond.reshape(B, C, num_tokens)
#             y = torch.cat([y, cond_flat, sc], dim=1)
#             y = self.selfcond_conv(y)
#         else:
#             y = torch.cat([y, cond_flat], dim=1)
#             y = self.cond_conv(y)
        
#         # Gating
#         g2, r2 = y.chunk(2, dim=1)
#         y = torch.sigmoid(g2) * torch.tanh(r2)
        
#         # Time mixer
#         tm = y.reshape(B, C, K, L).permute(3,0,2,1).reshape(L, B*K, C)
#         tm = self.time_mixer(tm)
#         gs_time = graph_seq.permute(2,0,1).unsqueeze(2).expand(-1,-1,K,-1).reshape(num_tokens, B*K, C)
#         tm, _ = self.cross_attn_time(tm, gs_time, gs_time)
#         y = tm.reshape(L, B, K, C).permute(1,3,2,0).reshape(B, C, num_tokens)
        
#         # Feature mixer
#         fm = y.reshape(B, C, K, L).permute(2,0,3,1).reshape(K, B*L, C)
#         fm = self.feat_mixer(fm)
#         gs_feat = graph_seq.permute(2,0,1).reshape(num_tokens, B*K, C)
#         fm, _ = self.cross_attn_feat(fm, gs_feat, gs_feat)
#         y = fm.reshape(K, B, L, C).permute(1,3,0,2).reshape(B, C, num_tokens)
        
#         # Output + skip
#         out = self.out_conv(y)
#         res, skip = out.chunk(2, dim=1)
#         res = res.reshape(B, C, K, L)
#         skip= skip.reshape(B, C, K, L)
#         return (input_feat + res)/math.sqrt(2.0), skip

# class DiffCSDI(nn.Module):
#     def __init__(self, config, input_dim=2):
#         super().__init__()
#         self.input_dim = input_dim
#         self.hidden_dim = config['channels']
#         self.config = config
#         self.diffusion_embedder = DiffusionEmbedding(
#             config['num_steps'], config['diffusion_embedding_dim']
#         )
#         self.graph_encoder = None
#         self.input_conv = nn.Conv1d(self.input_dim, self.hidden_dim, 1)
#         self.mid_conv1 = nn.Conv1d(self.hidden_dim, self.hidden_dim, 1)
#         self.mid_conv2 = nn.Conv1d(self.hidden_dim, self.input_dim, 1)
#         nn.init.zeros_(self.mid_conv2.weight)
        
#         self.res_blocks = nn.ModuleList([
#             ResidualBlock(
#                 config['side_dim'], self.hidden_dim,
#                 config['diffusion_embedding_dim'], config['nheads']
#             ) for _ in range(config['layers'])
#         ])
        
#     def _make_graph_seq(self, graph_batch, num_tokens):
#         if self.graph_encoder is None:
#             feat_dim = graph_batch['Node'].x.size(1)
#             self.graph_encoder = InteractionGraphEncoder(
#                 feat_dim, self.hidden_dim, self.config['nheads']
#             ).to(graph_batch['Node'].x.device)
#         rep = self.graph_encoder(graph_batch)
#         return rep.unsqueeze(-1).repeat(1, 1, num_tokens)
    
#     def forward(self, traj, cond_info, timesteps, graph_batch, self_cond=None):
#         B = traj.size(0)
        
#         if traj.size(1) != self.input_dim:
#             traj = traj.permute(0, 3, 2, 1)
#         _, C_in, K, L = traj.size()
#         num_tokens = K * L
#         graph_seq = self._make_graph_seq(graph_batch, num_tokens)
#         # project trajectory
#         h = traj.reshape(B, C_in, num_tokens)
#         h = F.relu(self.input_conv(h))
#         h = h.reshape(B, self.hidden_dim, K, L)
#         diffusion_embed = self.diffusion_embedder(timesteps)
        
#         skips = []
#         for block in self.res_blocks:
#             h, skip = block(h, cond_info, diffusion_embed, graph_seq, self_cond)
#             skips.append(skip)
#         h = torch.stack(skips).sum(0) / math.sqrt(len(skips))
#         h = h.reshape(B, self.hidden_dim, num_tokens)
#         h = F.relu(self.mid_conv1(h))
#         h = h.reshape(B, self.hidden_dim, K, L)
        
#         h_flat = h.reshape(B, self.hidden_dim, num_tokens)
#         h_flat = F.relu(self.mid_conv1(h_flat))
#         out_flat = self.mid_conv2(h_flat)
        
#         return out_flat.reshape(B, self.input_dim, K, L)


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_torch_trans(heads=4, layers=1, channels=128):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=channels * 2,
        activation="gelu", dropout=0.0, batch_first=True
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


class ResidualBlock(nn.Module):
    def __init__(self, channels, diffusion_embedding_dim, nheads, side_dim=None, K=11, L=125):
        super().__init__()
        self.channels = channels
        self.side_dim = side_dim
        self.K = K
        self.L = L

        # Diffusion embedding projection
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)

        # Time & feature encoders
        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

        # Condition encoders and projections
        if side_dim is not None:
            self.cond_time_encoder = get_torch_trans(nheads, 1, side_dim)
            self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
            self.cond_feat_encoder = get_torch_trans(nheads, 1, side_dim)
            self.cond_feat_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
            # FiLM scale & bias layers
            self.film_scale = nn.Linear(side_dim + diffusion_embedding_dim, 2 * channels)
            self.film_bias = nn.Linear(side_dim + diffusion_embedding_dim, 2 * channels)
            # FC summarization for cond_info
            self.cond_summary_fc = nn.Linear(side_dim * K * L, side_dim)
        else:
            self.cond_time_encoder = None
            self.cond_projection = None
            self.cond_feat_encoder = None
            self.cond_feat_projection = None
            self.film_scale = None
            self.film_bias = None
            self.cond_summary_fc = None

        # Projections
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

    def forward_time(self, y, base_shape):
        B, C, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, C, K, L).permute(0, 2, 1, 3).reshape(B * K, C, L)
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, C, L).permute(0, 2, 1, 3).reshape(B, C, K * L)
        return y

    def forward_feature(self, y, base_shape):
        B, C, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, C, K, L).permute(0, 3, 1, 2).reshape(B * L, C, K)
        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, C, K).permute(0, 2, 3, 1).reshape(B, C, K * L)
        return y

    def forward(self, x, cond_info, diffusion_emb):
        B, C, K, L = x.shape
        base_shape = x.shape
        # Reshape
        y = x.reshape(B, C, K * L)

        # Diffusion embedding
        diff = self.diffusion_projection(diffusion_emb).unsqueeze(-1)
        y = y + diff

        # Temporal and feature transformations
        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)

        # Mid projection for gate/filter
        y = self.mid_projection(y)

        # Conditional encoding injection
        if cond_info is not None:
            # Time-axis condition
            c_t = cond_info.reshape(B * K, self.side_dim, L).permute(2, 0, 1)
            c_t = self.cond_time_encoder(c_t)
            c_t = c_t.permute(1, 2, 0).reshape(B, self.side_dim, K * L)
            y = y + self.cond_projection(c_t)

            # Feature-axis condition
            c_f = cond_info.reshape(B * L, self.side_dim, K).permute(2, 0, 1)
            c_f = self.cond_feat_encoder(c_f)
            c_f = c_f.permute(1, 2, 0).reshape(B, self.side_dim, K * L)
            y = y + self.cond_feat_projection(c_f)

            # FC-based summary for cond_info
            flat = cond_info.reshape(B, self.side_dim * K * L)
            cond_summary = self.cond_summary_fc(flat)

            # FiLM modulation combining cond_summary and diffusion_emb
            c_k = torch.cat([cond_summary, diffusion_emb], dim=1)
            gamma = self.film_scale(c_k).unsqueeze(-1)
            beta  = self.film_bias(c_k).unsqueeze(-1)
            y = gamma * y + beta

        # Gated activation
        gate, filt = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filt)

        # Output projection
        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        out = (x + residual.reshape(base_shape)) / math.sqrt(2.0)
        skip = skip.reshape(B, C, K, L)
        return out, skip


class diff_CSDI(nn.Module):
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = config["channels"]
        self.num_steps = config["num_steps"]
        self.diffusion_embedding_dim = config["diffusion_embedding_dim"]
        self.side_dim = config.get("side_dim", None)

        # Diffusion embedding module
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=self.num_steps,
            embedding_dim=self.diffusion_embedding_dim
        )
        # Input projection
        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        # Self-conditioning projection
        self.self_cond_projection = Conv1d_with_init(2, self.channels, 1)
        # Residual blocks
        self.residual_layers = nn.ModuleList([
            ResidualBlock(
                channels=self.channels,
                diffusion_embedding_dim=self.diffusion_embedding_dim,
                nheads=config["nheads"],
                side_dim=self.side_dim
            ) for _ in range(config["layers"])
        ])
        # Output projections
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 2, 1)
        nn.init.xavier_uniform_(self.output_projection2.weight, gain=0.01)
        if self.output_projection2.bias is not None:
            nn.init.zeros_(self.output_projection2.bias)

    def forward(self, x, diffusion_step, cond_info=None, self_cond=None):
        B, inputdim, K, L = x.shape
        # Flatten time and feature
        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = torch.tanh(x)

        # Add self-conditioning if provided
        if self_cond is not None:
            sc = self.self_cond_projection(self_cond.reshape(B, 2, K * L))
            x = x + sc

        # Prepare diffusion embedding
        diffusion_emb = self.diffusion_embedding(diffusion_step)

        # Reshape for blocks
        x = x.reshape(B, self.channels, K, L)
        skip_connections = []
        for block in self.residual_layers:
            x, skip = block(x, cond_info, diffusion_emb)
            skip_connections.append(skip)

        # Aggregate skips
        x = torch.sum(torch.stack(skip_connections), dim=0) / math.sqrt(len(self.residual_layers))

        # Final projections
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)
        x = torch.tanh(x)
        x = self.output_projection2(x)
        return x.reshape(B, 2, K, L)

