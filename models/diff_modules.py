import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATConv
from torch_geometric.utils import softmax
from torch.nn import MultiheadAttention

class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim) * 0.02)
        
    def forward(self, x, batch):
        scores = (x * self.query).sum(-1)
        weights = softmax(scores, batch).unsqueeze(-1)
        weighted = weights * x
        num_graphs = int(batch.max().item()) + 1
        pooled = weighted.new_zeros((num_graphs, x.size(-1)))
        pooled.index_add_(0, batch, weighted)
        return pooled

class InteractionGraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden, heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        edge_types = [("Node", et, "Node") for et in [
            'attk_and_attk','attk_and_def','def_and_def',
            'attk_and_ball','def_and_ball','temporal'
        ]]
        conv1 = {rel: GATConv(in_dim, hidden//heads, heads, concat=True, dropout=0.1,
                               add_self_loops=False) for rel in edge_types}
        conv2 = {rel: GATConv(hidden, hidden//heads, heads, concat=True, dropout=0.1,
                               add_self_loops=False) for rel in edge_types}
        self.layer1 = HeteroConv(conv1, aggr='sum')
        self.layer2 = HeteroConv(conv2, aggr='sum')
        self.pool = AttentionPooling(hidden)
        self.proj = nn.Linear(hidden, hidden)
        
    def forward(self, graph: HeteroData):
        x = {'Node': graph['Node'].x}
        x = self.layer1(x, graph.edge_index_dict)['Node']
        x = F.softsign(x); x = self.norm1(x)
        x = self.layer2({'Node': x}, graph.edge_index_dict)['Node']
        x = F.softsign(x); x = self.norm2(x)
        pooled = self.pool(x, graph['Node'].batch)
        return self.proj(pooled)

class DiffusionEmbedding(nn.Module):
    def __init__(self, steps, dim):
        super().__init__()
        t = torch.arange(steps).unsqueeze(1)
        freqs = 10 ** (torch.arange(dim//2) / (dim//2 - 1) * 4)
        table = torch.cat([torch.sin(t * freqs), torch.cos(t * freqs)], dim=1)
        self.register_buffer('table', table)
        self.proj1 = nn.Linear(dim, dim)
        self.proj2 = nn.Linear(dim, dim)
        
    def forward(self, timesteps):
        timesteps = timesteps.to(self.table.device)
        emb = self.table[timesteps]
        emb = F.silu(self.proj1(emb))
        return F.silu(self.proj2(emb))

class ResidualBlock(nn.Module):
    def __init__(self, cond_dim, channels, diff_dim, heads):
        super().__init__()
        self.diffusion_proj = nn.Linear(diff_dim, channels)
        # Temporal Conv for local time patterns
        self.temporal_conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False)
        # Self-conditioning + cond_info gating
        self.cond_conv = nn.Conv1d(cond_dim + channels, 2*channels, 1)
        self.selfcond_conv = nn.Conv1d(cond_dim + 2*channels, 2*channels, 1)
        self.mid_conv = nn.Conv1d(channels, 2*channels, 1)
        self.out_conv = nn.Conv1d(channels, 2*channels, 1)
        
        self.time_mixer = nn.TransformerEncoderLayer(d_model=channels, nhead=heads,
            dim_feedforward=channels*4, dropout=0.0, activation='gelu')
        self.feat_mixer = self.time_mixer
        
        self.cross_attn_time = MultiheadAttention(channels, heads)
        self.cross_attn_feat = MultiheadAttention(channels, heads)
        
    def forward(self, input_feat, cond_info, diff_emb, graph_seq, self_cond=None):
        B, C, K, L = input_feat.shape
        num_tokens = K * L
        y = input_feat.reshape(B, C, num_tokens) + self.diffusion_proj(diff_emb).unsqueeze(-1)
        
        # TCN
        y_t = y.reshape(B*K, C, L)
        y_t = F.gelu(self.temporal_conv(y_t))
        y = y_t.reshape(B, C, num_tokens)
        # self-conditioning
        cond_flat = cond_info.reshape(B, cond_info.size(1), num_tokens)
        if self_cond is not None:
            sc = self_cond.reshape(B, C, num_tokens)
            y = torch.cat([y, cond_flat, sc], dim=1)
            y = self.selfcond_conv(y)
        else:
            y = torch.cat([y, cond_flat], dim=1)
            y = self.cond_conv(y)
        
        # Gating
        g2, r2 = y.chunk(2, dim=1)
        y = torch.sigmoid(g2) * torch.tanh(r2)
        
        # Time mixer
        tm = y.reshape(B, C, K, L).permute(3,0,2,1).reshape(L, B*K, C)
        tm = self.time_mixer(tm)
        gs_time = graph_seq.permute(2,0,1).unsqueeze(2).expand(-1,-1,K,-1).reshape(num_tokens, B*K, C)
        tm, _ = self.cross_attn_time(tm, gs_time, gs_time)
        y = tm.reshape(L, B, K, C).permute(1,3,2,0).reshape(B, C, num_tokens)
        
        # Feature mixer
        fm = y.reshape(B, C, K, L).permute(2,0,3,1).reshape(K, B*L, C)
        fm = self.feat_mixer(fm)
        gs_feat = graph_seq.permute(2,0,1).reshape(num_tokens, B*K, C)
        fm, _ = self.cross_attn_feat(fm, gs_feat, gs_feat)
        y = fm.reshape(K, B, L, C).permute(1,3,0,2).reshape(B, C, num_tokens)
        
        # Output + skip
        out = self.out_conv(y)
        res, skip = out.chunk(2, dim=1)
        res = res.reshape(B, C, K, L)
        skip= skip.reshape(B, C, K, L)
        return (input_feat + res)/math.sqrt(2.0), skip

class DiffCSDI(nn.Module):
    def __init__(self, config, input_dim=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = config['channels']
        self.config = config
        self.diffusion_embedder = DiffusionEmbedding(
            config['num_steps'], config['diffusion_embedding_dim']
        )
        self.graph_encoder = None
        self.input_conv = nn.Conv1d(self.input_dim, self.hidden_dim, 1)
        self.mid_conv1 = nn.Conv1d(self.hidden_dim, self.hidden_dim, 1)
        self.mid_conv2 = nn.Conv1d(self.hidden_dim, self.input_dim, 1)
        nn.init.zeros_(self.mid_conv2.weight)
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(
                config['side_dim'], self.hidden_dim,
                config['diffusion_embedding_dim'], config['nheads']
            ) for _ in range(config['layers'])
        ])
        
    def _make_graph_seq(self, graph_batch, num_tokens):
        if self.graph_encoder is None:
            feat_dim = graph_batch['Node'].x.size(1)
            self.graph_encoder = InteractionGraphEncoder(
                feat_dim, self.hidden_dim, self.config['nheads']
            ).to(graph_batch['Node'].x.device)
        rep = self.graph_encoder(graph_batch)
        return rep.unsqueeze(-1).repeat(1, 1, num_tokens)
    
    def forward(self, traj, cond_info, timesteps, graph_batch, self_cond=None):
        B = traj.size(0)
        
        if traj.size(1) != self.input_dim:
            traj = traj.permute(0, 3, 2, 1)
        _, C_in, K, L = traj.size()
        num_tokens = K * L
        graph_seq = self._make_graph_seq(graph_batch, num_tokens)
        # project trajectory
        h = traj.reshape(B, C_in, num_tokens)
        h = F.relu(self.input_conv(h))
        h = h.reshape(B, self.hidden_dim, K, L)
        diffusion_embed = self.diffusion_embedder(timesteps)
        
        skips = []
        for block in self.res_blocks:
            h, skip = block(h, cond_info, diffusion_embed, graph_seq, self_cond)
            skips.append(skip)
        h = torch.stack(skips).sum(0) / math.sqrt(len(skips))
        h = h.reshape(B, self.hidden_dim, num_tokens)
        h = F.relu(self.mid_conv1(h))
        h = h.reshape(B, self.hidden_dim, K, L)
        
        h_flat = h.reshape(B, self.hidden_dim, num_tokens)
        h_flat = F.relu(self.mid_conv1(h_flat))
        out_flat = self.mid_conv2(h_flat)
        
        return out_flat.reshape(B, self.input_dim, K, L)
