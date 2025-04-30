import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim = 128, projection_dim = None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim

        half_dim = embedding_dim // 2
        steps = torch.arange(num_steps).unsqueeze(1)                # [num_steps, 1]
        freqs = 10.0 ** (torch.arange(half_dim) / (half_dim - 1) * 4.0).unsqueeze(0)  # [1, half_dim]
        table = steps * freqs                                      # [num_steps, half_dim]
        emb = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # [num_steps, embedding_dim]
        self.register_buffer('embedding', emb, persistent=False)
        
        self.proj1 = nn.Linear(embedding_dim, projection_dim)
        self.proj2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = F.silu(self.proj1(x))
        return F.silu(self.proj2(x))


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups = 8):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool1d(kernel_size=2)

    def forward(self, x):
        conv_out = self.conv(x)           # [batch, out_ch, length]

        norm_out = self.norm(conv_out)    # [batch, out_ch, length]

        activated = self.act(norm_out)  # [batch, out_ch, length]

        skip = activated                 

        pooled = self.pool(activated)    # [batch, out_ch, length//2]

        return pooled, skip


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups = 8):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.ReLU(inplace=True)
        self.conv = nn.Conv1d(out_channels * 2, out_channels, kernel_size=3, padding=1)
        self.conv_norm = nn.GroupNorm(groups, out_channels)

    def forward(self, x, skip):
        upsampled = self.up(x)                         

        normalized = self.norm(upsampled)              

        activated = self.act(normalized)               

        concat_feat = torch.cat([activated, skip], dim=1)  

        conv_out = self.conv(concat_feat)               

        norm2 = self.conv_norm(conv_out)     
        final = self.act(norm2)                    
 
        return final


class CrossAttentionBlock(nn.Module):
    def __init__(self, feature_dim, heads=4, dropout_rate=0.0):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(feature_dim)
        self.multihead_attn1 = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=heads, dropout=dropout_rate)
        self.layer_norm2 = nn.LayerNorm(feature_dim)
        self.multihead_attn2 = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=heads, dropout=dropout_rate)
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Linear(feature_dim * 4, feature_dim)
        )

    def forward(self, query, graph_encoded, hist_encoded):
        # All inputs: [sequence_length, B, feature_dim]
        res = query
        norm_query = self.layer_norm1(query)
        attn_out1, _ = self.multihead_attn1(norm_query, graph_encoded, graph_encoded)
        res1 = attn_out1 + res

        norm_mid = self.layer_norm2(res1)
        attn_out2, _ = self.multihead_attn2(norm_mid, hist_encoded, hist_encoded)
        res2 = attn_out2 + res1

        ff_out = self.feed_forward(res2)
        return ff_out + res2


class TrajectoryUNetDenoiser(nn.Module):
    def __init__(self, num_steps, embedding_dim = 128, feature_dim = 2, base_channels = 64, depth = 4, heads = 4):
        super().__init__()
        self.diff_emb = DiffusionEmbedding(num_steps, embedding_dim, projection_dim=base_channels)
        self.input_proj = nn.Linear(feature_dim, base_channels)

        self.downs = nn.ModuleList([
            DownsampleBlock(
                in_channels = base_channels * (2 ** i) if i > 0 else base_channels,
                out_channels = base_channels * (2 ** (i + 1))
            )
            for i in range(depth)
        ])

        bottleneck_ch = base_channels * (2**depth)
        self.bottleneck = nn.Sequential(
            nn.Conv1d(bottleneck_ch, bottleneck_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, bottleneck_ch),
            nn.ReLU(inplace=True)
        )
        self.cross_attn = CrossAttentionBlock(bottleneck_ch, heads)

        self.ups = nn.ModuleList([
            UpsampleBlock(
                in_channels = bottleneck_ch if i == 0 else base_channels * (2 ** (depth- i + 1)),
                out_channels = base_channels * (2 ** (depth - i))
            )
            for i in range(1, depth+1)
        ])

        self.output_proj = nn.Linear(base_channels, feature_dim)

    def forward(self, traj, t, cond_info, self_cond=None): # cond_info = (graph_emb, hist_emb)
        graph_emb, hist_emb = cond_info
        B, T, N, F = traj.shape
        
        x = traj.view(B, T*N, F)
        x = F.silu(self.input_proj(x))           # [B, seq, C]
        
        if self_cond is not None:
            sc = self_cond.view(B, T*N, F)         # [B, T*N, F]
            sc = F.silu(self.input_proj(sc))       # [B, T*N, C]
            x = x + sc                             # [B, T*N, C]

        x = x.transpose(1, 2).reshape(B, -1, T * N) # [B, C, L]

        # inject diffusion embedding
        diff = self.diff_emb(t).unsqueeze(-1)   # [B, C, 1]
        x = x + diff

        # encode with skips
        skips = []
        for down in self.downs:
            x, skip = down(x)
            skips.append(skip)

        # bottleneck + cross-attention
        x = self.bottleneck(x)                 # [B, C, L_b]
        L_b = x.size(-1)
        query = x.view(B, -1, L_b).permute(2, 0, 1)   # [L_b, B, C]
        graph_cond = graph_emb.unsqueeze(0).expand(L_b, B, -1)
        hist_cond = hist_emb.unsqueeze(0).expand(L_b, B, -1)
        x = self.cross_attn(query, graph_cond, hist_cond).permute(1, 2, 0)  # [B, C, L_b]

        # decode
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip)

        # project back to [B, T, N, F]
        x = x.view(B, -1, T, N).permute(0,2,3,1)
        return self.output_proj(x)            # [B, T, N, F]
