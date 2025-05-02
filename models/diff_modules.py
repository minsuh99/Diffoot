import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding2D(nn.Module):
    def __init__(self, num_players, seq_len, dim):
        super().__init__()
        # 시간축 SinCos 인코딩
        pe_time = torch.zeros(seq_len, dim)
        pos = torch.arange(0, seq_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pe_time[:, 0::2] = torch.sin(pos * div)
        pe_time[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe_time', pe_time)  # [T, D]
        # 선수별(공간) learnable 인코딩
        self.pe_space = nn.Parameter(torch.randn(num_players, dim))

    def forward(self, x):
        _, _, _, T = x.shape
        # Time encoding: [T, D] -> [1, D, 1, T]
        pe_t = self.pe_time[:T].permute(1, 0).unsqueeze(0).unsqueeze(2)
        # Spatial encoding: [N, D] -> [1, D, N, 1]
        pe_s = self.pe_space.permute(1, 0).unsqueeze(0).unsqueeze(-1)

        pe = pe_s + pe_t

        return x + pe
    
class DilatedCausalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), dilation=(1, 1)):
        super().__init__()
        pad_time = (kernel_size[1] - 1) * dilation[1]
        pad_space = (kernel_size[0] - 1) // 2 * dilation[0]
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size,
                              dilation = dilation,
                              padding = (pad_space, pad_time))
    def forward(self, x):
        x = self.conv(x)
        return x[:, :, :, :x.shape[-1] - (self.conv.padding[1])]


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim = 128, projection_dim = None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim

        half_dim = embedding_dim // 2
        steps = torch.arange(num_steps).unsqueeze(1)                # [num_steps, 1]
        freqs = 10.0 ** (torch.arange(half_dim) / (half_dim - 1) * 4.0).unsqueeze(0)  # [1, half_dim]
        table = steps * freqs                                      # [num_steps, half_dim]
        emb = torch.cat([torch.sin(table), torch.cos(table)], dim = 1)  # [num_steps, embedding_dim]
        self.register_buffer('embedding', emb, persistent=False)
        
        self.proj1 = nn.Linear(embedding_dim, projection_dim)
        self.proj2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = F.silu(self.proj1(x))
        return F.silu(self.proj2(x))


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups = 8, dilation = (1, 1)):
        super().__init__()
        self.conv1 = DilatedCausalConv2d(in_channels, out_channels, kernel_size = (3, 3), dilation = dilation)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = DilatedCausalConv2d(out_channels, out_channels, kernel_size = (3, 3), dilation = dilation)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.act  = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d((1, 2), ceil_mode=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.norm1(x1)
        x1 = self.act(x1)
        x2 = self.conv2(x1)
        x2 = self.norm2(x2)
        x2 = self.act(x2 + x1)
        skip = x2
        out = self.pool(x2)
        return out, skip


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups = 8, dilation = (1, 1)):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = (1, 2), stride = (1, 2))
        self.conv1 = DilatedCausalConv2d(in_channels + out_channels, out_channels, kernel_size = (3, 3), dilation = dilation)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = DilatedCausalConv2d(out_channels, out_channels, kernel_size = (3, 3), dilation = dilation)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x1 = self.conv1(x)
        x1 = self.norm1(x1)
        x1 = self.act(x1)
        x2 = self.conv2(x1)
        x2 = self.norm2(x2)
        out = self.act(x2 + x1)         
        return out


class CrossAttentionBlock(nn.Module):
    def __init__(self, feature_dim, heads = 4, dropout_rate = 0.0):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(feature_dim)
        self.multihead_attn1 = nn.MultiheadAttention(embed_dim = feature_dim, num_heads = heads, dropout = dropout_rate)
        self.layer_norm2 = nn.LayerNorm(feature_dim)
        self.multihead_attn2 = nn.MultiheadAttention(embed_dim = feature_dim, num_heads = heads, dropout = dropout_rate)
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
    def __init__(self, num_steps, num_players, seq_len, embedding_dim = 128, feature_dim = 2, base_channels = 64, depth = 4, heads = 4):
        super().__init__()
        self.diff_emb = DiffusionEmbedding(num_steps, embedding_dim, projection_dim=base_channels)
        self.input_conv2d = nn.Conv2d(feature_dim, base_channels, kernel_size=(1, 1))
        self.pos_enc = PositionalEncoding2D(num_players, seq_len, base_channels)
        
        # dilation schedule
        dilations = [(1, 2**i) for i in range(depth)]

        self.downs = nn.ModuleList([
            DownsampleBlock(
                in_channels = base_channels * (2 ** i) if i > 0 else base_channels,
                out_channels = base_channels * (2 ** (i + 1)),
                dilation = dilations[i]
            ) for i in range(depth)
        ])

        bottleneck_ch = base_channels * (2 ** depth)
        self.bottleneck = nn.Sequential(
            nn.Conv1d(bottleneck_ch, bottleneck_ch, kernel_size = 3, padding = 1),
            nn.GroupNorm(8, bottleneck_ch),
            nn.ReLU(inplace=True)
        )
        
        self.graph_proj = nn.Linear(128, bottleneck_ch)   # 128은 graph_encoder.out_dim
        self.hist_proj  = nn.Linear(128, bottleneck_ch)   # 128은 history_encoder.out_dim
        
        self.cross_attn = CrossAttentionBlock(bottleneck_ch, heads)

        self.ups = nn.ModuleList([
            UpsampleBlock(
                in_channels = bottleneck_ch if i == 0 else base_channels * (2 ** (depth- i + 1)),
                out_channels = base_channels * (2 ** (depth - i)),
                dilation = dilations[depth - i]
            ) for i in range(1, depth+1)
        ])

        self.output_conv2d = nn.Conv2d(base_channels, feature_dim, kernel_size=(1, 1))

    def forward(self, traj, t, cond_info, self_cond = None): # cond_info = (graph_emb, hist_emb)
        graph_emb, hist_emb = cond_info
        B, T, N, D = traj.shape
        
        pad_t = (16 - (T % 16)) % 16   # e.g. 125 -> (16 - 13) % 16 = 3
        if pad_t > 0:
            traj = F.pad(traj, (0,0, 0,0, 0, pad_t), value=0.0)
            T = T + pad_t
        
        graph_emb = self.graph_proj(graph_emb)
        hist_emb  = self.hist_proj(hist_emb)
        
        if self_cond is not None and pad_t > 0:
            self_cond = F.pad(self_cond, (0,0, 0,0, 0, pad_t), value=0.0)
        
        x = traj.permute(0, 3, 2, 1)
        x = self.input_conv2d(x)
        x = self.pos_enc(x)
        
        if self_cond is not None:
            sc = self_cond.permute(0, 3, 2, 1)  # [B, D, N, T]
            sc = self.input_conv2d(sc)
            sc = self.pos_enc(sc)
            x = x + sc

        # inject diffusion embedding
        diff = self.diff_emb(t).view(B, -1, 1, 1)
        x = x + diff

        # encode with skips
        skips = []
        for down in self.downs:
            x, skip = down(x)
            skips.append(skip)

        # bottleneck + cross-attention
        B_, C_, N_, T_ = x.shape
        x = x.reshape(B_, C_, N_ * T_)
        x = self.bottleneck(x)                 # [B, C, L_b]
        L_b = x.size(-1)
        query = x.view(B, -1, L_b).permute(2, 0, 1)   # [L_b, B, C]
        graph_cond = graph_emb.unsqueeze(0).expand(L_b, B, -1)
        hist_cond = hist_emb.unsqueeze(0).expand(L_b, B, -1)
        x = self.cross_attn(query, graph_cond, hist_cond).permute(1, 2, 0)  # [B, C, L_b]

        x = x.reshape(B_, C_, N_, T_)
        # decode
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip)

        out = self.output_conv2d(x)            # [B, D, N, T]
        out = out.permute(0, 3, 2, 1)         # [B, T, N, D]
        
        # remove padding
        if pad_t > 0:
            out = out[:, :T-pad_t]          # remove the last pad_t frames

        return out                           # [B, original_T, N, D]
