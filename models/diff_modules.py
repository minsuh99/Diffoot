import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def get_torch_trans(heads=4, layers=2, channels=256):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=channels * 2, activation="gelu", dropout=0.1, batch_first=True
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim // 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step.long()]
        x = F.silu(self.projection1(x))
        x = F.silu(self.projection2(x))
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)
        table = steps * frequencies
        return torch.cat([torch.sin(table), torch.cos(table)], dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, channels, diffusion_embedding_dim, nheads, side_dim=None):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        
        if side_dim is not None:
            self.cond_time_encoder = get_torch_trans(
                heads=nheads, layers=1, channels=side_dim
            )
            self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        else:
            self.cond_time_encoder = None
            self.cond_projection = None

        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.time_layer = get_torch_trans(heads=nheads, layers=2, channels=channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=2, channels=channels)

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, cond_info, diffusion_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)
        y = self.mid_projection(y)

        if cond_info is not None and self.cond_time_encoder is not None:
            c = cond_info.reshape(B * K, cond_info.size(1), L).permute(2, 0, 1)
            c = self.cond_time_encoder(c)
            c = c.permute(1, 2, 0).reshape(B, K, cond_info.size(1), L)
            c = c.permute(0, 2, 1, 3).reshape(B, cond_info.size(1), K * L)
            c = self.cond_projection(c)
            y = y + c

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        return (x.reshape(base_shape) + residual.reshape(base_shape)) / math.sqrt(2.0), skip.reshape(base_shape)


class diff_CSDI(nn.Module):
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = config["channels"]
        self.num_players = 11
        self.output_dim = 2  # (x, y)

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, self.output_dim, 1)
        nn.init.xavier_uniform_(self.output_projection2.weight, gain=0.01)
        if self.output_projection2.bias is not None:
            nn.init.zeros_(self.output_projection2.bias)
            
        # ↓ Self‑conditioning용 projection (x_{t-1} → feature 차원)
        self.self_cond_projection = Conv1d_with_init(self.output_dim, self.channels,1)            

        self.residual_layers = nn.ModuleList([
            ResidualBlock(
                channels=self.channels,
                diffusion_embedding_dim=config["diffusion_embedding_dim"],
                nheads=config["nheads"],
                side_dim=config.get("side_dim", None)
            ) for _ in range(config["layers"])
        ])

    def forward(self, x, diffusion_step, cond_info=None, self_cond=None):       
        B, inputdim, K, L = x.shape  # [B, 2, 11, 125]

        x = x.reshape(B, inputdim, K * L)              # [B, 2, 1375]
        x = self.input_projection(x)
        # x = F.relu(x)           # [B, C, 1375]
        x = F.tanh(x)
        x = x.reshape(B, self.channels, K, L)          # [B, C, 11, 125]
        
        if self_cond is not None:
            sc = self_cond.reshape(B, self.output_dim, K*L)    # [B, 2, 1375]
            sc = self.self_cond_projection(sc)                # [B, C, 1375]
            sc = sc.reshape(B, self.channels, K, L)           # [B, C, 11, 125]
            x = x + sc 

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))  # [B, C, 11, 125]

        x = x.reshape(B, self.channels, K * L)  # [B, C, 1375]
        x = self.output_projection1(x)
        # x = F.relu(x)
        x = F.tanh(x)
        x = self.output_projection2(x)  # [B, 2, 1375]
        x = x.reshape(B, self.output_dim, K, L)  # [B, 2, 11, 125]
        return x

