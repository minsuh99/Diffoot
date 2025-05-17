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
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x
    
    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class ResidualBlock(nn.Module):
    def __init__(self, channels, diffusion_embedding_dim, nheads, side_dim=None):
        super().__init__()
        self.channels = channels
        self.side_dim = side_dim

        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)

        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        
        if side_dim is not None:
            self.cond_proj = nn.Linear(side_dim, channels)
            # Cross-Attention module
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=channels,
                num_heads=nheads,
                batch_first=True,
                dropout=0.1
            )
            # FiLM projection from attended cond_info
            self.film_proj = nn.Sequential(
                nn.Linear(channels, 2 * channels),
                nn.SiLU(),
                nn.Linear(2 * channels, 2 * channels)
            )

        # Projections
        self.mid_projection = Conv1d_with_init(channels, channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)
        
        self.dropout = nn.Dropout(0.1)

    def forward_time(self, y, base_shape):
        B, C, K, L = base_shape
        y_t = y.reshape(B * K, C, L)
        y_t = self.time_layer(y_t.permute(2, 0, 1)).permute(1, 2, 0)
        return y_t.reshape(B, C, K * L)

    def forward_feature(self, y, base_shape):
        B, C, K, L = base_shape
        y_f = y.reshape(B * L, C, K)
        y_f = self.feature_layer(y_f.permute(2, 0, 1)).permute(1, 2, 0)
        return y_f.reshape(B, C, K * L)

    def forward(self, x, cond_info, diffusion_emb):
        B, C, K, L = x.shape
        base_shape = x.shape
        
        # Flatten spatial dims and add diffusion embedding
        y = x.reshape(B, C, K * L)
        diff = self.diffusion_projection(diffusion_emb).unsqueeze(-1)
        y = y + diff

        # Temporal + feature mixing
        yt = self.forward_time(y, base_shape)
        yf = self.forward_feature(y, base_shape)
        y = yt + yf

        # Mid projection
        y = self.mid_projection(y)
        y = self.dropout(y)

        # Cross-Attention based FiLM
        if cond_info is not None:
            # Prepare Q and KV
            x_flat = y.reshape(B, C, K * L).permute(0, 2, 1)  # (B, K*L, C)
            c_flat = cond_info.reshape(B, self.side_dim, K * L).permute(0, 2, 1)  # (B, K*L, side_dim)
            c_proj = self.cond_proj(c_flat)
            
            attn_out, _ = self.cross_attn(
                query=x_flat,
                key=c_proj,
                value=c_proj
            )

            # Pool attended output
            pooled = attn_out.mean(dim=1)  # (B, C)

            # FiLM parameters
            gamma_beta = self.film_proj(pooled)  # (B, 2C)
            gamma, beta = gamma_beta.chunk(2, dim=1)

            # Apply FiLM
            y = y.reshape(B, C, K * L)
            y = gamma.unsqueeze(-1) * y + beta.unsqueeze(-1)
            y = y.reshape(B, C, K * L)

        # Gated activation and skip
        z = self.output_projection(y)
        gate, filt = z.chunk(2, dim=1)
        activated = torch.sigmoid(gate) * torch.tanh(filt)

        out = (x + activated.reshape(B, C, K, L)).div(math.sqrt(2.0))
        skip = activated.reshape(B, C, K, L)
        return out, skip


class diff_CSDI(nn.Module):
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = config["channels"]
        self.num_steps = config["num_steps"]
        self.diffusion_embedding_dim = config["diffusion_embedding_dim"]
        self.side_dim = config.get("side_dim", None)
        
        self.dropout = nn.Dropout(0.1)

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
        self.output_projection2 = Conv1d_with_init(self.channels, 4, 1)  # noise(2) + log_sigma(2 channels)
        nn.init.xavier_uniform_(self.output_projection2.weight, gain=0.01)
        if self.output_projection2.bias is not None:
            nn.init.zeros_(self.output_projection2.bias)

    def forward(self, x, diffusion_step, cond_info=None, self_cond=None):
        B, inputdim, K, L = x.shape
        # Flatten time and feature
        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.silu(x)

        # Add self-conditioning if provided
        if self_cond is not None:
            sc = self.self_cond_projection(self_cond.reshape(B, 2, K * L))
            x = x + sc

        # Prepare diffusion embedding
        diffusion_emb = self.diffusion_embedding(diffusion_step)

        # Apply residual blocks
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
        x = self.dropout(x)
        x = self.output_projection2(x)
        x = x.reshape(B, 4, K, L)
        eps, log_sigma = x[:, :2], x[:, 2:]
        log_sigma = torch.clamp(log_sigma, -5, 5)
        x = torch.cat([eps, log_sigma], dim=1)
        return x
