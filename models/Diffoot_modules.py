import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# def get_torch_trans(heads=4, layers=1, channels=128):
#     encoder_layer = nn.TransformerEncoderLayer(
#         d_model=channels, nhead=heads, dim_feedforward=channels * 2,
#         activation="gelu", dropout=0.0, batch_first=True
#     )
#     return nn.TransformerEncoder(encoder_layer, num_layers=layers)

# TransformerEncoder -> Linformer
def get_EF(input_size, dim, bias=True):
    lin = nn.Linear(input_size, dim, bias)
    torch.nn.init.xavier_normal_(lin.weight)
    return lin

class LinearAttentionHead(nn.Module):
    def __init__(self, dim, dropout, E_proj, F_proj, causal=False):
        super().__init__()
        self.E = E_proj
        self.F = F_proj
        self.dim = dim
        self.dropout = nn.Dropout(dropout)
        self.causal = causal

    def forward(self, Q, K, V):
        K = K.transpose(1, 2)
        K = self.E(K)
        Q = torch.matmul(Q, K)

        P_bar = Q / torch.sqrt(torch.tensor(self.dim, dtype=Q.dtype, device=Q.device))

        if self.causal:
            seq_len, comp_dim = P_bar.size(1), P_bar.size(2)
            causal_mask = torch.triu(torch.ones(seq_len, comp_dim, device=P_bar.device)) == 1
            P_bar = P_bar.masked_fill(~causal_mask, -1e10)
        
        P_bar = P_bar.softmax(dim=-1)
        P_bar = self.dropout(P_bar)

        V = V.transpose(1, 2)
        V = self.F(V)
        V = V.transpose(1, 2)
        
        out_tensor = torch.matmul(P_bar, V)
        return out_tensor

class LinformerMultiHead(nn.Module):
    def __init__(self, channels, nheads, seq_len, compressed_dim=32, dropout=0.2, causal=False):
        super().__init__()
        self.nheads = nheads
        self.head_dim = channels // nheads
        self.channels = channels
        
        self.to_q = nn.ModuleList([nn.Linear(channels, self.head_dim, bias=False) for _ in range(nheads)])
        self.to_k = nn.ModuleList([nn.Linear(channels, self.head_dim, bias=False) for _ in range(nheads)])
        self.to_v = nn.ModuleList([nn.Linear(channels, self.head_dim, bias=False) for _ in range(nheads)])
        
        E_proj = get_EF(seq_len, compressed_dim)
        F_proj = get_EF(seq_len, compressed_dim)
        
        self.heads = nn.ModuleList([
            LinearAttentionHead(
                self.head_dim, 
                dropout, 
                E_proj, 
                F_proj, 
                causal
            ) for _ in range(nheads)
        ])

        self.w_o = nn.Linear(channels, channels)
        
    def forward(self, x):
        head_outputs = []
        for i, head in enumerate(self.heads):
            Q = self.to_q[i](x)
            K = self.to_k[i](x)
            V = self.to_v[i](x)

            head_out = head(Q, K, V)
            head_outputs.append(head_out)
        
        # Concatenate heads
        out = torch.cat(head_outputs, dim=-1)
        return self.w_o(out)

def get_linformer_trans(heads=4, layers=1, channels=128, seq_len=100, compressed_dim=32, causal=False):
    return LinformerTransformer(
        channels=channels,
        nheads=heads,
        seq_len=seq_len,
        compressed_dim=compressed_dim,
        causal=causal,
        dropout=0.2
    )

class LinformerTransformer(nn.Module):
    def __init__(self, channels, nheads, seq_len, compressed_dim=32, causal=False, dropout=0.2):
        super().__init__()
        self.attention = LinformerMultiHead(
            channels=channels,
            nheads=nheads, 
            seq_len=seq_len,
            compressed_dim=compressed_dim,
            dropout=dropout,
            causal=causal
        )
        
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels)
        )
        
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self-attention with residual connection
        x = self.norm1(x)
        attn_out = self.attention(x)
        x = x + self.dropout(attn_out)
        
        # Feed-forward with residual connection
        x = self.norm2(x)
        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        
        return x


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
        steps = torch.arange(num_steps).unsqueeze(1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)
        table = steps * frequencies
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class ResidualBlock(nn.Module):
    def __init__(self, channels, diffusion_embedding_dim, nheads, side_dim=None, time_seq_len=50, feature_seq_len=11, compressed_dim=32):
        super().__init__()
        self.channels = channels
        self.side_dim = side_dim

        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)

        self.time_layer = get_linformer_trans(heads=nheads, layers=1, channels=channels, seq_len=time_seq_len, 
                                              compressed_dim=compressed_dim, causal=True)
        self.feature_layer = get_linformer_trans(heads=nheads, layers=1, channels=channels, seq_len=feature_seq_len, 
                                                 compressed_dim=min(compressed_dim, feature_seq_len), causal=False)
        
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        
        if side_dim is not None:
            self.cond_proj = nn.Linear(side_dim, channels)

            self.cross_attn = nn.MultiheadAttention(
                embed_dim=channels,
                num_heads=nheads,
                batch_first=True,
                dropout=0.2
            )

            self.film_proj = nn.Sequential(
                nn.Linear(channels, 2 * channels),
                nn.SiLU(),
                nn.Linear(2 * channels, 2 * channels)
            )

            with torch.no_grad():
                nn.init.zeros_(self.film_proj[-1].weight)
                if self.film_proj[-1].bias is not None:
                    self.film_proj[-1].bias[:channels] = 1.0  # gamma
                    self.film_proj[-1].bias[channels:] = 0.0  # beta

        # Projections
        self.mid_projection = Conv1d_with_init(channels, channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)
        
        self.dropout = nn.Dropout(0.2)

    def forward_time(self, y, base_shape):
        B, C, K, L = base_shape
        y_t = y.reshape(B * K, C, L)
        y_t = self.time_layer(y_t.permute(0, 2, 1)).permute(0, 2, 1)
        return y_t.reshape(B, C, K * L)

    def forward_feature(self, y, base_shape):
        B, C, K, L = base_shape
        y_f = y.reshape(B * L, C, K)
        y_f = self.feature_layer(y_f.permute(0, 2, 1)).permute(0, 2, 1)
        return y_f.reshape(B, C, K * L)

    def forward(self, x, cond_info, diffusion_emb):
        B, C, K, L = x.shape
        base_shape = x.shape

        y = x.reshape(B, C, K * L)
        diff = self.diffusion_projection(diffusion_emb).unsqueeze(-1)
        y = y + diff
        
        y = self.norm1(y.permute(0, 2, 1)).permute(0, 2, 1)

        yt = self.forward_time(y, base_shape)
        yf = self.forward_feature(y, base_shape)
        y = yt + yf

        y = self.mid_projection(y)
        y = self.norm2(y.permute(0, 2, 1)).permute(0, 2, 1)
        y = self.dropout(y)

        if cond_info is not None:
            x_flat = y.reshape(B, C, K * L).permute(0, 2, 1)  # (B, K*L, C)
            c_flat = cond_info.reshape(B, self.side_dim, K * L).permute(0, 2, 1)  # (B, K*L, side_dim)
            c_proj = self.cond_proj(c_flat)
            
            attn_out, _ = self.cross_attn(
                query=x_flat,
                key=c_proj,
                value=c_proj
            )
            # Apply FiLM
            film_params = self.film_proj(attn_out)
            film_params = film_params.permute(0, 2, 1).reshape(B, 2*C, K, L)
            gamma, beta = film_params.chunk(2, dim=1)

            y = y.reshape(B, C, K, L)
            y = gamma * y + beta
            y = y.reshape(B, C, K * L)

        z = self.output_projection(y)
        gate, filt = z.chunk(2, dim=1)
        activated = torch.sigmoid(gate) * torch.tanh(filt)

        out = (x + activated.reshape(B, C, K, L)).div(math.sqrt(2.0))
        skip = activated.reshape(B, C, K, L)
        return out, skip

# CSDI based Denoising Network
class Diffoot_DenoisingNetwork(nn.Module):
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = config["channels"]
        self.num_steps = config["num_steps"]
        self.diffusion_embedding_dim = config["diffusion_embedding_dim"]
        self.side_dim = config.get("side_dim", None)
        
        self.time_seq_len = config.get("time_seq_len", 50)
        self.feature_seq_len = config.get("feature_seq_len", 11) 
        self.compressed_dim = config.get("compressed_dim", 32)
        
        self.dropout = nn.Dropout(0.2)
        self.norm = nn.LayerNorm(self.channels)

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=self.num_steps,
            embedding_dim=self.diffusion_embedding_dim
        )

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)

        self.residual_layers = nn.ModuleList([
            ResidualBlock(
                channels=self.channels,
                diffusion_embedding_dim=self.diffusion_embedding_dim,
                nheads=config["nheads"],
                side_dim=self.side_dim,
                time_seq_len=self.time_seq_len,
                feature_seq_len=self.feature_seq_len,
                compressed_dim=self.compressed_dim
            ) for _ in range(config["layers"])
        ])
        self.inv_sqrt_layers = 1.0 / math.sqrt(len(self.residual_layers))

        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, inputdim * 2, 1)  # noise(2) + log_sigma(2 channels)
        nn.init.xavier_uniform_(self.output_projection2.weight)
        if self.output_projection2.bias is not None:
            nn.init.zeros_(self.output_projection2.bias)

    def forward(self, x, diffusion_step, cond_info=None):
        B, inputdim, K, L = x.shape
        # Flatten time and feature
        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.silu(x)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        # # Aggregate skips
        x = x.reshape(B, self.channels, K, L)
        skip_sum = torch.zeros_like(x)

        for block in self.residual_layers:
            x, skip = block(x, cond_info, diffusion_emb)
            skip_sum.add_(skip)
        x = skip_sum.mul_(self.inv_sqrt_layers)
        
        # Final projections
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)
        x = F.silu(x)
        x = self.dropout(x)
        x = self.output_projection2(x)
        x = x.reshape(B, inputdim * 2, K, L)
        eps, log_sigma = x[:, :inputdim], x[:, inputdim:]
        log_sigma = torch.clamp(log_sigma, -10, 2)
        x = torch.cat([eps, log_sigma], dim=1)
        return x
