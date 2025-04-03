import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=250):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [T, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [T, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, T, d_model]

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class DefenseTrajectoryTransformer(nn.Module):
    def __init__(self, input_dim=158, hidden_dim=128, output_dim=22, num_layers=4, nhead=4, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim*4, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)  # 11 defenders × (x, y)
        )

    def forward(self, x):
        # x: [B, T, 158]
        x = self.norm(x)
        x = self.input_proj(x)              # [B, T, hidden_dim]
        x = self.pos_encoder(x)             # [B, T, hidden_dim]
        x = self.transformer(x)             # [B, T, hidden_dim]
        out = self.projection(x)            # [B, T, 22]
        return out
