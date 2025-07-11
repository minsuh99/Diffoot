import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=250):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class DefenseTrajectoryTransformer(nn.Module):
    def __init__(self, input_dim=22, hidden_dim=256, output_dim=22, projection_dim=64, num_layers=6, nhead=8, seq_len=100):
        super().__init__()

        self.encoder_input_proj = nn.Linear(input_dim, hidden_dim)
        self.decoder_input_proj = nn.Linear(output_dim, hidden_dim)
        self.encoder_pos_encoding = PositionalEncoding(hidden_dim, max_len=seq_len)
        self.decoder_pos_encoding = PositionalEncoding(hidden_dim, max_len=seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=0.2,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=0.2,
            activation='gelu',
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(projection_dim, output_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, condition, target=None):
        B, T, _ = condition.size()

        encoder_input = self.encoder_input_proj(condition)
        encoder_input = self.encoder_pos_encoding(encoder_input)
        memory = self.encoder(encoder_input)

        if self.training and target is not None:
            decoder_input = self.decoder_input_proj(target)
            decoder_input = self.decoder_pos_encoding(decoder_input)

            tgt_mask = self._generate_square_subsequent_mask(T, condition.device)
            
            decoder_output = self.decoder(
                tgt=decoder_input, 
                memory=memory, 
                tgt_mask=tgt_mask
            )
            output_seq = self.output_proj(decoder_output)
            
        else:
            output_seq = self._autoregressive_decode(memory, T, condition.device)
            
        return output_seq

    def _autoregressive_decode(self, memory, T, device):
        B = memory.size(0)

        decoder_input = torch.zeros(B, 1, self.output_proj[-1].out_features, device=device)
        outputs = []
        
        for t in range(T):
            decoder_input_proj = self.decoder_input_proj(decoder_input)
            decoder_input_proj = self.decoder_pos_encoding(decoder_input_proj)
            
            # Decode
            decoder_output = self.decoder(
                tgt=decoder_input_proj, 
                memory=memory
            )

            output_t = self.output_proj(decoder_output[:, -1:, :])  # [B, 1, output_dim]
            outputs.append(output_t)

            decoder_input = torch.cat([decoder_input, output_t], dim=1)
        
        return torch.cat(outputs, dim=1)  # [B, T, output_dim]

    def _generate_square_subsequent_mask(self, sz, device):
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask