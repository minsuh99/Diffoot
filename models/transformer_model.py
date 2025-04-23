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
    def __init__(self, input_dim=158, hidden_dim=256, output_dim=22, projection_dim=64, num_layers=4, nhead=8, seq_len=125):
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

    def forward(self, condition, target=None, teacher_forcing_ratio=0.5):
        B, T, _ = condition.size()

        # Encode
        encoder_input = self.encoder_input_proj(condition)
        encoder_input = self.encoder_pos_encoding(encoder_input)
        memory = self.encoder(encoder_input)

        # Decode with teacher forcing or autoregressive
        outputs = []
        hidden_dim = memory.size(-1)
        decoder_input = torch.zeros(B, 1, hidden_dim, device=condition.device)

        for t in range(T):
            use_teacher_forcing = (
                self.training and 
                target is not None and 
                torch.rand(1).item() < teacher_forcing_ratio
            )

            if t == 0:
                decoder_input_t = decoder_input
            else:
                if use_teacher_forcing:
                    prev_gt = target[:, t - 1:t]
                    decoder_input_t = self.decoder_input_proj(prev_gt)
                else:
                    prev_pred = self.output_proj(decoder_output)
                    decoder_input_t = self.decoder_input_proj(prev_pred)

            decoder_input_t = self.decoder_pos_encoding(decoder_input_t)

            decoder_output = self.decoder(tgt=decoder_input_t, memory=memory)

            output_t = self.output_proj(decoder_output)
            outputs.append(output_t)


        output_seq = torch.cat(outputs, dim=1)  # [B, T, output_dim]
        return output_seq
