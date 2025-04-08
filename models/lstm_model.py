import torch
import torch.nn as nn

class DefenseTrajectoryPredictor(nn.Module):
    def __init__(self, input_dim=158, hidden_dim=128, projection_dim=64, output_dim=22, num_layers=2, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, output_dim)
        )

        # Linear 초기화
        self.projection.apply(self._init_linear_weights)
        
        # LSTM 초기화
        self._init_lstm_weights()
    
    def _init_linear_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _init_lstm_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
    
                    
    def forward(self, x):
        # x: [B, T, 158]
        x = self.norm(x)
        lstm_out, _ = self.lstm(x)                 # [B, T, hidden_dim]
        out = self.projection(lstm_out)            # [B, T, 22]
        return out