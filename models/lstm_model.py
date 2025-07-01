import torch
import torch.nn as nn

class DefenseTrajectoryPredictorLSTM(nn.Module):
    def __init__(self, input_dim=44, hidden_dim=256, num_layers=2, output_dim=2200, dropout=0.1):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )

        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)  # 100 * 22 = 2200
        )
        
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, condition):
        B = condition.size(0)

        _, (h_n, _) = self.lstm(condition)  # [B, 100, hidden_dim]

        final_hidden = h_n[-1]  # [B, hidden_dim]
        flat_output = self.output_projection(final_hidden)  # [B, 2200]
        out = flat_output.view(B, 100, 22)
        
        return out