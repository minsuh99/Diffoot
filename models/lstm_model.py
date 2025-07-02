import torch
import torch.nn as nn

class DefenseTrajectoryPredictorLSTM(nn.Module):
    def __init__(self, input_dim=22, hidden_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )

        self.output_layer = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize LSTM weights properly"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.zero_()
                n = param.data.size(0)
                param.data[n//4 : n//2].fill_(1.0)

        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # [B, 100, hidden_dim]
        lstm_out = self.dropout(lstm_out)
        output = self.output_layer(lstm_out)  # [B, 100, 22]
        
        return output