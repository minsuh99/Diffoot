import torch
import torch.nn as nn

class DefenseTrajectoryPredictorLSTM(nn.Module):
    def __init__(self, input_dim=22, hidden_dim=128, num_layers=2, dropout=0.1, target_len=100):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.target_len = target_len

        # Simple LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # Direct prediction of entire future sequence
        self.fc = nn.Linear(hidden_dim, target_len * input_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.init_weights()
    
    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.zero_()
                n = param.data.size(0)
                param.data[n//4 : n//2].fill_(1.0)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, past_sequence, target_sequence=None):
        batch_size = past_sequence.shape[0]
        
        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(past_sequence)  # [B, T_past, hidden_dim]
        
        # Use last hidden state
        last_hidden = h_n[-1]  # [B, hidden_dim] - last layer's hidden state
        
        # Predict entire future sequence at once
        output_flat = self.fc(self.dropout(last_hidden))  # [B, target_len * input_dim]
        
        # Reshape to sequence format
        output = output_flat.view(batch_size, self.target_len, self.input_dim)  # [B, T_future, 22]
        
        return output