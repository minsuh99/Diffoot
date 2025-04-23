import torch
import torch.nn as nn

class VanillaLSTM(nn.Module):
    def __init__(self, input_dim=158, hidden_dim=256, projection_dim=64, output_dim=22, num_layers=2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=0.2,
            batch_first=True
        )

        self.projection = nn.Sequential(
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
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.zeros_(param.data)

    def forward(self, condition):
        # condition: [B, T, input_dim]
        lstm_out, _ = self.lstm(condition)  # [B, T, hidden_dim]
        pred = self.projection(lstm_out)    # [B, T, output_dim]
        return pred



import torch
import torch.nn as nn

class DefenseTrajectoryPredictorLSTM(nn.Module):
    def __init__(self, input_dim=158, hidden_dim=256, projection_dim=64, output_dim=22, num_layers=3):
        super().__init__()

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=0.2,
            batch_first=True
        )

        self.decoder = nn.LSTM(
            input_size=output_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=0.2,
            batch_first=True
        )

        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(projection_dim, output_dim)
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.zeros_(param.data)

    def forward(self, condition, target=None, teacher_forcing_ratio=0.5):
        B = condition.size(0)
        T_out = target.size(1) if target is not None else 125

        # Encoding
        _, (h_n, c_n) = self.encoder(condition)

        # Decoding
        outputs = []
        decoder_input = torch.zeros(B, 1, 22, device=condition.device) # First input => 0
        hidden = (h_n, c_n)

        for t in range(T_out):
            out, hidden = self.decoder(decoder_input, hidden)
            pred = self.projection(out)
            outputs.append(pred)

            # teacher forcing (yes / no)
            use_teacher_forcing = (
                self.training and 
                target is not None and 
                torch.rand(1).item() < teacher_forcing_ratio
            )
            if use_teacher_forcing:
                decoder_input = target[:, t:t+1]  # Next input -> target
            else:
                decoder_input = pred.detach()  # Next input -> pred

        output_seq = torch.cat(outputs, dim=1)  # [B, T_out, 22]
        return output_seq
