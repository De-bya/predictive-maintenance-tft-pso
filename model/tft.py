import torch
import torch.nn as nn

class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.fc1     = nn.Linear(input_dim, hidden_dim)
        self.fc2     = nn.Linear(hidden_dim, output_dim)
        self.gate    = nn.Linear(hidden_dim, output_dim)
        self.skip    = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(output_dim)
        self.elu     = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = self.skip(x)
        h  = self.elu(self.fc1(x))
        h  = self.dropout(h)
        h2 = self.fc2(h)
        g  = self.sigmoid(self.gate(h))
        return self.norm(g * h2 + (1 - g) * residual)


class TemporalFusionTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim=128,
                 num_heads=4, num_layers=2,
                 dropout=0.1, num_classes=2):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.var_selection    = GatedResidualNetwork(
            hidden_dim, hidden_dim, hidden_dim, dropout
        )
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.post_attention_grn = GatedResidualNetwork(
            hidden_dim, hidden_dim, hidden_dim, dropout
        )
        self.norm1      = nn.LayerNorm(hidden_dim)
        self.dropout_   = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        x = self.input_projection(x)
        x = self.var_selection(x)

        lstm_out, _ = self.lstm(x)

        # Equation 3: Softmax(QK^T / sqrt(d_k)) V  (self-attention: Q = K = V)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        attn_out = self.dropout_(attn_out)
        attn_out = self.norm1(attn_out + lstm_out)
        attn_out = self.post_attention_grn(attn_out)

        # Equation 5: y_hat_t = TFT(x_1:t) — predict from the last timestep
        out = attn_out[:, -1, :]
        return self.classifier(out)
