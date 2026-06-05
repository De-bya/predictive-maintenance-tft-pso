import torch
import torch.nn as nn
import numpy as np

class GatedResidualNetwork(nn.Module):
    """Core building block of TFT"""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.fc1      = nn.Linear(input_dim, hidden_dim)
        self.fc2      = nn.Linear(hidden_dim, output_dim)
        self.gate     = nn.Linear(hidden_dim, output_dim)
        self.skip     = nn.Linear(input_dim, output_dim)
        self.dropout  = nn.Dropout(dropout)
        self.norm     = nn.LayerNorm(output_dim)
        self.elu      = nn.ELU()
        self.sigmoid  = nn.Sigmoid()

    def forward(self, x):
        residual = self.skip(x)
        h  = self.elu(self.fc1(x))
        h  = self.dropout(h)
        h2 = self.fc2(h)
        g  = self.sigmoid(self.gate(h))
        return self.norm(g * h2 + (1 - g) * residual)

class TemporalFusionTransformer(nn.Module):
    """
    Simplified TFT for binary fault classification.
    Implements equation 3 (attention mechanism) from the paper.
    """
    def __init__(self, input_dim, hidden_dim=128,
                 num_heads=4, num_layers=2,
                 dropout=0.1, num_classes=2):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Variable Selection via GRN
        self.var_selection = GatedResidualNetwork(
            hidden_dim, hidden_dim, hidden_dim, dropout
        )

        # LSTM encoder for local temporal context
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Multi-head self-attention (Equation 3)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Post-attention GRN
        self.post_attention_grn = GatedResidualNetwork(
            hidden_dim, hidden_dim, hidden_dim, dropout
        )

        self.norm1   = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Output classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        x = self.input_projection(x)        # → (B, T, H)
        x = self.var_selection(x)           # Variable selection

        lstm_out, _ = self.lstm(x)          # → (B, T, H)

        # Self-attention (Equation 3): Softmax(QK^T / sqrt(dk)) * V
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.dropout(attn_out)
        attn_out = self.norm1(attn_out + lstm_out)  # residual

        attn_out = self.post_attention_grn(attn_out)

        # Use last timestep for classification
        out = attn_out[:, -1, :]            # → (B, H)
        logits = self.classifier(out)       # → (B, num_classes)
        return logits

def build_model(config):
    """Build TFT from config or custom params"""
    model = TemporalFusionTransformer(
        input_dim=68,   # 17 sensors × 4 stats each
        hidden_dim=config['model']['hidden_size'],
        num_heads=config['model']['attention_heads'],
        dropout=config['model']['dropout'],
        num_classes=2
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ TFT model built — {total_params:,} parameters")
    return model