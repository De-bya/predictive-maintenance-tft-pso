import torch
import torch.nn as nn
import numpy as np

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

        # ── Explainability: store attention weights ──────────────
        self._last_attn_weights = None   # shape: (B, T, T)

    def forward(self, x, return_attention=False):
        x = self.input_projection(x)
        x = self.var_selection(x)

        lstm_out, _ = self.lstm(x)

        # need_weights=True → get attention weights back
        attn_out, attn_weights = self.attention(
            lstm_out, lstm_out, lstm_out,
            need_weights=True,
            average_attn_weights=True   # average across heads → (B, T, T)
        )
        # Store for later extraction
        self._last_attn_weights = attn_weights.detach()

        attn_out = self.dropout_(attn_out)
        attn_out = self.norm1(attn_out + lstm_out)
        attn_out = self.post_attention_grn(attn_out)

        out    = attn_out[:, -1, :]
        logits = self.classifier(out)

        if return_attention:
            return logits, attn_weights
        return logits

    def get_attention_weights(self):
        """Return last stored attention weights (B, T, T)"""
        return self._last_attn_weights

class MultiLabelTFT(nn.Module):
    """
    TFT with 5 independent output heads — one per target:
    cooler (3 classes), valve (4), pump (3), accumulator (4), stable_flag (2)
    """
    TARGET_CLASSES = {
        'cooler': 3, 'valve': 4, 'pump': 3,
        'accumulator': 4, 'stable_flag': 2
    }

    def __init__(self, input_dim, hidden_dim=128, num_heads=4,
                 num_layers=2, dropout=0.1):
        super().__init__()

        # Shared TFT backbone
        self.backbone = TemporalFusionTransformer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            num_classes=hidden_dim   # backbone outputs features not classes
        )
        # Override the classifier with an identity-like projection
        self.backbone.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 5 independent output heads
        self.heads = nn.ModuleDict({
            name: nn.Linear(hidden_dim // 2, n_classes)
            for name, n_classes in self.TARGET_CLASSES.items()
        })

        self._last_attn_weights = None

    def forward(self, x, return_attention=False):
        # Run backbone (returns features from the intermediate layer)
        x_proj = self.backbone.input_projection(x)
        x_proj = self.backbone.var_selection(x_proj)
        lstm_out, _ = self.backbone.lstm(x_proj)

        attn_out, attn_weights = self.backbone.attention(
            lstm_out, lstm_out, lstm_out,
            need_weights=True, average_attn_weights=True
        )
        self._last_attn_weights = attn_weights.detach()
        attn_out = self.backbone.dropout_(attn_out)
        attn_out = self.backbone.norm1(attn_out + lstm_out)
        attn_out = self.backbone.post_attention_grn(attn_out)

        features = self.backbone.classifier(attn_out[:, -1, :])

        # 5 heads
        outputs = {name: head(features) for name, head in self.heads.items()}

        if return_attention:
            return outputs, attn_weights
        return outputs

    def get_attention_weights(self):
        return self._last_attn_weights


def build_model(config):
    model = TemporalFusionTransformer(
        input_dim=68,
        hidden_dim=config['model']['hidden_size'],
        num_heads=config['model']['attention_heads'],
        dropout=config['model']['dropout'],
        num_classes=2
    )
    total = sum(p.numel() for p in model.parameters())
    print(f"✅ TFT model built — {total:,} parameters")
    return model
