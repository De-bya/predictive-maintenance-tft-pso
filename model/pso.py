import json
import os
import numpy as np
import torch
import yaml
from model.tft import TemporalFusionTransformer
from torch.utils.data import DataLoader, TensorDataset

HISTORY_PATH = "data/processed/pso_history.json"

# Valid attention-head counts: hidden_size must be divisible by num_heads
HEAD_CHOICES = [1, 2, 4, 8]

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def decode_particle(position):
    """Map a continuous particle position to concrete hyperparameters.
    Position = [learning_rate, num_layers, head_index]"""
    lr         = float(position[0])
    num_layers = int(round(position[1]))
    num_heads  = HEAD_CHOICES[int(round(position[2]))]
    return lr, num_layers, num_heads

def quick_train_eval(lr, num_layers, num_heads, hidden_size, dropout,
                     X_train, y_train, X_val, y_val,
                     epochs=5, batch_size=32):
    """Fast train + eval used inside PSO objective"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TemporalFusionTransformer(
        input_dim=X_train.shape[2],
        hidden_dim=hidden_size,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        num_classes=2
    ).to(device)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    X_v = torch.tensor(X_val, dtype=torch.float32)

    loader = DataLoader(TensorDataset(X_t, y_t),
                        batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    # Validation accuracy
    model.eval()
    with torch.no_grad():
        logits = model(X_v.to(device))
        preds  = logits.argmax(dim=1).cpu().numpy()
    val_acc = (preds == y_val).mean()
    return 1.0 - val_acc   # PSO minimizes, so return error

class PSO:
    """
    Particle Swarm Optimization — Equation 4 from the paper.
    v_i(t+1) = w*v_i(t) + c1*r1*(pbest_i - x_i) + c2*r2*(gbest - x_i)

    Each particle is a point in the paper's search space:
    [learning_rate, num_layers, num_attention_heads]
    """
    def __init__(self, n_particles=10, iterations=20,
                 w=0.7, c1=1.5, c2=1.5):
        self.n_particles = n_particles
        self.iterations  = iterations
        self.w, self.c1, self.c2 = w, c1, c2

        # Search bounds: [lr, num_layers, index into HEAD_CHOICES]
        self.bounds = {
            'low':  np.array([1e-4, 1, 0]),
            'high': np.array([1e-2, 3, len(HEAD_CHOICES) - 1])
        }

    def optimize(self, X_train, y_train, X_val, y_val, hidden_size, dropout):
        dim = 3
        low, high = self.bounds['low'], self.bounds['high']

        # Initialize particles
        pos = np.random.uniform(low, high, (self.n_particles, dim))
        vel = np.zeros_like(pos)
        pbest_pos   = pos.copy()
        pbest_score = np.full(self.n_particles, np.inf)
        gbest_pos   = pos[0].copy()
        gbest_score = np.inf
        history     = []

        print(f"\n🔍 PSO starting: {self.n_particles} particles × {self.iterations} iterations")

        for it in range(self.iterations):
            for i in range(self.n_particles):
                lr, num_layers, num_heads = decode_particle(pos[i])
                score = quick_train_eval(
                    lr=lr, num_layers=num_layers, num_heads=num_heads,
                    hidden_size=hidden_size, dropout=dropout,
                    X_train=X_train, y_train=y_train,
                    X_val=X_val,     y_val=y_val
                )

                if score < pbest_score[i]:
                    pbest_score[i] = score
                    pbest_pos[i]   = pos[i].copy()

                if score < gbest_score:
                    gbest_score = score
                    gbest_pos   = pos[i].copy()

            # Equation 4: update velocity and position
            r1  = np.random.rand(self.n_particles, dim)
            r2  = np.random.rand(self.n_particles, dim)
            vel = (self.w * vel
                   + self.c1 * r1 * (pbest_pos - pos)
                   + self.c2 * r2 * (gbest_pos  - pos))
            pos = np.clip(pos + vel, low, high)

            best_acc = (1 - gbest_score) * 100
            history.append(round(float(best_acc), 2))
            print(f"  Iter {it+1:02d}/{self.iterations} | "
                  f"Best error: {gbest_score:.4f} | "
                  f"Best acc: {best_acc:.2f}%")

        lr, num_layers, num_heads = decode_particle(gbest_pos)
        best = {
            'hidden_size':     hidden_size,
            'attention_heads': num_heads,
            'num_layers':      num_layers,
            'dropout':         dropout,
            'learning_rate':   lr,
            'val_accuracy':    float(1 - gbest_score)
        }
        print(f"\n✅ PSO complete — Best params: {best}")

        # Saved so the convergence plot uses the real run
        os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
        with open(HISTORY_PATH, "w") as f:
            json.dump(history, f)
        return best

def run_pso(X_train, y_train):
    config  = load_config()
    pso_cfg = config['pso']

    # Use 80% of train for PSO training, 20% for PSO validation
    split = int(len(X_train) * 0.8)
    X_tr, X_val = X_train[:split], X_train[split:]
    y_tr, y_val = y_train[:split], y_train[split:]

    pso = PSO(
        n_particles=pso_cfg['n_particles'],
        iterations=pso_cfg['iterations'],
        w=pso_cfg['options']['w'],
        c1=pso_cfg['options']['c1'],
        c2=pso_cfg['options']['c2']
    )
    return pso.optimize(X_tr, y_tr, X_val, y_val,
                        hidden_size=config['model']['hidden_size'],
                        dropout=config['model']['dropout'])
