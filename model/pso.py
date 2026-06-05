import numpy as np
import torch
import yaml
from model.tft import build_model, TemporalFusionTransformer
from torch.utils.data import DataLoader, TensorDataset

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def quick_train_eval(hidden_size, num_heads, dropout, lr,
                     X_train, y_train, X_val, y_val,
                     epochs=5, batch_size=32):
    """Fast train + eval used inside PSO objective"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TemporalFusionTransformer(
        input_dim=X_train.shape[2],
        hidden_dim=int(hidden_size),
        num_heads=int(num_heads),
        dropout=float(dropout),
        num_classes=2
    ).to(device)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    X_v = torch.tensor(X_val, dtype=torch.float32)
    y_v = torch.tensor(y_val, dtype=torch.long)

    loader = DataLoader(TensorDataset(X_t, y_t),
                        batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
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
    """
    def __init__(self, n_particles=10, iterations=20,
                 w=0.7, c1=1.5, c2=1.5):
        self.n_particles = n_particles
        self.iterations  = iterations
        self.w, self.c1, self.c2 = w, c1, c2

        # Search bounds: [hidden_size, num_heads, dropout, lr]
        self.bounds = {
            'low':  np.array([32,   2, 0.05, 1e-4]),
            'high': np.array([256,  8, 0.30, 1e-2])
        }

    def optimize(self, X_train, y_train, X_val, y_val):
            dim = 4  # hidden_size, num_heads, dropout, lr
            low, high = self.bounds['low'], self.bounds['high']

            # Initialize particles
            pos = np.random.uniform(low, high, (self.n_particles, dim))
            vel = np.zeros_like(pos)
            pbest_pos   = pos.copy()
            pbest_score = np.full(self.n_particles, np.inf)
            gbest_pos   = pos[0].copy()
            gbest_score = np.inf

            print(f"\n🔍 PSO starting: {self.n_particles} particles × {self.iterations} iterations")

            for it in range(self.iterations):
                for i in range(self.n_particles):

                    # ── FIX: snap hidden_size to a multiple of 8 ──────────────
                    raw_hidden = int(np.clip(pos[i, 0], 32, 256))
                    hidden = max(8, round(raw_hidden / 8) * 8)   # nearest multiple of 8

                    # Pick largest valid head count that divides hidden evenly
                    for heads in [8, 4, 2, 1]:
                        if hidden % heads == 0:
                            break
                    # ──────────────────────────────────────────────────────────

                    score = quick_train_eval(
                        hidden_size=hidden,
                        num_heads=heads,
                        dropout=float(np.clip(pos[i, 2], 0.05, 0.30)),
                        lr=float(np.clip(pos[i, 3], 1e-4, 1e-2)),
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

                print(f"  Iter {it+1:02d}/{self.iterations} | "
                    f"Best error: {gbest_score:.4f} | "
                    f"Best acc: {(1-gbest_score)*100:.2f}%")

            # Snap final best params too
            final_hidden = max(8, round(int(gbest_pos[0]) / 8) * 8)
            for heads in [8, 4, 2, 1]:
                if final_hidden % heads == 0:
                    break

            best = {
                'hidden_size':     final_hidden,
                'attention_heads': heads,
                'dropout':         float(np.clip(gbest_pos[2], 0.05, 0.30)),
                'learning_rate':   float(np.clip(gbest_pos[3], 1e-4, 1e-2)),
                'val_accuracy':    float(1 - gbest_score)
            }
            print(f"\n✅ PSO complete — Best params: {best}")
            return best

def run_pso(X_train, y_train):
    config = load_config()
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
    return pso.optimize(X_tr, y_tr, X_val, y_val)