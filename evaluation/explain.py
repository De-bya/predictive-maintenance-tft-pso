"""
TFT Attention Explainability
- Extracts attention weights after inference
- Maps timestep attention back to sensor importance
- Produces sensor importance heatmaps
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Sensor names matching our 68 features (17 sensors × 4 stats)
SENSOR_NAMES = [
    'PS1','PS2','PS3','PS4','PS5','PS6',
    'EPS1','FS1','FS2',
    'TS1','TS2','TS3','TS4',
    'VS1','CE','CP','SE'
]
STATS        = ['mean','std','min','max']
FEATURE_COLS = [f"{s}_{st}" for s in SENSOR_NAMES for st in STATS]  # 68 cols


def extract_sensor_importance(model, x_window):
    """
    Given a single input window (1, 60, 68), run forward pass
    and extract sensor importance from attention weights.

    Returns:
        sensor_scores  : dict  {sensor_name: importance_score}
        attn_weights   : np.ndarray (60, 60)  raw attention matrix
        top3_sensors   : list of (sensor_name, score) — top 3
    """
    model.eval()
    with torch.no_grad():
        logits, attn_weights = model(x_window, return_attention=True)

    # attn_weights: (1, T, T) → (T, T)
    attn = attn_weights[0].cpu().numpy()   # (60, 60)

    # For each timestep t, sum attention it receives from all other steps
    # → (60,) — how much each timestep is "attended to"
    timestep_importance = attn.sum(axis=0)   # (60,)
    timestep_importance /= timestep_importance.sum() + 1e-8

    # Map timestep importance back to sensor importance:
    # Each timestep has 68 features. We use the raw input values
    # (after projection) as a proxy for which sensors drove attention.
    x_np = x_window[0].cpu().numpy()        # (60, 68)

    # Weighted average of absolute feature values across timesteps
    # weighted by how much attention that timestep received
    weighted_features = (np.abs(x_np) * timestep_importance[:, None]).sum(axis=0)  # (68,)

    # Aggregate by sensor (4 stats per sensor → 1 score per sensor)
    sensor_scores = {}
    for i, sensor in enumerate(SENSOR_NAMES):
        start = i * 4
        sensor_scores[sensor] = float(weighted_features[start:start+4].mean())

    # Normalize to [0, 1]
    max_score = max(sensor_scores.values()) + 1e-8
    sensor_scores = {k: v / max_score for k, v in sensor_scores.items()}

    # Sort and get top 3
    sorted_sensors = sorted(sensor_scores.items(), key=lambda x: x[1], reverse=True)
    top3 = sorted_sensors[:3]

    return sensor_scores, attn, top3


def plot_sensor_heatmap(sensor_scores, title="Sensor Importance", save_path=None):
    """Bar chart of sensor importance scores"""
    sensors = list(sensor_scores.keys())
    scores  = list(sensor_scores.values())

    # Color by sensor group
    group_colors = {
        'PS': '#E74C3C', 'EPS': '#E67E22', 'FS': '#F1C40F',
        'TS': '#2ECC71', 'VS': '#1ABC9C', 'CE': '#3498DB',
        'CP': '#9B59B6', 'SE': '#E91E63'
    }
    colors = []
    for s in sensors:
        prefix = ''.join(c for c in s if not c.isdigit())
        colors.append(group_colors.get(prefix, '#95A5A6'))

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # ── Left: bar chart ──────────────────────────────────────────
    ax = axes[0]
    bars = ax.barh(sensors[::-1], scores[::-1], color=colors[::-1],
                   edgecolor='white', linewidth=0.5)
    ax.set_xlabel("Importance Score (normalized)", fontsize=11)
    ax.set_title("Sensor Importance", fontsize=12, fontweight='bold')
    ax.set_xlim([0, 1.1])
    ax.grid(True, axis='x', alpha=0.3)
    ax.spines[['top','right']].set_visible(False)

    # Annotate top 3
    sorted_pairs = sorted(zip(sensors, scores), key=lambda x: x[1], reverse=True)
    for rank, (sensor, score) in enumerate(sorted_pairs[:3]):
        idx = sensors[::-1].index(sensor)
        ax.text(score + 0.02, idx, f" #{rank+1}", va='center',
                fontsize=9, fontweight='bold', color='#2C3E50')

    # ── Right: grouped heatmap ────────────────────────────────────
    ax2 = axes[1]
    groups = {
        'Pressure\n(PS1-6)':   [sensor_scores.get(f'PS{i}', 0) for i in range(1, 7)],
        'Motor\n(EPS1)':       [sensor_scores.get('EPS1', 0)],
        'Flow\n(FS1-2)':       [sensor_scores.get(f'FS{i}', 0) for i in range(1, 3)],
        'Temperature\n(TS1-4)':[sensor_scores.get(f'TS{i}', 0) for i in range(1, 5)],
        'Other\n(VS,CE,CP,SE)':[sensor_scores.get(s, 0) for s in ['VS1','CE','CP','SE']],
    }
    group_names = list(groups.keys())
    group_avgs  = [np.mean(v) for v in groups.values()]

    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=0, vmax=1)
    for i, (name, avg) in enumerate(zip(group_names, group_avgs)):
        ax2.bar(i, avg, color=cmap(norm(avg)), edgecolor='white', linewidth=1.5)
        ax2.text(i, avg + 0.02, f"{avg:.2f}", ha='center',
                 fontsize=10, fontweight='bold')

    ax2.set_xticks(range(len(group_names)))
    ax2.set_xticklabels(group_names, fontsize=9)
    ax2.set_ylabel("Average Group Importance", fontsize=11)
    ax2.set_title("Sensor Group Importance", fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 1.2])
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.spines[['top','right']].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"✅ Heatmap saved to {save_path}")
    plt.close()


def plot_attention_matrix(attn_weights, save_path=None):
    """Plot the raw 60×60 attention matrix"""
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(attn_weights, cmap='Blues', aspect='auto')
    plt.colorbar(im, ax=ax, label='Attention weight')
    ax.set_xlabel("Key timestep (attended to)", fontsize=11)
    ax.set_ylabel("Query timestep (attending)", fontsize=11)
    ax.set_title("TFT Self-Attention Matrix (60 × 60)", fontsize=13, fontweight='bold')

    # Tick every 10 steps
    ticks = list(range(0, 60, 10))
    ax.set_xticks(ticks); ax.set_xticklabels([f"t-{60-t}" for t in ticks])
    ax.set_yticks(ticks); ax.set_yticklabels([f"t-{60-t}" for t in ticks])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"✅ Attention matrix saved to {save_path}")
    plt.close()


def run_explainability_demo():
    """Run on a real test sample and save all plots"""
    from model.tft import TemporalFusionTransformer

    os.makedirs("evaluation/plots", exist_ok=True)

    # Load model
    model = TemporalFusionTransformer(
        input_dim=68, hidden_dim=128,
        num_heads=8, dropout=0.068, num_classes=2
    )
    model.load_state_dict(
        torch.load("data/processed/best_model.pt", map_location="cpu")
    )

    # Load a real test sample
    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")

    # Pick one fault and one normal sample
    fault_idx  = np.where(y_test == 0)[0][0]
    normal_idx = np.where(y_test == 1)[0][0]

    for label, idx in [("FAULT", fault_idx), ("NORMAL", normal_idx)]:
        x = torch.tensor(X_test[idx:idx+1], dtype=torch.float32)
        sensor_scores, attn, top3 = extract_sensor_importance(model, x)

        print(f"\n{'='*45}")
        print(f"  Sample: {label}")
        print(f"{'='*45}")
        print(f"  Top-3 influential sensors:")
        for rank, (sensor, score) in enumerate(top3, 1):
            print(f"    #{rank}  {sensor:<6} — importance: {score:.4f}")

        plot_sensor_heatmap(
            sensor_scores,
            title=f"Sensor Importance — {label} Sample",
            save_path=f"evaluation/plots/sensor_importance_{label.lower()}.png"
        )
        plot_attention_matrix(
            attn,
            save_path=f"evaluation/plots/attention_matrix_{label.lower()}.png"
        )

    print("\n✅ Explainability demo complete — check evaluation/plots/")


if __name__ == "__main__":
    run_explainability_demo()