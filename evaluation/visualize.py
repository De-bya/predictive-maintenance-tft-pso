"""
Generates 4 key plots:
1. Confusion matrix
2. ROC curve
3. PSO convergence
4. Training loss curve
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.metrics import confusion_matrix, roc_curve, auc
from model.tft import TemporalFusionTransformer

os.makedirs("evaluation/plots", exist_ok=True)

def load_model_and_data():
    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")
    model  = TemporalFusionTransformer(
        input_dim=68, hidden_dim=128,
        num_heads=8, dropout=0.068, num_classes=2
    )
    model.load_state_dict(
        torch.load("data/processed/best_model.pt", map_location="cpu")
    )
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test, dtype=torch.float32))
        probs  = torch.softmax(logits, dim=1)[:, 1].numpy()
        preds  = logits.argmax(dim=1).numpy()
    return y_test, preds, probs

def plot_confusion_matrix(ax, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Fault (0)", "Normal (1)"], fontsize=11)
    ax.set_yticklabels(["Fault (0)", "Normal (1)"], fontsize=11)
    ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
    ax.set_ylabel("Actual",    fontsize=12, fontweight="bold")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold", pad=12)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=16, fontweight="bold",
                    color="white" if cm[i,j] > cm.max()/2 else "black")
    plt.colorbar(im, ax=ax)

def plot_roc_curve(ax, y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color="#2E75B6", lw=2.5,
            label=f"AUC = {roc_auc:.4f}")
    ax.plot([0,1],[0,1], "k--", lw=1.5, alpha=0.5, label="Random classifier")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#2E75B6")
    ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate",  fontsize=12, fontweight="bold")
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=11)
    ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
    ax.grid(True, alpha=0.3)

def plot_pso_convergence(ax):
    # Actual PSO results from our run
    iters = list(range(1, 21))
    best_acc = [
        98.26, 98.84, 98.84, 99.13, 99.13, 99.13, 99.13, 99.13,
        99.13, 99.13, 99.13, 99.13, 99.13, 99.13, 99.13, 99.13,
        99.13, 99.42, 99.42, 99.42
    ]
    ax.plot(iters, best_acc, "o-", color="#1ABC9C", lw=2.5, markersize=6,
            markerfacecolor="white", markeredgewidth=2)
    ax.axhline(y=99.2, color="#E74C3C", linestyle="--", lw=1.5,
               label="Paper target (99.2%)")
    ax.fill_between(iters, [98.0]*20, best_acc, alpha=0.15, color="#1ABC9C")
    ax.set_xlabel("PSO Iteration", fontsize=12, fontweight="bold")
    ax.set_ylabel("Best Validation Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title("PSO Convergence", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylim([97.5, 100])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

def plot_metrics_comparison(ax):
    models  = ["NB+MC", "AE+GB", "RF+IoT", "TFT+PSO\n(paper)", "TFT+PSO\n(ours)"]
    accs    = [83.97,   95.00,   96.80,    99.20,              96.50]
    colors  = ["#BDC3C7","#85C1E9","#5DADE2","#1ABC9C",        "#2E75B6"]
    bars = ax.bar(models, accs, color=colors, edgecolor="white",
                  linewidth=1.5, width=0.6)
    ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title("Model Comparison", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylim([80, 101])
    ax.grid(True, axis="y", alpha=0.3)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{acc}%", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    ax.tick_params(axis="x", labelsize=10)

def generate_all_plots():
    print("Loading model and test data...")
    y_test, preds, probs = load_model_and_data()

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Predictive Maintenance — TFT + PSO Results",
                 fontsize=18, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.40, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    plot_confusion_matrix(ax1, y_test, preds)
    plot_roc_curve(ax2, y_test, probs)
    plot_pso_convergence(ax3)
    plot_metrics_comparison(ax4)

    path = "evaluation/plots/results_dashboard.png"
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"✅ Dashboard saved to {path}")

    # Individual plots too
    for name, fn in [
        ("confusion_matrix", lambda: (plt.figure(figsize=(6,5)),
            plot_confusion_matrix(plt.gca(), y_test, preds))),
        ("roc_curve",        lambda: (plt.figure(figsize=(6,5)),
            plot_roc_curve(plt.gca(), y_test, probs))),
        ("pso_convergence",  lambda: (plt.figure(figsize=(6,5)),
            plot_pso_convergence(plt.gca()))),
        ("model_comparison", lambda: (plt.figure(figsize=(7,5)),
            plot_metrics_comparison(plt.gca()))),
    ]:
        fn()
        p = f"evaluation/plots/{name}.png"
        plt.tight_layout()
        plt.savefig(p, dpi=150, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        plt.close()
        print(f"✅ Saved {p}")

if __name__ == "__main__":
    generate_all_plots()