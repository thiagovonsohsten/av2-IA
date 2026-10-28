"""Figuras adicionais para o relatório (dataset, PR, calibração, limiar, etc.)."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Caminhos replicados de config.py (sem importar, pois config importa torch)
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIGS_DIR = RESULTS_DIR / "figs"
RESULTS_CSV = RESULTS_DIR / "results.csv"

sns.set_theme(style="whitegrid", font_scale=1.05)

COLOR_H0 = "#4C72B0"
COLOR_H1 = "#55A868"


def _load_results() -> pd.DataFrame:
    return pd.read_csv(RESULTS_CSV).drop_duplicates(subset=["seed", "scenario"], keep="last")


def _load_probs():
    npz = np.load(RESULTS_DIR / "seed0_probs.npz")
    return npz["y_test"], npz["probs_h0"], npz["probs_h1"]


def _load_dataset():
    """Carrega o pneumoniamnist.npz; usa medmnist se disponível, senão baixa direto."""
    cache = Path.home() / ".medmnist" / "pneumoniamnist.npz"
    if not cache.exists():
        try:
            from medmnist import PneumoniaMNIST

            PneumoniaMNIST(split="test", download=True)
        except ImportError:
            import urllib.request

            cache.parent.mkdir(parents=True, exist_ok=True)
            url = "https://zenodo.org/records/10519652/files/pneumoniamnist.npz"
            print(f"Baixando {url} ...")
            urllib.request.urlretrieve(url, cache)
    return np.load(cache)


def plot_dataset_samples():
    """Grade de amostras normal vs pneumonia do conjunto de teste."""
    data = _load_dataset()
    imgs = data["test_images"]
    labels = data["test_labels"].squeeze()
    rng = np.random.default_rng(0)
    idx_norm = rng.choice(np.where(labels == 0)[0], 5, replace=False)
    idx_pneu = rng.choice(np.where(labels == 1)[0], 5, replace=False)

    fig, axes = plt.subplots(2, 5, figsize=(9, 4))
    for ax, i in zip(axes[0], idx_norm):
        ax.imshow(imgs[i], cmap="gray")
        ax.axis("off")
    for ax, i in zip(axes[1], idx_pneu):
        ax.imshow(imgs[i], cmap="gray")
        ax.axis("off")
    axes[0, 0].set_title("Normal", loc="left", fontsize=11)
    axes[1, 0].set_title("Pneumonia", loc="left", fontsize=11)
    fig.suptitle("Amostras do PneumoniaMNIST (teste)")
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "amostras_dataset.png", dpi=150)
    plt.close(fig)


def plot_class_distribution():
    """Distribuição de classes por divisão oficial."""
    data = _load_dataset()
    counts = {}
    for split in ("train", "val", "test"):
        labels = data[f"{split}_labels"].squeeze()
        counts[split] = (int((labels == 0).sum()), int((labels == 1).sum()))

    splits = ["train", "val", "test"]
    normal = [counts[s][0] for s in splits]
    pneumonia = [counts[s][1] for s in splits]
    x = np.arange(len(splits))
    w = 0.38

    fig, ax = plt.subplots(figsize=(6, 4))
    b0 = ax.bar(x - w / 2, normal, w, label="Normal", color=COLOR_H0)
    b1 = ax.bar(x + w / 2, pneumonia, w, label="Pneumonia", color="#C44E52")
    ax.bar_label(b0, padding=2, fontsize=9)
    ax.bar_label(b1, padding=2, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(["Treino", "Validação", "Teste"])
    ax.set_ylabel("Número de imagens")
    ax.set_title("Distribuição de classes por divisão")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "dist_classes.png", dpi=150)
    plt.close(fig)


def plot_acc_auc_box(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, col, lab in zip(axes, ["accuracy", "auc"], ["Acurácia", "AUC-ROC"]):
        sns.boxplot(data=df, x="scenario", y=col, hue="scenario", ax=ax, legend=False)
        ax.set_xlabel("Cenário")
        ax.set_ylabel(lab)
        ax.set_title(f"{lab} por cenário")
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "acc_auc_boxplot.png", dpi=150)
    plt.close(fig)


def plot_f1_scatter(df: pd.DataFrame):
    h0 = df[df["scenario"] == "H0"].sort_values("seed")["f1"].values
    h1 = df[df["scenario"] == "H1"].sort_values("seed")["f1"].values
    lo = min(h0.min(), h1.min()) - 0.005
    hi = max(h0.max(), h1.max()) + 0.005

    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.scatter(h0, h1, s=45, alpha=0.8, color=COLOR_H1, edgecolor="black", linewidth=0.5)
    ax.plot([lo, hi], [lo, hi], "r--", label="y = x (empate)")
    ax.set_xlabel("F1 — H0 (modelo único)")
    ax.set_ylabel("F1 — H1 (ensemble)")
    ax.set_title("F1 pareado por seed")
    n_above = int((h1 > h0).sum())
    ax.legend(title=f"H1 vence em {n_above}/{len(h0)} seeds")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "f1_scatter_pareado.png", dpi=150)
    plt.close(fig)


def plot_pr_curves():
    y, p0, p1 = _load_probs()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for probs, label, color in [(p0, "H0 (modelo único)", COLOR_H0), (p1, "H1 (ensemble)", COLOR_H1)]:
        prec, rec, _ = precision_recall_curve(y, probs)
        ap = average_precision_score(y, probs)
        ax.plot(rec, prec, label=f"{label} (AP={ap:.3f})", color=color)
    ax.set_xlabel("Revocação (recall)")
    ax.set_ylabel("Precisão")
    ax.set_title("Curvas precisão-revocação (seed 0)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "pr_curves.png", dpi=150)
    plt.close(fig)


def plot_calibration():
    y, p0, p1 = _load_probs()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot([0, 1], [0, 1], "k:", label="Calibração perfeita")
    for probs, label, color in [(p0, "H0 (modelo único)", COLOR_H0), (p1, "H1 (ensemble)", COLOR_H1)]:
        frac_pos, mean_pred = calibration_curve(y, probs, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=label, color=color)
    ax.set_xlabel("Probabilidade média predita")
    ax.set_ylabel("Fração de positivos observada")
    ax.set_title("Curva de calibração (seed 0)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "calibracao.png", dpi=150)
    plt.close(fig)


def plot_threshold_sensitivity():
    y, p0, p1 = _load_probs()
    thresholds = np.linspace(0.05, 0.95, 91)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for probs, label, color in [(p0, "H0 (modelo único)", COLOR_H0), (p1, "H1 (ensemble)", COLOR_H1)]:
        f1s = [f1_score(y, (probs >= t).astype(int)) for t in thresholds]
        ax.plot(thresholds, f1s, label=label, color=color)
        best_t = thresholds[int(np.argmax(f1s))]
        ax.axvline(best_t, color=color, linestyle=":", alpha=0.6)
    ax.axvline(0.5, color="red", linestyle="--", alpha=0.7, label="Limiar usado (0,5)")
    ax.set_xlabel("Limiar de decisão")
    ax.set_ylabel("F1-score")
    ax.set_title("Sensibilidade do F1 ao limiar (seed 0)")
    ax.legend(loc="lower center")
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "limiar_f1.png", dpi=150)
    plt.close(fig)


def plot_time_per_seed(df: pd.DataFrame):
    h0 = df[df["scenario"] == "H0"].sort_values("seed")
    h1 = df[df["scenario"] == "H1"].sort_values("seed")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(h0["seed"], h0["train_time_sec"], marker="o", label="H0 (modelo único)", color=COLOR_H0, alpha=0.85)
    ax.plot(h1["seed"], h1["train_time_sec"], marker="s", label="H1 (ensemble)", color=COLOR_H1, alpha=0.85)
    ax.set_xlabel("Seed (simulação)")
    ax.set_ylabel("Tempo de treino (s)")
    ax.set_title("Tempo de treinamento por simulação")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "tempo_por_seed.png", dpi=150)
    plt.close(fig)


def plot_diff_ci(df: pd.DataFrame):
    """Diferença média do F1 com IC 95% (visual do teste de hipótese)."""
    h0 = df[df["scenario"] == "H0"].sort_values("seed")["f1"].values
    h1 = df[df["scenario"] == "H1"].sort_values("seed")["f1"].values
    diff = h1 - h0
    n = len(diff)
    mean = diff.mean()
    sem = diff.std(ddof=1) / np.sqrt(n)
    ci = stats.t.interval(0.95, n - 1, loc=mean, scale=sem)

    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.errorbar(
        [mean], [0],
        xerr=[[mean - ci[0]], [ci[1] - mean]],
        fmt="o", color=COLOR_H1, capsize=6, markersize=9,
        label=f"Média = {mean:.4f}\nIC 95% = [{ci[0]:.4f}; {ci[1]:.4f}]",
    )
    ax.axvline(0, color="red", linestyle="--", label="H0: diferença = 0")
    ax.set_yticks([])
    ax.set_xlabel("Diferença média F1(H1) − F1(H0)")
    ax.set_title("Diferença pareada com IC 95%")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "diff_ic95.png", dpi=150)
    plt.close(fig)


def plot_pipeline():
    """Diagrama do pipeline experimental."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axis("off")

    def box(x, y, w, h, text, fc="#E8EEF7", ec="#4C72B0"):
        rect = plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=1.5, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5, zorder=3)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color="#444", lw=1.4), zorder=1)

    # Entrada
    box(0.02, 0.40, 0.17, 0.20, "PneumoniaMNIST\n(28×28 → 64×64,\n3 canais, norm. ImageNet)")
    # Treino compartilhado
    box(0.27, 0.62, 0.18, 0.16, "ResNet18\n(fine-tuning)")
    box(0.27, 0.40, 0.18, 0.16, "MobileNetV2\n(fine-tuning)", fc="#EAF4EA", ec="#55A868")
    box(0.27, 0.18, 0.18, 0.16, "EfficientNet-B0\n(fine-tuning)", fc="#EAF4EA", ec="#55A868")
    # Cenários
    box(0.55, 0.68, 0.18, 0.14, "H0: predição\nResNet18")
    box(0.55, 0.26, 0.18, 0.14, "H1: votação suave\n(média das probs)", fc="#EAF4EA", ec="#55A868")
    # Avaliação
    box(0.81, 0.47, 0.17, 0.18, "Teste (624 img)\nAcc, F1, AUC\n× 30 seeds")

    arrow(0.19, 0.50, 0.27, 0.70)
    arrow(0.19, 0.50, 0.27, 0.48)
    arrow(0.19, 0.50, 0.27, 0.26)
    arrow(0.45, 0.70, 0.55, 0.74)
    arrow(0.45, 0.70, 0.55, 0.35)
    arrow(0.45, 0.48, 0.55, 0.33)
    arrow(0.45, 0.26, 0.55, 0.31)
    arrow(0.73, 0.75, 0.81, 0.58)
    arrow(0.73, 0.33, 0.81, 0.52)

    ax.text(0.36, 0.92, "Pipeline experimental (repetido para cada seed s = 0, ..., 29)",
            ha="center", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "pipeline.png", dpi=150)
    plt.close(fig)


def main():
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    df = _load_results()
    plot_dataset_samples()
    plot_class_distribution()
    plot_acc_auc_box(df)
    plot_f1_scatter(df)
    plot_pr_curves()
    plot_calibration()
    plot_threshold_sensitivity()
    plot_time_per_seed(df)
    plot_diff_ci(df)
    plot_pipeline()
    print(f"Figuras extras salvas em {FIGS_DIR}")


if __name__ == "__main__":
    main()
