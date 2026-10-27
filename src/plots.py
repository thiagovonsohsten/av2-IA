"""Figuras para relatório e apresentação."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import FIGS_DIR, RESULTS_CSV, RESULTS_DIR

sns.set_theme(style="whitegrid", font_scale=1.05)


def _load():
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"Execute simulate.py antes. Falta {RESULTS_CSV}")
    return pd.read_csv(RESULTS_CSV).drop_duplicates(subset=["seed", "scenario"], keep="last")


def plot_f1_box(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x="scenario", y="f1", hue="scenario", ax=ax, legend=False)
    ax.set_xlabel("Cenário")
    ax.set_ylabel("F1-score")
    ax.set_title("F1 por cenário (30 simulações)")
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "f1_boxplot.png", dpi=150)
    plt.close(fig)


def plot_f1_lines(df: pd.DataFrame):
    h0 = df[df["scenario"] == "H0"].sort_values("seed")
    h1 = df[df["scenario"] == "H1"].sort_values("seed")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(h0["seed"], h0["f1"], marker="o", label="H0 (modelo único)", alpha=0.8)
    ax.plot(h1["seed"], h1["f1"], marker="s", label="H1 (ensemble)", alpha=0.8)
    ax.set_xlabel("Seed (simulação)")
    ax.set_ylabel("F1-score")
    ax.legend()
    ax.set_title("F1 por simulação")
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "f1_por_seed.png", dpi=150)
    plt.close(fig)


def plot_diff_hist(df: pd.DataFrame):
    h0 = df[df["scenario"] == "H0"].sort_values("seed")["f1"].values
    h1 = df[df["scenario"] == "H1"].sort_values("seed")["f1"].values
    diff = h1 - h0
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(diff, bins=12, edgecolor="black", alpha=0.75)
    ax.axvline(0, color="red", linestyle="--", label="zero")
    ax.axvline(diff.mean(), color="green", linestyle="-", label=f"média={diff.mean():.3f}")
    ax.set_xlabel("F1(H1) - F1(H0)")
    ax.set_ylabel("Frequência")
    ax.legend()
    ax.set_title("Distribuição das diferenças de F1")
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "f1_diff_hist.png", dpi=150)
    plt.close(fig)


def plot_metrics_bars(df: pd.DataFrame):
    metrics = ["accuracy", "f1", "auc"]
    labels = ["Acurácia", "F1", "AUC-ROC"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    for ax, col, lab in zip(axes, metrics, labels):
        means = df.groupby("scenario")[col].mean()
        stds = df.groupby("scenario")[col].std()
        x = np.arange(len(means))
        ax.bar(x, means, yerr=stds, capsize=4, color=["#4C72B0", "#55A868"])
        ax.set_xticks(x)
        ax.set_xticklabels(means.index)
        ax.set_ylabel(lab)
        ax.set_title(lab)
    fig.suptitle("Média ± desvio padrão por cenário")
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "metricas_barras.png", dpi=150)
    plt.close(fig)


def plot_train_time(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x="scenario", y="train_time_sec", hue="scenario", ax=ax, legend=False)
    ax.set_xlabel("Cenário")
    ax.set_ylabel("Tempo de treino (s)")
    ax.set_title("Custo computacional")
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "tempo_treino.png", dpi=150)
    plt.close(fig)


def plot_roc_confusion():
    npz_path = RESULTS_DIR / "seed0_probs.npz"
    if not npz_path.exists():
        return
    data = np.load(npz_path)
    y = data["y_test"]
    for key, title, fname in [
        ("probs_h0", "H0 — ROC e matriz (seed 0)", "roc_conf_h0"),
        ("probs_h1", "H1 — ROC e matriz (seed 0)", "roc_conf_h1"),
    ]:
        probs = data[key]
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        RocCurveDisplay.from_predictions(y, probs, ax=axes[0])
        axes[0].set_title("Curva ROC")
        cm = confusion_matrix(y, (probs >= 0.5).astype(int))
        ConfusionMatrixDisplay(cm).plot(ax=axes[1], colorbar=False)
        axes[1].set_title("Matriz de confusão")
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(FIGS_DIR / f"{fname}.png", dpi=150)
        plt.close(fig)


def main():
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    df = _load()
    plot_f1_box(df)
    plot_f1_lines(df)
    plot_diff_hist(df)
    plot_metrics_bars(df)
    plot_train_time(df)
    plot_roc_confusion()
    print(f"Figuras salvas em {FIGS_DIR}")


if __name__ == "__main__":
    main()
