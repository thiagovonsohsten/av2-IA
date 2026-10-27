"""30 simulações pareadas: H0 (ResNet18) vs H1 (ensemble)."""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from config import MODELS_ENSEMBLE, N_RUNS, RESULTS_CSV, RESULTS_DIR
from data import get_loaders
from ensemble import soft_voting
from metrics import compute_metrics
from models import build_model
from train import predict_probs, set_seed, train_model


def _load_done_seeds() -> set[int]:
    if not RESULTS_CSV.exists():
        return set()
    df = pd.read_csv(RESULTS_CSV).drop_duplicates(subset=["seed", "scenario"], keep="last")
    if "seed" not in df.columns:
        return set()
    # seed completo quando H0 e H1 existem
    counts = df.groupby("seed")["scenario"].nunique()
    return set(counts[counts >= 2].index.astype(int))


def _append_rows(rows: list[dict]):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame(rows)
    if RESULTS_CSV.exists():
        df = pd.concat([pd.read_csv(RESULTS_CSV), df_new], ignore_index=True)
    else:
        df = df_new
    df = df.drop_duplicates(subset=["seed", "scenario"], keep="last")
    df.to_csv(RESULTS_CSV, index=False)


def run_seed(seed: int) -> list[dict]:
    set_seed(seed)
    train_loader, val_loader, test_loader, y_test = get_loaders(seed)

    # ResNet18 compartilhado (mesmo treino para H0 e parte do H1)
    r18 = build_model("resnet18")
    r18, t_r18 = train_model(r18, train_loader, val_loader)
    probs_r18 = predict_probs(r18, test_loader)
    m_h0 = compute_metrics(y_test, probs_r18)

    rows = [
        {
            "seed": seed,
            "scenario": "H0",
            "accuracy": m_h0["accuracy"],
            "f1": m_h0["f1"],
            "auc": m_h0["auc"],
            "train_time_sec": t_r18,
        }
    ]

    prob_list = [probs_r18]
    total_time = t_r18

    for name in MODELS_ENSEMBLE:
        if name == "resnet18":
            continue
        model = build_model(name)
        model, t = train_model(model, train_loader, val_loader)
        prob_list.append(predict_probs(model, test_loader))
        total_time += t

    probs_ens = soft_voting(prob_list)
    m_h1 = compute_metrics(y_test, probs_ens)
    rows.append(
        {
            "seed": seed,
            "scenario": "H1",
            "accuracy": m_h1["accuracy"],
            "f1": m_h1["f1"],
            "auc": m_h1["auc"],
            "train_time_sec": total_time,
        }
    )

    # Guarda probs da seed 0 para gráficos ROC/confusão
    if seed == 0:
        import numpy as np

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(
            RESULTS_DIR / "seed0_probs.npz",
            y_test=y_test,
            probs_h0=probs_r18,
            probs_h1=probs_ens,
        )

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=N_RUNS)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    if args.epochs is not None:
        config.EPOCHS = args.epochs

    done = _load_done_seeds()
    print(f"Device: {config.DEVICE}")
    print(f"Épocas: {config.EPOCHS} | Runs: {args.runs}")

    for seed in range(args.runs):
        if seed in done:
            print(f"Seed {seed}: já concluída, pulando.")
            continue
        print(f"Seed {seed}...", flush=True)
        rows = run_seed(seed)
        _append_rows(rows)
        print(f"  H0 F1={rows[0]['f1']:.4f} | H1 F1={rows[1]['f1']:.4f}", flush=True)

    print(f"Resultados em {RESULTS_CSV}")


if __name__ == "__main__":
    main()
