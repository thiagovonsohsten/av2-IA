"""Teste de hipótese pareado sobre F1 (H1: ensemble > modelo único)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import RESULTS_CSV, STATS_TXT, RESULTS_DIR


def cohens_d(diff: np.ndarray) -> float:
    return float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else 0.0


def main():
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"Execute simulate.py antes. Falta {RESULTS_CSV}")

    df = pd.read_csv(RESULTS_CSV).drop_duplicates(subset=["seed", "scenario"], keep="last")
    h0 = df[df["scenario"] == "H0"].sort_values("seed")
    h1 = df[df["scenario"] == "H1"].sort_values("seed")
    if len(h0) != len(h1):
        raise ValueError("Número de linhas H0 e H1 difere.")

    if not (h0["seed"].values == h1["seed"].values).all():
        raise ValueError("Seeds de H0 e H1 não estão alinhadas; pareamento inválido.")
    f0 = h0["f1"].values
    f1 = h1["f1"].values
    diff = f1 - f0  # positivo favorece H1

    alpha = 0.05
    if len(diff) >= 3:
        shapiro_stat, shapiro_p = stats.shapiro(diff)
        normal = shapiro_p > alpha
    else:
        shapiro_stat, shapiro_p = float("nan"), float("nan")
        normal = False

    if normal:
        t_stat, p_two = stats.ttest_rel(f1, f0)
        p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2
        test_name = "t pareado (unilateral)"
    else:
        try:
            res = stats.wilcoxon(f1, f0, alternative="greater")
            t_stat, p_one = res.statistic, res.pvalue
        except ValueError:
            t_stat, p_one = float("nan"), 1.0
        test_name = "Wilcoxon (unilateral, H1 > H0)"

    mean_diff = diff.mean()
    se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else 0.0
    # IC 95% com distribuição t (gl = n-1), adequado a amostras pequenas
    t_crit = stats.t.ppf(0.975, len(diff) - 1) if len(diff) > 1 else float("nan")
    ci_low = mean_diff - t_crit * se
    ci_high = mean_diff + t_crit * se
    d = cohens_d(diff)

    reject = p_one < alpha
    conclusion = (
        "Rejeita-se H0: o ensemble (H1) é estatisticamente mais eficaz em F1 (alpha=0,05)."
        if reject
        else "Não se rejeita H0: não há evidência suficiente de que o ensemble supere o modelo único."
    )

    lines = [
        "=== Teste de hipótese (métrica principal: F1) ===",
        f"Amostras pareadas (seeds): n = {len(diff)}",
        f"Média F1 H0 (modelo único): {f0.mean():.4f} (dp={f0.std(ddof=1):.4f})",
        f"Média F1 H1 (ensemble):     {f1.mean():.4f} (dp={f1.std(ddof=1):.4f})",
        f"Diferença média (H1 - H0): {mean_diff:.4f}",
        f"IC 95% da diferença: [{ci_low:.4f}, {ci_high:.4f}]",
        f"Cohen's d (pareado): {d:.4f}",
        "",
        f"Shapiro-Wilk (normalidade das diferenças): W={shapiro_stat:.4f}, p={shapiro_p:.4f}",
        f"Teste escolhido: {test_name}",
        f"Estatística: {t_stat} | p (unilateral): {p_one:.6f}",
        "",
        conclusion,
    ]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    STATS_TXT.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
