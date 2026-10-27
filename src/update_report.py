"""Atualiza main.tex e slides.tex com métricas de results/."""
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RESULTS_CSV, ROOT, STATS_TXT


def _br_float(x: float, dec: int = 4) -> str:
    return f"{x:.{dec}f}".replace(".", ",")


def _load_stats() -> dict:
    text = STATS_TXT.read_text(encoding="utf-8") if STATS_TXT.exists() else ""
    conclusion = ""
    for line in text.splitlines():
        if line.startswith("Rejeita-se") or line.startswith("Não se rejeita"):
            conclusion = line.strip()
            break
    p_match = re.search(r"p \(unilateral\): ([0-9.]+)", text)
    d_match = re.search(r"Cohen's d \(pareado\): ([0-9.-]+)", text)
    return {
        "conclusion": conclusion or "Conclusão não encontrada em stats.txt.",
        "p": p_match.group(1) if p_match else "—",
        "d": d_match.group(1) if d_match else "—",
    }


def _load_metrics() -> dict:
    df = pd.read_csv(RESULTS_CSV).drop_duplicates(subset=["seed", "scenario"], keep="last")
    out = {}
    for scen in ("H0", "H1"):
        sub = df[df["scenario"] == scen]
        out[scen] = {
            "acc": (sub["accuracy"].mean(), sub["accuracy"].std(ddof=1)),
            "f1": (sub["f1"].mean(), sub["f1"].std(ddof=1)),
            "auc": (sub["auc"].mean(), sub["auc"].std(ddof=1)),
        }
    out["n"] = len(df[df["scenario"] == "H0"])
    return out


def patch_main_tex(m: dict, s: dict):
    path = ROOT / "relatorio" / "main.tex"
    tex = path.read_text(encoding="utf-8")
    h0_f1 = m["H0"]["f1"]
    h1_f1 = m["H1"]["f1"]
    replacements = [
        (r"\\newcommand\{\\mediaFZero\}\{[^}]*\}", f"\\newcommand{{\\mediaFZero}}{{{_br_float(h0_f1[0])} $\\pm$ {_br_float(h0_f1[1])}}}"),
        (r"\\newcommand\{\\mediaFOne\}\{[^}]*\}", f"\\newcommand{{\\mediaFOne}}{{{_br_float(h1_f1[0])} $\\pm$ {_br_float(h1_f1[1])}}}"),
        (r"\\newcommand\{\\pvalor\}\{[^}]*\}", f"\\newcommand{{\\pvalor}}{{{s['p'].replace('.', ',')}}}"),
        (r"\\newcommand\{\\cohend\}\{[^}]*\}", f"\\newcommand{{\\cohend}}{{{s['d'].replace('.', ',')}}}"),
    ]
    for pat, repl in replacements:
        tex = re.sub(pat, lambda _m, r=repl: r, tex, count=1)
    concl = s["conclusion"].replace("_", "\\_")
    concl_line = f"\\newcommand{{\\conclusaoStats}}{{{concl}}}\n"
    tex = re.sub(
        r"\\newcommand\{\\conclusaoStats\}\{[\s\S]*?\n\}",
        lambda _m: concl_line,
        tex,
        count=1,
    )
    acc0, acc1 = m["H0"]["acc"], m["H1"]["acc"]
    auc0, auc1 = m["H0"]["auc"], m["H1"]["auc"]
    table_block = f"""H0 (modelo único) & {_br_float(acc0[0])} $\\pm$ {_br_float(acc0[1])} & {_br_float(h0_f1[0])} $\\pm$ {_br_float(h0_f1[1])} & {_br_float(auc0[0])} $\\pm$ {_br_float(auc0[1])} \\\\
H1 (ensemble)     & {_br_float(acc1[0])} $\\pm$ {_br_float(acc1[1])} & {_br_float(h1_f1[0])} $\\pm$ {_br_float(h1_f1[1])} & {_br_float(auc1[0])} $\\pm$ {_br_float(auc1[1])} \\\\"""
    tex = re.sub(
        r"H0 \(modelo único\) & .*? \\\\\nH1 \(ensemble\).*?\\\\",
        lambda _m: table_block,
        tex,
        count=1,
    )
    tex = tex.replace(
        "30 simulações). Atualize com \\texttt{results/results.csv}.",
        f"{m['n']} simulações).",
    )
    path.write_text(tex, encoding="utf-8")
    print(f"Atualizado: {path}")


def patch_slides(s: dict):
    path = ROOT / "apresentacao" / "slides.tex"
    tex = path.read_text(encoding="utf-8")
    bullet = s["conclusion"].replace("_", "\\_")
    new_frame = f"""\\begin{{frame}}{{Conclusão estatística}}
  \\begin{{itemize}}
    \\item {bullet}
    \\item Trade-off: ensemble treina três arquiteturas; tempo de treino tipicamente maior que H0.
  \\end{{itemize}}
\\end{{frame}}"""
    tex = re.sub(
        r"\\begin\{frame\}\{Conclusão estatística\}[\s\S]*?\\end\{frame\}",
        lambda _m: new_frame,
        tex,
        count=1,
    )
    path.write_text(tex, encoding="utf-8")
    print(f"Atualizado: {path}")


def main():
    if not RESULTS_CSV.exists():
        raise FileNotFoundError("Execute simulate.py antes.")
    m = _load_metrics()
    s = _load_stats()
    patch_main_tex(m, s)
    patch_slides(s)


if __name__ == "__main__":
    main()
