"""Executa stats, gráficos e atualização do LaTeX após simulate.py."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def main():
    for script in ("stats_test.py", "plots.py", "extra_plots.py", "update_report.py"):
        subprocess.check_call([PY, str(ROOT / "src" / script)], cwd=str(ROOT))
    print("Pipeline concluído: results/stats.txt, results/figs/, relatorio/, apresentacao/")


if __name__ == "__main__":
    main()
