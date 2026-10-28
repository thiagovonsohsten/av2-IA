# Estudo comparativo: modelo único vs Ensemble (pneumonia)

Compara **Classificação + Transfer Learning + modelo único (H0)** com **Classificação + Transfer Learning + Ensemble (H1)** no dataset PneumoniaMNIST (MedMNIST).

## Requisitos

- Python 3.10+
- GPU NVIDIA recomendada (CUDA)

## Instalação

```bash
pip install -r requirements.txt
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

Se `CUDA: False` e você tem GPU NVIDIA, instale o PyTorch com CUDA em [pytorch.org](https://pytorch.org/get-started/locally/) (no Windows, ative caminhos longos se o `pip install torch` falhar com WinError 206).

## Execução

Teste rápido (2 seeds, 1 época):

```bash
python src/simulate.py --runs 2 --epochs 1
```

Experimento completo (30 simulações):

```bash
python src/simulate.py
python src/finish_pipeline.py   # roda stats_test, plots, extra_plots e update_report
```

Ou, passo a passo:

```bash
python src/stats_test.py
python src/plots.py
python src/extra_plots.py   # figuras adicionais (dataset, PR, calibração, limiar, IC, pipeline)
python src/update_report.py
```

Saídas:

- `results/results.csv` — métricas por seed e cenário
- `results/stats.txt` — testes estatísticos e conclusão H0/H1
- `results/seed0_probs.npz` — probabilidades da seed 0 (insumo de `extra_plots.py`)
- `results/figs/` — 17 gráficos para o relatório e os slides

## Relatório e apresentação

- `relatorio/main.tex` — IEEE coluna dupla (Overleaf), números preenchidos via `update_report.py`
- `apresentacao/slides.tex` — Beamer
- Autores, instituição e e-mails já preenchidos no `main.tex` e no `slides.tex`; confira antes de entregar

### Resultado do experimento (30 simulações, GPU)

| Cenário | F1 (média ± dp) |
|---------|-----------------|
| H0 (ResNet18) | 0,8980 ± 0,0184 |
| H1 (ensemble) | 0,9050 ± 0,0077 |

**Conclusão:** rejeita-se H0 ($p = 0{,}011$, $t$ pareado unilateral); **H1 (ensemble) é mais eficaz** em F1. Detalhes em `results/stats.txt`.

## Estrutura

```
src/
  config.py      # hiperparâmetros
  data.py        # PneumoniaMNIST
  models.py      # CNNs com transfer learning
  train.py       # treino e inferência
  ensemble.py    # votação suave
  metrics.py     # acurácia, F1, AUC
  simulate.py    # 30 simulações
  stats_test.py  # Shapiro, t pareado / Wilcoxon, IC 95% (t de Student)
  plots.py       # figuras principais
  extra_plots.py # figuras extras (dataset, PR, calibração, limiar, IC, pipeline)
  update_report.py   # injeta métricas no main.tex e slides.tex
  finish_pipeline.py # roda stats_test + plots + extra_plots + update_report
```
