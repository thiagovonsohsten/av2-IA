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
python src/stats_test.py
python src/plots.py
```

Saídas:

- `results/results.csv` — métricas por seed e cenário
- `results/stats.txt` — testes estatísticos e conclusão H0/H1
- `results/figs/` — gráficos para o relatório

## Relatório e apresentação

- `relatorio/main.tex` — IEEE coluna dupla (Overleaf)
- `apresentacao/slides.tex` — Beamer

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
  stats_test.py  # Shapiro, t pareado / Wilcoxon
  plots.py       # figuras
```
