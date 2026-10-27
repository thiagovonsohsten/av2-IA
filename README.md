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
python src/update_report.py
```

Saídas:

- `results/results.csv` — métricas por seed e cenário
- `results/stats.txt` — testes estatísticos e conclusão H0/H1
- `results/figs/` — gráficos para o relatório

## Relatório e apresentação

- `relatorio/main.tex` — IEEE coluna dupla (Overleaf), números preenchidos via `update_report.py`
- `apresentacao/slides.tex` — Beamer
- Ajuste **nome, instituição e e-mail** no `main.tex` antes de entregar

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
  stats_test.py  # Shapiro, t pareado / Wilcoxon
  plots.py       # figuras
```
