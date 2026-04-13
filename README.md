# Redes Neurais Convolucionais na Moda

Estudo comparativo de 13 modelos de Redes Neurais Convolucionais (CNNs) para classificação binária de itens de moda — **Bolsas vs. Calças** — utilizando a metodologia de Repeated Holdout com 10 sementes aleatórias.

> **Trabalho de Conclusão de Curso (TCC)**
> Autora: Anna Victoria

---

## Visão Geral

O projeto avalia 11 modelos de Transfer Learning (pré-treinados com ImageNet) e 2 CNNs treinadas do zero, comparando métricas de desempenho (acurácia, precisão, recall, F1-score) e tempo de execução em cada configuração.

### Modelos Avaliados

| Categoria | Modelos |
|-----------|---------|
| Transfer Learning | MobileNetV2, EfficientNetB0, EfficientNetB1, EfficientNetB2, ResNet50, DenseNet121, MobileNetV3Small, MobileNetV3Large, NASNetMobile, InceptionV3, Xception |
| CNN do Zero | cnn_from_scratch_lr5e4, cnn_from_scratch_lr1e4 |

---

## Dataset

Classificação binária de imagens de moda:

| Conjunto | Classe | Quantidade |
|----------|--------|------------|
| Treino | Bolsas | 50 imagens |
| Treino | Calças | 50 imagens |
| Teste | Bolsas | 30 imagens |
| Teste | Calças | 30 imagens |

**Total:** 100 imagens de treino · 60 imagens de teste

---

## Metodologia

- **Validação:** Repeated Holdout com 10 sementes aleatórias (seeds 0–9)
- **Métricas:** Acurácia, Precisão, Recall, F1-Score (média ponderada)
- **Estatísticas:** Média, Desvio Padrão, Coeficiente de Variação, IC 95% (t-Student)
- **Augmentação:** rotação ±15°, zoom 20%, shift horizontal/vertical 10%, flip horizontal
- **Early Stopping:** patience=5 (monitorado na perda de treino)

### Configuração por Modelo

| Parâmetro | Maioria dos Modelos | InceptionV3 / Xception |
|-----------|--------------------|-----------------------|
| Resolução | 150×150 px | 299×299 px |
| Batch Size | 32 | 16 |
| Épocas | 30 (TL) / 50 (scratch) | 30 |
| Learning Rate | 1×10⁻⁴ | 1×10⁻⁴ |

---

## Estrutura do Projeto

```
.
├── notebooks/
│   └── cnn_fashion_classification.ipynb   # Notebook principal (Google Colab)
├── data/
│   ├── train/
│   │   ├── bags/                          # 50 imagens de treino (bolsas)
│   │   └── pants/                         # 50 imagens de treino (calças)
│   └── test/
│       ├── bags/                          # 30 imagens de teste (bolsas)
│       └── pants/                         # 30 imagens de teste (calças)
├── results/
│   ├── excel/                             # Tabelas de resultados (.xlsx)
│   ├── plots/                             # Gráficos gerados (.pdf)
│   └── latex/                             # Tabelas para LaTeX (.tex)
├── report/
│   ├── relatorio_13_modelos_CNN.tex       # Relatório completo (LaTeX)
│   ├── relatorio_13_modelos_CNN.pdf       # Relatório compilado (PDF)
│   └── model_parameters/                 # Documentação de hiperparâmetros
└── backups/                              # Arquivos de backup (.zip)
```

---

## Ambiente de Execução

O experimento foi executado no **Google Colab** com acelerador de hardware **GPU NVIDIA T4**.

### Dependências Principais

```
tensorflow >= 2.x
scikit-learn
pandas
numpy
matplotlib
scipy
openpyxl
```

Veja [`requirements.txt`](requirements.txt) para a lista completa.

---

## Como Reproduzir

1. Faça upload do dataset para o Google Drive mantendo a estrutura de pastas:
   ```
   data/
     train/bags/
     train/pants/
     test/bags/
     test/pants/
   ```

2. Abra o notebook [`notebooks/cnn_fashion_classification.ipynb`](notebooks/cnn_fashion_classification.ipynb) no Google Colab.

3. No menu do Colab: **Runtime → Change runtime type → T4 GPU**.

4. Atualize a variável `BASE_DIR` no notebook para apontar ao diretório do projeto no seu Google Drive.

5. Execute todas as células em ordem.

> **Reprodutibilidade:** Sementes fixas são definidas para `random`, `numpy`, `tensorflow` e variáveis de ambiente (`PYTHONHASHSEED`, `TF_DETERMINISTIC_OPS`).

---

## Resultados

Os resultados completos estão em `results/`:

- **`results/excel/`** — planilhas por modelo e tabela comparativa ordenada por desempenho
- **`results/plots/`** — 3 gráficos PDF por modelo (acurácia por rodada, boxplot de métricas, média ± desvio padrão)
- **`results/latex/`** — tabelas formatadas para inclusão direta em documentos LaTeX/Overleaf

O relatório completo com análise e discussão dos resultados está em [`report/relatorio_13_modelos_CNN.pdf`](report/relatorio_13_modelos_CNN.pdf).

---

## Arquitetura CNN do Zero

```python
Sequential([
    Conv2D(32,  (3,3), activation='relu', input_shape=(150, 150, 3)),
    MaxPool2D((2,2)),
    Conv2D(64,  (3,3), activation='relu'),
    MaxPool2D((2,2)),
    Conv2D(128, (3,3), activation='relu'),
    MaxPool2D((2,2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(2,   activation='softmax')
])
```

---

## Licença

Este projeto é um trabalho acadêmico (TCC). Todos os direitos reservados à autora.
