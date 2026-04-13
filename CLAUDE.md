# CLAUDE.md

## Project Overview

This is a Brazilian academic research project (TCC — Undergraduate Thesis) by Anna Victória Duarte Barbosa, advised by Fabrízzio Condé de Oliveira. It evaluates 13 CNN architectures for binary fashion image classification (bags vs. pants) using a Repeated Holdout methodology with 10 random seeds.

**Language:** All code, comments, variable names, and documentation are written in Brazilian Portuguese.

**Execution environment:** Google Colab with NVIDIA T4 GPU. The notebook uses `drive.mount` to access files from Google Drive.

---

## Repository Structure

```
notebooks/   → Main Jupyter notebook (run on Google Colab)
data/        → Dataset: 100 train + 60 test images
  train/bags/        (50 bag images)
  train/pants/       (50 pants images)
  test/bags/         (30 bag images)
  test/pants/        (30 pants images)
results/     → All experiment outputs
  excel/     → Per-model .xlsx tables + comparative ranking
  plots/     → 3 PDFs per model (accuracy, boxplot, mean±std)
  latex/     → LaTeX-ready tables for the academic report
report/      → Full academic report (LaTeX source + compiled PDF)
  model_parameters/  → Hyperparameter documentation (.tex, .docx)
backups/     → Original zip backups from Google Drive (gitignored)
```

---

## Key Files

- **`notebooks/cnn_fashion_classification.ipynb`** — The entire experiment lives here. It defines all model builders, the generic experiment runner (`rodar_experimento_generico`), and the result export functions.
- **`results/excel/tabela_comparativa_modelos_ordenada.xlsx`** — Final ranked comparison of all 13 models.
- **`report/relatorio_13_modelos_CNN.pdf`** — Full academic report with analysis and discussion.
- **`report/model_parameters/replicabilidade_modelos_TCC.tex`** — Complete hyperparameter table for replicability.

---

## Core Functions in the Notebook

| Function | Description |
|----------|-------------|
| `rodar_experimento_generico(...)` | Main experiment loop: 10 seeds × train/evaluate cycle |
| `build_transfer_learning_model(...)` | Builds a TL model with frozen base + custom head |
| `build_cnn_from_scratch(...)` | Builds custom 3-block CNN |
| `salvar_graficos_em_pdfs(...)` | Generates 3 PDFs per model |
| `salvar_tabelas_excel_latex(...)` | Exports results to Excel and LaTeX |
| `ic95_t(media, desvio, n)` | Computes 95% CI using t-Student distribution |
| `reconstruir_consolidados()` | Rebuilds consolidated results from per-model Excel files |

---

## Important Conventions

- **Seeds:** All experiments use seeds 0–9. Reproducibility is enforced via `random.seed`, `np.random.seed`, `tf.random.set_seed`, `PYTHONHASHSEED`, and `TF_DETERMINISTIC_OPS=1`.
- **Metrics:** `accuracy_score`, `precision_score`, `recall_score`, `f1_score` from scikit-learn (weighted average for multi-class safety).
- **Image pipeline:** Keras `ImageDataGenerator.flow_from_directory` — folder names are the class labels.
- **Model saving:** Models are NOT saved to disk between runs; only metrics and generated figures are persisted.

---

## What NOT to Change

- Do not rename the `data/train/bags` and `data/train/pants` directories without updating the notebook paths — Keras uses folder names as class labels.
- Do not change the `results/` subdirectory names (`excel/`, `plots/`, `latex/`) without updating the notebook's output paths.

---

## Running the Notebook

1. Open `notebooks/cnn_fashion_classification.ipynb` in Google Colab.
2. Set runtime to **GPU T4** (Runtime → Change runtime type → T4 GPU).
3. Mount Google Drive and update `BASE_DIR` to your Drive path.
4. Run all cells sequentially. Each model experiment takes several minutes on T4.

---

## Dependencies

```
tensorflow >= 2.x   (pre-installed on Colab)
scikit-learn        (pre-installed on Colab)
pandas              (pre-installed on Colab)
numpy               (pre-installed on Colab)
matplotlib          (pre-installed on Colab)
scipy               (pre-installed on Colab)
openpyxl            (pip install openpyxl, done inside notebook)
```
