# Notebooks

This directory contains the analysis notebooks for the HIV-1 drug-resistance prediction project. The notebooks are designed to be read and, when needed, executed in numerical order.

| Notebook | Purpose |
|---|---|
| `01_data_preparation_and_task_definition.ipynb` | Prepares the HIV-1 genotype-phenotype resistance data, defines prediction tasks, and checks dataset structure. |
| `02_indistribution_benchmark.ipynb` | Evaluates baseline methods under standard in-distribution random-split settings. |
| `03_baseline_limitations_and_mutation_signal_motivation.ipynb` | Examines why mutation-based features are strong baselines and motivates the need for more careful evaluation. |
| `04_construct_mutation_aware_features.ipynb` | Constructs mutation-aware feature representations for downstream model comparison. |
| `05_train_mutation_aware_models.ipynb` | Trains and evaluates mutation-aware and fusion models to test whether explicit feature combinations improve performance. |
| `06_leakage_and_ood_generalization.ipynb` | Tests model robustness under leakage-aware, out-of-distribution, and scarce-drug evaluation settings. This is the core generalization analysis. |
| `07_interpretability_drm_enrichment.ipynb` | Analyzes whether model signals align with known drug-resistance mutation sites. |
| `08_complexity_and_epistasis.ipynb` | Studies model complexity and mutational epistasis to explain why strong representations with simple classifiers can be robust. |
| `09_manuscript_figures_and_summary.ipynb` | Assembles manuscript-ready figures, summary tables, and final evidence for the paper narrative. |

## Suggested reading path

- For the main result, focus on `06`.
- For mechanistic interpretation, continue with `07` and `08`.
- For full reproduction, run the notebooks from `01` to `09` in order.
