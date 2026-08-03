# HIV-ESM-2

HIV drug-resistance prediction with **ESM-2 embeddings**, under **honest evaluation protocols** (Random / Patient / Cluster OOD / Subtype OOD).

This repository contains the **reproducible compute scripts**, reference result tables, and manuscript figures for the OOD-focused analysis pipeline.

## What this repo provides

- End-to-end scripts under `notebooks/scripts/`
- Prepared tables + ESM embeddings under `work/`
- Reference CSVs under `results/notebooks/`
- Manuscript figures under `figures/manuscript/`
- Plotting notebooks `notebooks/fig1.ipynb` … `fig5.ipynb`, `figS1.ipynb`

## Quick start

```bash
git clone <YOUR_REPO_URL>
cd HIV-ESM-2

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt

# Recommended: reuse shipped prepared data + embeddings
bash notebooks/scripts/run_all.sh --skip-heavy

# Full rebuild from HIVDB Full text (slow; ESM embedding step needs time/GPU optional)
# bash notebooks/scripts/run_all.sh
```

Plot:

```bash
for nb in notebooks/fig1.ipynb notebooks/fig2.ipynb notebooks/fig3.ipynb \
          notebooks/fig4.ipynb notebooks/fig5.ipynb notebooks/figS1.ipynb; do
  venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace "$nb"
done
```

Detailed order and notes: [`notebooks/README.md`](notebooks/README.md).

## Pipeline (short)

```
fig0_prepare_data.py
        → work/prepared/*.csv
fig0_extract_embeddings.py
        → work/emb_full/*
CLUSTER_K = 12   # fixed in notebooks/scripts/_common.py
fig3_protocol_benchmark.py   # main AUROC table
        → work/results/probe_protocol_full_selected_k.csv
        ├─ fig4_compute.py
        ├─ fig5_compute.py
        ├─ fig2_cross_split_similarity.py
        └─ fig3_protocol_summary.py
```

**Main result table:** `work/results/probe_protocol_full_selected_k.csv`

## Reproducibility contracts

| Item | Value |
|---|---|
| Seed | `SEED = 42` |
| Cluster OOD k | **fixed `CLUSTER_K = 12`** (`_common.selected_k()`) |
| Resistance label | fold-change ≥ 3.0 |
| LightGBM | `n_jobs=1, deterministic=True, force_col_wise=True` + fixed bagging/feature seeds |
| Thread pinning | `pin_compute_threads()` before heavy numeric work |

Bit-exact CSV identity is expected **in the same environment** after the determinism fixes. Across machines/OS/BLAS builds, prefer numerical tolerance checks (e.g. AUROC to 4 decimals).

## Data

Canonical raw inputs (Stanford HIV Drug Resistance Database genotype–phenotype Full sets):

```
work/hivdb_full/{PI,NRTI,NNRTI,INI,CAI}_DataSet.Full.txt
work/hivdb_full/refs/*.fasta
```

Please cite HIVDB when using these data:

> Rhee SY et al. Human immunodeficiency virus reverse transcriptase and protease sequence database. *Nucleic Acids Research*.
> Website: https://hivdb.stanford.edu/

Derived artifacts:

| Path | Role |
|---|---|
| `work/prepared/*.csv` | sequences + labels |
| `work/emb_full/*` | ESM-2 mean embeddings (can be recomputed) |
| `results/notebooks/fig4/pdb/*` | static structure files |
| `results/notebooks/fig4/fig4_ram_to_ligand_3d_distance.csv` | static distance table |

## Repository layout

```
HIV-ESM-2/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── README.md              # run order (source of truth)
│   ├── fig1.ipynb … fig5.ipynb / figS1.ipynb
│   └── scripts/               # all compute entrypoints
├── work/
│   ├── hivdb_full/            # raw Full tables + refs
│   ├── prepared/              # cleaned tables
│   ├── emb_full/              # embeddings
│   └── results/               # main probe CSVs
├── results/notebooks/         # per-figure intermediate CSVs
├── figures/manuscript/        # paper figures
└── docs/                      # notes + upload guide
```

## License

Code license is not finalized in-repo yet. Add a `LICENSE` before making the repository public.
