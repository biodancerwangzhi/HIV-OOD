# HIV-OOD

HIV drug-resistance prediction with **ESM-2 embeddings**, under **honest evaluation protocols** (Random / Patient / Cluster OOD / Subtype OOD).

This repository contains the **reproducible compute scripts** and plotting notebooks for the OOD-focused analysis pipeline. Result tables and manuscript figures are not tracked here; you regenerate them by running the pipeline.

## What this repo provides

Shipped in the repository:

- End-to-end scripts under `notebooks/scripts/`
- Plotting notebooks `notebooks/fig1.ipynb` … `fig5.ipynb`, `figS1.ipynb`
- Raw HIVDB Full tables under `notebooks/data/hivdb_full/`
- Reference sequences under `notebooks/data/refs/`

Generated locally when you run the pipeline (git-ignored):

- Prepared tables + ESM embeddings under `work/`
- Per-figure intermediate CSVs under `results/notebooks/`
- Manuscript figures under `figures/manuscript/`

## Quick start

```bash
git clone https://github.com/biodancerwangzhi/HIV-OOD.git
cd HIV-OOD

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt

# Full run from the shipped HIVDB Full tables.
# Required on a fresh clone. The ESM embedding step is slow (GPU optional).
bash notebooks/scripts/run_all.sh

# Later runs, once work/prepared + work/emb_full exist locally:
# skips data prep, embedding extraction, and the O(n^2) fig2 Hamming step.
# bash notebooks/scripts/run_all.sh --skip-heavy
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
step01_prepare_data.py
        → work/prepared/*.csv
step02_extract_embeddings.py
        → work/emb_full/*
CLUSTER_K = 12   # fixed in notebooks/scripts/_common.py
step03_protocol_benchmark.py   # main AUROC table
        → work/results/probe_protocol_full_selected_k.csv
        ├─ step04_fig4_compute.py
        ├─ step05_fig5_compute.py
        ├─ step06_cross_split_similarity.py
        └─ step07_protocol_summary.py
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
notebooks/data/hivdb_full/{PI,NRTI,NNRTI,INI,CAI}_DataSet.Full.txt
```

These are shipped in the repository (~9.8 MB), so a fresh clone can rebuild everything.

Please cite HIVDB when using these data:

> Rhee SY et al. Human immunodeficiency virus reverse transcriptase and protease sequence database. *Nucleic Acids Research*.
> Website: https://hivdb.stanford.edu/

Derived artifacts, all rebuilt by the pipeline and none of them tracked in git:

| Path | Role |
|---|---|
| `work/prepared/*.csv` | sequences + labels |
| `work/emb_full/*` | ESM-2 mean embeddings (recomputed by `step02`) |
| `work/results/*.csv` | main probe/protocol tables |
| `results/notebooks/*` | per-figure intermediate CSVs |

## Repository layout

Tracked in the repository:

```
HIV-OOD/
├── README.md
├── requirements.txt
├── .gitignore
└── notebooks/
    ├── README.md              # run order (source of truth)
    ├── fig1.ipynb … fig5.ipynb / figS1.ipynb
    ├── scripts/               # all compute entrypoints
    └── data/
        ├── hivdb_full/        # raw HIVDB Full tables
        └── refs/              # P04585 / P04591 reference FASTA
```

Created locally when you run the pipeline (git-ignored):

```
HIV-OOD/
├── work/
│   ├── prepared/              # cleaned tables
│   ├── emb_full/              # embeddings
│   └── results/               # main probe CSVs
├── results/notebooks/         # per-figure intermediate CSVs
└── figures/manuscript/        # paper figures
```

## License

Code license is not finalized in-repo yet. Add a `LICENSE` before making the repository public.

### 参考序列 (已移到 notebooks/data/refs/)
- P04585.fasta (PR/RT/IN reference)
- P04591.fasta (CA reference)

### 原始数据下载（HIVDB Full genotype-phenotype tables）
```bash
python notebooks/scripts/step00_download_hivdb.py
```
下载后运行 `step01_prepare_data.py` 即可。

**注意**：Stanford HIVDB 会在定期更新这些表。复现论文结果时建议使用固定快照（若有），否则 AUROC 会略有浮动。
