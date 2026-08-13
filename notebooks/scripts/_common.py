"""Shared constants and model builders for reproducible HIV-ESM-2 scripts."""
from __future__ import annotations

import os


def pin_compute_threads() -> None:
    """Force single-thread BLAS/OpenMP before heavy numeric work.

    Multi-thread OpenBLAS/MKL makes sklearn LR / PCA / XGBoost non-bit-exact
    across process runs even with fixed random_state.
    """
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[key] = "1"


# Must run before numpy/sklearn/lightgbm are imported in this process.
pin_compute_threads()

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJ = Path(__file__).resolve().parent.parent.parent
PREP = PROJ / "work" / "prepared"
EMB = PROJ / "work" / "emb_full"
RES = PROJ / "work" / "results"
REF_DIR = PROJ / "notebooks" / "data" / "refs"
HIVDB_FULL = PROJ / "notebooks" / "data" / "hivdb_full"
CLUSTER_DIR = PROJ / "results" / "notebooks" / "cluster_validation"
FIG2_DIR = PROJ / "results" / "notebooks" / "fig2"
FIG3_DIR = PROJ / "results" / "notebooks" / "fig3"
FIG4_DIR = PROJ / "results" / "notebooks" / "fig4"
FIG5_DIR = PROJ / "results" / "notebooks" / "fig5"
K_CSV = CLUSTER_DIR / "selected_cluster_k.csv"  # optional legacy artifact / figS1 output
PROTOCOL_CSV = RES / "probe_protocol_full_selected_k.csv"

SEED = 42
CLUSTER_K = 12  # fixed contract for Cluster OOD (no longer selected at runtime)

N_SPLITS = 5
FC_RESISTANT = 3.0

GENE_OF = {"PI": "PR", "NRTI": "RT", "NNRTI": "RT", "INI": "IN", "CAI": "CA"}
DRUGS = {
    "PI": ["FPV", "ATV", "IDV", "LPV", "NFV", "SQV", "TPV", "DRV"],
    "NRTI": ["3TC", "ABC", "AZT", "D4T", "DDI", "TDF"],
    "NNRTI": ["EFV", "ETR", "NVP", "RPV"],
    "INI": ["RAL", "EVG", "DTG", "BIC", "CAB"],
}
MAIN_CLASSES = ["PI", "NRTI", "NNRTI", "INI"]
METHODS = ["binary", "esm_lr", "esm_lgb"]


def read_fasta_seq(path: Path) -> str:
    return "".join(
        line.strip()
        for line in path.read_text().splitlines()
        if line and not line.startswith(">")
    )


def build_refs() -> dict[str, str]:
    pol = read_fasta_seq(REF_DIR / "P04585.fasta")
    gag = read_fasta_seq(REF_DIR / "P04591.fasta")
    spec = {
        "PR": (pol, 489, 587, {3: "I"}),
        "RT": (pol, 588, 1187, {214: "F", 570: "E"}),
        "IN": (pol, 1148, 1435, {10: "E", 72: "I", 123: "S", 124: "T", 127: "K", 232: "D"}),
        "CA": (gag, 133, 363, {}),
    }
    out: dict[str, str] = {}
    for gene, (src, start, end, patch) in spec.items():
        seq = list(src[start - 1 : end])
        for pos, aa in patch.items():
            seq[pos - 1] = aa
        out[gene] = "".join(seq)
    return out


def binary_encode(seqs: list[str], ref: str) -> np.ndarray:
    length = min(min(len(s) for s in seqs), len(ref))
    x = np.zeros((len(seqs), length), dtype=np.float32)
    for i, seq in enumerate(seqs):
        for j in range(length):
            if seq[j] != ref[j]:
                x[i, j] = 1.0
    return x


def load_emb(gene: str) -> tuple[dict[str, int], np.ndarray]:
    mean = np.load(EMB / f"{gene}_mean.npy")
    seqs = json.loads((EMB / f"{gene}_seqs.json").read_text(encoding="utf-8"))["sequences"]
    return {s: i for i, s in enumerate(seqs)}, mean


def make_binary_model():
    return make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=SEED,
            solver="lbfgs",
            tol=1e-4,
        ),
    )


def make_esm_lr_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=SEED,
            solver="lbfgs",
            tol=1e-4,
        ),
    )


def make_esm_lgb_model():
    """Deterministic LightGBM used by every main evaluation script."""
    return lgb.LGBMClassifier(
        n_estimators=300,
        num_leaves=31,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        bagging_seed=SEED,
        feature_fraction_seed=SEED,
        data_random_seed=SEED,
        deterministic=True,
        force_col_wise=True,
        n_jobs=1,
        verbose=-1,
    )


def fit_predict(method: str, xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray) -> np.ndarray:
    if method == "binary":
        clf = make_binary_model()
    elif method == "esm_lr":
        clf = make_esm_lr_model()
    elif method == "esm_lgb":
        clf = make_esm_lgb_model()
    else:
        raise ValueError(f"Unknown method: {method}")
    clf.fit(xtr, ytr)
    return clf.predict_proba(xte)[:, 1]


def selected_k() -> int:
    """Return the fixed Cluster OOD k.

    Historically this was read from results/notebooks/cluster_validation/selected_cluster_k.csv
    (figS1 reliability scan). The main pipeline now hardcodes CLUSTER_K=12 as the contract.
    figS1 may still regenerate the CSV as a sensitivity/supplement artifact.
    """
    return int(CLUSTER_K)


def pd_read_k() -> int:
    """Legacy alias of selected_k(); kept for old call sites."""
    return int(CLUSTER_K)
