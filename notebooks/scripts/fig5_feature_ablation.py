"""Fig5 mechanism probes (Full set). Reuses embeddings/mutscore; patient-grouped CV only (fast).

Outputs work/results/probe_fig5.csv with per-drug AUROC for:
  5B fusion:  esm_lr, esm+mutscore, binary_only
  5C matrix:  binary_lr, binary_lgb, esm_lr, esm_lgb  (repr x classifier complexity)
  5D epistasis: binary_lr, binary_pair (binary + pairwise-interaction on top-var positions), esm_lr
Also 5A DRM enrichment is computed separately in the notebook (no model).
"""
from __future__ import annotations

# Pin threads before numpy/sklearn/lightgbm import for bit-exact reruns.
import sys
from pathlib import Path as _Path
_scripts = _Path(__file__).resolve().parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))
from _common import SEED, pin_compute_threads

pin_compute_threads()

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJ = Path(__file__).resolve().parent.parent.parent
EMB = PROJ / "work" / "emb_full"
PREP = PROJ / "work" / "prepared"
REF_DIR = PROJ / "notebooks" / "data" / "refs"
OUT = PROJ / "work" / "results" / "probe_fig5.csv"
GENE_OF = {"PI": "PR", "NRTI": "RT", "NNRTI": "RT", "INSTI": "IN"}
DRUGS = {
    "PI": ["FPV", "ATV", "IDV", "LPV", "NFV", "SQV", "TPV", "DRV"],
    "NRTI": ["3TC", "ABC", "AZT", "D4T", "DDI", "TDF"],
    "NNRTI": ["EFV", "ETR", "NVP", "RPV"],
    "INSTI": ["RAL", "EVG", "DTG", "BIC", "CAB"],
}


def read_fasta_seq(p):
    return "".join(l.strip() for l in p.read_text().splitlines() if l and not l.startswith(">"))


def build_refs():
    pol = read_fasta_seq(REF_DIR / "P04585.fasta")
    gag = read_fasta_seq(REF_DIR / "P04591.fasta")
    spec = {
        "PR": (pol, 489, 587, {3: "I"}),
        "RT": (pol, 588, 1187, {214: "F", 570: "E"}),
        "IN": (pol, 1148, 1435, {10: "E", 72: "I", 123: "S", 124: "T", 127: "K", 232: "D"}),
    }
    out = {}
    for g, (src, s, e, patch) in spec.items():
        seq = list(src[s - 1 : e])
        for p, aa in patch.items():
            seq[p - 1] = aa
        out[g] = "".join(seq)
    return out


REFS = build_refs()


def binary_encode(seqs, ref):
    L = min(min(len(s) for s in seqs), len(ref))
    X = np.zeros((len(seqs), L), dtype=np.float32)
    for i, s in enumerate(seqs):
        X[i] = [1.0 if s[j] != ref[j] else 0.0 for j in range(L)]
    return X


def pairwise_top(Xb, k=15):
    # Stable top-k by variance; tie-break by lower index for bit-exact feature sets.
    var = Xb.mean(0) * (1 - Xb.mean(0))
    top = np.argsort(-var, kind="stable")[:k]
    inter = []
    for a in range(len(top)):
        for b in range(a + 1, len(top)):
            inter.append(Xb[:, top[a]] * Xb[:, top[b]])
    return np.column_stack([Xb] + inter) if inter else Xb


def make_lr(kind: str):
    # Fixed solver + seed; binary features skip mean centering.
    return make_pipeline(
        StandardScaler(with_mean=("binary" not in kind)),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=SEED,
            solver="lbfgs",
            tol=1e-4,
        ),
    )


def make_lgb():
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


def cv_auc(kind, X, y, groups):
    k = min(5, len(np.unique(groups)))
    if k < 2 or len(np.unique(y)) < 2:
        return np.nan
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=k).split(X, y, groups):
        clf = make_lgb() if "lgb" in kind else make_lr(kind)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, oof)


def load_emb(g):
    mean = np.load(EMB / f"{g}_mean.npy")
    ms = np.load(EMB / f"{g}_mutscore.npy")
    idx = {s: i for i, s in enumerate(json.load(open(EMB / f"{g}_seqs.json"))["sequences"])}
    return idx, mean, ms


def main():
    pin_compute_threads()
    cache, rows = {}, []
    for cls, drugs in DRUGS.items():
        g = GENE_OF[cls]
        if g not in cache:
            cache[g] = load_emb(g)
        idx, mean, ms = cache[g]
        df = pd.read_csv(PREP / f"{cls}.csv", dtype={"SeqID": str, "PtID": str})
        seqs = df["sequence"].tolist()
        er = np.array([idx.get(s, -1) for s in seqs])
        pt = df["PtID"].fillna("NA").values
        Xb_all = binary_encode(seqs, REFS[g])
        for drug in drugs:
            y = pd.to_numeric(df[f"{drug}_label"], errors="coerce").values
            v = (~np.isnan(y)) & (er >= 0)
            yv = y[v].astype(int)
            if len(np.unique(yv)) < 2:
                continue
            r = er[v]
            gr = pt[v]
            Xb = Xb_all[v]
            Xe = mean[r]
            Xm = ms[r].reshape(-1, 1)
            rec = {"class": cls, "drug": drug, "n": len(yv), "n_res": int(yv.sum())}
            rec["binary_lr"] = cv_auc("binary_lr", Xb, yv, gr)
            rec["binary_lgb"] = cv_auc("binary_lgb", Xb, yv, gr)
            rec["esm_lr"] = cv_auc("esm_lr", Xe, yv, gr)
            rec["esm_lgb"] = cv_auc("esm_lgb", Xe, yv, gr)
            rec["esm_mut_lr"] = cv_auc("esm_lr", np.hstack([Xe, Xm]), yv, gr)
            rec["binary_pair_lr"] = cv_auc("binary_lr", pairwise_top(Xb), yv, gr)
            rows.append(rec)
            print(
                f"  {cls}/{drug:4s} bLR={rec['binary_lr']:.3f} bLGB={rec['binary_lgb']:.3f} "
                f"eLR={rec['esm_lr']:.3f} eLGB={rec['esm_lgb']:.3f} "
                f"e+mut={rec['esm_mut_lr']:.3f} b+pair={rec['binary_pair_lr']:.3f}",
                flush=True,
            )
    R = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    R.to_csv(OUT, index=False)
    print("\n=== MEAN ===")
    for c in ["binary_lr", "binary_lgb", "esm_lr", "esm_lgb", "esm_mut_lr", "binary_pair_lr"]:
        print(f"  {c:14s}: {R[c].mean():.4f}")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
