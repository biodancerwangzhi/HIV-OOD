"""Full-set protocol probe with reliability-selected cluster k.

Uses fixed CLUSTER_K=12 from _common.selected_k() (historically from selected_cluster_k.csv)
and computes the full 3-method x 4-protocol table:
  random_* : stratified random 5-fold CV
  rg_*     : patient-grouped 5-fold CV
  clu_*    : selected-k k-means leave-one-cluster-out
  ood_*    : subtype OOD (train B, test non-B)

Output: work/results/probe_protocol_full_selected_k.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    DRUGS,
    EMB,
    GENE_OF,
    METHODS,
    N_SPLITS,
    PREP,
    PROTOCOL_CSV,
    RES,
    SEED,
    binary_encode,
    build_refs,
    fit_predict,
    load_emb,
    selected_k,
)

REFS = build_refs()


def random_cv_auc(method: str, X: np.ndarray, y: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return np.nan
    oof = np.full(len(y), np.nan)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for tr, te in cv.split(X, y):
        oof[te] = fit_predict(method, X[tr], y[tr], X[te])
    return float(roc_auc_score(y, oof))


def group_cv_auc(method: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    k = min(N_SPLITS, len(np.unique(groups)))
    if k < 2 or len(np.unique(y)) < 2:
        return np.nan
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=k).split(X, y, groups):
        oof[te] = fit_predict(method, X[tr], y[tr], X[te])
    return float(roc_auc_score(y, oof))


def holdout_auc(method: str, X: np.ndarray, y: np.ndarray, is_test: np.ndarray) -> float:
    tr, te = ~is_test, is_test
    if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
        return np.nan
    return float(roc_auc_score(y[te], fit_predict(method, X[tr], y[tr], X[te])))


def cluster_labels(X_emb: np.ndarray, requested_k: int) -> tuple[np.ndarray, int]:
    k_eff = min(requested_k, max(2, len(X_emb) // 20))
    labels = KMeans(n_clusters=k_eff, n_init=10, random_state=SEED).fit_predict(X_emb)
    return labels, k_eff


def loco_auc(method: str, X: np.ndarray, y: np.ndarray, clusters: np.ndarray) -> float:
    oof = np.full(len(y), np.nan)
    for c in np.unique(clusters):
        te = clusters == c
        tr = ~te
        if len(np.unique(y[tr])) < 2 or te.sum() < 3:
            continue
        oof[te] = fit_predict(method, X[tr], y[tr], X[te])
    m = ~np.isnan(oof)
    if m.sum() > 10 and len(np.unique(y[m])) == 2:
        return float(roc_auc_score(y[m], oof[m]))
    return np.nan


def main() -> None:
    k_star = selected_k()
    print(f"selected k = {k_star}", flush=True)
    cache: dict = {}
    rows = []
    for cls, drugs in DRUGS.items():
        gene = GENE_OF[cls]
        if gene not in cache:
            cache[gene] = load_emb(gene)
        idx, mean = cache[gene]
        ref = REFS[gene]
        df = pd.read_csv(PREP / f"{cls}.csv", dtype={"SeqID": str, "PtID": str})
        seqs_all = df["sequence"].tolist()
        emb_row = np.array([idx.get(s, -1) for s in seqs_all])
        st_all = df["Subtype"].fillna("Unknown").values
        pt_all = df["PtID"].fillna("NA").values
        Xb_all = binary_encode(seqs_all, ref)
        for drug in drugs:
            y = pd.to_numeric(df[f"{drug}_label"], errors="coerce").values
            valid = (~np.isnan(y)) & (emb_row >= 0)
            yv = y[valid].astype(int)
            if len(np.unique(yv)) < 2:
                continue
            r = emb_row[valid]
            groups = pt_all[valid]
            st = st_all[valid]
            is_nonB = (st != "B") & (st != "Unknown") & (st != "U")
            feat = {"binary": Xb_all[valid], "esm_lr": mean[r], "esm_lgb": mean[r]}
            clu, k_eff = cluster_labels(mean[r], k_star)
            rec = {
                "class": cls,
                "drug": drug,
                "gene": gene,
                "n": len(yv),
                "n_nonB": int(is_nonB.sum()),
                "n_res": int(yv.sum()),
                "clu_method": f"kmeans_euclidean_k{k_star}_reliability_selected",
                "clu_k_eff": int(k_eff),
            }
            for method in METHODS:
                rec[f"random_{method}"] = random_cv_auc(method, feat[method], yv)
                rec[f"rg_{method}"] = group_cv_auc(method, feat[method], yv, groups)
                rec[f"clu_{method}"] = loco_auc(method, feat[method], yv, clu)
                if is_nonB.sum() >= 10 and yv[is_nonB].sum() >= 2 and (yv[is_nonB] == 0).sum() >= 2:
                    rec[f"ood_{method}"] = holdout_auc(method, feat[method], yv, is_nonB)
                else:
                    rec[f"ood_{method}"] = np.nan
            rows.append(rec)
            print(
                f"  {cls}/{drug:4s} n={len(yv):4d} k_eff={k_eff:2d} | "
                f"random bin={rec['random_binary']:.3f} esmlr={rec['random_esm_lr']:.3f} esmlgb={rec['random_esm_lgb']:.3f} | "
                f"ood bin={rec['ood_binary']:.3f} esmlr={rec['ood_esm_lr']:.3f} esmlgb={rec['ood_esm_lgb']:.3f}",
                flush=True,
            )
    result = pd.DataFrame(rows)
    cols = [
        "class", "drug", "gene", "n", "n_nonB", "n_res",
        "random_binary", "rg_binary", "clu_binary", "ood_binary",
        "random_esm_lr", "rg_esm_lr", "clu_esm_lr", "ood_esm_lr",
        "random_esm_lgb", "rg_esm_lgb", "clu_esm_lgb", "ood_esm_lgb",
        "clu_method", "clu_k_eff",
    ]
    result = result[cols]
    RES.mkdir(parents=True, exist_ok=True)
    result.to_csv(PROTOCOL_CSV, index=False)
    # Keep legacy dual-source consumers aligned with the single source of truth.
    legacy = RES / "probe_ood_full_selected_k.csv"
    legacy_cols = [
        "class", "drug", "gene", "n", "n_nonB", "n_res",
        "rg_binary", "clu_binary", "ood_binary",
        "rg_esm_lr", "clu_esm_lr", "ood_esm_lr",
        "rg_esm_lgb", "clu_esm_lgb", "ood_esm_lgb",
        "clu_method", "clu_k_eff",
    ]
    result[legacy_cols].to_csv(legacy, index=False)
    print("\n===== MEAN AUC by protocol x method =====")
    for proto in ["random", "rg", "clu", "ood"]:
        print(proto, " ".join(f"{method}={result[f'{proto}_{method}'].mean():.4f}" for method in METHODS))
    print(f"saved {PROTOCOL_CSV}")
    print(f"synced {legacy}")


if __name__ == "__main__":
    main()
