"""
step06_cross_split_similarity.py
=======================================
计算 fig2 所需的跨 split 序列相似度统计。

输入（均为预计算产物，必须已存在）：
  work/prepared/{PI,NRTI,NNRTI,INI}.csv
  work/emb_full/{PR,RT,IN}_mean.npy
  work/emb_full/{PR,RT,IN}_seqs.json
  work/results/probe_protocol_full_selected_k.csv
  (Cluster OOD uses fixed CLUSTER_K=12 from _common.selected_k)

输出（写入 results/notebooks/fig2/）：
  cross_split_similarity.csv   -- 每 (drug_class, scheme) 的最近邻相似度统计
  leakage_vs_binary_auc.csv    -- 泄漏率 vs binary-baseline AUROC

运行方式（从项目根目录）：
  python notebooks/scripts/step06_cross_split_similarity.py
"""

from __future__ import annotations

import json
from pathlib import Path

import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import selected_k  # noqa: E402

# ── path discovery ────────────────────────────────────────────────────────
CWD = Path(__file__).resolve().parent.parent.parent   # project root
PROJECT_ROOT = CWD if (CWD / "work" / "prepared").exists() else CWD.parent
assert (PROJECT_ROOT / "work" / "prepared").exists(), (
    f"Cannot locate work/prepared from {CWD}. Run this script from the project root."
)

PREP    = PROJECT_ROOT / "work" / "prepared"
EMB     = PROJECT_ROOT / "work" / "emb_full"
RES     = PROJECT_ROOT / "work" / "results"
OUT_DIR = PROJECT_ROOT / "results" / "notebooks" / "fig2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAIN_CLASSES = ["PI", "NRTI", "NNRTI", "INI"]
GENE_OF      = {"PI": "PR", "NRTI": "RT", "NNRTI": "RT", "INI": "IN"}


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    return path


# ── similarity helpers ────────────────────────────────────────────────────

def seq_matrix(seqs: list[str]) -> np.ndarray:
    L = min(len(s) for s in seqs)
    return np.array([list(s[:L]) for s in seqs])


def max_sim_to_train(test_M: np.ndarray, train_M: np.ndarray) -> np.ndarray:
    out = np.empty(len(test_M))
    for i in range(len(test_M)):
        eq = (train_M == test_M[i]).mean(axis=1)
        out[i] = eq.max()
    return out


# ── embedding cache ───────────────────────────────────────────────────────

EMB_CACHE: dict[str, tuple[dict[str, int], np.ndarray]] = {}


def load_emb(gene: str) -> tuple[dict[str, int], np.ndarray]:
    if gene not in EMB_CACHE:
        mean_path = _require(EMB / f"{gene}_mean.npy")
        seqs_path = _require(EMB / f"{gene}_seqs.json")
        mean  = np.load(mean_path)
        seqs  = json.load(seqs_path.open())["sequences"]
        EMB_CACHE[gene] = ({s: i for i, s in enumerate(seqs)}, mean)
    return EMB_CACHE[gene]


# ── cluster labels (matches probe_cluster_selected.py) ───────────────────

def cluster_labels_for_class(
    df: pd.DataFrame, cls: str, k_star: int
) -> tuple[np.ndarray, np.ndarray, int]:
    gene = GENE_OF[cls]
    seq_to_i, mean = load_emb(gene)
    emb_row = np.array([seq_to_i.get(s, -1) for s in df["sequence"].tolist()])
    valid   = emb_row >= 0
    labels  = np.full(len(df), -1, dtype=int)
    k_eff   = min(k_star, max(2, int(valid.sum()) // 20))
    labels[valid] = KMeans(n_clusters=k_eff, n_init=10, random_state=42).fit_predict(
        mean[emb_row[valid]]
    )
    return labels, valid, k_eff


# ── split definitions ─────────────────────────────────────────────────────

def split_indices(
    df: pd.DataFrame, scheme: str, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    n   = len(df)
    idx = np.arange(n)
    if scheme == "random":
        te = rng.choice(idx, size=int(n * 0.2), replace=False)
        tr = np.setdiff1d(idx, te)
    elif scheme == "patient":
        pts  = df["PtID"].fillna("NA").values
        uniq = pd.unique(pts)
        te_pt = set(rng.choice(uniq, size=max(1, int(len(uniq) * 0.2)), replace=False))
        te = idx[np.array([p in te_pt for p in pts])]
        tr = np.setdiff1d(idx, te)
    elif scheme == "subtype":
        st    = df["Subtype"].fillna("Unknown").values
        is_nonB = (st != "B") & (st != "Unknown") & (st != "U")
        te = idx[is_nonB]
        tr = idx[st == "B"]
    else:
        raise ValueError(f"Unknown scheme: {scheme}")
    return tr, te


def cluster_loco_sims(
    M: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    sims = []
    idx  = np.arange(len(labels))
    for c in np.unique(labels[labels >= 0]):
        te = idx[labels == c]
        tr = idx[(labels >= 0) & (labels != c)]
        if len(tr) < 5 or len(te) < 5:
            continue
        sims.append(max_sim_to_train(M[te], M[tr]))
    return np.concatenate(sims) if sims else np.array([])


# ── main computation ──────────────────────────────────────────────────────

def compute_cross_split_similarity(
    k_star: int,
) -> tuple[pd.DataFrame, dict[tuple[str, str], np.ndarray]]:
    """
    Compute per-(class, scheme) nearest-neighbour similarity statistics.
    Returns (summary_df, dist) where dist[(cls, scheme)] = array of per-sample max sims.
    """
    rng  = np.random.default_rng(42)
    rows: list[dict] = []
    dist: dict[tuple[str, str], np.ndarray] = {}
    for cls in MAIN_CLASSES:
        df = pd.read_csv(_require(PREP / f"{cls}.csv"), dtype={"SeqID": str, "PtID": str})
        M  = seq_matrix(df["sequence"].tolist())
        for scheme in ["random", "patient", "cluster", "subtype"]:
            if scheme == "cluster":
                labels, valid, k_eff = cluster_labels_for_class(df, cls, k_star)
                sims    = cluster_loco_sims(M, labels)
                n_test  = int(len(sims))
                n_train = int(valid.sum())
            else:
                tr, te = split_indices(df, scheme, rng)
                if len(tr) < 5 or len(te) < 5:
                    continue
                sims    = max_sim_to_train(M[te], M[tr])
                n_test  = len(te)
                n_train = len(tr)
            if len(sims) < 5:
                continue
            dist[(cls, scheme)] = sims
            rows.append({
                "drug_class":    cls,
                "scheme":        scheme,
                "n_test":        n_test,
                "n_train":       n_train,
                "mean_maxsim":   round(float(sims.mean()), 4),
                "median_maxsim": round(float(np.median(sims)), 4),
                "frac_ge_0.99":  round(float((sims >= 0.99).mean()), 4),
            })
        print(f"  {cls} done")
    return pd.DataFrame(rows), dist


def save_dist(dist: dict[tuple[str, str], np.ndarray], out_path: Path) -> None:
    """Save dist dict as a compressed .npz archive (keys: cls_scheme)."""
    np.savez_compressed(out_path, **{f"{cls}_{sc}": arr for (cls, sc), arr in dist.items()})


def load_dist(npz_path: Path) -> dict[tuple[str, str], np.ndarray]:
    """Load dist dict from a .npz archive."""
    npz = np.load(npz_path)
    result: dict[tuple[str, str], np.ndarray] = {}
    for key in npz.files:
        # key format: cls_scheme  (e.g., PI_random, NRTI_patient)
        # scheme can be multi-word? no – all are single-word; split on first underscore only
        # but class names have no underscore, scheme names have no underscore → split at last '_'
        idx = key.rfind("_")
        cls, sc = key[:idx], key[idx + 1:]
        result[(cls, sc)] = npz[key]
    return result


def compute_leakage_vs_auc(leak: pd.DataFrame) -> pd.DataFrame:
    """Merge leakage statistics with binary-baseline AUROC per protocol."""
    protocol_path = _require(RES / "probe_protocol_full_selected_k.csv")
    R = pd.read_csv(protocol_path)
    leak_summary = leak.groupby("scheme", as_index=False)["frac_ge_0.99"].mean()
    leak_summary["leak_pct"] = 100 * leak_summary["frac_ge_0.99"]
    auc_rows = [
        {"scheme": "random",  "auroc": float(R["random_binary"].mean())},
        {"scheme": "patient", "auroc": float(R["rg_binary"].mean())},
        {"scheme": "cluster", "auroc": float(R["clu_binary"].mean())},
        {"scheme": "subtype", "auroc": float(R["ood_binary"].mean())},
    ]
    link = leak_summary.merge(pd.DataFrame(auc_rows), on="scheme", how="inner")
    link["label"] = link["scheme"].map(
        {"random": "Random", "patient": "Patient", "cluster": "Cluster", "subtype": "Subtype"}
    )
    return link


def verify_against_cache(df: pd.DataFrame, cache_path: Path) -> None:
    """Print a quick diff summary to confirm reproducibility."""
    if not cache_path.exists():
        print(f"  [INFO] No cache at {cache_path.name} to compare against.")
        return
    old = pd.read_csv(cache_path)
    key_cols = [c for c in ["mean_maxsim", "median_maxsim", "frac_ge_0.99"] if c in df.columns and c in old.columns]
    if old.shape == df.shape:
        diffs = (df[key_cols].values - old[key_cols].values)
        max_diff = float(np.abs(diffs).max())
        print(f"  [VERIFY] {cache_path.name}: max numeric diff = {max_diff:.6f} "
              f"({'OK ✓' if max_diff < 1e-4 else 'MISMATCH ⚠'})")
    else:
        print(f"  [VERIFY] {cache_path.name}: shape changed {old.shape} → {df.shape}")


# ── entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    k_star = selected_k()
    print(f"Selected cluster k = {k_star} (fixed contract)")

    print("\n[1/2] Computing cross-split Hamming similarity …")
    leak, dist = compute_cross_split_similarity(k_star)

    out1 = OUT_DIR / "cross_split_similarity.csv"
    out_npz = OUT_DIR / "cross_split_dist.npz"
    cache_prev = OUT_DIR / "cross_split_similarity.prev.csv"
    if out1.exists():
        # keep previous artifact for true reproducibility check
        out1.replace(cache_prev)
    leak.to_csv(out1, index=False)
    save_dist(dist, out_npz)
    verify_against_cache(leak, cache_prev if cache_prev.exists() else out1)
    print(f"  Saved → {out1.relative_to(PROJECT_ROOT)}")
    print(f"  Saved → {out_npz.relative_to(PROJECT_ROOT)}  ({len(dist)} arrays)")
    print(leak.to_string(index=False))

    print("\n[2/2] Computing leakage vs AUROC …")
    link = compute_leakage_vs_auc(leak)
    out2 = OUT_DIR / "leakage_vs_binary_auc.csv"
    link.to_csv(out2, index=False)
    print(f"  Saved → {out2.relative_to(PROJECT_ROOT)}")
    print(link[["scheme", "leak_pct", "auroc"]].to_string(index=False))

    print("\nDone.")
