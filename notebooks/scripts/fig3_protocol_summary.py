"""
fig3_protocol_summary.py
=================================
从预计算的 AUROC 表和相似度统计中，产出 fig3 所需的所有中间结果 CSV。

输入（必须已存在）：
  work/results/probe_protocol_full_selected_k.csv
  results/notebooks/fig2/cross_split_similarity.csv
  (Cluster OOD uses fixed CLUSTER_K=12 from _common.selected_k)

输出（写入 results/notebooks/fig3/）：
  protocol_method_auc_summary.csv   -- 各方法×协议的 mean AUROC + 95% CI
  protocol_pressure_metrics.csv     -- 3协议的压力评分（原始值）
  protocol_pressure_scaled.csv      -- percentile-rank 标准化后的压力评分
  random_to_strict_protocol_drop.csv-- Random→严格协议的 AUROC drop + CI
  subtype_ood_method_auc.csv        -- Subtype OOD 下逐药×方法 AUROC（long format）
  random_minus_subtype_esm_nonlinear_drop.csv -- 逐药 AUROC 差值（ESM-nonlinear）

运行方式（从项目根目录）：
  python notebooks/scripts/fig3_protocol_summary.py
"""

from __future__ import annotations

from pathlib import Path

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import selected_k  # noqa: E402

# ── path discovery ────────────────────────────────────────────────────────
CWD = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = CWD if (CWD / "work" / "results").exists() else CWD.parent
assert (PROJECT_ROOT / "work" / "results").exists(), (
    f"Cannot locate work/results from {CWD}."
)

RES     = PROJECT_ROOT / "work" / "results"
FIG2_OUT= PROJECT_ROOT / "results" / "notebooks" / "fig2"
OUT_DIR = PROJECT_ROOT / "results" / "notebooks" / "fig3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METHODS      = ["binary", "esm_lr", "esm_lgb"]
M_LABEL      = {"binary": "binary mutation", "esm_lr": "ESM-linear", "esm_lgb": "ESM-nonlinear"}
PROTO_LABEL  = {"random": "Random", "rg": "Patient", "clu": "Cluster", "ood": "Subtype"}
STRICT_ORDER = ["rg", "clu", "ood"]


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    return path


def boot_ci(vals: np.ndarray, n: int = 2000, seed: int = 42) -> tuple[float, float, float]:
    vals = np.asarray(vals, float)
    vals = vals[~np.isnan(vals)]
    if len(vals) < 2:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    bs  = [rng.choice(vals, len(vals), replace=True).mean() for _ in range(n)]
    return float(vals.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


# ── sub-computations ──────────────────────────────────────────────────────

def compute_protocol_method_summary(auc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for proto in ["random", *STRICT_ORDER]:
        for method in METHODS:
            mean, lo, hi = boot_ci(auc[f"{proto}_{method}"].values)
            rows.append({
                "protocol":       proto,
                "protocol_label": PROTO_LABEL[proto],
                "method":         method,
                "method_label":   M_LABEL[method],
                "mean":           mean,
                "ci_low":         lo,
                "ci_high":        hi,
            })
    return pd.DataFrame(rows)


def compute_protocol_pressure(
    auc: pd.DataFrame, leak: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    leak_summary = leak.groupby("scheme", as_index=False).agg(
        mean_maxsim  = ("mean_maxsim",  "mean"),
        frac_ge_099  = ("frac_ge_0.99", "mean"),
    )
    leak_map = leak_summary.set_index("scheme")

    stress_rows = []
    for proto, scheme in [("rg", "patient"), ("clu", "cluster"), ("ood", "subtype")]:
        drops    = [auc[f"random_{m}"] - auc[f"{proto}_{m}"] for m in METHODS]
        all_drop = pd.concat(drops, ignore_index=True)
        stress_rows.append({
            "protocol":        proto,
            "protocol_label":  PROTO_LABEL[proto],
            "anti_leakage":    1 - float(leak_map.loc[scheme, "frac_ge_099"]),
            "sequence_novelty":1 - float(leak_map.loc[scheme, "mean_maxsim"]),
            "subtype_shift":   1.0 if proto == "ood" else 0.0,
            "auroc_stress":    float(all_drop.mean()),
        })
    stress       = pd.DataFrame(stress_rows)
    metric_cols  = ["anti_leakage", "sequence_novelty", "subtype_shift", "auroc_stress"]
    stress_scaled = stress.copy()
    for col in metric_cols:
        stress_scaled[col] = stress[col].rank(method="average", pct=True)
    stress["pressure_score"] = stress_scaled[metric_cols].mean(axis=1)
    return stress, stress_scaled[["protocol", "protocol_label", *metric_cols]]


def compute_random_to_strict_drop(auc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for proto in STRICT_ORDER:
        for method in METHODS:
            vals = auc[f"random_{method}"] - auc[f"{proto}_{method}"]
            mean, lo, hi = boot_ci(vals.values)
            rows.append({
                "protocol":       proto,
                "protocol_label": PROTO_LABEL[proto],
                "method":         method,
                "method_label":   M_LABEL[method],
                "mean_drop":      mean,
                "ci_low":         lo,
                "ci_high":        hi,
            })
    return pd.DataFrame(rows)


def compute_subtype_ood_per_drug(auc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        for _, r in auc.iterrows():
            rows.append({
                "class":        r["class"],
                "drug":         r["drug"],
                "method":       method,
                "method_label": M_LABEL[method],
                "auroc":        r[f"ood_{method}"],
            })
    return pd.DataFrame(rows)


def compute_random_minus_subtype_drop(auc: pd.DataFrame) -> pd.DataFrame:
    drop_df = auc[["class", "drug", "n_res", "random_esm_lgb", "ood_esm_lgb"]].copy()
    drop_df["drop"]     = drop_df["random_esm_lgb"] - drop_df["ood_esm_lgb"]
    drop_df["abs_drop"] = drop_df["drop"].abs()
    drop_df["status"]   = np.where(drop_df["drop"] > 0.05, "overestimated", "stable")
    return drop_df.sort_values("drop", ascending=True).reset_index(drop=True)


# ── entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    k_star       = selected_k()
    protocol_path = _require(RES / "probe_protocol_full_selected_k.csv")
    leak_path     = _require(FIG2_OUT / "cross_split_similarity.csv")

    auc  = pd.read_csv(protocol_path)
    leak = pd.read_csv(leak_path)

    required_cols = [f"{p}_{m}" for p in ["random", "rg", "clu", "ood"] for m in METHODS]
    missing = [c for c in required_cols if c not in auc.columns]
    if missing:
        raise ValueError(f"Protocol CSV missing columns: {missing}")

    print(f"Loaded: {len(auc)} drugs | k={k_star} | source: {protocol_path.name}")

    # 1. Protocol × method summary (boot CI)
    summary = compute_protocol_method_summary(auc)
    summary.to_csv(OUT_DIR / "protocol_method_auc_summary.csv", index=False)
    print("  Saved protocol_method_auc_summary.csv")

    # 2. Protocol pressure heatmap data
    stress, stress_scaled = compute_protocol_pressure(auc, leak)
    stress.to_csv(OUT_DIR / "protocol_pressure_metrics.csv", index=False)
    stress_scaled.to_csv(OUT_DIR / "protocol_pressure_scaled.csv", index=False)
    print("  Saved protocol_pressure_metrics.csv + protocol_pressure_scaled.csv")

    # 3. AUROC drop: Random → each strict protocol
    drop_summary = compute_random_to_strict_drop(auc)
    drop_summary.to_csv(OUT_DIR / "random_to_strict_protocol_drop.csv", index=False)
    print("  Saved random_to_strict_protocol_drop.csv")

    # 4. Per-drug subtype OOD AUROC (long format)
    subtype = compute_subtype_ood_per_drug(auc)
    subtype.to_csv(OUT_DIR / "subtype_ood_method_auc.csv", index=False)
    print("  Saved subtype_ood_method_auc.csv")

    # 5. Per-drug AUROC drop (ESM-nonlinear)
    drop_df = compute_random_minus_subtype_drop(auc)
    drop_df.to_csv(OUT_DIR / "random_minus_subtype_esm_nonlinear_drop.csv", index=False)
    print("  Saved random_minus_subtype_esm_nonlinear_drop.csv")

    print("\nProtocol pressure scores:")
    print(stress[["protocol_label", "anti_leakage", "sequence_novelty",
                  "subtype_shift", "auroc_stress", "pressure_score"]].round(3).to_string(index=False))
    print("\nDone.")
