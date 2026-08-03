"""
fig5_benchmark_matrix.py
==================================
计算 fig5 所需的 Brier score 和 benchmark 矩阵。

输入（必须已存在）：
  work/results/probe_ood_full_selected_k.csv
  work/results/probe_q5_epistasis.csv
  results/notebooks/fig4/fig4_subtype_ood_predictions.csv

输出（写入 results/notebooks/fig5/）：
  fig5_mat.csv                  -- 每药 benchmark 矩阵（AUROC + Brier + 推荐模型）
  fig5_deployment_metrics.csv   -- OOD 部署指标（esm_gain, nonB_pct 等）
  fig5_epistasis_metrics.csv    -- 上位性探针结果

运行方式（从项目根目录）：
  python notebooks/scripts/fig5_benchmark_matrix.py
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path
_scripts=_Path(__file__).resolve().parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))
from _common import pin_compute_threads
pin_compute_threads()

from pathlib import Path

import numpy as np
import pandas as pd

# ── path discovery ────────────────────────────────────────────────────────
CWD = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = CWD if (CWD / "work" / "results").exists() else CWD.parent
assert (PROJECT_ROOT / "work" / "results").exists(), (
    f"Cannot locate work/results from {CWD}."
)

OOD_DATA  = PROJECT_ROOT / "work" / "results" / "probe_protocol_full_selected_k.csv"
OOD_FALLBACK = PROJECT_ROOT / "work" / "results" / "probe_ood_full_selected_k.csv"
EPI_DATA  = PROJECT_ROOT / "work" / "results" / "probe_q5_epistasis.csv"
PRED_DATA = PROJECT_ROOT / "results" / "notebooks" / "fig4" / "fig4_subtype_ood_predictions.csv"
OUT_DIR   = PROJECT_ROOT / "results" / "notebooks" / "fig5"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_ORDER = ["PI", "NRTI", "NNRTI", "INI"]


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    return path


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_true.astype(float) - y_prob) ** 2))


def compute_brier_per_drug(pred: pd.DataFrame) -> pd.DataFrame:
    """Compute per-drug Brier scores for binary and ESM-LGB models."""
    rows = []
    for (cls, drug), sub in pred.groupby(["class", "drug"]):
        yt     = sub["y_true"].to_numpy(float)
        yp_bin = pd.to_numeric(sub["binary_prob"],  errors="coerce").to_numpy(float)
        yp_esm = pd.to_numeric(sub["esm_lgb_prob"], errors="coerce").to_numpy(float)
        row    = {"drug": drug, "class": cls}
        ok_bin = np.isfinite(yp_bin)
        ok_esm = np.isfinite(yp_esm)
        if ok_bin.sum() >= 5:
            row["brier_binary"] = brier_score(yt[ok_bin], yp_bin[ok_bin])
        if ok_esm.sum() >= 5:
            row["brier_esm"] = brier_score(yt[ok_esm], yp_esm[ok_esm])
        rows.append(row)
    return pd.DataFrame(rows)


def get_recommendation(row: pd.Series) -> str:
    cls, gain, auroc = row["class"], row["esm_gain"], row["ood_binary"]
    if cls == "PI":
        return "Binary-LR"
    if gain >= 0.01:
        return "ESM-LGB"
    if gain > -0.01 and auroc >= 0.85:
        return "Binary-LR"
    if auroc < 0.78:
        return "⚠ Caution"
    return "Binary-LR"


def build_benchmark_matrix(ood: pd.DataFrame, brier_df: pd.DataFrame) -> pd.DataFrame:
    mat = ood[["drug", "class", "ood_binary", "ood_esm_lgb", "esm_gain"]].copy()
    mat = mat.merge(brier_df[["drug", "brier_binary", "brier_esm"]], on="drug", how="left")
    mat["recommendation"] = mat.apply(get_recommendation, axis=1)
    cls_sort = {c: i for i, c in enumerate(CLASS_ORDER)}
    mat["_cs"] = mat["class"].map(cls_sort)
    mat = (mat.sort_values(["_cs", "ood_binary"], ascending=[True, False])
              .drop("_cs", axis=1)
              .reset_index(drop=True))
    return mat


# ── entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    pin_compute_threads()
    ood_path = OOD_DATA if OOD_DATA.exists() else OOD_FALLBACK
    ood  = pd.read_csv(_require(ood_path))
    print(f"Using OOD source: {ood_path}")
    epi  = pd.read_csv(_require(EPI_DATA))
    pred = pd.read_csv(_require(PRED_DATA))

    # Derived columns used by the figure
    ood["esm_gain"] = ood["ood_esm_lgb"] - ood["ood_binary"]
    ood["nonB_pct"] = ood["n_nonB"] / ood["n"] * 100
    epi["int_gain"] = epi["binary_int"] - epi["binary"]
    epi["esm_gain"] = epi["esm_xgb"]   - epi["binary"]

    print(f"Loaded: ood={ood.shape}, epi={epi.shape}, pred={pred.shape}")

    print("Computing per-drug Brier scores …")
    brier_df = compute_brier_per_drug(pred)

    print("Building benchmark matrix …")
    mat = build_benchmark_matrix(ood, brier_df)

    # Save outputs
    out_mat  = OUT_DIR / "fig5_mat.csv"
    out_ood  = OUT_DIR / "fig5_deployment_metrics.csv"
    out_epi  = OUT_DIR / "fig5_epistasis_metrics.csv"
    mat.to_csv(out_mat,  index=False)
    ood.to_csv(out_ood,  index=False)
    epi.to_csv(out_epi,  index=False)

    print(f"  Saved → {out_mat.relative_to(PROJECT_ROOT)}  ({len(mat)} rows)")
    print(f"  Saved → {out_ood.relative_to(PROJECT_ROOT)}  ({len(ood)} rows)")
    print(f"  Saved → {out_epi.relative_to(PROJECT_ROOT)}  ({len(epi)} rows)")

    print("\nBenchmark matrix preview:")
    print(mat[["drug","class","ood_binary","esm_gain","brier_binary","brier_esm","recommendation"]].to_string(index=False))
    print("\nDone.")
