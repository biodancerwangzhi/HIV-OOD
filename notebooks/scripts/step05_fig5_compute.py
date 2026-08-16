"""Fig5 full computation pipeline.

Single entry point. Runs scripts in dependency order:
  1. fig5_feature_ablation.py
  2. fig5_epistasis.py
  3. fig5_benchmark_matrix.py

Usage (from project root):
  python notebooks/scripts/step05_fig5_compute.py
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

# Pin before importing project modules that pull in numpy/sklearn.
for _k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[_k] = "1"

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _common import pin_compute_threads  # noqa: E402

PROJ = Path(__file__).resolve().parent.parent.parent

STEPS = [
    "fig5_feature_ablation.py",
    "fig5_epistasis.py",
    "fig5_benchmark_matrix.py",
]


def main() -> None:
    pin_compute_threads()
    if not (PROJ / "work").exists():
        raise SystemExit(f"Cannot locate project work/ from {PROJ}")
    for name in STEPS:
        path = SCRIPTS_DIR / name
        if not path.exists():
            raise SystemExit(f"Missing step script: {path}")
        print(f"\n===== [{path.stem}] start =====", flush=True)
        runpy.run_path(str(path), run_name="__main__")
        print(f"===== [{path.stem}] done =====\n", flush=True)
    print(f"All steps in {Path(__file__).name} finished.")


if __name__ == "__main__":
    main()
