#!/usr/bin/env bash
# run_all.sh — dependency-ordered computation pipeline
# From project root:
#   bash notebooks/scripts/run_all.sh
#   bash notebooks/scripts/run_all.sh --skip-heavy
#
# --skip-heavy  skip data prep, embedding extraction, and fig2 Hamming O(n^2)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON:-$PROJECT_ROOT/venv/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON="$(command -v python3)"
SKIP_HEAVY=0
for arg in "$@"; do [[ "$arg" == "--skip-heavy" ]] && SKIP_HEAVY=1; done

echo "=== HIV-ESM-2 pipeline ==="
echo "Project root : $PROJECT_ROOT"
echo "Python       : $PYTHON"
echo "Skip heavy   : $SKIP_HEAVY"
echo ""
cd "$PROJECT_ROOT"
S="notebooks/scripts"

# 0) Data + embeddings (heavy / one-time)
if [[ $SKIP_HEAVY -eq 0 ]]; then
  echo "[0a] Prepare Full HIVDB tables → work/prepared/*.csv + results/notebooks/00_data/"
  $PYTHON $S/step01_prepare_data.py

  echo "[0b] Extract ESM-2 embeddings → work/emb_full/*.npy"
  $PYTHON $S/step02_extract_embeddings.py
else
  echo "[0a/0b] SKIPPED (--skip-heavy); requiring existing work/prepared + work/emb_full"
  [[ -f work/prepared/PI.csv ]] || { echo "missing work/prepared/PI.csv"; exit 1; }
  [[ -f work/emb_full/PR_mean.npy ]] || { echo "missing work/emb_full/PR_mean.npy"; exit 1; }
fi
echo ""

# 1) Cluster k contract is fixed in notebooks/scripts/_common.py (CLUSTER_K=12).
# figS1 may still produce selected_cluster_k.csv as a sensitivity artifact, but it is
# not required to run the main protocol benchmark.
echo "[1] Cluster OOD k fixed at CLUSTER_K=12 (_common.selected_k); figS1 optional"
echo ""

# 2) Single source of truth for protocol AUROCs (fig3 owns probe_* tables)
echo "[2] Protocol benchmark (3 methods × 4 protocols) → probe_protocol_full_selected_k.csv"
$PYTHON $S/step03_protocol_benchmark.py
echo ""

# 3) Fig4 (single entry)
echo "[3] Fig4 pipeline → results/notebooks/fig4/"
$PYTHON $S/step04_fig4_compute.py
echo ""

# 4) Fig5 (single entry)
echo "[4] Fig5 pipeline → work/results/probe_fig5.csv + probe_q5_epistasis.csv + results/notebooks/fig5/"
$PYTHON $S/step05_fig5_compute.py
echo ""

# 5) Fig2 similarity then fig3 summaries
if [[ $SKIP_HEAVY -eq 0 ]]; then
  echo "[5] Cross-split Hamming similarity → results/notebooks/fig2/"
  $PYTHON $S/step06_cross_split_similarity.py
else
  echo "[5] SKIPPED fig2 Hamming (--skip-heavy)"
fi
echo "[6] Fig3 protocol summary tables"
$PYTHON $S/step07_protocol_summary.py
echo ""

echo "=== All computation scripts finished ==="
echo "Plot with:"
echo "  for nb in notebooks/fig1.ipynb notebooks/fig2.ipynb notebooks/fig3.ipynb notebooks/fig4.ipynb notebooks/fig5.ipynb notebooks/figS1.ipynb; do"
echo "    $PYTHON -m jupyter nbconvert --to notebook --execute --inplace \"\$nb\""
echo "  done"
