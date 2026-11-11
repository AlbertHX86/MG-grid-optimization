#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/MG_Step2_Congestion_Relief"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
bash RUN_STEP2.sh
