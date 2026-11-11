#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/MG_Step2_Congestion_Relief/MG_Nodal_Connection_Optimization"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
bash RUN_ALL.sh
