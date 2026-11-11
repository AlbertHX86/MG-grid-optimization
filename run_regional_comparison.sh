#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
# The regional comparison imports MG_Step2_Congestion_Relief from its own working directory.
# An alias is created locally without mutating the nested stage-2 source data.
ln -sfn ../MG_Step2_Congestion_Relief regional_comparison/MG_Step2_Congestion_Relief
python regional_comparison/optimize_scope.py
python regional_comparison/evaluate_parallel_lines.py
