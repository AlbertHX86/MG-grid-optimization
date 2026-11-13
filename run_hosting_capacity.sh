#!/usr/bin/env bash
# Deliverability of the fixed 2035/2050 portfolios on the existing MG main grid (issue #3).
# Needs the private inputs restored (tools/restore_local_inputs.py) and the committed
# regional_comparison/results (MILP siting sensitivities). Environment: planning_study/requirements.txt
# (set PYTHON to that interpreter; defaults to python3).
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$PY" hosting_capacity/check_engine.py
"$PY" hosting_capacity/check_pypsa.py
"$PY" hosting_capacity/run_hosting.py
"$PY" hosting_capacity/check_results.py
printf '\nDELIVERABILITY COMPLETE: see hosting_capacity/RESULTS.md\n'
