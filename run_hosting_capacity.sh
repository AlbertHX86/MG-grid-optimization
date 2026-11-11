#!/usr/bin/env bash
# Hosting capacity of the existing MG grid vs the fixed 2035/2050 portfolios (issue #3).
# Needs the private inputs restored (tools/restore_local_inputs.py) and the committed
# regional_comparison/results (MILP siting sensitivities).
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 hosting_capacity/check_engine.py
python3 hosting_capacity/run_hosting.py
printf '\nHOSTING CAPACITY COMPLETE: see hosting_capacity/RESULTS.md\n'
