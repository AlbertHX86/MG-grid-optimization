#!/usr/bin/env bash
# Phases 2-5 of the planning study (phase 1: run_hosting_capacity.sh). Each step exits non-zero if its gate fails.
# Needs the private inputs restored (tools/restore_local_inputs.py). Environment: planning_study/requirements.txt;
# set PYTHON to that interpreter (defaults to python3).
set -euo pipefail
trap 'p=$(jobs -p); [ -n "$p" ] && kill $p 2>/dev/null; true' EXIT   # a failed step stops any Method B jobs still running
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$PY" planning_study/portfolio_pypsa.py                  # 2a: PyPSA portfolio LP + gate + sensitivity portfolios
"$PY" planning_study/sensitivity_deliverability.py       #     HSN/SSN deliverability of sensitivity portfolios
mkdir -p planning_study/results/adequacy_parts
"$PY" planning_study/adequacy.py A 2035                  # 2b: resource adequacy, method A (COPT, ELCC)
"$PY" planning_study/adequacy.py A 2050
pids=()                                                  #     method B (energy-limited Monte Carlo): the four
for y in 2035 2050; do for h in normal dry; do           #     cases run in parallel (each 2050 case is ~1,100 solves)
  "$PY" planning_study/adequacy.py B "$y" "$h" 100 50 > "planning_study/results/adequacy_parts/log_B_${y}_${h}.txt" 2>&1 &
  pids+=("$!")
done; done
for p in "${pids[@]}"; do wait "$p"; done                # waiting on each PID propagates a failed job (set -e)
"$PY" planning_study/adequacy.py merge                   #     merges parts and checks the adequacy gates
"$PY" planning_study/mitigation.py                       # 3: costed alternatives, verified plan
"$PY" planning_study/pcm.py                              # 4: network-constrained production cost
"$PY" planning_study/check_pcm.py                        #    gate: monthly copper plate = annual dispatch
"$PY" planning_study/bcr.py                              #    benefit-cost ratios
"$PY" planning_study/contingency_p7.py                   # 5: P7 check
"$PY" planning_study/ac_check.py                         #    AC gate + voltage screen
"$PY" planning_study/matching.py                         #    hourly vs annual matching
"$PY" planning_study/report/make_report.py               #    tables, numbers and figures of the report
(cd planning_study/report && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex poster.tex)
printf '\nPLANNING STUDY COMPLETE: planning_study/report/main.pdf and poster.pdf\n'
