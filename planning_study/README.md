# Planning study: can the existing MG main grid deliver the 2035/2050 portfolios?

The study follows the CAISO policy-driven assessment flow taught in ENERGY 176/276 (Lecture 11):
portfolio and resource adequacy → busbar mapping → peak and off-peak deliverability, reliability and mitigation →
production-cost economics → findings. Phase 1 (deliverability) lives in `../hosting_capacity/`; this folder holds
phases 2–5, the report and the poster. The compiled report is `report/main.pdf`; the A0 poster is `report/poster.pdf`.

| Step | Script | What it does | Gate (script exits non-zero if it fails) |
|---|---|---|---|
| 1 | `../run_hosting_capacity.sh` | DC N-0/N-1 deliverability with pro-rata absorption and the CAISO 5% circle; POI hosting map; constraint subset | PR #4 reproduced exactly; PyPSA PTDF/BODF/lpf cross-check |
| 2a | `portfolio_pypsa.py` | The single-year portfolio LP rebuilt in PyPSA; CAISO/PJM sensitivity portfolios | Reproduces the saved 2035/2050 objectives and capacities, and the original code's dry-hydro solves |
| 2a | `sensitivity_deliverability.py` | HSN/SSN deliverability of the sensitivity portfolios | Base portfolios reproduce phase 1 |
| 2b | `adequacy.py` | Method A: COPT with NERC GADS EFORd, LOLE/LOLH/EUE, ELCC. Method B: sequential Monte Carlo with an energy-limited PyPSA dispatch | Zero-EFOR COPT matches deterministic shortfalls; no-outage dispatch has zero shed (normal) and equals the original code's fixed-portfolio ENS (dry) |
| 3 | `mitigation.py` | Alternatives per attributable upgrade group, costed from published EPE/ANEEL unit costs; plan verified on a network containing the new elements | Every main-case study condition deliverable with the verified plan |
| 4 | `pcm.py`, `check_pcm.py`, `bcr.py` | 8,760-h network-constrained PyPSA dispatch with lazy flowgates on all N-0/N-1 rows; system cost and benefit-cost ratios | Without flowgates the monthly runs reproduce the annual dispatch; no violated row left in any hour |
| 5 | `contingency_p7.py`, `ac_check.py`, `matching.py` | P7 common-structure check on the existing grid and on the phase-3 plan networks, with a costed, verified P7 remedy; PyPSA AC power flow and voltage screen; hourly vs annual matching in 2050 | Multiple-outage factors vs explicit re-solve; P1 α on the plan networks ≥ 1; P1+P7 α ≥ 1 with the remedy; AC model reproduces the PWF flows; H0 reproduces the LP |
| 5 | `report/make_report.py`, `report/main.tex`, `report/poster.tex` | Every table, number and figure of the report and the poster is generated from the CSVs in `results/` | Asserts in `make_report.py` guard the claims the text makes |

Run everything after restoring the private inputs: `PYTHON=<env> bash run_hosting_capacity.sh && PYTHON=<env> bash planning_study/run_all.sh`.
The four Method B cases run in parallel (about 75 min each for 2050; on an 8 GB machine memory is tight).
Data sources: `data/gads_class_averages.csv` (NERC GADS Brochure 4, 2016–2020) and `data/unit_costs.csv` (EPE/ANEEL documents),
each with its citation. No hourly series or network files are written to `results/`.
