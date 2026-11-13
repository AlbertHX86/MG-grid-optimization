# ⚠️ DO NOT MERGE — OPEN A PR AND AN ISSUE FIRST

> **REVIEW-ONLY RESEARCH WORKSPACE. DO NOT COMMIT DIRECTLY TO `main`.**  
> **Create an Issue and a DRAFT Pull Request. Obtain data-redistribution approval, review assumptions, run all tests, and request explicit approval before merging.**  
> Tracking issue: [#1 — DO NOT MERGE](https://github.com/AlbertHX86/MG-grid-optimization/issues/1).

# Minas Gerais Electricity Portfolio and Grid Optimization

**Status:** research prototype / preliminary numerical screening (11 November 2026). **Not a construction-ready grid plan or independently validated resource assessment.**

This repository package contains the existing computational workflows and saved results for two planning horizons:

- **Part I — Generation and storage portfolios (2035 / 2050):** hourly linear programming, policy/fossil-free scenario constraints, and optimization of candidate grid connections.
- **Part II — Network congestion and expansion:** DC/PTDF screening, candidate parallel-circuit simulations, and comparison of a 4-area local case with a 12-area regional candidate set.

## Results snapshot

| Model result | 2035 | 2050 |
|---|---:|---:|
| Incremental capacity used in the latest network study, including storage | 3.217 GW | 30.101 GW |
| Model buses / branches in the reference 2035 PWF | 1,489 / 1,890 | 1,489 / 1,890 (frozen 2035 topology) |
| High-voltage physical lines monitored for selected thermal constraints | 283 | 283 |

The 2050 case **reuses the 2035 network as a stress-test proxy**, not an official 2050 planning case. Reported connection costs depend on assumptions about candidate siting and routes. Residual overloads remain after parallel-circuit screening. Values from alternative scenarios and optimization stages must not be combined without reconciling their definitions.

## Part II (revised): deliverability of the fixed portfolios on the existing main grid ([#3](https://github.com/AlbertHX86/MG-grid-optimization/issues/3))

The project driver is: **can the existing MG main grid deliver the fixed 2035/2050 portfolios, and which reinforcements do they need?** The flow follows the CAISO policy-driven assessment taught in ENERGY176/276 (Lecture 11): portfolio and resource adequacy → busbar mapping → peak/off-peak deliverability, reliability and mitigation → production-cost economics → findings. The plan and status are in [`planning_study/PROGRESS.md`](planning_study/PROGRESS.md).

- **Code:** `hosting_capacity/` (v2).
  - Scope: main grid only, ≥230 kV lines (283) and network transformers (272 branches), 437 LODF contingencies.
  - Study conditions: off-peak from the LP's midday hours; peak from the LP's evening hours and the CAISO HSN/SSN exceedance definitions.
  - Absorption, main case: all other online generation scaled down pro rata (MG hydro/thermal plus boundary inflow equivalents), load at the target-year level. MG-only back-down, boundary-only displacement and export are sensitivities.
  - Attribution: CAISO 5% circle (DFAX ≥ 5% or flow impact ≥ 5% of the rating). Pre-existing overloads are reported separately as worsened reliability needs.
  - Gates: `check_engine.py`, `check_pypsa.py` (PyPSA PTDF/BODF/linear power flow cross-check) and `check_results.py` (reproduces PR #4's all-rows α exactly).
- **Results:** [`hosting_capacity/RESULTS.md`](hosting_capacity/RESULTS.md) (generated). Summary, reference siting, main case:
  - **2035:** deliverable in every study condition (α 5.6–7.5).
  - **2050 midday (off-peak):** deliverable after LP curtailment (α 1.22); before curtailment α 0.95 with one transformer group (Irapé) binding.
  - **2050 evening (peak):** α 0.58–0.62, with 8–9 attributable upgrade groups, mostly transformation at Irapé, Montes Claros 2, Janaúba 3 and São Gonçalo do Pará, driven by 3.7 GW of BESS sited in the north. With BESS at load buses, α rises to 0.89–1.18.
  - The export sensitivity scales every existing outflow tie in proportion; the Itajubá 3–Cachoeira Paulista result in PR #4 depends on that convention.
- **Run** (after restoring the private inputs, see below): `PYTHON=<planning_study env> bash run_hosting_capacity.sh`.

> **Note (2026-11-11):** `regional_comparison/results/` was regenerated after making the substation/kV tie-break in `candidate_table` explicit (lowest eligible kV, then bus id). The connection costs differ from the PDF draft (e.g. 2035 Local-4 R$1,285.10M vs 1,357.88M).
> - The committed PDF numbers could not be reproduced with the original code on the regeneration machine (R$1,299.77M, pandas 3.1.1).
>   - The original code's candidate table is identical under pandas 2.3.3, so the pandas version is not the cause.
>   - The cause of the gap is unknown.
> - The tie rule then moves 11 of 102 candidate rows to a lower-kV bus at the same substation (Pirapora 2, Marimbondo, Itumbiara).
> - The PDF, the PNGs in `regional_comparison/figures/` (which have no generator script in this repository) and `SHA256SUMS.txt` predate these changes.

## Project layout

```text
README.md
PR_DRAFT.md
DATA_PROVENANCE_AND_LICENSE.md
ISSUE_CHECKLIST.md
run_portfolio_and_siting.sh
run_congestion.sh
run_regional_comparison.sh
MG_Step2_Congestion_Relief/
  MG_Nodal_Connection_Optimization/
    inputs/                  # portfolios, scenario assumptions, candidate locations, network data
    macro_reopt/             # hourly LP code, inputs and saved solutions
    code/                    # nodal siting and audit scripts
    outputs/                 # archived nodal siting results
    requirements.txt
    RUN_ALL.sh
  step2_model.py             # DC/PTDF and physical parallel circuits
  local_only_run.py
  redispatch_compare.py
  audit_step2.py
  create_outputs.py
  results/                    # 2035/2050 congestion results
  requirements.txt
  RUN_STEP2.sh
regional_comparison/
  optimize_scope.py
  evaluate_parallel_lines.py
  results/                    # local-vs-regional scenario comparisons
```

## Quickstart (Linux, macOS, or WSL)

Requires **Python 3.10+**, bash, and SciPy's MILP-compatible installation. Run from the package root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r MG_Step2_Congestion_Relief/requirements.txt
python -m pip install -r MG_Step2_Congestion_Relief/MG_Nodal_Connection_Optimization/requirements.txt

# Restore authorized local-only inputs from the separate handoff folder first.
# (From this repository root; adjust the path to the private ZIP.)
python tools/restore_local_inputs.py ../02_LOCAL_ONLY_DO_NOT_UPLOAD/PRIVATE_INPUTS_FOR_LOCAL_REPRODUCTION.zip
python tools/check_readiness.py

# Part I: reproduce portfolio refresh + candidate nodal siting
bash run_portfolio_and_siting.sh

# Part II: re-run congestion and parallel-circuit screening
bash run_congestion.sh

# Compare 4-area northern case and 12-area regional candidate optimization
bash run_regional_comparison.sh
```

**Run-order note:** The local-vs-regional model imports the previously produced siting/network inputs. Its launcher creates a **local symlink** to the bundled Step 2 project. On Windows, use WSL for the shell launchers, or run the Python commands manually after making the same relative directory available. The MILP scripts may take time and require substantial memory. Numerical output can depend on Python/SciPy/BLAS versions.

**Repository checkouts intentionally omit third-party-derived input tables.** A complete handoff ZIP contains `02_LOCAL_ONLY_DO_NOT_UPLOAD/PRIVATE_INPUTS_FOR_LOCAL_REPRODUCTION.zip`, but it is outside the GitHub upload folder. Restore it locally only if authorized. A public GitHub clone will **not** reproduce the numeric results until these inputs are separately sourced with appropriate permission. Do not circumvent source restrictions.

## Important scientific limitations

1. **Candidate resource regions:** the 12-area set represents assumed municipal candidate locations. It is **not** a verified inventory of buildable wind, solar, hydro, biomass or biogas resources. Regional capacity upper bounds include assumptions.
2. **Connection distance:** location-to-bus lengths approximate straight-line projected distances multiplied by an assumed routing factor (1.25); they are not surveyed transmission routes.
3. **PTDF assumptions:** fixed-topology DC/PTDF calculations change net injections, not the network matrix; physical parallel circuits require recomputation of network susceptance and flows.
4. **Line screening:** only 283 filtered high-voltage physical lines are checked for the chosen thermal-rating proxy; this is not comprehensive AC, reactive-power, transformer, N-1 or stability validation.
5. **2050 grid:** there is no independent official 2050 PWF topology in this package.
6. **Cost accounting:** connection CAPEX, parallel-circuit CAPEX, and discounted portfolio-model objectives are different cost conventions. Do not directly add unlike quantities.
7. **Engineering status:** all candidate interconnection points and corridor investments are **research proxies**, not approved projects.

## Data provenance and redistribution

**STOP BEFORE PUBLIC PUSH:** This handoff separates the six most obvious third-party-derived raw/historical files into a private, non-upload folder. Other model outputs and source-derived data still require a human review for redistribution and confidential information. See [DATA_PROVENANCE_AND_LICENSE.md](DATA_PROVENANCE_AND_LICENSE.md). The `.gitignore` flags several potentially restricted paths, but a human review must inspect **all** source files and derived products before any public upload. A `.gitignore` does **not** remove already tracked files.

## Repository setup and Draft PR

See [`GITHUB_SETUP_EN.md`](GITHUB_SETUP_EN.md) for the instructions for the initially empty repository, branch creation, staged upload, and Draft PR. The existing tracking issue is [#1](https://github.com/AlbertHX86/MG-grid-optimization/issues/1).

## Review and contributions

- Create an **Issue** to capture changes and evidence: [issue #1](https://github.com/AlbertHX86/MG-grid-optimization/issues/1).
- Create a **Draft PR** targeting `main`. Keep it unmerged until the owner approves data sources, reproducibility, and scientific labeling.
- Check `ISSUE_CHECKLIST.md` and `PR_DRAFT.md` for suggested descriptions.

**DO NOT MERGE WITHOUT EXPLICIT REVIEW AND APPROVAL.**
