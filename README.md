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

## Part II (revised): hosting capacity of the existing grid ([#3](https://github.com/AlbertHX86/MG-grid-optimization/issues/3))

The project driver is now: **how much new generation can the existing MG grid deliver, and does the fixed 2035/2050 portfolio need major grid improvements?** This follows the course case studies (NREL Oregon offshore wind: "the existing system may support up to 2.6 GW"; CAISO portfolio-driven transmission planning).

- **Code:** `hosting_capacity/`.
  - DC N-0/N-1 screen on 283 lines + 272 network-transformer branches, with 437 LODF contingencies.
  - Study conditions come from the LP's own hourly dispatch.
  - Absorption of the new MW is explicit, as one of three cases:
    - **A1:** MG hydro/thermal back-down.
    - **A2:** displaced boundary inflows, including the Estreito HVDC infeed.
    - **B:** export to the rest of the SIN.
  - This replaces the load-proportional convention that produced the 663 MW Vale do Aço result.
- **Results:** [`hosting_capacity/RESULTS.md`](hosting_capacity/RESULTS.md) (generated). Summary of the current run (reference siting):
  - **2035 new additions on the PWF (evening) snapshot:** deliverable under every absorption case with 3–7× margin.
    - The data contain no midday base case.
    - Adding today's existing PV/DG at midday (a sensitivity) makes the 2035 export case bind on the Itajubá 3–Cachoeira Paulista 500 kV interface.
  - **2050 midday (8.2 GW of LP-used new output):**
    - MG's own hydro/thermal back-down (5.6 GW in the snapshot) cannot absorb it.
    - Export delivers ≈7.5 GW. The gap closes with +86 MW (+3.7%) on the 52.5 km Itajubá 3–Cachoeira Paulista 500 kV line.
    - With 2050 load growth absorbing ≈2.2 GW locally, export needs no upgrade.
  - **2050 evening:** 3.7 GW of BESS discharging in the north needs 5–9 upgrade groups.
    - Transformer banks at Irapé, Montes Claros 2, Janaúba 3 and Marimbondo.
    - 345 kV line reinforcements.
    - Placing BESS near load reduces this to ≤1.
  - **Minor vs major depends on the threshold** (both are reported).
    - Counting only short (≤50 km) parallel circuits as minor makes the 2050 export and evening cases major.
    - Also counting line uprates of ≤25% as minor leaves only the existing-PV export sensitivity major. This rule was added after the first results.
- **Run** (after restoring the private inputs, see below): `bash run_hosting_capacity.sh`. It runs the regression gate `hosting_capacity/check_engine.py` first.

> **Note (2026-11-11):** `regional_comparison/results/` was regenerated after making the substation/kV tie-break in `candidate_table` deterministic. The connection costs therefore differ from the PDF draft (e.g. 2035 Local-4 R$1,285.10M vs 1,357.88M). The PDF has not been updated. `SHA256SUMS.txt` predates these changes.

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
