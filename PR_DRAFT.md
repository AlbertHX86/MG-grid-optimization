# [DRAFT — DO NOT MERGE] Add Minas Gerais portfolio and congestion optimization research workflows

**Related issue:** #1

## Summary
This is a research-only import of our Minas Gerais electricity-planning prototype for 2035/2050. It contains the hourly portfolio model, candidate nodal siting MILP, selected DC/PTDF grid screening, hypothetical parallel-circuit additions, and a local-versus-regional scenario comparison. It also includes saved calculation outputs and an English quickstart.

## What's included
- Portfolio and nodal siting study, source code and archived results.
- Congestion and hypothetical transmission-upgrade scripts/results.
- 4-area northern local-case vs 12-area candidate-region optimization.
- README, data provenance warnings, and reproduction commands.

## Explicit exclusions / caveats
- No verified site-level resource reserves or approved transmission construction plans.
- No independently verified 2050 grid model; the 2035 topology is held fixed for 2050 screening.
- No full AC feasibility, N-1 or reliability certification.
- Third-party input data must be removed or cleared before a **public** PR.

## Reviewer checklist
- [ ] All uploaded inputs have redistribution permission or are excluded.
- [ ] `bash run_portfolio_and_siting.sh` reproduces the portfolio/siting outputs.
- [ ] `bash run_congestion.sh` reproduces the congestion screening.
- [ ] `bash run_regional_comparison.sh` reproduces the scope comparison.
- [ ] Scenario assumptions and different cost units are clearly labeled.
- [ ] Provenance, solver output, hashes and missing validations are documented.

**DO NOT MERGE. Leave this PR as a draft until explicitly approved.**
