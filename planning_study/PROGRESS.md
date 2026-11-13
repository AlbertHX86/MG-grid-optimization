# Planning study — progress log

Plan agreed on 2026-11-12 (follows the CAISO policy-driven assessment flow, ENERGY176/276 Lecture 11):
portfolio + resource adequacy → busbar mapping → peak/off-peak deliverability + reliability + mitigation
→ economic assessment (production cost, BCR) → findings. Scope: main grid only (≥230 kV lines and network
transformers). Attribution: CAISO 5% circle. Forced outage rates: NERC GADS class averages.

| Phase | Content | Gate | Status |
|---|---|---|---|
| 1 | Deliverability redo: pro-rata absorption, 5% circle, CAISO HSN/SSN check, POI hosting, constraint subset | `check_engine.py` passes; circle threshold 0 reproduces PR #4's DFAX-0% α | done 2026-11-12: all gates pass (18/18 α exact; PyPSA PTDF/BODF/lpf ≤2e-9; subset 106/106 in sample, 71/71 out of sample) |
| 2 | Portfolio LP in PyPSA (1a) + sensitivities; resource adequacy (1b) | PyPSA reproduces 15,752.63 / 20,284.67; no-outage dispatch has zero shed | — |
| 3 | Mitigation alternatives and costs (3f); BESS location (2c) | every attributable violation has a costed or dispositioned alternative | — |
| 4 | 8,760-hour production-cost run with flowgates (4a); BCR | flowgates off reproduces LP annual cost/curtailment | — |
| 5 | Optional checks (P7, AC voltage, annual vs hourly matching); LaTeX report | compiled PDF reviewed page by page | — |
