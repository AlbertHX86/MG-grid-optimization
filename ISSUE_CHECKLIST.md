# Issue: DO NOT MERGE — research reproducibility and data release review

Existing issue: https://github.com/AlbertHX86/MG-grid-optimization/issues/1

- [ ] Verify authority to publish third-party EPE, ONS, CEMIG and geographic inputs.
- [ ] Inspect generated CSV/JSON for embedded copyrighted/restricted source values.
- [ ] Verify latest portfolio inputs, installed-capacity assumptions, and technology totals.
- [ ] Re-run the hourly portfolio LP, check solver log and hourly feasibility.
- [ ] Re-run nodal MILP and compare selected bus coordinates and connection costs.
- [ ] Re-run 2035/2050 DC/PTDF and physical-parallel-line screening.
- [ ] Re-run northern 4-area and regional 12-area comparison.
- [ ] Document exact dependencies, runtime, hardware and SHA256 checksums.
- [ ] Confirm that 2050 has no independently validated grid topology.
- [ ] Confirm candidate resource regions are not certified resource reserves.
- [ ] Review AC/reactive constraints, N-1, site eligibility, costs and reliability as open limitations.
- [ ] Request human sign-off on Draft PR; never merge automatically.
