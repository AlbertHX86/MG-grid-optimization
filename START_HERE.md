# MG Grid Optimization — Complete Handoff (English)

## ⚠️ DO NOT MERGE — OPEN A PR AND AN ISSUE FIRST

**Existing issue:** https://github.com/AlbertHX86/MG-grid-optimization/issues/1  
**Repository:** https://github.com/AlbertHX86/MG-grid-optimization

This package is for the project owner to perform a manual, review-first GitHub upload. It contains:

1. `01_GITHUB_REVIEW_BRANCH_UPLOAD/` — English README, model code, scenario assumptions, saved results, provenance checklist, `.gitignore`, and run scripts. **Review data licensing and outputs before publicly pushing.**
2. `02_LOCAL_ONLY_DO_NOT_UPLOAD/` — third-party-derived inputs and all original research calculation archives. **Keep this outside any public Git repository.**
3. `03_GITHUB_INSTRUCTIONS/GITHUB_SETUP_EN.md` — complete clone/initialize/branch/push/Draft PR instructions, including special handling for the repository's initially empty `main` branch.
4. `04_REFERENCE_REPORTS/` — earlier arXiv-style PDF/LaTeX draft and congestion technical PDF, for reference; not necessary to upload to GitHub.
5. `MANIFEST_SHA256SUMS.txt` — hashes of files inside this handoff, for transport integrity.

### Fast path

Read `03_GITHUB_INSTRUCTIONS/GITHUB_SETUP_EN.md`. Only copy the **contents** of `01_GITHUB_REVIEW_BRANCH_UPLOAD/` into the `review/mg-planning-2035-2050` branch, not into `main`. Create a **Draft** PR with `PR_DRAFT.md`, and link existing issue #1. Never merge without approval.

### Development environment

After installing the requirements and restoring authorized private data, run:

```bash
python tools/check_readiness.py
bash run_portfolio_and_siting.sh
bash run_congestion.sh
bash run_regional_comparison.sh
```

This handoff performs repository-packaging and basic syntax/data consistency tests only; it is not a new full 8-scenario optimization run.
