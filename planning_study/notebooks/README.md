# Preliminary presentation notebooks

Run these notebooks from a checkout of this branch (for example, with JupyterLab) after installing `planning_study/requirements.txt`. Both notebooks find the repository root from the current working directory.

1. `01_preliminary_portfolio.ipynb` reads the saved PyPSA portfolio CSV and writes `figures/preliminary_portfolio_mix.png`. It runs without the private grid inputs.
2. `02_ptdf_heatmap.ipynb` uses `hosting_capacity/hc_engine.py`, the saved POI table, and the private PWF bus/branch CSVs. Restore those inputs locally at the paths shown in its first code cell. It writes `figures/preliminary_ptdf_heatmap.png`.

These are presentation aids for the unmerged planning-study branch. New capacity means new build in each model case, not total installed capacity. The PTDF plot is a DC transfer sensitivity; siting, resource potential, and full engineering feasibility are later questions. No private inputs or generated figures are committed here.
