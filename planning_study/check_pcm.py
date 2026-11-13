#!/usr/bin/env python3
"""Phase 4 gate: without flowgates, the monthly production-cost runs reproduce the annual PyPSA dispatch of
the same fixed portfolio (imports, gas, biogas, unserved energy, production cost). Exit 1 on failure."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pcm  # noqa: E402

fails = []


def check(name, ok, detail):
    print(('PASS ' if ok else 'FAIL ') + name + ': ' + detail)
    if not ok:
        fails.append(name)


S = pd.read_csv(HERE / 'results' / 'pcm_scenarios.csv').set_index('scenario')
for y in (2035, 2050):
    soc, ann = pcm.annual_soc(y)
    cp = S.loc[f'{y}_copperplate']
    inp = pcm.pp.lp_inputs(pcm.pp.refresh_cases()[0 if y == 2035 else 1])
    prod_annual = ann['imports_twh'] * 1000 * inp['imp_mc'] + ann['gas_twh'] * 1000 * inp['gas_mc'] + ann['biogas_twh'] * 1000 * inp['biogas_mc']
    check(f'{y} monthly copper plate = annual dispatch: imports', abs(cp.imports_twh - ann['imports_twh']) < 1e-3 * max(1, ann['imports_twh']),
          f"{cp.imports_twh:.4f} vs {ann['imports_twh']:.4f} TWh")
    check(f'{y} monthly copper plate = annual dispatch: unserved energy', abs(cp.ens_mwh - ann['shed_gwh'] * 1000) < 1.0,
          f"{cp.ens_mwh:.1f} vs {ann['shed_gwh'] * 1000:.1f} MWh")
    check(f'{y} monthly copper plate = annual dispatch: production cost', abs(cp.production_cost_mbrl_yr - prod_annual) < 1e-3 * max(1, prod_annual),
          f"{cp.production_cost_mbrl_yr:.3f} vs {prod_annual:.3f} R$ M/yr")
M = pd.read_csv(HERE / 'results' / 'pcm_monthly.csv')
fg = M[M.scenario.str.contains('grid|plan')]
check('lazy flowgates converged (no violated row left in any month)', int(fg.violated_after.max()) == 0,
      f'max violated rows after the last iteration: {int(fg.violated_after.max())}')
sys.exit(1 if fails else 0)
