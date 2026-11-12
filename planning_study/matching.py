#!/usr/bin/env python3
"""Phase 5 (optional check): what does "fossil-free 2050" cost under hourly vs annual matching?

The course (Lecture 15, PCE 24/7 case) contrasts annual 100 % clean matching with hourly matching.
The 2050 LP imposes hourly local balance (no imports, no gas). Definitions compared (PyPSA capacity
expansion, otherwise identical to the 2050 LP):
  H0  hourly local balance: no imports, no exports (the LP itself);
  H1  hourly local balance with exports: no imports, surplus may be exported (<= the LP import bound);
  A1  annual matching: imports allowed hour by hour (<= the import bound) but annual imports <= annual
      exports (MG is a net annual exporter of clean energy). Gas stays at zero in every case.
Imports cost and exports earn the LP's 2025 SE CMO price (R$ 209.47/MWh).
"""
import os
import sys
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import portfolio_pypsa as pp  # noqa: E402

OUT = HERE / 'results'


def run(label, netzero_mode, allow_export, annual):
    c = pp.Case(label, 2050, 'scenario_addon_2050.json', netzero_mode=netzero_mode)
    d = pp.lp_inputs(c)
    n = pp.build(c, d)
    bound = d['p']['import_bound_gw']
    if allow_export:
        n.add('Generator', 'export', bus='MG', p_nom=bound, p_max_pu=0.0, p_min_pu=-1.0, marginal_cost=d['imp_mc'])
    base = pp.extra(c, d)

    def ef(n, sns):
        base(n, sns)
        if annual:
            p = n.model.variables['Generator-p']
            n.model.add_constraints(p.sel(name='imports').sum() + p.sel(name='export').sum() <= 0, name='annual_net_exporter')
    st, cond = n.optimize(solver_name='highs', extra_functionality=ef, include_objective_constant=False,
                          solver_options={'output_flag': False})
    assert st == 'ok', (label, st, cond)
    g = n.generators_t.p
    caps = {t: float(n.generators.at[t, 'p_nom_opt']) for t in pp.CF_TECH + ['biogas']}
    caps['bess'] = float(n.storage_units.at['bess', 'p_nom_opt'])
    return {'definition': label, 'objective_mbrl': float(n.objective), **{f'cap_{t}_gw': v for t, v in caps.items()},
            'imports_twh': float(g['imports'].sum() / 1000) if 'imports' in g else 0.0,
            'exports_twh': float(-g['export'].sum() / 1000) if 'export' in g else 0.0,
            'hours_importing': int((g['imports'] > 1e-6).sum()) if 'imports' in g else 0,
            'gas_twh': float(g['gas'].sum() / 1000)}


def main():
    rows = [run('H0 hourly local balance (LP)', 'local_zero_2050', False, False),
            run('H1 hourly balance, exports allowed', 'local_zero_2050', True, False),
            run('A1 annual matching (net annual exporter)', 'annual_match', True, True)]
    df = pd.DataFrame(rows)
    df['cost_vs_H0_pct'] = 100 * (df.objective_mbrl / df.objective_mbrl.iloc[0] - 1)
    df.to_csv(OUT / 'matching_2050.csv', index=False, float_format='%.6f')
    print(df.round(3).to_string())
    ok = abs(df.objective_mbrl.iloc[0] - 20284.666350353113) / 20284.67 < 1e-6
    print('GATE H0 reproduces the LP objective:', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
