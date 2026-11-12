#!/usr/bin/env python3
"""Peak deliverability (CAISO HSN/SSN) of the base and sensitivity portfolios (phase 2a).

CAISO uses sensitivity portfolios to inform "Category 2" transmission (Lecture 11). The HSN/SSN
output levels depend only on the availability profiles and the storage convention, not on an LP
dispatch, so base and sensitivity portfolios are compared on the same footing. Main case of the
deliverability study: reference siting, pro-rata absorption, target-year load, 5% circle.
"""
import os
import sys
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'hosting_capacity'))
import run_hosting as rh  # noqa: E402

OUT = HERE / 'results'


def main():
    port = pd.read_csv(OUT / 'portfolio_pypsa.csv')
    net = rh.Network(); crd = rh.coords(); groups = rh.upgrade_groups(net); prof = rh.availability_profiles()
    lp_load = {2035: 80.42508377485512, 2050: 97.18599192842592}          # REFRESH annual_energy.csv
    for y in lp_load:
        lp_load[y] = float(pd.read_csv(rh.NODAL / 'macro_reopt' / 'output' / f'REFRESH_{y}' / 'annual_energy.csv').load_twh.iloc[0])
    pwf_load = float(net.B.loc[(net.B.state == 11) & (net.B.pd > 0), 'pd'].sum())
    rows = []
    for r in port.itertuples(index=False):
        if r.status != 'ok':
            continue
        year = int(r.year)
        cap = {t: max(0.0, getattr(r, f'cap_{t}_gw')) * 1000 for t in ['solar_central', 'solar_distributed', 'bio',
               'small_hydro', 'wind', 'wind_site', 'biogas', 'bess']}
        scale = 1.06 if 'load_plus6' in r.case else 0.94 if 'load_minus6' in r.case else 1.0
        # target-year load relative to the 2035 PWF snapshot, at PWF scale
        growth = (lp_load[year] * scale / lp_load[2035] - 1.0) * pwf_load
        sites, _ = rh.reference_siting(year, cap)
        fac_df = rh.caiso_factors(year, cap, prof)
        for cond in ('HSN_caiso', 'SSN_caiso'):
            fac = dict(fac_df[fac_df.condition == cond][['technology', 'factor']].itertuples(index=False))
            units, P, meta = rh.study_units(net, sites, fac, cap)
            res, ups, _ = rh.assess(net, units, P, meta, rh.MAIN, groups, crd, growth_mw=growth)
            rows.append({'portfolio': r.case, 'year': year, 'condition': cond, 'load_growth_mw': growth,
                         'bess_gw': cap['bess'] / 1000, 'solar_gw': (cap['solar_central'] + cap['solar_distributed']) / 1000,
                         **{k: res[k] for k in ('need_mw', 'alpha', 'deliverable_mw', 'status', 'n_upgrades_to_alpha1',
                                                'binding_element', 'worsened_preexisting_added_mw')},
                         'upgrades': '; '.join(f"{u['upgrade']} +{u['required_increase_pct']:.0f}%" for u in ups)})
            print(r.case, cond, round(res['need_mw']), round(res['alpha'], 2), res['n_upgrades_to_alpha1'], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'sensitivity_portfolio_deliverability.csv', index=False, float_format='%.6f')
    base = df[df.portfolio.str.startswith('REFRESH')]
    ref = pd.read_csv(ROOT / 'hosting_capacity' / 'results' / 'portfolio_vs_hosting.csv')
    ref = ref[(ref.siting == 'reference') & ref.main & (ref['mode'] == 'circle') & ref.condition.isin(['HSN_caiso', 'SSN_caiso'])]
    m = base.merge(ref, on=['year', 'condition'], suffixes=('', '_phase1'))
    d = (np.abs(m.alpha - m.alpha_phase1) / np.maximum(1, m.alpha_phase1)).max()   # phase-1 CSV has 6 decimals
    ok = d < 1e-6 and len(m) == 4
    print('GATE base portfolios reproduce phase-1 HSN/SSN alpha:', 'PASS' if ok else 'FAIL', f'{d:.1e}')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
