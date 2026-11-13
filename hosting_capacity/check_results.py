#!/usr/bin/env python3
"""Post-run gates for run_hosting.py (v2). Exit code 1 on any failure.

1. With every row tested (mode 'all') and the PR #4 absorption cases, alpha reproduces PR #4's
   DFAX-0% column (fixture) exactly.
2. The 5% circle only removes rows (and pre-existing overloads), so alpha(circle) >= alpha(all).
3. Pro-rata headroom = A1 + A2 headroom; deliverable MW <= need and <= headroom.
4. The constraint subset reproduces alpha in every in-sample case with alpha < 1.
5. Study factors lie in [0, 1] (BESS in [-1, 1]).
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results'
fails = []


def check(name, ok, detail):
    print(('PASS ' if ok else 'FAIL ') + name + ': ' + detail)
    if not ok:
        fails.append(name)


chk = pd.read_csv(OUT / 'portfolio_vs_hosting.csv')
old = pd.read_csv(HERE / 'tests' / 'fixture_pr4_portfolio_vs_hosting.csv')
old = old[(old.siting == 'reference') & (old.dfax_cut == 0.0) & ~old.case.str.contains(r'\+')]
new = chk[(chk.siting == 'reference') & (chk['mode'] == 'all') & chk.case.isin(old.case.unique())
          & (((chk.year == 2035) & (chk.load_level == 'target-year load')) |
             ((chk.year == 2050) & (chk.load_level == '2035 PWF load')))]
m = old.merge(new, on=['year', 'condition', 'case'], suffixes=('_pr4', '_v2'))
dif = np.abs(m.alpha_pr4 - m.alpha_v2) / np.maximum(1, np.abs(m.alpha_pr4))
check('mode "all" reproduces PR #4 DFAX-0% alpha', len(m) == len(old) == 18 and dif.max() < 1e-9,
      f'{len(m)} of {len(old)} rows matched, max rel. diff {dif.max():.1e}')
dn = np.abs(m.need_mw_pr4 - m.need_mw_v2).max()
check('same study need as PR #4', dn < 1e-6, f'max |diff| {dn:.1e} MW')

ref = chk[(chk.siting == 'reference') & (chk.condition != 'S1_plus_existing_pv_dg')]
pv = ref.pivot_table(index=['year', 'condition', 'case', 'load_level'], columns='mode', values='alpha')
bad = pv[pv['circle'] < pv['all'] - 1e-9]
check('alpha(circle) >= alpha(all)', len(bad) == 0, f'{len(pv)} cases, {len(bad)} violations')

info = json.loads((OUT / 'run_info.json').read_text())
h = info['energy_headroom_mw']
check('pro-rata headroom = A1 + A2', abs(h['P_prorata'] - h['A1_mg_backdown'] - h['A2_displace_inflow']) < 1e-6,
      f"{h['P_prorata']:.1f} = {h['A1_mg_backdown']:.1f} + {h['A2_displace_inflow']:.1f}")
c = chk[np.isfinite(chk.alpha)]
bound = c.need_mw * np.maximum(1, c.alpha)          # CSVs carry 6 decimals: compare relatively
ok = (c.deliverable_mw <= bound * (1 + 1e-6) + 1e-6).all() and \
     (c.deliverable_mw <= c.energy_headroom_mw * (1 + 1e-6) + 1e-6).all()
check('deliverable MW bounded by need*alpha and headroom', bool(ok), f'{len(c)} rows')

s = info['subset']
check('subset exact in sample for alpha < 1', s['in_sample_exact_alpha_lt1'] == s['cases_alpha_lt1'] and
      s['in_sample_differ_alpha_lt1'] == 0, f"{s['in_sample_exact_alpha_lt1']} of {s['cases_alpha_lt1']}")
check('subset verdict in sample', s['in_sample_same_verdict'] == s['cases'], f"{s['in_sample_same_verdict']} of {s['cases']}")

f = pd.read_csv(OUT / 'study_factors.csv')
nb = f[f.technology != 'bess']
ok = nb.factor.between(-1e-9, 1 + 1e-9).all() and f[f.technology == 'bess'].factor.between(-1 - 1e-9, 1 + 1e-9).all()
check('study factors in range', bool(ok), f'{len(f)} rows')
sys.exit(1 if fails else 0)
