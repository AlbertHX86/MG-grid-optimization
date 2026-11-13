#!/usr/bin/env python3
"""Regression gate for hc_engine. Run before trusting any hosting-capacity number.

1. Reproduces the paper's 2050 MG-local pre-build overload (663.31 MW, Local-4 siting,
   PV90, 283 lines) and the generator-weighted attribution test (8.62 MW).
2. Cross-checks PTDF columns against the original engine (optimize_siting.ptdf_bus).
3. Checks LODF post-contingency flows against an explicit outage re-solve.
Exit code 1 on any failure.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from hc_engine import Network, ROOT  # noqa: E402

PFAC = {'solar_central': .70, 'bio': .50, 'small_hydro': .60, 'wind_site': .50, 'biogas': .70, 'bess': 0.0}
fails = []


def check(name, ok, detail):
    print(('PASS ' if ok else 'FAIL ') + name + ': ' + detail)
    if not ok:
        fails.append(name)


net = Network()
lines = net.kind_mon == 'line'
assert lines.sum() == 283, lines.sum()

# 1 -- paper reproduction ------------------------------------------------------
sites = pd.read_csv(HERE / 'tests' / 'fixture_paper_2050_local4_chosen_sites.csv')
port = pd.read_csv(ROOT / 'MG_Step2_Congestion_Relief/MG_Nodal_Connection_Optimization/inputs/portfolio_2050.csv')
dg = float(port.loc[port.technology == 'solar_distributed', 'new_build_gw'].iloc[0]) * 1000
pv = 0.90
inj = net.bus_vector({})
for r in sites.itertuples(index=False):
    inj[net.idx[int(r.bus_id)]] += r.new_capacity_mw * (pv if r.tech == 'solar_central' else PFAC[r.tech])
lw = net.load_weights()
inj_dg = inj + lw * dg * pv          # paper convention: DG at load buses (cancels under load balancing)
M = net.monitored[lines]
# The attribution test in the 2026-11-11 review moved only the central-site injections.
for case, vec, expect in [('legacy_load', inj_dg, 663.31), ('legacy_gen', inj, 8.62)]:
    wdr, _, _ = net.balancing(case)
    dp = vec - wdr * vec.sum()
    flow = net.F[M] + net.PTDF[M] @ dp
    over = np.maximum(0, np.abs(flow) - net.rate[M]).sum()
    check(f'paper reproduction ({case})', abs(over - expect) < 0.01, f'{over:.2f} MW vs {expect:.2f} MW')

# 2 -- independent PTDF implementation ----------------------------------------------
sys.path.insert(0, str(ROOT / 'MG_Step2_Congestion_Relief/MG_Nodal_Connection_Optimization/code'))
import optimize_siting as ns  # noqa: E402
rng = np.random.default_rng(0)
buses = rng.choice(net.bus_ids, 25, replace=False)
old = ns.ptdf_bus(buses)                                     # 283 monitored lines x buses
new = net.PTDF[M][:, [net.idx[int(b)] for b in buses]]
order = np.argsort(ns.monitor)                               # both index the same branches
check('PTDF vs optimize_siting.ptdf_bus', np.allclose(np.sort(ns.monitor), M) and
      np.max(np.abs(old[order] - new)) < 1e-9, f'max |diff| {np.max(np.abs(old[order] - new)):.2e}')

# 3 -- LODF vs explicit outage ---------------------------------------------------------
dp = rng.normal(size=net.n); dp -= dp.mean()
f_pre = net.PTDF @ dp
worst = 0.0
for jk in rng.choice(len(net.contingencies), 8, replace=False):
    k = net.contingencies[jk]
    keep = np.arange(net.m) != k
    A2, w2 = net.A[keep], net.w[keep]
    Bb = A2.T @ (w2[:, None] * A2)
    ref = net.idx[585]; free = np.array([j for j in range(net.n) if j != ref])
    th = np.zeros(net.n); th[free] = np.linalg.solve(Bb[np.ix_(free, free)], dp[free])
    f_post = np.zeros(net.m); f_post[keep] = w2 * (A2 @ th)
    pred = f_pre[net.monitored] + net.LODF[:, jk] * f_pre[k]
    mask = net.monitored != k
    worst = max(worst, float(np.max(np.abs(pred[mask] - f_post[net.monitored][mask]))))
check('LODF vs explicit outage re-solve (8 outages)', worst < 1e-6, f'max |diff| {worst:.2e} MW')

print(f'monitored: {len(net.monitored)} ({int(lines.sum())} lines, {int((~lines).sum())} transformer branches); '
      f'contingencies: {len(net.contingencies)} (+{len(net.bridges)} radial outages skipped)')
sys.exit(1 if fails else 0)
