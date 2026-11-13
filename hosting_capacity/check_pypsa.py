#!/usr/bin/env python3
"""Independent cross-check of hc_engine's DC network algebra with PyPSA.

Builds the PWF as a PyPSA network (every branch as a Line with the engine's effective
reactance x*tap, v_nom = 1 so x_pu = x) and compares
  1. PyPSA's PTDF (re-referenced to bus 585) with hc_engine.PTDF on all monitored branches,
  2. PyPSA's BODF with hc_engine.LODF on monitored branches x contingencies,
  3. a PyPSA linear power flow (n.lpf) with PTDF @ dp for a random balanced injection.
Exit code 1 on any failure.
"""
import os
import sys
import logging
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from hc_engine import Network, REF_BUS  # noqa: E402

import pypsa  # noqa: E402
logging.getLogger('pypsa').setLevel(logging.ERROR)

fails = []


def check(name, ok, detail):
    print(('PASS ' if ok else 'FAIL ') + name + ': ' + detail)
    if not ok:
        fails.append(name)


net = Network()
L = net.L
n = pypsa.Network()
bus = [str(int(b)) for b in net.bus_ids]
n.add('Bus', bus, v_nom=1.0)
names = [f'br{i}' for i in range(net.m)]
n.add('Line', names, bus0=[bus[i] for i in net.f], bus1=[bus[i] for i in net.t],
      x=(L.x * L.tap).to_numpy(float), r=0.0, s_nom=1e6)
n.determine_network_topology()
subs = n.sub_networks.index
check('single synchronous island', len(subs) == 1, f'{len(subs)} sub-network(s)')
sn = n.sub_networks.obj.iloc[0] if hasattr(n.sub_networks, 'obj') else n.sub_networks.at[subs[0], 'obj']

sn.calculate_PTDF()
br = sn.branches_i()                       # MultiIndex (component, name)
bi = [int(name[2:]) for _, name in br]
cols = list(sn.buses_o)
ptdf = pd.DataFrame(sn.PTDF, index=bi, columns=cols)
ref = str(REF_BUS)
ptdf_ref = ptdf.sub(ptdf[ref], axis=0)     # injection at bus, withdrawal at bus 585
M = net.monitored
P_py = ptdf_ref.loc[M, bus].to_numpy()
P_en = net.PTDF[M]
d1 = float(np.max(np.abs(P_py - P_en)))
check('PTDF: PyPSA vs hc_engine (monitored branches x all buses)', d1 < 1e-9, f'max |diff| {d1:.2e}')

sn.calculate_BODF()
bodf = pd.DataFrame(sn.BODF, index=bi, columns=bi)
K = net.contingencies
B_py = bodf.loc[M, K].to_numpy()
mask = ~net.self_mask
d2 = float(np.max(np.abs(B_py[mask] - net.LODF[mask])))
check('LODF: PyPSA BODF vs hc_engine (monitored x contingencies)', d2 < 1e-8, f'max |diff| {d2:.2e}')

rng = np.random.default_rng(7)
dp = rng.normal(scale=100.0, size=net.n); dp -= dp.mean()
n.add('Load', [f'ld{b}' for b in bus], bus=bus, p_set=-dp)
n.add('Generator', 'slack', bus=ref, control='Slack', p_nom=1e6)
n.lpf()
flow_py = n.lines_t.p0.iloc[0].reindex(names).to_numpy()
flow_en = net.PTDF @ dp
d3 = float(np.max(np.abs(flow_py - flow_en)))
check('linear power flow: PyPSA lpf vs PTDF @ dp (all branches)', d3 < 1e-6, f'max |diff| {d3:.2e} MW')

print(f'pypsa {pypsa.__version__}; buses {len(n.buses)}, branches {len(n.lines)}')
sys.exit(1 if fails else 0)
