#!/usr/bin/env python3
"""Phase 5 (optional check): AC power-flow screening with PyPSA (Newton-Raphson).

Gate first: the PWF branch model (π lines with charging and end reactors, transformers with off-nominal
tap and phase shift, bus shunts) must reproduce the PWF's own AC branch flows. The bus angles in the
export are rounded (more than half are whole degrees), so flows cannot be recomputed from them; instead
PyPSA solves the AC power flow with the PWF injections: internal buses use their generation and load,
boundary equivalents inject the net power implied by the AC branch flows and hold their solved voltage
(their reactive injections are not in the export), generator buses hold their solved voltage (no
reactive limits; PyPSA's power flow does not enforce them). If the solved flows match the PWF flows the
model is used to screen voltages (0.95-1.05 pu normal range) with the portfolio's study injections added
under the main absorption case. If the gate fails, no AC result is reported.
"""
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import pandas as pd
import pypsa

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'hosting_capacity'))
import run_hosting as rh  # noqa: E402
from hc_engine import Network, REF_BUS  # noqa: E402

logging.getLogger('pypsa').setLevel(logging.ERROR)
OUT = HERE / 'results'
VNOM = 10.0      # with v_nom = 10 kV and PyPSA's 1 MVA base, line r/x/b in ohm/siemens equal the PWF 100 MVA per-unit values


def build(net, shunt_sign=1.0, extra_p=None):
    B, L = net.B, net.L
    n = pypsa.Network()
    bus = [str(int(b)) for b in B.id]
    n.add('Bus', bus, v_nom=VNOM)
    ln = L.kind == 'line'
    il = np.flatnonzero(ln.to_numpy()); it = np.flatnonzero(~ln.to_numpy())
    n.add('Line', [f'l{i}' for i in il], bus0=[str(int(L.f.iloc[i])) for i in il], bus1=[str(int(L.t.iloc[i])) for i in il],
          r=L.r.to_numpy()[il], x=L.x.to_numpy()[il], b=L.b.to_numpy()[il], s_nom=1e5)
    n.add('Transformer', [f't{i}' for i in it], bus0=[str(int(L.f.iloc[i])) for i in it], bus1=[str(int(L.t.iloc[i])) for i in it],
          r=L.r.to_numpy()[it], x=L.x.to_numpy()[it], s_nom=100.0, tap_ratio=L.tap.to_numpy()[it],
          phase_shift=L['shift'].to_numpy()[it], tap_side=0, model='pi')
    # bus shunts (Mvar at 1 pu) and line end reactors (pu on 100 MVA)
    sh = B.bs.to_numpy(float).copy()
    end = np.zeros(len(B))
    np.add.at(end, net.f, L.bfr.to_numpy() * 100.0); np.add.at(end, net.t, L.bto.to_numpy() * 100.0)
    tot = sh + end
    k = np.flatnonzero(np.abs(tot) > 1e-9)
    n.add('ShuntImpedance', [f'sh{i}' for i in k], bus=[bus[i] for i in k], b=shunt_sign * tot[k] / VNOM ** 2)
    boundary = B.boundary.to_numpy()
    p_int = (B.pg - B.pd).to_numpy(float)
    p = np.where(boundary, net.p_implied, p_int)
    if extra_p is not None:
        p = p + extra_p
    q = np.where(boundary, 0.0, (B.qg - B.qd).to_numpy(float))
    pv = boundary | (B.type.to_numpy() == 1)
    slack = net.idx[REF_BUS]
    ctrl = np.where(pv, 'PV', 'PQ').astype(object); ctrl[slack] = 'Slack'
    n.add('Generator', [f'g{i}' for i in range(len(B))], bus=bus, p_set=p, q_set=q, control=list(ctrl), p_nom=1e5)
    for i in np.flatnonzero(pv):
        n.buses.at[bus[i], 'v_mag_pu_set'] = float(B.solved_vm.iloc[i])
    n.buses.at[bus[slack], 'v_mag_pu_set'] = float(B.solved_vm.iloc[slack])
    return n, il, it


def flows(n, net, il, it):
    p0 = np.zeros(net.m)
    p0[il] = n.lines_t.p0.iloc[0][[f'l{i}' for i in il]].to_numpy()
    p0[it] = n.transformers_t.p0.iloc[0][[f't{i}' for i in it]].to_numpy()
    return p0


def gate(net):
    P = net.L.p_from_mw.to_numpy()
    best = None
    for sign in (1.0, -1.0):
        n, il, it = build(net, sign)
        info = n.pf()
        conv = bool(np.all(info.converged.values))
        f = flows(n, net, il, it)
        err = np.abs(f - P)
        ok = err <= 2 + 0.02 * np.abs(P)
        vm = n.buses_t.v_mag_pu.iloc[0].to_numpy()
        verr = np.abs(vm - net.B.solved_vm.to_numpy())
        rec = {'shunt_sign': sign, 'converged': conv, 'median_err_mw': float(np.median(err)), 'p95_err_mw': float(np.percentile(err, 95)),
               'share_within_2mw_2pct': float(ok.mean()), 'median_v_err_pu': float(np.median(verr)), 'p95_v_err_pu': float(np.percentile(verr, 95))}
        print(rec, flush=True)
        if conv and (best is None or rec['share_within_2mw_2pct'] > best[0]['share_within_2mw_2pct']):
            best = (rec, sign)
    return best


def main():
    net = Network()
    rec, sign = gate(net)
    passed = rec['converged'] and rec['share_within_2mw_2pct'] >= 0.90 and rec['median_v_err_pu'] < 0.01
    pd.DataFrame([{**rec, 'gate_passed': passed}]).to_csv(OUT / 'ac_gate.csv', index=False, float_format='%.6f')
    print('GATE AC model reproduces PWF flows:', 'PASS' if passed else 'FAIL', flush=True)
    if not passed:
        print('No AC screening is reported (gate failed).')
        return 0
    # voltage screening of the main-case study injections (reference siting), N-0 only
    rows = []
    lp_load = {y: float(pd.read_csv(rh.NODAL / 'macro_reopt' / 'output' / f'REFRESH_{y}' / 'annual_energy.csv').load_twh.iloc[0])
               for y in (2035, 2050)}
    pwf_load = float(net.B.loc[(net.B.state == 11) & (net.B.pd > 0), 'pd'].sum())
    prof = rh.availability_profiles()
    mg = (net.B.state == 11).to_numpy() & (net.B.kv.to_numpy() >= 230) & (net.B.kv.to_numpy() < 999)
    n0, il, it = build(net, sign); n0.pf(); v0 = n0.buses_t.v_mag_pu.iloc[0].to_numpy()
    for year in (2035, 2050):
        cap = rh.portfolio_mw(year)
        fac_df, _ = rh.lp_factors(year, cap)
        fac_df = pd.concat([fac_df, rh.caiso_factors(year, cap, prof)], ignore_index=True)
        growth = (lp_load[year] / lp_load[2035] - 1.0) * pwf_load if year == 2050 else 0.0
        sites, _ = rh.reference_siting(year, cap)
        for siting in ('reference', 'reference_bess_at_load'):
            if siting.endswith('load') and cap.get('bess', 0) <= 0:
                continue
            for cond in fac_df.condition.unique():
                fac = dict(fac_df[fac_df.condition == cond][['technology', 'factor']].itertuples(index=False))
                units, P, meta = rh.study_units(net, sites, fac, cap, bess_at_load=siting.endswith('load'))
                need = P.sum(); w, _, _ = net.balancing(rh.MAIN); g_abs = min(growth, need)
                dp = units @ P - (g_abs * net.load_weights() + (need - g_abs) * w)
                n, il, it = build(net, sign, extra_p=dp)
                info = n.pf()
                conv = bool(np.all(info.converged.values))
                vm = n.buses_t.v_mag_pu.iloc[0].to_numpy()
                # bus-specific normal-condition limits of the PWF (DGLT groups; e.g. 500 kV 1.00-1.10 pu)
                vmin = net.B.vmin.to_numpy(float); vmax = net.B.vmax.to_numpy(float)
                low = mg & (vm < vmin - 1e-4); high = mg & (vm > vmax + 1e-4)
                low0 = mg & (v0 < vmin - 1e-4); high0 = mg & (v0 > vmax + 1e-4)
                new_low = low & ~low0; new_high = high & ~high0
                nm = net.B.name.to_numpy()
                margin = np.minimum(vm - vmin, vmax - vm)
                rows.append({'year': year, 'siting': siting, 'condition': cond, 'converged': conv, 'min_vm_mg_230plus': float(vm[mg].min()),
                             'max_vm_mg_230plus': float(vm[mg].max()), 'buses_below_limit': int(low.sum()), 'buses_above_limit': int(high.sum()),
                             'base_below_limit': int(low0.sum()), 'base_above_limit': int(high0.sum()),
                             'new_below_limit': int(new_low.sum()), 'new_above_limit': int(new_high.sum()),
                             'min_margin_pu': float(margin[mg].min()), 'tightest_bus': nm[np.argmin(np.where(mg, margin, 9))],
                             'max_dv_vs_base_pu': float(np.abs(vm - v0)[mg].max())})
                print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / 'ac_voltage_screen.csv', index=False, float_format='%.6f')
    return 0


if __name__ == '__main__':
    sys.exit(main())
