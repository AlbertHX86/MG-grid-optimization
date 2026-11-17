#!/usr/bin/env python3
"""Phase 6: the joint capacity and siting LP of joint_siting.py, solved by monthly Benders decomposition.

The annual single LP (joint_siting.py) solves 2035 but not 2050 on an 8 GB machine: 8,760 hours of dispatch for many
sites with identical profiles plus thousands of dense flow cuts stall the simplex. Here the same problem is split:

* master: capacity of every candidate station (and distributed PV, BESS at load buses), statewide policy floors and
  ceilings, area caps, the area tie-break of joint_siting.py, one cost variable theta_m per month, Benders cuts and a
  box trust region around the incumbent;
* subproblem m: dispatch of month m with the capacities fixed (zero-cost capacity columns fixed by their bounds), the
  month's hydro budget, hydro ramp within the month, storage cyclic within the month, unserved energy at the LP's
  penalty (always feasible) and DC N-0/N-1 flow cuts on all 242,708 rows added lazily per (row, hour) until no row is
  violated in any hour of the month. phi_m(P) is its weighted operating cost and the reduced costs of the fixed
  capacity columns are a subgradient, so theta_m >= phi_m(P*) + g_m (P - P*) is a valid cut.

Every iterate is evaluated with complete flow cuts (all rows, all hours), so its total cost is a valid upper bound;
the master without the trust region gives the lower bound. Differences from the annual LP, both small and measured
by the copper-plate gate (which must return the one-bus LP objective): storage is cyclic within each month (a
restriction) and the first hour of each month has no ramp limit relative to the previous month (a relaxation).
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import pandas as pd
import xarray as xr
import pypsa

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import joint_siting as js  # noqa: E402
from joint_siting import pp, pcm, Network  # noqa: E402

OUT = js.OUT / 'joint_benders'
TOL_GAP = float(os.environ.get('BENDERS_GAP', 1e-5))       # relative gap (about R$ 0.2 M)
MAX_IT = int(os.environ.get('BENDERS_MAX_IT', 60))
DELTA0 = 2.0                                                # GW, initial trust-region half-width


# ------------------------------------------------------------------ subproblem
class Month:
    """Dispatch of one month with fixed capacities as a persistent HiGHS model."""

    def __init__(self, P: 'Problem', month: int):
        self.P, self.month = P, month
        d = P.d
        self.idx = idx = np.flatnonzero(d['month'] == month)
        sn = pd.Index(idx, name='snapshot')
        n = pypsa.Network(); n.set_snapshots(sn)
        n.snapshot_weightings.loc[:, 'objective'] = d['w']
        n.add('Bus', 'MG')
        n.add('Load', 'load', bus='MG', p_set=pd.Series(d['load'][idx], sn))
        gens, stores = {}, {}

        def fixed(name, series, pat, mc=0.0):
            if np.ndim(series):
                mx = float(np.max(series))              # annual maximum, as joint_siting.py
                if mx <= 1e-9:
                    return
                prof = pd.Series(np.asarray(series)[idx] / mx, sn)
            else:
                mx = float(series)
                if mx <= 1e-9:
                    return
                prof = 1.0
            n.add('Generator', name, bus='MG', p_nom=mx, p_max_pu=prof, marginal_cost=mc); gens[name] = pat
        fixed('ex_pv', P.ex['pv'], 'ufv'); fixed('ex_mmgd', P.ex['mmgd'], 'load'); fixed('ex_thermal', P.ex['thermal'], 'ute')
        fixed('hydro', d['hydro_upper'], 'hydro')
        fixed('gas', d['gas_cap'], 'ute', d['gas_mc']); fixed('imports', d['imp_cap'], 'imp', d['imp_mc'])
        for r in P.cand.itertuples(index=False):
            if r.tech == 'bess':
                continue
            prof = d['biogas_cf'] if r.tech == 'biogas' else pd.Series(d['cf'][r.tech][idx], sn)
            n.add('Generator', r.name, bus='MG', p_nom_extendable=True, capital_cost=0.0, p_max_pu=prof,
                  marginal_cost=d['biogas_mc'] if r.tech == 'biogas' else 0.0)
            gens[r.name] = f'bus{int(r.bus_id)}'
        n.add('Generator', 'solar_distributed', bus='MG', p_nom_extendable=True, capital_cost=0.0,
              p_max_pu=pd.Series(d['cf']['solar_distributed'][idx], sn)); gens['solar_distributed'] = 'load'
        lm = float(d['load'].max())
        n.add('Generator', 'shed', bus='MG', p_nom=lm, p_max_pu=pd.Series(d['load'][idx] / lm, sn), marginal_cost=js.LP_SHED)
        gens['shed'] = 'load'
        kw = dict(bus='MG', max_hours=d['dur'], efficiency_store=d['eta'], efficiency_dispatch=d['eta'],
                  cyclic_state_of_charge=True, marginal_cost=0.001, p_nom_extendable=True, capital_cost=0.0)
        for r in P.cand[P.cand.tech == 'bess'].itertuples(index=False):
            n.add('StorageUnit', r.name, **kw); stores[r.name] = f'bus{int(r.bus_id)}'
        n.add('StorageUnit', 'bess@load', **kw); stores['bess@load'] = 'load'
        m = n.optimize.create_model(include_objective_constant=False)
        p = m.variables['Generator-p']
        hyd = p.sel(name='hydro')
        m.add_constraints(hyd.sum() <= d['energy'][month], name='hydro_energy')
        first = xr.DataArray(n.snapshots != n.snapshots[0], coords={'snapshot': n.snapshots})
        prev = hyd.roll(snapshot=1)
        m.add_constraints(hyd - prev <= d['ramp'], name='ramp_up', mask=first)
        m.add_constraints(prev - hyd <= d['ramp'], name='ramp_dn', mask=first)
        ps, pdis = m.variables['StorageUnit-p_store'], m.variables['StorageUnit-p_dispatch']
        m.add_constraints(ps + pdis - m.variables['StorageUnit-p_nom'] <= 0, name='bess_single_converter')
        m.add_objective(m.objective.expression + (d['w'] * 0.001) * ps.sum(), overwrite=True)
        self._to_highs(m, gens, stores)

    def _to_highs(self, m, gens, stores):
        from linopy import solvers
        M = m.matrices
        vl = np.asarray(M.vlabels)
        colmap = np.full(int(vl.max()) + 1, -1, dtype=np.int64); colmap[vl] = np.arange(len(vl))
        h = solvers.Highs._build_solver_model(m, set_names=False)
        h.setOptionValue('output_flag', False)
        for k, v in js.SOLVER.items():
            h.setOptionValue(k, v)
        self.h = h
        cols = lambda v: colmap[m.variables[v].labels.transpose('name', 'snapshot').values]
        self.gn, self.sn_ = list(gens), list(stores)
        nameP = list(m.variables['Generator-p'].labels.coords['name'].values)
        self.colP = cols('Generator-p')[[nameP.index(g) for g in self.gn]]
        nameS = list(m.variables['StorageUnit-p_store'].labels.coords['name'].values)
        iS = [nameS.index(x) for x in self.sn_]
        self.colPS = cols('StorageUnit-p_store')[iS]; self.colPD = cols('StorageUnit-p_dispatch')[iS]
        g = self.P.grid
        self.patg = np.array([g.col[gens[x]] for x in self.gn]); self.pats = np.array([g.col[stores[x]] for x in self.sn_])
        lab = lambda v: m.variables[v].labels
        self.cap_col = {str(k): int(colmap[int(lab('Generator-p_nom').sel(name=k))]) for k in lab('Generator-p_nom').coords['name'].values}
        self.cap_col.update({str(k): int(colmap[int(lab('StorageUnit-p_nom').sel(name=k))])
                             for k in lab('StorageUnit-p_nom').coords['name'].values})
        assert set(self.cap_col) == set(self.P.names), set(self.cap_col) ^ set(self.P.names)
        self.cap_idx = np.array([self.cap_col[k] for k in self.P.names], np.int32)
        self.shed_cols = self.colP[self.gn.index('shed')].astype(np.int32)
        self.cst = self.P.cst[:, self.idx]
        self.pairs = {}; self.n_pairs = 0

    def set_shed_price(self, price):
        self.h.changeColsCost(len(self.shed_cols), self.shed_cols, np.full(len(self.shed_cols), self.P.d['w'] * price))

    def evaluate(self, x: np.ndarray, network: bool, max_rounds=60):
        """phi(x), subgradient, unserved GWh, flow-cut rounds. x: capacities (GW) in Problem.names order."""
        import highspy
        h = self.h
        h.changeColsBounds(len(self.cap_idx), self.cap_idx, x, x)
        rounds, t0 = 0, time.time()
        while True:
            h.run()
            st = h.getModelStatus()
            assert st == highspy.HighsModelStatus.kOptimal, (self.month, h.modelStatusToString(st))
            sol = h.getSolution()
            xv = np.asarray(sol.col_value)
            if not network:
                break
            inj = self.cst.copy()
            np.add.at(inj, self.patg, xv[self.colP]); np.add.at(inj, self.pats, xv[self.colPD] - xv[self.colPS])
            assert np.abs(inj.sum(0)).max() < 1e-5
            R, T, S, nviol, worst, self.worst_pre = self.P.check(inj)
            if nviol == 0:
                break
            self.add_cuts(R, T, S); rounds += 1
            assert rounds <= max_rounds, f'month {self.month}: flow cuts did not converge'
        phi = float(h.getInfo().objective_function_value)
        g = np.asarray(sol.col_dual)[self.cap_idx]
        self.last = {'shed_gwh': float(xv[self.shed_cols].sum()), 'rounds': rounds, 'seconds': time.time() - t0,
                     'x': xv}
        return phi, g

    def add_cuts(self, R, T, S):
        g = self.P.grid
        new = []
        for r, t, sg in zip(R, T, S):
            hs = self.pairs.setdefault((int(r), int(sg)), set())
            if int(t) not in hs:
                hs.add(int(t)); new.append((int(r), int(t), int(sg)))
        if not new:
            return
        R = np.array([x[0] for x in new]); T = np.array([x[1] for x in new]); S = np.array([x[2] for x in new])
        DR = g.D[R]
        idx = np.hstack([self.colP[:, T].T, self.colPD[:, T].T, self.colPS[:, T].T])
        val = np.hstack([DR[:, self.patg], DR[:, self.pats], -DR[:, self.pats]])
        nz = np.abs(val) > 1e-12
        starts = np.r_[0, np.cumsum(nz.sum(1))[:-1]].astype(np.int32)
        const = g.f0[R] / 1000.0 + np.einsum('kp,pk->k', DR, self.cst[:, T])
        lim = g.lim[R] / 1000.0
        lower = np.where(S > 0, -np.inf, -lim - const); upper = np.where(S > 0, lim - const, np.inf)
        self.h.addRows(len(R), lower, upper, int(nz.sum()), starts, idx[nz].astype(np.int32), val[nz])
        self.n_pairs += len(new)


# ------------------------------------------------------------------ problem / master
class Problem:
    def __init__(self, year: int):
        self.year = year
        self.case = pp.refresh_cases()[0 if year == 2035 else 1]
        self.d = d = pp.lp_inputs(self.case)
        self.cand = cand = js.candidates(year)
        f = d['f']; retire = max(0, (2050 - year) / 25) if self.case.netzero_mode != 'none' else 1.
        self.ex = {'pv': f.existing_solar_central_mw.to_numpy() / 1000, 'mmgd': f.existing_mmgd_mw.to_numpy() / 1000,
                   'thermal': f.existing_biomass_observed_mw.to_numpy() / 1000
                   + f.existing_mixed_thermal_mw.to_numpy() / 1000 * retire}
        self.net = Network()
        self.grid = pcm.Grid(self.net, cand.assign(bus_id=cand.bus_id.astype(int)), bess_at_load=True)
        self.cst = pcm.constants(self.grid, d, self.ex, pcm.references(), np.arange(8760))
        self.names = list(cand.name) + ['solar_distributed', 'bess@load']
        self.tech = {k: (k.split('@')[0] if '@' in k else k) for k in self.names}
        self.cost_site = np.array([d['capcoef'][self.tech[k]] + (js.extra_coef(self.tech[k], float(cand.set_index('name').at[k, 'extra_conn_brl_kw']), d, year)
                                   if k in set(cand.name) else 0.0) for k in self.names])
        self.cost_base = np.array([d['capcoef'][self.tech[k]] for k in self.names])
        # reference capacities of the sequential method (joint_siting.Joint.reference_caps)
        cap = js.rh.portfolio_mw(year); sites, _ = js.rh.reference_siting(year, cap)
        ref = {k: 0.0 for k in self.names}
        for r in sites.itertuples(index=False):
            ref[f'{r.tech}@{r.area}@{int(r.bus_id)}'] += r.new_capacity_mw / 1000
        ref['solar_distributed'] = cap.get('solar_distributed', 0.0) / 1000
        self.ref = np.array([ref[k] for k in self.names])
        self.lp_total = {t: cap.get(t, 0.0) / 1000 for t in pp.TECH}
        self.months = None

    def check(self, inj):
        """All rows, every hour of `inj` (patterns x hours). Same rule as joint_siting.Joint.check."""
        return js.Joint.check(self, inj)

    def build_months(self):
        t0 = time.time()
        self.months = [Month(self, m) for m in range(1, 13)]
        print(f'  {self.year}: 12 monthly models built in {time.time() - t0:.0f} s', flush=True)

    def evaluate(self, x, network):
        tot, G, info = 0.0, [], []
        phis = []
        for M in self.months:
            phi, g = M.evaluate(x, network)
            phis.append(phi); G.append(g); info.append(M.last)
        return np.array(phis), np.array(G), info


class Master:
    """min cost.x + TIE.dev + sum theta_m; floors/ceilings, area caps, tie-break, cuts, box trust region."""

    def __init__(self, P: Problem, cost, area_caps=True, fix_totals=False):
        import highspy
        self.P = P; d = P.d
        nx, names = len(P.names), P.names
        groups = [(f'{t}|{a}', [list(names).index(n) for n in g.name]) for (t, a), g in P.cand.groupby(['tech', 'area'])]
        groups += [('bess|load', [names.index('bess@load')])]
        self.ng = len(groups)
        h = highspy.Highs(); h.setOptionValue('output_flag', False)
        nv = nx + self.ng + 12
        self.ix, self.idev, self.ith = np.arange(nx), nx + np.arange(self.ng), nx + self.ng + np.arange(12)
        lb = np.zeros(nv); ub = np.full(nv, np.inf)
        dg = names.index('solar_distributed')
        lb[dg] = d['floor']['solar_distributed']; ub[dg] = d['ceil']['solar_distributed']
        h.addVars(nv, lb, ub)
        c = np.r_[cost, np.full(self.ng, js.TIE), np.ones(12)]
        h.changeColsCost(nv, np.arange(nv, dtype=np.int32), c)
        self.lb0, self.ub0 = lb.copy(), ub.copy()

        def row(cols, vals, lo, hi):
            h.addRow(lo, hi, len(cols), np.asarray(cols, np.int32), np.asarray(vals, float))
        for t in js.SITED + ['bess']:
            cols = [i for i, n in enumerate(names) if P.tech[n] == t]
            if not cols:
                continue
            if fix_totals:
                v = max(0.0, P.lp_total.get(t, 0.0)); row(cols, np.ones(len(cols)), v, v)
            else:
                lo = d['floor'][t]; hi = d['ceil'][t]
                if lo > 0 or np.isfinite(hi):
                    row(cols, np.ones(len(cols)), lo, hi if np.isfinite(hi) else np.inf)
        if fix_totals:
            v = P.lp_total['solar_distributed']; h.changeColBounds(dg, v, v); self.lb0[dg] = self.ub0[dg] = v
        if area_caps:
            for (t, a), g in P.cand[np.isfinite(P.cand.area_cap_mw)].groupby(['tech', 'area']):
                cols = [names.index(n) for n in g.name]
                row(cols, np.ones(len(cols)), -np.inf, float(g.area_cap_mw.iloc[0]) / 1000)
        for j, (_, cols) in enumerate(groups):       # dev_j >= |sum x - ref|
            r0 = float(P.ref[cols].sum())
            row(list(cols) + [self.idev[j]], np.r_[-np.ones(len(cols)), 1.0], -r0, np.inf)
            row(list(cols) + [self.idev[j]], np.r_[np.ones(len(cols)), 1.0], r0, np.inf)
        self.h = h; self.ncuts = 0

    def add_cut(self, m, phi, g, x):
        cols = np.r_[self.ith[m], self.ix]; vals = np.r_[1.0, -g]
        self.h.addRow(phi - float(g @ x), np.inf, len(cols), cols.astype(np.int32), vals)
        self.ncuts += 1

    def solve(self, center=None, delta=None):
        import highspy
        h = self.h
        lb, ub = self.lb0.copy(), self.ub0.copy()
        if center is not None:
            lb[self.ix] = np.maximum(lb[self.ix], center - delta); ub[self.ix] = np.minimum(ub[self.ix], center + delta)
        h.changeColsBounds(len(lb), np.arange(len(lb), dtype=np.int32), lb, ub)
        h.run()
        st = h.getModelStatus()
        assert st == highspy.HighsModelStatus.kOptimal, h.modelStatusToString(st)
        v = np.asarray(h.getSolution().col_value)
        return float(h.getInfo().objective_function_value), v[self.ix].copy(), v


def benders(P: Problem, label, network=True, area_caps=True, fix_totals=False, shed='lp', site_cost=True,
            seeds=(), max_it=MAX_IT, log=print):
    cost = P.cost_site if site_cost else P.cost_base
    for M in P.months:
        M.set_shed_price(js.SHED_PRICE[shed])
    Ms = Master(P, cost, area_caps=area_caps, fix_totals=fix_totals)
    devs = lambda x: sum(abs(float(x[c].sum()) - float(P.ref[c].sum())) for c in _groups(P))
    hist, best = [], None
    queue = [np.asarray(s, float) for s in seeds]
    delta = DELTA0; t0 = time.time()
    for it in range(max_it):
        if queue:
            x = queue.pop(0)
        else:
            _, x, _ = Ms.solve(best['x'], delta)
        x = np.maximum(x, 0.0)
        phis, G, info = P.evaluate(x, network)
        for m in range(12):
            Ms.add_cut(m, phis[m], G[m], x)
        ub = float(cost @ x) + js.TIE * devs(x) + float(phis.sum())
        lb, _, _ = Ms.solve()
        serious = best is None or ub < best['ub'] - 1e-9
        if serious:
            best = {'x': x, 'ub': ub, 'phis': phis, 'info': info, 'it': it}
            delta = min(4 * DELTA0, delta * 1.5) if it > len(seeds) else delta
        else:
            delta = max(0.02, delta * 0.5)
        gap = (best['ub'] - lb) / abs(best['ub'])
        shed = sum(i['shed_gwh'] for i in info)
        rec = {'case': label, 'iteration': it, 'ub_iterate_mbrl': ub, 'best_ub_mbrl': best['ub'], 'lb_mbrl': lb, 'gap': gap,
               'delta_gw': delta, 'serious': serious, 'shed_gwh': shed, 'cut_pairs': sum(M.n_pairs for M in P.months),
               'flow_rounds': sum(i['rounds'] for i in info), 'seconds': time.time() - t0}
        hist.append(rec)
        log(f'  {label} it {it}: iterate {ub:.3f}, best {best["ub"]:.3f}, LB {lb:.3f}, gap {gap:.2e}, '
            f'shed {shed:.2f} GWh, delta {delta:.3g}, pairs {rec["cut_pairs"]}, {rec["seconds"]:.0f} s', flush=True)
        if gap <= TOL_GAP and not queue:
            break
    return best, pd.DataFrame(hist)


def _groups(P):
    names = P.names
    g = [[names.index(n) for n in gg.name] for _, gg in P.cand.groupby(['tech', 'area'])]
    return g + [[names.index('bess@load')]]


def report(P, label, best, hist, extra=None):
    x = best['x']; d = P.d
    sites = P.cand[['tech', 'area', 'rank', 'bus_id', 'station', 'kv', 'extra_conn_brl_kw', 'area_cap_mw', 'name']].copy()
    sites = pd.concat([sites, pd.DataFrame([{'tech': 'solar_distributed', 'area': 'MG load buses', 'rank': -1, 'bus_id': -1,
                                             'station': 'load buses pro rata', 'kv': -1, 'extra_conn_brl_kw': 0.0,
                                             'area_cap_mw': np.inf, 'name': 'solar_distributed'},
                                            {'tech': 'bess', 'area': 'MG load buses', 'rank': -1, 'bus_id': -1,
                                             'station': 'load buses pro rata', 'kv': -1, 'extra_conn_brl_kw': 0.0,
                                             'area_cap_mw': np.inf, 'name': 'bess@load'}])], ignore_index=True)
    sites['capacity_mw'] = [x[P.names.index(n)] * 1000 for n in sites.name]
    sites.insert(0, 'case', label)
    tot = sites.groupby('tech').capacity_mw.sum() / 1000
    shed = sum(i['shed_gwh'] for i in best['info'])
    w = d['w']
    pen = w * js.LP_SHED * shed
    out = {'case': label, 'year': P.year, 'objective_mbrl': best['ub'], 'lower_bound_mbrl': float(hist.lb_mbrl.iloc[-1]),
           'gap': float(hist.gap.iloc[-1]), 'iterations': int(len(hist)), 'best_iteration': int(best['it']),
           'shed_gwh': shed, 'objective_voll_valued_mbrl': best['ub'] - pen + w * js.VOLL * shed if extra is None or extra.get('shed', 'lp') == 'lp' else best['ub'],
           'cut_pairs': int(hist.cut_pairs.iloc[-1]), 'seconds': float(hist.seconds.iloc[-1]),
           **{f'cap_{t}_gw': float(tot.get(t, 0.0)) for t in pp.TECH}, 'bess_at_load_gw': float(x[P.names.index('bess@load')]),
           **(extra or {})}
    return out, sites


CASES = {  # label: (network, area caps, fix statewide totals at the LP mix, unserved-energy price, site cost, seeds)
    'copperplate': (False, True, False, 'lp', True, ('ref',)),
    'sequential': (True, True, True, 'lp', True, ('ref',)),          # one evaluation of the reference siting
    'siting': (True, True, True, 'lp', True, ('ref', 'ref_bess_load')),
    'joint': (True, True, False, 'lp', True, ('ref', 'ref_bess_load')),
    'joint_uncapped': (True, False, False, 'lp', True, ('ref', 'ref_bess_load')),
    'joint_voll': (True, True, False, 'voll', True, ('ref', 'ref_bess_load')),
}


def seeds_of(P, keys):
    out = []
    for k in keys:
        x = P.ref.copy()
        if k == 'ref_bess_load':
            b = [i for i, n in enumerate(P.names) if P.tech[n] == 'bess']
            tot = float(x[b].sum()); x[b] = 0.0; x[P.names.index('bess@load')] = tot
        out.append(x)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--year', type=int, required=True)
    ap.add_argument('--cases', nargs='+', default=['copperplate', 'sequential', 'siting', 'joint'])
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    P = Problem(a.year); P.build_months()
    for c in a.cases:
        network, caps, fixt, shed, site, seeds = CASES[c]
        label = f'{a.year}_{c}'
        print(f'CASE {label}', flush=True)
        best, hist = benders(P, label, network=network, area_caps=caps, fix_totals=fixt, shed=shed, site_cost=site,
                             seeds=seeds_of(P, seeds), max_it=1 if c == 'sequential' else MAX_IT)
        out, sites = report(P, label, best, hist, extra={'network': network, 'area_caps': caps, 'fix_totals': fixt, 'shed': shed})
        pd.DataFrame([out]).to_csv(OUT / f'{label}_summary.csv', index=False, float_format='%.6f')
        sites.to_csv(OUT / f'{label}_sites.csv', index=False, float_format='%.6f')
        hist.to_csv(OUT / f'{label}_iterations.csv', index=False, float_format='%.6f')
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}, flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
