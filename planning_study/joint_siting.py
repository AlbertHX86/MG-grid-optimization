#!/usr/bin/env python3
"""Phase 6: joint capacity and siting optimisation on the full MG network (PTDF, N-0/N-1, 8,760 h).

One LP per target year chooses how much of each technology to build and at which candidate bus,
together with the hourly dispatch, subject to DC flow limits on the main grid of the full 1,489-bus PWF
network: 555 monitored elements, N-0 at normal ratings and 437 single contingencies at emergency
ratings (242,708 rows), in every hour. Flow limits are generated lazily per (row, hour): solve, check
every row in every hour, add the violated (row, hour) pairs, re-solve, until no row is violated. The final
solution is optimal for a relaxation of the full problem and feasible for it, hence optimal for the full
problem; no constraint subset is needed.

Cost basis and every other constraint are the one-bus portfolio LP's (portfolio_pypsa.lp_inputs): policy
floors and ceilings become statewide sums over sites; the monthly hydro budgets, hydro ramp, gas/import
bounds, BESS single converter, cyclic storage and tie-break are unchanged. With no network, no shedding
and no site cost, the model reproduces the one-bus LP exactly (gate).

Siting (study assumptions):
* candidate buses are the three nearest distinct eligible substations of every (technology, area) of the
  allocation file (optimize_siting.candidate_table, same voltage eligibility); BESS may also sit at MG load
  buses pro rata to load. 2035 has no BESS or wind rows in the allocation file, so the 2050 BESS and wind
  areas are used; distributed PV injects at load buses pro rata to load, as before;
* each area's capacity of a technology is at most ALPHA x its reference allocation (ALPHA = 1.5, the
  siting MILP's assumption); BESS has no area cap; the 'uncapped' case drops the area caps;
* the LP's generic per-kW interconnection add-on pays for the area's cheapest candidate station; another
  station adds its extra cost per MW (route length x line cost and bays per linearised circuit, plus
  R$ 0.08 M/MW transformation), passed through the LP's capital-coefficient formula;
* every area has the same hourly profile per technology, so without binding flows the split across areas
  is not unique; a tie-break of TIE M BRL per GW of deviation from the reference allocation (the sequential
  method's siting) keeps each area at its reference unless the network pays for a move.
Network injections follow pcm.py: changes relative to the PWF snapshot (2035 LP hour 2349) are injected on
their bus patterns; pre-existing overloads of the base case are reported, not constrained.
Unserved energy is priced at the portfolio LP's own penalty (R$ 1,000,000/MWh), i.e. the main cases keep the LP's
rule of serving all load; any unserved energy left is unavoidable on the existing grid and is valued ex post at VOLL
(R$ 8,327.76/MWh, CCEE). Pricing it at VOLL inside the expansion changes the portfolio even without a network
(2050: 297 GWh unserved, 2.2 GW less PV, 1.5 GW less BESS), so that is a labelled sensitivity ('network_voll').
"""
from __future__ import annotations
import argparse
import json
import logging
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
ROOT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / 'hosting_capacity'))
import portfolio_pypsa as pp  # noqa: E402
import run_hosting as rh  # noqa: E402
import pcm  # noqa: E402
from hc_engine import Network  # noqa: E402

logging.getLogger('pypsa').setLevel(logging.ERROR); logging.getLogger('linopy').setLevel(logging.ERROR)
OUT = HERE / 'results'
PARTS = OUT / 'joint_parts'           # hourly dispatch (not committed)
ALPHA = 1.5
TIE = 0.1                             # M BRL per GW of area deviation from the reference allocation (tie-break)
VOLL = pcm.VOLL / 1000.0              # M BRL per GWh (ex-post valuation and the 'voll' sensitivity)
LP_SHED = 1000.0                      # M BRL per GWh: the portfolio LP's own penalty (effectively no unserved energy)
SHED_PRICE = {'lp': LP_SHED, 'voll': VOLL}
TOL = 1e-3                            # MW, violation tolerance (as pcm.py)
NEAR = 0.97                           # a violated row enters for every hour loaded >= 97% of its limit
CHUNK = 4000                          # rows per flow-check block
# Dual simplex warm-starts every re-solve after cuts or case switches. (Interior point, JOINT_SOLVER=ipm, was
# tested and is far slower on this model.)
if os.environ.get('JOINT_SOLVER', 'simplex') == 'ipm':
    SOLVER = {'solver': 'ipm', 'run_crossover': os.environ.get('JOINT_CROSSOVER', 'off'), 'ipm_optimality_tolerance': 1e-8,
              'primal_feasibility_tolerance': 1e-7, 'dual_feasibility_tolerance': 1e-7}
else:
    SOLVER = {'solver': 'simplex', 'simplex_strategy': 1,
              'primal_feasibility_tolerance': 1e-7, 'dual_feasibility_tolerance': 1e-7}
SOLVER['time_limit'] = float(os.environ.get('JOINT_TIME_LIMIT', 4 * 3600))    # per solve, seconds
MAX_ROWS = 150                        # violated rows added per round (largest violation first), as pcm.py
MAX_PAIRS = 200_000                   # and at most this many (row, hour) pairs per round (memory)
SITED = ['solar_central', 'bio', 'small_hydro', 'wind_site', 'biogas', 'wind']


# ------------------------------------------------------------------ candidates
def candidates(year: int) -> pd.DataFrame:
    """Candidate (technology, area, bus) rows with area caps (MW) and extra connection cost (R$/kW)."""
    _, c = rh.reference_siting(year, rh.portfolio_mw(year))
    _, c50 = rh.reference_siting(2050, rh.portfolio_mw(2050))
    if year == 2035:                  # no 2035 BESS / wind rows in the allocation file
        c = pd.concat([c, c50[c50.tech.isin(['bess', 'wind_site'])]], ignore_index=True)
    w = c50[c50.tech == 'wind_site'].copy(); w['tech'] = 'wind'    # generic wind on the wind areas (LP: never built)
    c = pd.concat([c, w], ignore_index=True)
    conn = c.per_circuit_mbrl / c.max_circuit_mw + c.variable_cost_mbrl_mw          # M BRL per MW
    cheapest = conn.groupby([c.tech, c.area]).transform('min')                      # the area's cheapest station
    c['extra_conn_brl_kw'] = (conn - cheapest) * 1000.0
    c['area_cap_mw'] = np.where(c.tech.isin(['bess', 'wind']), np.inf, c.max_area_mw)
    c['name'] = c.tech + '@' + c.area + '@' + c.bus_id.astype(str)
    assert c.name.is_unique
    return c.reset_index(drop=True)


def extra_coef(t: str, extra_brl_kw: float, d: dict, year: int) -> float:
    """The LP's capital coefficient (portfolio_pypsa.lp_inputs) of an extra capex per kW, no O&M."""
    p = d['p']; tp = p['technology'][t]; rate = p['discount_rate']; start = year - 4
    life = tp['life_years']; used = min(life, 2051 - start); remaining = max(0, life - used) / life
    capital = extra_brl_kw * (1 + tp['construction_interest_fraction'])
    return capital / (1 + rate) ** (start - 2026) - capital * remaining / (1 + rate) ** 25


# ------------------------------------------------------------------ model
class Joint:
    """One model per year holding every case: unserved-energy tranches, site costs and area caps are all
    built in; a case only changes column bounds, column costs or row bounds of the persistent HiGHS model
    (set_case), so every case after the first warm-starts from the previous basis. Flow cuts are valid
    constraints of the full problem in every network case, so they accumulate across those cases."""

    def __init__(self, year: int):
        self.year = year
        self.network = self.fixed = False; self.shed = self.site_cost = self.area_caps = True
        self.case = pp.refresh_cases()[0 if year == 2035 else 1]
        self.d = d = pp.lp_inputs(self.case)
        self.cand = cand = candidates(year)
        f = d['f']; retire = max(0, (2050 - year) / 25) if self.case.netzero_mode != 'none' else 1.
        self.ex = {'pv': f.existing_solar_central_mw.to_numpy() / 1000, 'mmgd': f.existing_mmgd_mw.to_numpy() / 1000,
                   'thermal': f.existing_biomass_observed_mw.to_numpy() / 1000
                   + f.existing_mixed_thermal_mw.to_numpy() / 1000 * retire}
        assert np.allclose(self.ex['pv'] + self.ex['mmgd'] + self.ex['thermal'], d['vrebase'])
        self.net = Network()
        self.grid = pcm.Grid(self.net, cand.assign(bus_id=cand.bus_id.astype(int)), bess_at_load=True)
        self.ref = pcm.references()
        self.cst = pcm.constants(self.grid, d, self.ex, self.ref, np.arange(8760))
        self.pairs = {}               # (row, sign) -> set of hours already constrained
        self.n_pairs = 0
        self.cut_idx, self.cut_lo, self.cut_hi = [], [], []     # HiGHS rows of the flow cuts and their bounds
        # Stations open at the start: the nearest and the cheapest of every (technology, area), and BESS at load.
        # The others are priced after convergence and opened if their reduced cost is negative (column generation).
        first = cand.groupby(['tech', 'area']).head(1).name
        cheap = cand.loc[cand.extra_conn_brl_kw <= 1e-9].name
        self.open = set(first) | set(cheap) | {'bess@load'}
        if os.environ.get('JOINT_OPEN_ALL'):
            self.open |= set(cand.name)
        self._build()

    def _build(self):
        d, cand, year = self.d, self.cand, self.year
        sn = pd.RangeIndex(8760, name='snapshot')
        n = pypsa.Network(); n.set_snapshots(sn)
        n.snapshot_weightings.loc[:, 'objective'] = d['w']
        n.add('Bus', 'MG')
        n.add('Load', 'load', bus='MG', p_set=pd.Series(d['load'], sn))
        gens, stores = {}, {}

        def fixed(name, series, pat, mc=0.0):
            mx = float(np.max(series)) if np.ndim(series) else float(series)
            if mx <= 1e-9:
                return
            prof = pd.Series(np.asarray(series) / mx, sn) if np.ndim(series) else 1.0
            n.add('Generator', name, bus='MG', p_nom=mx, p_max_pu=prof, marginal_cost=mc); gens[name] = pat
        fixed('ex_pv', self.ex['pv'], 'ufv'); fixed('ex_mmgd', self.ex['mmgd'], 'load')
        fixed('ex_thermal', self.ex['thermal'], 'ute')
        fixed('hydro', d['hydro_upper'], 'hydro')
        fixed('gas', d['gas_cap'], 'ute', d['gas_mc']); fixed('imports', d['imp_cap'], 'imp', d['imp_mc'])
        self.cost_site, self.cost_base = {}, {}
        for r in cand.itertuples(index=False):
            self.cost_base[r.name] = d['capcoef'][r.tech]
            self.cost_site[r.name] = d['capcoef'][r.tech] + extra_coef(r.tech, r.extra_conn_brl_kw, d, year)
        self.cap_cost = self.cost_site
        for r in cand.itertuples(index=False):
            if r.tech == 'bess':
                continue
            prof = d['biogas_cf'] if r.tech == 'biogas' else pd.Series(d['cf'][r.tech], sn)
            mc = d['biogas_mc'] if r.tech == 'biogas' else 0.0
            n.add('Generator', r.name, bus='MG', p_nom_extendable=True, capital_cost=self.cap_cost[r.name],
                  p_max_pu=prof, marginal_cost=mc)
            gens[r.name] = f'bus{int(r.bus_id)}'
        n.add('Generator', 'solar_distributed', bus='MG', p_nom_extendable=True, p_nom_min=d['floor']['solar_distributed'],
              p_nom_max=d['ceil']['solar_distributed'], capital_cost=d['capcoef']['solar_distributed'],
              p_max_pu=pd.Series(d['cf']['solar_distributed'], sn))
        gens['solar_distributed'] = 'load'
        # unserved energy at VOLL, one column per hour (switched off by column bounds in the gate case)
        lm = float(d['load'].max())
        n.add('Generator', 'shed', bus='MG', p_nom=lm, p_max_pu=pd.Series(d['load'] / lm, sn), marginal_cost=VOLL)
        gens['shed'] = 'load'
        kw = dict(bus='MG', max_hours=d['dur'], efficiency_store=d['eta'], efficiency_dispatch=d['eta'],
                  cyclic_state_of_charge=True, marginal_cost=0.001, p_nom_extendable=True)
        for r in cand[cand.tech == 'bess'].itertuples(index=False):
            n.add('StorageUnit', r.name, capital_cost=self.cap_cost[r.name], **kw); stores[r.name] = f'bus{int(r.bus_id)}'
        n.add('StorageUnit', 'bess@load', capital_cost=d['capcoef']['bess'], **kw); stores['bess@load'] = 'load'
        self.cost_site['bess@load'] = self.cost_base['bess@load'] = d['capcoef']['bess']
        self.n, self.gens, self.stores = n, gens, stores
        m = n.optimize.create_model()
        p = m.variables['Generator-p']
        hyd = p.sel(name='hydro')
        mon = pd.Series(d['month'], index=n.snapshots)
        for mm in range(1, 13):
            m.add_constraints(hyd.sel(snapshot=mon.index[mon.values == mm]).sum() <= d['energy'][mm], name=f'hydro_energy_m{mm}')
        prev = hyd.roll(snapshot=1)
        m.add_constraints(hyd - prev <= d['ramp'], name='hydro_ramp_up')
        m.add_constraints(prev - hyd <= d['ramp'], name='hydro_ramp_dn')
        ps, pdis = m.variables['StorageUnit-p_store'], m.variables['StorageUnit-p_dispatch']
        m.add_constraints(ps + pdis - m.variables['StorageUnit-p_nom'] <= 0, name='bess_single_converter')
        m.add_objective(m.objective.expression + (d['w'] * 0.001) * ps.sum(), overwrite=True)
        gp = m.variables['Generator-p_nom']; sp = m.variables['StorageUnit-p_nom']
        for t in SITED + ['bess']:
            names = list(cand.name[cand.tech == t]) + (['bess@load'] if t == 'bess' else [])
            if not names:
                continue
            tot = (sp if t == 'bess' else gp).sel(name=names).sum()
            if d['floor'][t] > 0:
                m.add_constraints(tot >= d['floor'][t], name=f'floor_{t}')
            if np.isfinite(d['ceil'][t]):
                m.add_constraints(tot <= d['ceil'][t], name=f'ceil_{t}')
        self.area_cons = []           # (constraint name, cap GW)
        for (t, a), g in cand[np.isfinite(cand.area_cap_mw)].groupby(['tech', 'area']):
            nm = f'area_{t}_{a}'.replace(' ', '_'); cap = float(g.area_cap_mw.iloc[0]) / 1000
            m.add_constraints(gp.sel(name=list(g.name)).sum() <= cap, name=nm); self.area_cons.append((nm, cap))
        # tie-break: deviation of each (technology, area) total from the reference allocation
        ref = self.reference_caps()
        groups = [(f'{t}|{a}', list(g.name)) for (t, a), g in cand.groupby(['tech', 'area'])] + [('bess|load', ['bess@load'])]
        keys = [k for k, _ in groups]
        dev = m.add_variables(lower=0, coords=[pd.Index(keys, name='area_group')], name='area_deviation')
        for k, names in groups:
            v = (sp if k.startswith('bess|') else gp).sel(name=names).sum()
            r0 = sum(ref.get(x, 0.0) for x in names)
            m.add_constraints(dev.sel(area_group=k) - v >= -r0, name=f'dev_hi_{k}'.replace(' ', '_'))
            m.add_constraints(dev.sel(area_group=k) + v >= r0, name=f'dev_lo_{k}'.replace(' ', '_'))
        m.add_objective(m.objective.expression + TIE * dev.sum(), overwrite=True)
        self.tie_names = keys
        self.ref_caps = ref
        self.m = m

    def reference_caps(self) -> dict:
        """GW per component of the sequential method: the one-bus LP's statewide mix (REFRESH) placed
        at the reference siting (nearest eligible station at the allocation weights)."""
        cap = rh.portfolio_mw(self.year)
        sites, _ = rh.reference_siting(self.year, cap)
        ref = {}
        for r in sites.itertuples(index=False):
            k = f'{r.tech}@{r.area}@{int(r.bus_id)}'
            assert k in set(self.cand.name), k
            ref[k] = ref.get(k, 0.0) + r.new_capacity_mw / 1000
        ref['solar_distributed'] = cap.get('solar_distributed', 0.0) / 1000
        return ref

    # -------------------------------------------------------------- solve / check
    def _highs(self):
        """Persistent HiGHS model mirroring the linopy model; cut rows are added to it directly, so
        every re-solve warm-starts from the previous basis (dual simplex)."""
        import highspy
        from linopy import solvers
        m = self.m
        M = m.matrices
        vl = np.asarray(M.vlabels)
        self.colmap = np.full(int(vl.max()) + 1, -1, dtype=np.int64); self.colmap[vl] = np.arange(len(vl))
        self.obj_const = float(np.asarray(m.objective.expression.const).sum()) if hasattr(m.objective.expression, 'const') else 0.0
        h = solvers.Highs._build_solver_model(m, set_names=False)
        PARTS.mkdir(parents=True, exist_ok=True)
        h.setOptionValue('output_flag', True); h.setOptionValue('log_to_console', False)
        h.setOptionValue('log_file', str(PARTS / f'highs_{self.year}.log'))
        for k, v in SOLVER.items():
            h.setOptionValue(k, v)
        self.h = h
        rl = np.asarray(M.clabels)
        rowmap = np.full(int(rl.max()) + 1, -1, dtype=np.int64); rowmap[rl] = np.arange(len(rl))
        self.area_rows = np.array([rowmap[int(m.constraints[nm].labels.values)] for nm, _ in self.area_cons], dtype=np.int32)
        self.area_ub = np.array([c for _, c in self.area_cons])
        lab = lambda v: m.variables[v].labels
        self.pnom_cols = {str(k): int(self.colmap[int(lab('Generator-p_nom').sel(name=k))]) for k in lab('Generator-p_nom').coords['name'].values}
        self.pnom_cols.update({str(k): int(self.colmap[int(lab('StorageUnit-p_nom').sel(name=k))])
                               for k in lab('StorageUnit-p_nom').coords['name'].values})
        lp = h.getLp(); cost0 = np.asarray(lp.col_cost_)
        for k, c in self.pnom_cols.items():           # the HiGHS cost of a capacity column is its capital cost
            if k in self.cost_site:
                assert abs(cost0[c] - self.cost_site[k]) < 1e-6 * max(1, abs(cost0[c])), (k, cost0[c], self.cost_site[k])
        self.pnom_lb0 = {k: float(lp.col_lower_[c]) for k, c in self.pnom_cols.items()}
        self.pnom_ub0 = {k: float(lp.col_upper_[c]) for k, c in self.pnom_cols.items()}
        cols = lambda v: self.colmap[m.variables[v].labels.transpose('name', 'snapshot').values]
        self.gn, self.sn_ = list(self.gens), list(self.stores)
        self.colP = cols('Generator-p'); self.nameP = list(m.variables['Generator-p'].labels.coords['name'].values)
        iP = [self.nameP.index(g) for g in self.gn]; self.colP = self.colP[iP]
        nameS = list(m.variables['StorageUnit-p_store'].labels.coords['name'].values)
        iS = [nameS.index(x) for x in self.sn_]
        self.colPS = cols('StorageUnit-p_store')[iS]; self.colPD = cols('StorageUnit-p_dispatch')[iS]
        self.patg = np.array([self.grid.col[self.gens[g]] for g in self.gn])
        self.pats = np.array([self.grid.col[self.stores[x]] for x in self.sn_])
        shed = [i for i, g in enumerate(self.gn) if g == 'shed']
        self.shed_cols = self.colP[shed].ravel().astype(np.int32)
        self.shed_ub = np.asarray(lp.col_upper_)[self.shed_cols]
        # keep only the labels read back later, then free the PyPSA/linopy copy of the problem (memory)
        self.labels = {v: m.variables[v].labels for v in ('Generator-p_nom', 'StorageUnit-p_nom', 'area_deviation')}
        self.m = None; self.n = None
        import gc; gc.collect()

    def set_case(self, case: str):
        network, shed, site_cost, area_caps, fixed = CASES[case]
        if not hasattr(self, 'h'):
            self._highs()
        h = self.h
        z = np.zeros(len(self.shed_cols))
        h.changeColsBounds(len(self.shed_cols), self.shed_cols, z, self.shed_ub if shed else z)
        if shed:                          # objective weight x price, as PyPSA writes marginal costs
            h.changeColsCost(len(self.shed_cols), self.shed_cols, np.full(len(self.shed_cols), self.d['w'] * SHED_PRICE[shed]))
        cost = self.cost_site if site_cost else self.cost_base
        ks = [k for k in self.pnom_cols if k in cost]
        h.changeColsCost(len(ks), np.array([self.pnom_cols[k] for k in ks], np.int32), np.array([cost[k] for k in ks]))
        nr = len(self.area_rows)
        h.changeRowsBounds(nr, self.area_rows, np.full(nr, -np.inf), self.area_ub if area_caps else np.full(nr, np.inf))
        self.network, self.shed, self.site_cost, self.area_caps, self.fixed = network, shed, site_cost, area_caps, fixed
        self.cap_cost = cost
        self.apply_capacity_bounds()
        if self.cut_idx:                  # flow cuts are active only in network cases
            k = len(self.cut_idx); ix = np.array(self.cut_idx, np.int32)
            lo = np.array(self.cut_lo) if network else np.full(k, -np.inf)
            hi = np.array(self.cut_hi) if network else np.full(k, np.inf)
            h.changeRowsBounds(k, ix, lo, hi)

    def apply_capacity_bounds(self):
        ks = list(self.pnom_cols)
        closed = lambda k: k in self.cost_site and k not in self.open
        lb = np.array([self.ref_caps.get(k, 0.0) if self.fixed else self.pnom_lb0[k] for k in ks])
        ub = np.array([self.ref_caps.get(k, 0.0) if self.fixed else (0.0 if closed(k) else self.pnom_ub0[k]) for k in ks])
        self.h.changeColsBounds(len(ks), np.array([self.pnom_cols[k] for k in ks], np.int32), lb, ub)

    def price_closed(self, tol=1e-6):
        """Reduced costs of the closed stations' capacity columns at the current optimum. If none is negative
        (and HiGHS reports no dual infeasibility), the solution is optimal over all candidate stations."""
        info = self.h.getInfo()
        rc = np.asarray(self.h.getSolution().col_dual)
        closed = [k for k in self.cost_site if k not in self.open]
        neg = [k for k in closed if rc[self.pnom_cols[k]] < -tol * max(1.0, abs(self.cap_cost[k]))]
        return neg, int(info.num_dual_infeasibilities), (min(rc[self.pnom_cols[k]] for k in closed) if closed else 0.0)

    def solve(self):
        import highspy
        if not hasattr(self, 'h'):
            self._highs()
        bas = PARTS / f'basis_{self.year}_cold.bas'
        first = not getattr(self, '_solved', False)
        if first and bas.exists() and self.n_pairs == 0 and SOLVER['solver'] == 'simplex':
            import highspy
            try:
                st = self.h.readBasis(str(bas))
                ok = st == highspy.HighsStatus.kOk and self.h.getBasis().valid
            except Exception:
                ok = False
            print(f'  basis checkpoint {bas.name}: {"warm start" if ok else "not used (solving cold)"}', flush=True)
            if not ok:
                self.h.setBasis()       # discard any partially read basis
        t0 = time.time()
        self.h.run()
        if first:
            if SOLVER['solver'] == 'simplex':
                self.h.writeBasis(str(bas))
            self._solved = True
        st = self.h.getModelStatus()
        assert st == highspy.HighsModelStatus.kOptimal, self.h.modelStatusToString(st)
        self.x = np.asarray(self.h.getSolution().col_value)
        self.obj = float(self.h.getInfo().objective_function_value) + self.obj_const
        return time.time() - t0

    def value(self, var):
        lab = self.labels[var]
        return xr.DataArray(self.x[self.colmap[lab.values]], coords=lab.coords, dims=lab.dims)

    def solution(self):
        P = xr.DataArray(self.x[self.colP], dims=('name', 'snapshot'), coords={'name': self.gn})
        PS = xr.DataArray(self.x[self.colPS], dims=('name', 'snapshot'), coords={'name': self.sn_})
        PD = xr.DataArray(self.x[self.colPD], dims=('name', 'snapshot'), coords={'name': self.sn_})
        inj = self.cst.copy()
        np.add.at(inj, self.patg, P.values)
        np.add.at(inj, self.pats, PD.values - PS.values)
        assert np.abs(inj.sum(0)).max() < 1e-5, 'pattern injections must balance'
        return P, PS, PD, inj

    def check(self, inj):
        """All rows, all hours. Returns violated (row, hour, sign) arrays, the near-binding hours of the
        violated rows, max violation (MW) and the worst increase of a pre-existing overload (MW)."""
        g = self.grid
        found, worst, worst_pre, nviol = [], 0.0, 0.0, 0
        for a in range(0, len(g.f0), CHUNK):
            b = min(len(g.f0), a + CHUNK)
            fl = g.f0[a:b, None] + (g.D[a:b] @ inj) * 1000.0
            lim = g.lim[a:b, None]
            pre = g.pre[a:b]
            if pre.any():
                worst_pre = max(worst_pre, float((np.abs(fl[pre]).max(1) - np.abs(g.f0[a:b][pre])).max()))
            over = (np.abs(fl) - lim) > TOL
            over[pre] = False
            rows_v = np.flatnonzero(over.any(1))
            if not len(rows_v):
                continue
            nviol += int(over.sum())
            for q in rows_v:
                f = fl[q]; v = float((np.abs(f) - lim[q, 0]).max()); worst = max(worst, v)
                tt = np.flatnonzero(np.abs(f) >= NEAR * lim[q, 0])
                found.append((v, a + int(q), tt, np.sign(f[tt])))
        found.sort(key=lambda x: -x[0])
        keep, npairs = [], 0
        for x in found[:MAX_ROWS]:
            if keep and npairs + len(x[2]) > MAX_PAIRS:
                break
            keep.append(x); npairs += len(x[2])
        found = keep
        R = np.concatenate([np.full(len(x[2]), x[1]) for x in found]) if found else np.zeros(0, int)
        T = np.concatenate([x[2] for x in found]) if found else np.zeros(0, int)
        S = np.concatenate([x[3] for x in found]) if found else np.zeros(0)
        return R, T, S, nviol, worst, worst_pre

    def add_flowgates(self, R, T, S, tag):
        g = self.grid
        new = []
        for r, t, sg in zip(R, T, S):
            hs = self.pairs.setdefault((int(r), int(sg)), set())
            if int(t) not in hs:
                hs.add(int(t)); new.append((int(r), int(t), int(sg)))
        if not new:
            return 0
        R = np.array([x[0] for x in new]); T = np.array([x[1] for x in new]); S = np.array([x[2] for x in new])
        DR = g.D[R]
        idx = np.hstack([self.colP[:, T].T, self.colPD[:, T].T, self.colPS[:, T].T])            # pair x terms
        val = np.hstack([DR[:, self.patg], DR[:, self.pats], -DR[:, self.pats]])
        nz = np.abs(val) > 1e-12
        starts = np.r_[0, np.cumsum(nz.sum(1))[:-1]].astype(np.int32)
        const = g.f0[R] / 1000.0 + np.einsum('kp,pk->k', DR, self.cst[:, T])
        lim = g.lim[R] / 1000.0
        lower = np.where(S > 0, -np.inf, -lim - const); upper = np.where(S > 0, lim - const, np.inf)
        r0 = self.h.getNumRow()
        self.h.addRows(len(R), lower, upper, int(nz.sum()), starts, idx[nz].astype(np.int32), val[nz])
        self.cut_idx += list(range(r0, r0 + len(R))); self.cut_lo += list(lower); self.cut_hi += list(upper)
        self.n_pairs += len(new)
        return len(new)

    def run(self, max_iter=40, log=print):
        hist = []
        for it in range(max_iter):
            sec = self.solve()
            P, PS, PD, inj = self.solution()
            R, T, S, nviol, worst, worst_pre = self.check(inj)
            rec = {'iteration': it, 'seconds': sec, 'objective_mbrl': self.obj, 'violations': nviol,
                   'max_violation_mw': worst, 'pairs': self.n_pairs, 'rows': len({k[0] for k in self.pairs}),
                   'worst_preexisting_increase_mw': worst_pre}
            hist.append(rec)
            log(f'  it {it}: {sec:.0f} s, obj {rec["objective_mbrl"]:.3f}, violated {nviol} (max {worst:.1f} MW), '
                f'pairs {self.n_pairs} on {rec["rows"]} rows', flush=True)
            if not self.network or nviol == 0:
                if self.fixed:
                    return P, PS, PD, inj, pd.DataFrame(hist)
                neg, dinf, rcmin = self.price_closed()
                hist[-1].update({'open_stations': len(self.open & set(self.cost_site)), 'opened_now': len(neg),
                                 'dual_infeasibilities': dinf, 'min_closed_reduced_cost': rcmin})
                log(f'  pricing: {len(neg)} closed stations with negative reduced cost (min {rcmin:.4g}), '
                    f'dual infeasibilities {dinf}', flush=True)
                if not neg:
                    return P, PS, PD, inj, pd.DataFrame(hist)
                self.open |= set(neg); self.apply_capacity_bounds()
                continue
            self.add_flowgates(R, T, S, it)
        raise RuntimeError(f'lazy flowgates did not converge in {max_iter} iterations')


# ------------------------------------------------------------------ results
def summarise(J: Joint, label: str, P, PS, PD, hist) -> tuple[dict, pd.DataFrame]:
    d, cand = J.d, J.cand
    gp = J.value('Generator-p_nom').to_series()
    sp = J.value('StorageUnit-p_nom').to_series()
    sites = []
    for r in cand.itertuples(index=False):
        v = float(sp.get(r.name, 0.0) if r.tech == 'bess' else gp.get(r.name, 0.0))
        sites.append({'case': label, 'year': J.year, 'tech': r.tech, 'area': r.area, 'rank': int(r.rank), 'bus_id': int(r.bus_id),
                      'station': r.station, 'kv': int(r.kv), 'capacity_mw': v * 1000,
                      'area_cap_mw': r.area_cap_mw if J.area_caps else np.inf, 'extra_conn_brl_kw': r.extra_conn_brl_kw})
    sites.append({'case': label, 'year': J.year, 'tech': 'bess', 'area': 'MG load buses', 'rank': -1, 'bus_id': -1,
                  'station': 'load buses pro rata', 'kv': -1, 'capacity_mw': float(sp['bess@load']) * 1000,
                  'area_cap_mw': np.inf, 'extra_conn_brl_kw': 0.0})
    sites.append({'case': label, 'year': J.year, 'tech': 'solar_distributed', 'area': 'MG load buses', 'rank': -1, 'bus_id': -1,
                  'station': 'load buses pro rata', 'kv': -1, 'capacity_mw': float(gp['solar_distributed']) * 1000,
                  'area_cap_mw': np.inf, 'extra_conn_brl_kw': 0.0})
    S = pd.DataFrame(sites)
    tot = S.groupby('tech').capacity_mw.sum() / 1000
    gen = P.to_pandas().T
    shed = gen['shed'].to_numpy() if J.shed else np.zeros(8760)
    w = d['w']; pen = w * SHED_PRICE[J.shed] * float(shed.sum()) if J.shed else 0.0
    names_new = [g for g in J.gens if ('@' in g or g == 'solar_distributed') and not g.startswith('biogas@')]
    avail = sum(float(gp[g]) * (d['cf'][g.split('@')[0]] if '@' in g else d['cf']['solar_distributed']) for g in names_new)
    avail = avail + d['vrebase']
    used = gen[names_new + [g for g in ('ex_pv', 'ex_mmgd', 'ex_thermal') if g in gen]].sum(1).to_numpy()
    capital = float(sum(J.cap_cost[k] * float(gp[k]) for k in gp.index if k in J.cap_cost)
                    + sum(J.cap_cost[k] * float(sp[k]) for k in sp.index))
    capital += float(gp['solar_distributed']) * d['capcoef']['solar_distributed']
    site_extra = float(sum((J.cap_cost[r.name] - d['capcoef'][r.tech]) * float((sp if r.tech == 'bess' else gp).get(r.name, 0.0))
                           for r in cand.itertuples(index=False)))
    tie = TIE * float(J.value('area_deviation').sum())
    last = hist.iloc[-1]
    out = {'case': label, 'year': J.year, 'network': J.network, 'shed_allowed': J.shed, 'site_cost': J.site_cost,
           'area_caps': J.area_caps, 'fixed_sequential': J.fixed, 'shed_price': J.shed or 'none',
           'objective_mbrl': J.obj - tie, 'tie_break_mbrl': tie, 'shed_penalty_pv_mbrl': pen,
           'objective_excl_shed_mbrl': J.obj - tie - pen,
           'objective_voll_valued_mbrl': J.obj - tie - pen + w * VOLL * float(shed.sum()),
           'capital_and_fixed_cost_pv_mbrl': capital,
           'site_connection_extra_pv_mbrl': site_extra, 'operating_cost_pv_mbrl': J.obj - tie - pen - capital,
           **{f'cap_{t}_gw': float(tot.get(t, 0.0)) for t in pp.TECH},
           'bess_at_load_gw': float(sp['bess@load']),
           'imports_twh': float(gen['imports'].sum() / 1000) if 'imports' in gen else 0.0,
           'gas_twh': float(gen['gas'].sum() / 1000) if 'gas' in gen else 0.0,
           'hydro_twh': float(gen['hydro'].sum() / 1000), 'curtailment_twh': float(np.maximum(0, avail - used).sum() / 1000),
           'shed_gwh': float(shed.sum()), 'shed_hours': int((shed > 1e-6).sum()),
           'iterations': int(last.iteration), 'flowgate_pairs': int(last.pairs), 'flowgate_rows': int(last.rows),
           'violations_final': int(last.violations), 'worst_preexisting_increase_mw': float(last.worst_preexisting_increase_mw),
           'seconds_total': float(hist.seconds.sum())}
    return out, S


def run_case(J: Joint, case: str, save_hourly=False):
    label = f'{J.year}_{case}'
    print(f'CASE {label}', flush=True)
    J.set_case(case)
    P, PS, PD, inj, hist = J.run()
    out, S = summarise(J, label, P, PS, PD, hist)
    if save_hourly:
        P.to_pandas().T.to_parquet(PARTS / f'{label}_gen.parquet')
        (PD - PS).to_pandas().T.to_parquet(PARTS / f'{label}_storage.parquet')
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}, flush=True)
    return out, S, hist.assign(case=label)


CASES = {  # label: (network, unserved-energy price: None / 'lp' / 'voll', site connection cost, area caps,
           #         fixed sequential capacities)
    'copperplate': (False, 'lp', True, True, False),
    'gate': (False, None, False, True, False),
    'network': (True, 'lp', True, True, False),
    'network_uncapped': (True, 'lp', True, False, False),
    'sequential': (True, 'lp', True, True, True),
    'network_voll': (True, 'voll', True, True, False),
}


CASE_DIR = OUT / 'joint_cases'


def one_bus_bound(year: int) -> dict:
    """Lower bound on every joint case with unserved energy at VOLL: the one-bus LP (no network, no site
    costs, no area caps) with the same VOLL shed column. Site costs and the tie-break are >= 0."""
    case = pp.refresh_cases()[0 if year == 2035 else 1]
    d = pp.lp_inputs(case); n = pp.build(case, d)
    lm = float(d['load'].max())
    n.add('Generator', 'shed', bus='MG', p_nom=lm, p_max_pu=pd.Series(d['load'] / lm, n.snapshots), marginal_cost=VOLL)
    status, cond = n.optimize(solver_name='highs', extra_functionality=pp.extra(case, d), include_objective_constant=False,
                              solver_options={'output_flag': False, 'primal_feasibility_tolerance': 1e-7,
                                              'dual_feasibility_tolerance': 1e-7})
    assert status == 'ok', (status, cond)
    g = n.generators_t.p
    caps = {t: float(n.generators.at[t, 'p_nom_opt']) for t in pp.CF_TECH + ['biogas']}
    caps['bess'] = float(n.storage_units.at['bess', 'p_nom_opt'])
    return {'case': f'{year}_one_bus_voll', 'year': year, 'objective_mbrl': float(n.objective),
            'shed_gwh': float(g['shed'].sum()), **{f'cap_{t}_gw': v for t, v in caps.items()}}


def merge():
    """Concatenate the per-case files into results/joint_summary.csv, joint_sites.csv, joint_iterations.csv."""
    for kind in ('summary', 'sites', 'iterations'):
        fs = sorted(CASE_DIR.glob(f'*_{kind}.csv'))
        if fs:
            pd.concat([pd.read_csv(f) for f in fs]).to_csv(OUT / f'joint_{kind}.csv', index=False, float_format='%.6f')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', nargs='+', type=int, default=[2035, 2050])
    ap.add_argument('--cases', nargs='+', default=list(CASES))
    ap.add_argument('--merge', action='store_true', help='only merge the per-case files')
    a = ap.parse_args(argv)
    if a.merge:
        merge(); return 0
    CASE_DIR.mkdir(parents=True, exist_ok=True)
    for y in a.years:
        t0 = time.time()
        J = Joint(y)
        print(f'{y}: built in {time.time() - t0:.0f} s: {len(J.gens)} generators, {len(J.stores)} storage units', flush=True)
        rec = json.loads((pp.MACRO / 'output' / f'REFRESH_{y}' / 'solver.json').read_text())
        pd.DataFrame([{'case': f'{y}_one_bus_lp', 'year': y, 'objective_mbrl': float(rec['objective']),
                       'note': 'saved portfolio LP (no unserved energy): lower bound of the main cases'},
                      {**one_bus_bound(y), 'note': 'one-bus LP with unserved energy at VOLL: lower bound of network_voll'}]
                     ).to_csv(CASE_DIR / f'{y}_bounds.csv', index=False, float_format='%.6f')
        for c in a.cases:
            label = f'{y}_{c}'
            out, S, H = run_case(J, c, save_hourly=c in ('network', 'network_uncapped', 'sequential'))
            pd.DataFrame([out]).to_csv(CASE_DIR / f'{label}_summary.csv', index=False, float_format='%.6f')
            S.to_csv(CASE_DIR / f'{label}_sites.csv', index=False, float_format='%.6f')
            H.to_csv(CASE_DIR / f'{label}_iterations.csv', index=False, float_format='%.6f')
    merge()
    return 0


if __name__ == '__main__':
    sys.exit(main())
