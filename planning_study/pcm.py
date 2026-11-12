#!/usr/bin/env python3
"""Phase 4: network-constrained 8,760-hour production-cost run (PyPSA) and benefit-cost ratios.

Hourly operation of the fixed portfolio is a PyPSA dispatch (one bus for the energy balance, exactly the
LP's costs and limits) with DC flowgate constraints on the MG main grid. Network flows are incremental
on the PWF snapshot, which is matched to the 2035 LP hour that reproduces it (evening, existing PV at 0,
MG load 10,731 MW, hydro 5,132 MW = PWF UHE+PCH output). In hour t every component's change relative to
that hour is injected on its PWF buses:
  load (and MMGD, distributed PV, unserved energy) -> MG load buses pro rata to PWF load;
  reservoir hydro -> MG UHE buses pro rata to snapshot output; imports -> boundary inflow equivalents
  (A2 pattern); existing central PV -> the 106 UFV buses equally; existing biomass/mixed thermal and gas
  -> MG UTE buses pro rata; new resources and BESS -> their sited buses.
Because both the LP hour and every simulated hour balance, the changes sum to zero.
Flowgates are added lazily: after each solve all 242,708 monitored rows (N-0 and N-1) are checked in
every hour and violated rows are added, until none is violated. Pre-existing overloads of the base case
are reliability needs (phase 1) and are reported, not constrained. Months are solved separately (monthly
hydro budgets are the LP's own; BESS state of charge is cyclic within each month).

Valuation, fixed before the runs: production cost as in the LP (imports at the 2025 SE CMO, gas, biogas
fuel, BESS tie-break), unserved energy at the 2025 Custo do Déficit (CCEE, R$ 8,327.76/MWh), curtailed
energy at the 2025 SE CMO (R$ 209.47/MWh) as its opportunity value in the SIN (sensitivity ±50 %).
"""
from __future__ import annotations
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
import mitigation as mt  # noqa: E402
from hc_engine import Network  # noqa: E402

logging.getLogger('pypsa').setLevel(logging.ERROR); logging.getLogger('linopy').setLevel(logging.ERROR)
OUT = HERE / 'results'
VOLL = 8327.76            # R$/MWh, CCEE Custo do Déficit 2025
P_CURT = 209.47           # R$/MWh, 2025 SE CMO (the LP's import price)
T_SNAP = 2349             # 2035 LP hour matching the PWF snapshot (see find_snapshot_hour)
MAX_ADD = 150             # rows added per lazy iteration
TOL = 1e-3                # MW


def find_snapshot_hour():
    """The 2035 LP hour closest to the PWF snapshot in MG load and MG hydro output (evening, PV at 0)."""
    d = pd.read_csv(rh.NODAL / 'macro_reopt' / 'output' / 'REFRESH_2035' / 'dispatch_2035.csv.gz', parse_dates=['hour_start_brt'])
    net = Network(); B = net.B
    L0 = float(B.loc[(B.state == 11) & (B.pd > 0), 'pd'].sum()) / 1000
    H0 = float(B.loc[(B.state == 11) & B.gen_type.isin([3, 5]), 'pg'].sum()) / 1000
    h = d.hour_start_brt.dt.hour
    ev = d[(h >= 18) & (h <= 21) & (d.used_central_pv_gw < 0.005)]       # PV below 5 MW
    score = (ev.load_gw - L0).abs() / L0 + (ev.hydro_gw - H0).abs() / H0
    return int(score.idxmin()), d


def references():
    t, d = find_snapshot_hour()
    r = d.loc[t]
    ref = {'load': r.load_gw, 'hydro': r.hydro_gw, 'mmgd': r.used_mmgd_gw, 'ex_pv': r.used_central_pv_gw,
           'ex_thermal': r.used_biomass_observed_gw + r.used_mixed_thermal_gw, 'gas': r.gas_gw}
    ref['imports'] = ref['load'] - (ref['hydro'] + ref['mmgd'] + ref['ex_pv'] + ref['ex_thermal'] + ref['gas'])
    ref['t_snap'] = t; ref['lp_imports'] = r.imports_gw
    return ref


class Grid:
    """Row coefficients of the injection patterns on every monitored N-0 / N-1 row."""

    def __init__(self, net, sites, bess_at_load):
        B = net.B
        self.net = net
        pats = {'load': net.load_weights()}
        w = np.where(((B.state == 11) & (B.gen_type == 5)).to_numpy(), B.pg, 0.0); pats['hydro'] = w / w.sum()
        w = ((B.state == 11) & (B.gen_type == 8)).to_numpy().astype(float); pats['ufv'] = w / w.sum()
        w = np.where(((B.state == 11) & (B.gen_type == 7) & (B.pg > 0)).to_numpy(), B.pg, 0.0); pats['ute'] = w / w.sum()
        w = np.where(B.boundary.to_numpy() & (net.p_implied > 1.0), net.p_implied, 0.0); pats['imp'] = w / w.sum()
        for b in sorted(set(int(x) for x in sites.bus_id)):
            e = np.zeros(net.n); e[net.idx[b]] = 1.0; pats[f'bus{b}'] = e
        self.names = list(pats)
        P = np.column_stack([pats[k] for k in self.names])
        G = net.PTDF @ P
        M, K = net.monitored, net.contingencies
        ee, kk = np.nonzero(~net.self_mask)
        self.D = np.vstack([G[M], G[M][ee] + net.LODF[ee, kk][:, None] * G[K][kk]])
        base = net.rows(np.zeros(net.n))
        self.f0, self.lim, self.elem, self.cont = base.f0, base.lim, base.elem, base.cont
        self.pre = np.abs(self.f0) > self.lim + 1e-6
        self.col = {k: i for i, k in enumerate(self.names)}


def build_month(case, inp, ex, sites, cap, bess_at_load, month, soc_frac):
    """One-bus PyPSA dispatch for one month with fixed capacities (GW)."""
    idx = np.flatnonzero(inp['month'] == month)
    sn = pd.Index(idx, name='snapshot')
    n = pypsa.Network(); n.set_snapshots(sn)
    n.add('Bus', 'MG')
    n.add('Load', 'load', bus='MG', p_set=pd.Series(inp['load'][idx], sn))
    gens = {}                                                    # name -> pattern

    def add(name, p_nom, prof, mc, pat):
        if p_nom <= 1e-9:
            return
        n.add('Generator', name, bus='MG', p_nom=p_nom, p_max_pu=prof, marginal_cost=mc)
        gens[name] = pat
    for nm, series, pat in [('ex_pv', ex['pv'], 'ufv'), ('ex_mmgd', ex['mmgd'], 'load'), ('ex_thermal', ex['thermal'], 'ute')]:
        mx = float(series.max())
        add(nm, mx, pd.Series(series[idx] / mx, sn), 0.0, pat)
    hu = inp['hydro_upper']; hm = float(hu.max())
    add('hydro', hm, pd.Series(hu[idx] / hm, sn), 0.0, 'hydro')
    add('gas', inp['gas_cap'], 1.0, inp['gas_mc'], 'ute')
    add('imports', inp['imp_cap'], 1.0, inp['imp_mc'], 'imp')
    for r in sites.itertuples(index=False):
        if r.tech == 'bess':
            continue
        cf = inp['biogas_cf'] if r.tech == 'biogas' else pd.Series(inp['cf'][r.tech][idx], sn)
        mc = inp['biogas_mc'] if r.tech == 'biogas' else 0.0
        add(f'{r.tech}@{r.bus_id}@{r.area}', r.new_capacity_mw / 1000, cf, mc, f'bus{int(r.bus_id)}')
    add('dg', cap.get('solar_distributed', 0) / 1000, pd.Series(inp['cf']['solar_distributed'][idx], sn), 0.0, 'load')
    lm = float(inp['load'].max())
    for i in range(pp.SHED_TRANCHES):
        n.add('Generator', f'shed_{i:02d}', bus='MG', p_nom=lm, p_max_pu=pd.Series(inp['load'][idx] / lm / pp.SHED_TRANCHES, sn),
              marginal_cost=1000.0 * (1 + 1e-4 * i))
        gens[f'shed_{i:02d}'] = 'load'
    stores = {}
    f0 = soc_frac[idx[0] - 1] if idx[0] > 0 else soc_frac[-1]     # state of charge entering the month
    kw = dict(bus='MG', max_hours=inp['dur'], efficiency_store=inp['eta'], efficiency_dispatch=inp['eta'],
              cyclic_state_of_charge=False, marginal_cost=0.001)
    if bess_at_load:
        if cap.get('bess', 0) > 0:
            n.add('StorageUnit', 'bess@load', p_nom=cap['bess'] / 1000, state_of_charge_initial=f0 * cap['bess'] / 1000 * inp['dur'],
                  **kw); stores['bess@load'] = 'load'
    else:
        for r in sites[sites.tech == 'bess'].itertuples(index=False):
            nm = f'bess@{r.bus_id}@{r.area}'
            n.add('StorageUnit', nm, p_nom=r.new_capacity_mw / 1000, state_of_charge_initial=f0 * r.new_capacity_mw / 1000 * inp['dur'],
                  **kw); stores[nm] = f'bus{int(r.bus_id)}'
    return n, gens, stores, idx


def constants(grid, inp, ex, ref, idx):
    """Pattern injections that do not depend on decisions: -(load change) and minus the reference levels."""
    T = len(idx); c = np.zeros((len(grid.names), T))
    c[grid.col['load']] = -(inp['load'][idx] - ref['load']) - ref['mmgd']
    c[grid.col['hydro']] = -ref['hydro']
    c[grid.col['imp']] = -ref['imports']
    c[grid.col['ufv']] = -ref['ex_pv']
    c[grid.col['ute']] = -ref['ex_thermal'] - ref['gas']
    return c


def solve_month(case, inp, ex, ref, grid, sites, cap, bess_at_load, month, soc_frac, rows_init=(), flowgates=True, max_iter=12):
    rows = list(rows_init)
    for it in range(max_iter):
        n, gens, stores, idx = build_month(case, inp, ex, sites, cap, bess_at_load, month, soc_frac)
        cst = constants(grid, inp, ex, ref, idx)
        E = inp['energy'][month]; ramp = inp['ramp']
        sel_rows = np.array(rows, int)

        def ef(n, sns):
            m = n.model
            p = m.variables['Generator-p']
            hyd = p.sel(name='hydro')
            m.add_constraints(hyd.sum() <= E, name='hydro_energy')
            first = xr.DataArray(n.snapshots != n.snapshots[0], coords={'snapshot': n.snapshots})
            prev = hyd.roll(snapshot=1)
            m.add_constraints(hyd - prev <= ramp, name='ramp_up', mask=first)
            m.add_constraints(prev - hyd <= ramp, name='ramp_dn', mask=first)
            if stores:
                ps = m.variables['StorageUnit-p_store']; pdis = m.variables['StorageUnit-p_dispatch']
                pn = xr.DataArray([n.storage_units.at[s, 'p_nom'] for s in stores], coords={'name': list(stores)})
                m.add_constraints(ps + pdis <= pn, name='bess_single_converter')
                soc = m.variables['StorageUnit-state_of_charge'].sel(snapshot=n.snapshots[-1])
                e_end = xr.DataArray([soc_frac[idx[-1]] * n.storage_units.at[s, 'p_nom'] * inp['dur'] for s in stores],
                                     coords={'name': list(stores)})
                m.add_constraints(soc >= e_end - 1e-6, name='soc_month_end')
                m.add_objective(m.objective.expression + 0.001 * ps.sum(), overwrite=True)
            if flowgates and len(sel_rows):
                gn = list(gens)
                Dg = xr.DataArray(np.column_stack([grid.D[sel_rows, grid.col[gens[g]]] for g in gn]),
                                  coords={'row': sel_rows, 'name': gn})
                expr = (p.sel(name=gn) * Dg).sum('name')
                if stores:
                    sn_ = list(stores)
                    Ds = xr.DataArray(np.column_stack([grid.D[sel_rows, grid.col[stores[s]]] for s in sn_]),
                                      coords={'row': sel_rows, 'name': sn_})
                    expr = expr + (pdis.sel(name=sn_) * Ds).sum('name') - (ps.sel(name=sn_) * Ds).sum('name')
                # all in GW of flow: decisions are GW, PWF flows and limits are MW
                const = grid.f0[sel_rows][:, None] / 1000.0 + grid.D[sel_rows] @ cst      # row x time
                lim = grid.lim[sel_rows][:, None] / 1000.0
                hi = xr.DataArray((lim - const).T, coords={'snapshot': n.snapshots, 'row': sel_rows})
                lo = xr.DataArray((-lim - const).T, coords={'snapshot': n.snapshots, 'row': sel_rows})
                m.add_constraints(expr <= hi, name='flowgate_hi')
                m.add_constraints(expr >= lo, name='flowgate_lo')
        status, cond = n.optimize(solver_name='highs', extra_functionality=ef, include_objective_constant=False,
                                  solver_options={'output_flag': False})
        assert status == 'ok', (month, it, status, cond)
        inj = cst.copy()
        gp = n.generators_t.p
        for g, pat in gens.items():
            inj[grid.col[pat]] += gp[g].to_numpy()
        if stores:
            for s, pat in stores.items():
                inj[grid.col[pat]] += (n.storage_units_t.p_dispatch[s] - n.storage_units_t.p_store[s]).to_numpy()
        assert np.abs(inj.sum(0)).max() < 1e-5, 'pattern injections must balance'
        viol = []
        worst_pre = 0.0
        for a in range(0, len(grid.f0), 20000):
            b = min(len(grid.f0), a + 20000)
            fl = grid.f0[a:b, None] + (grid.D[a:b] @ inj) * 1000.0      # pattern injections in GW -> MW
            over = np.abs(fl).max(1) - grid.lim[a:b]
            pre = grid.pre[a:b]
            if pre.any():
                worst_pre = max(worst_pre, float((np.abs(fl[pre]).max(1) - np.abs(grid.f0[a:b][pre])).max()))
            j = np.flatnonzero((over > TOL) & ~pre)
            viol += [(float(over[q]), a + int(q)) for q in j]
        if not flowgates or not viol:
            return n, gens, stores, idx, inj, rows, it, viol, worst_pre
        viol.sort(reverse=True)
        new = [r for _, r in viol if r not in set(rows)][:MAX_ADD]
        rows += new
    raise RuntimeError(f'month {month}: lazy flowgates did not converge ({len(viol)} violated rows)')


_SOC = {}


def annual_soc(year):
    """State-of-charge fraction of the aggregate BESS in the annual copper-plate dispatch (PyPSA, fixed
    REFRESH capacities). Month boundaries of every scenario inherit it, so the monthly decomposition keeps
    the inter-month energy shifting of the annual optimum."""
    if year not in _SOC:
        case = pp.refresh_cases()[0 if year == 2035 else 1]
        caps = {t: v / 1000 for t, v in rh.portfolio_mw(year).items()}
        c2 = pp.Case(case.name, year, case.scenario_config, netzero_mode=case.netzero_mode, allow_shed=True, fixed_caps=caps)
        r = pp.solve(c2)
        n = r['_network']
        E = caps.get('bess', 0.0) * r['_inputs']['dur']
        _SOC[year] = (n.storage_units_t.state_of_charge['bess'].to_numpy() / E if E > 0 else np.zeros(8760),
                      {k: r[k] for k in ('objective_mbrl', 'imports_twh', 'gas_twh', 'biogas_twh', 'shed_gwh', 'curtailment_twh')})
    return _SOC[year]


def run_scenario(name, year, siting, plan_siting, flowgates=True):
    """siting: 'reference' or 'reference_bess_at_load'; plan_siting: None or the mitigation plan to include."""
    t0 = time.time()
    case = pp.refresh_cases()[0 if year == 2035 else 1]
    inp = pp.lp_inputs(case)
    f = inp['f']
    retire = max(0, (2050 - year) / 25) if case.netzero_mode != 'none' else 1.
    ex = {'pv': f.existing_solar_central_mw.to_numpy() / 1000, 'mmgd': f.existing_mmgd_mw.to_numpy() / 1000,
          'thermal': f.existing_biomass_observed_mw.to_numpy() / 1000 + f.existing_mixed_thermal_mw.to_numpy() / 1000 * retire}
    cap = rh.portfolio_mw(year)
    sites, _ = rh.reference_siting(year, cap)
    bess_at_load = siting == 'reference_bess_at_load'
    if plan_siting:
        plan = pd.read_csv(OUT / 'mitigation_plan.csv'); plan = plan[plan.siting == plan_siting].to_dict('records')
        net0 = Network(); net = Network(extra=mt.extra_branches(net0, plan))
        mt.apply_reconductor(net, rh.upgrade_groups(net), plan)
    else:
        net = Network()
    grid = Grid(net, sites, bess_at_load)
    ref = references()
    months, rows_all = [], set()
    disp = []
    soc_frac, annual = annual_soc(year)
    for mth in range(1, 13):
        n, gens, stores, idx, inj, rows, its, viol, worst_pre = solve_month(case, inp, ex, ref, grid, sites, cap, bess_at_load,
                                                                             mth, soc_frac, flowgates=flowgates)
        gp = n.generators_t.p
        shed = gp.filter(like='shed_').sum(1).to_numpy()
        # zero-cost profile resources only (as the LP's curtailment); biogas has a fuel cost, so unused biogas is not curtailment
        new_names = [g for g in gens if ('@' in g or g == 'dg') and not g.startswith('biogas@')]
        avail_new = sum((n.generators.at[g, 'p_nom'] * n.get_switchable_as_dense('Generator', 'p_max_pu')[g]).to_numpy()
                        for g in new_names)
        used_new = sum(gp[g].to_numpy() for g in new_names)
        ex_names = [g for g in ('ex_pv', 'ex_mmgd', 'ex_thermal') if g in gens]
        avail_ex = sum((n.generators.at[g, 'p_nom'] * n.get_switchable_as_dense('Generator', 'p_max_pu')[g]).to_numpy() for g in ex_names)
        used_ex = sum(gp[g].to_numpy() for g in ex_names)
        prod = float(sum(gp[g].sum() * n.generators.at[g, 'marginal_cost'] for g in ('gas', 'imports') if g in gp)
                     + sum(gp[g].sum() * inp['biogas_mc'] for g in gens if g.startswith('biogas@')))
        months.append({'month': mth, 'iterations': its, 'flowgate_rows': len(rows), 'violated_after': len(viol),
                       'ens_gwh': float(shed.sum()), 'curt_new_gwh': float((avail_new - used_new).sum()),
                       'curt_existing_gwh': float((avail_ex - used_ex).sum()),
                       'profile_available_gwh': float((avail_new + avail_ex).sum()),
                       'biogas_gwh': float(sum(gp[g].sum() for g in gens if g.startswith('biogas@'))),
                       'hydro_gwh': float(gp['hydro'].sum()), 'hydro_budget_gwh': float(inp['energy'][mth]),
                       'imports_gwh': float(gp['imports'].sum()) if 'imports' in gp else 0.0,
                       'production_cost_mbrl': prod, 'worst_preexisting_increase_mw': worst_pre})
        rows_all |= set(rows)
        print(f'  {name} month {mth}: iters {its}, rows {len(rows)}, ENS {shed.sum():.3f} GWh, '
              f'curt new {months[-1]["curt_new_gwh"]:.1f} GWh', flush=True)
    M = pd.DataFrame(months)
    tot = {'scenario': name, 'year': year, 'siting': siting, 'plan': plan_siting or 'existing grid', 'flowgates': flowgates,
           'ens_mwh': M.ens_gwh.sum() * 1000, 'curt_new_twh': M.curt_new_gwh.sum() / 1000,
           'curt_existing_twh': M.curt_existing_gwh.sum() / 1000, 'profile_available_twh': M.profile_available_gwh.sum() / 1000,
           'biogas_twh': M.biogas_gwh.sum() / 1000, 'hydro_twh': M.hydro_gwh.sum() / 1000,
           'hydro_spill_twh': (M.hydro_budget_gwh - M.hydro_gwh).sum() / 1000, 'imports_twh': M.imports_gwh.sum() / 1000,
           'production_cost_mbrl_yr': M.production_cost_mbrl.sum(),
           'flowgate_rows_used': len(rows_all), 'max_iterations': int(M.iterations.max()),
           'worst_preexisting_increase_mw': M.worst_preexisting_increase_mw.max(), 'seconds': time.time() - t0}
    tot['ens_cost_mbrl_yr'] = tot['ens_mwh'] * VOLL / 1e6
    tot['curt_value_mbrl_yr'] = (tot['curt_new_twh'] + tot['curt_existing_twh']) * 1e6 * P_CURT / 1e6
    tot['total_cost_mbrl_yr'] = tot['production_cost_mbrl_yr'] + tot['ens_cost_mbrl_yr'] + tot['curt_value_mbrl_yr']
    sub = [{'scenario': name, 'row': int(r), 'element': '{from_name}-{to_name}'.format(**net.branch_label(grid.elem[r])),
            'contingency': 'N-0' if grid.cont[r] < 0 else '{from_name}-{to_name}'.format(**net.branch_label(grid.cont[r]))}
           for r in sorted(rows_all)]
    return tot, M.assign(scenario=name), pd.DataFrame(sub)


def main(only=None):
    scen = [('2035_copperplate', 2035, 'reference', None, False),
            ('2035_existing_grid', 2035, 'reference', None, True),
            ('2050_copperplate', 2050, 'reference', None, False),
            ('2050_existing_grid', 2050, 'reference', None, True),
            ('2050_with_plan', 2050, 'reference', 'reference', True),
            ('2050_bess_at_load_existing_grid', 2050, 'reference_bess_at_load', None, True),
            ('2050_bess_at_load_with_plan', 2050, 'reference_bess_at_load', 'reference_bess_at_load', True)]
    if only:
        scen = [s for s in scen if s[0] in only]
    T, Ms, Rs = [], [], []
    for s in scen:
        print('SCENARIO', s[0], flush=True)
        tot, M, R = run_scenario(*s)
        T.append(tot); Ms.append(M); Rs.append(R)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in tot.items()}, flush=True)
    T = pd.DataFrame(T); T.to_csv(OUT / 'pcm_scenarios.csv', index=False, float_format='%.6f')
    pd.concat(Ms).to_csv(OUT / 'pcm_monthly.csv', index=False, float_format='%.6f')
    pd.concat(Rs).to_csv(OUT / 'pcm_flowgates.csv', index=False)
    return T


if __name__ == '__main__':
    main(sys.argv[1:] or None)
