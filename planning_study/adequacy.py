#!/usr/bin/env python3
"""Phase 2b: resource adequacy of the fixed 2035/2050 portfolios (ENERGY176/276 Lectures 4, 5, 8).

Two complementary methods, both with NERC GADS class-average forced-outage data
(Generating Unit Statistical Brochure 4, 2016-2020; planning_study/data/gads_class_averages.csv):

A. COPT convolution (the course method). Dispatchable units (reservoir hydro, gas, biogas) form a
   capacity outage probability table; hydro availability in hour t is the LP's hourly upper bound
   scaled by the available share of installed hydro units. Profile-based resources (existing PV,
   MMGD, biomass, mixed thermal; new solar, wind, biomass, small hydro) and imports (the LP's firm
   bound) reduce the load; BESS shaves each day's net-load peaks (water-filling) and its 100 MW
   blocks are on forced outage independently. Hourly LOLP -> LOLE (days/yr, daily maximum LOLP),
   LOLH (h/yr), EUE (MWh/yr). Capacity only: monthly hydro energy budgets are NOT seen here.
   ELCC / firm-capacity gap by the CAISO calibration (Lecture 11): perfect capacity is added or
   removed until LOLE = 0.1 day/yr, with and without each resource class.
B. Sequential Monte Carlo with an energy-limited PyPSA dispatch. Two-state Markov outage histories
   (GADS EFORd and mean forced-outage duration) derate hydro, gas, biogas and BESS hour by hour; each
   sample is an 8,760-h minimum-cost dispatch with unserved energy priced at the LP's penalty, monthly
   hydro energy budgets, ramping and storage state of charge (portfolio_pypsa.py, fixed capacities).

Only deliverable capacity counts (Lecture 8): runs are repeated with every new resource scaled by
the portfolio's peak deliverable fraction d = min(1, alpha_HSN, alpha_SSN) from phase 1.
Unit data are not in the package: installed reservoir hydro (13.383 GW, the LP's value) is split over
the 37 MG UHE buses of the PWF in proportion to their reactive limit qmax and then into units of at
most 250 MW (sensitivity 125 / 500 MW); gas = two equal units; biogas = 20 MW engines; BESS = 100 MW blocks.
"""
from __future__ import annotations
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import portfolio_pypsa as pp  # noqa: E402

OUT = HERE / 'results'
GADS = pd.read_csv(HERE / 'data' / 'gads_class_averages.csv').set_index('model_class')
PWF_BUSES = ROOT / 'MG_Step2_Congestion_Relief' / 'MG_Nodal_Connection_Optimization' / 'inputs' / 'pwf_2035_buses.csv'
LOLE_TARGET = 0.1            # days per year ("one day in ten years")
NEW_PROFILE = ['solar_central', 'solar_distributed', 'bio', 'small_hydro', 'wind', 'wind_site']
CLASSES = {'new solar (central + DG)': ['solar_central', 'solar_distributed'], 'new BESS': ['bess'],
           'new wind': ['wind', 'wind_site'], 'new biomass + small hydro': ['bio', 'small_hydro'],
           'new biogas': ['biogas'], 'all new resources': ['solar_central', 'solar_distributed', 'bio',
                                                           'small_hydro', 'wind', 'wind_site', 'biogas', 'bess']}


# --------------------------------------------------------------------- fleet
from functools import lru_cache


@lru_cache(maxsize=None)
def _hydro_units_cached(total_mw, unit_max):
    return hydro_units_uncached(total_mw, unit_max)


def hydro_units(total_mw, unit_max=250.0):
    return _hydro_units_cached(float(total_mw), float(unit_max)).copy()


def hydro_units_uncached(total_mw, unit_max=250.0):
    B = pd.read_csv(PWF_BUSES)
    u = B[(B.state == 11) & (B.gen_type == 5)].copy()
    u['plant_mw'] = u.qmax / u.qmax.sum() * total_mw
    rows = []
    for r in u.itertuples(index=False):
        k = int(np.ceil(r.plant_mw / unit_max - 1e-9))
        size = r.plant_mw / k
        cls = 'hydro_30plus' if size >= 30 else 'hydro_1_29'
        rows += [{'plant_bus': r.name, 'unit': i, 'mw': size, 'class': cls} for i in range(k)]
    return pd.DataFrame(rows)


def fleet(case_inputs, caps_gw, unit_max=250.0, bess_block=100.0, biogas_unit=20.0):
    """Dispatchable units (MW) with GADS classes. caps_gw: new capacities actually counted."""
    p = case_inputs['p']
    hu = hydro_units(p['hydro_operating_capacity_gw'] * 1000, unit_max)
    rows = [dict(kind='hydro', mw=r.mw, cls=r['class']) for _, r in hu.iterrows()]
    gas = case_inputs['gas_cap'] * 1000
    if gas > 0:
        rows += [dict(kind='gas', mw=gas / 2, cls='combined_cycle')] * 2
    bg = caps_gw.get('biogas', 0.0) * 1000
    if bg > 1e-6:
        k = int(np.ceil(bg / biogas_unit - 1e-9))
        rows += [dict(kind='biogas', mw=bg / k, cls='diesel_ic')] * k
    bs = caps_gw.get('bess', 0.0) * 1000
    if bs > 1e-6:
        k = int(np.ceil(bs / bess_block - 1e-9))
        rows += [dict(kind='bess', mw=bs / k, cls='pumped_storage')] * k
    f = pd.DataFrame(rows)
    f['efor'] = f.cls.map(GADS.EFORd_pct) / 100.0
    f['mttr_h'] = f.cls.map(GADS.mean_forced_outage_h)
    return f


# ----------------------------------------------------------------- method A
def unit_pmf(sizes, q):
    """Exact pmf of available capacity on a 1-MW grid (sizes rounded to whole MW)."""
    s = np.rint(sizes).astype(int)
    pmf = np.zeros(int(s.sum()) + 1); pmf[0] = 1.0
    top = 0
    for si, qi in zip(s, q):
        new = pmf * qi
        new[si:top + si + 1] += pmf[:top + 1] * (1 - qi)
        pmf = new; top += si
    return pmf


def bess_plan(nl, p_mw, e_mwh, eta):
    """Daily water-filling: shave each day's net-load peaks with the deliverable energy, recharge in the
    day's lowest hours without creating a new peak. Returns the hourly discharge (MW, >= 0) and charge."""
    dis = np.zeros_like(nl); chg = np.zeros_like(nl)
    if p_mw <= 0:
        return dis, chg
    for d in range(len(nl) // 24):
        sl = slice(24 * d, 24 * d + 24); x = nl[sl]
        e_out = e_mwh * eta                     # usable discharge energy for one full cycle
        lo, hi = x.min() - p_mw, x.max()
        for _ in range(60):                     # shave level L: sum min(P, (x-L)+) = e_out
            L = (lo + hi) / 2
            if np.minimum(p_mw, np.maximum(0, x - L)).sum() > e_out:
                lo = L
            else:
                hi = L
        dd = np.minimum(p_mw, np.maximum(0, x - hi))
        need_in = dd.sum() / (eta * eta)
        free = dd <= 0                          # charge only in hours without discharge
        lo2, hi2 = x.min(), hi                  # charge fill level C <= L: sum_free min(P, (C-x)+) = need_in
        for _ in range(60):
            C = (lo2 + hi2) / 2
            if np.where(free, np.minimum(p_mw, np.maximum(0, C - x)), 0).sum() < need_in:
                lo2 = C
            else:
                hi2 = C
        cc = np.where(free, np.minimum(p_mw, np.maximum(0, hi2 - x)), 0.0)
        if cc.sum() + 1e-6 < need_in:           # cannot recharge fully without a new peak: scale down
            scale = cc.sum() / max(need_in, 1e-9); dd *= scale
        elif cc.sum() > need_in + 1e-6:         # trim the fill to the energy actually needed
            cc *= need_in / cc.sum()
        dis[sl] = dd; chg[sl] = cc
    return dis, chg


class COPT:
    """Hourly LOLP / EUE with hydro scaled hour by hour and independent BESS blocks."""

    def __init__(self, units: pd.DataFrame):
        h = units[units.kind == 'hydro']
        self.h_pmf = unit_pmf(h.mw.to_numpy(), h.efor.to_numpy())
        self.h_inst = float(np.rint(h.mw).sum())
        o = units[units.kind.isin(['gas', 'biogas'])]
        mw = o.mw.to_numpy() * np.where(o.kind == 'biogas', 0.7, 1.0)   # biogas availability factor
        self.y_pmf = unit_pmf(mw, o.efor.to_numpy()) if len(o) else np.array([1.0])
        b = units[units.kind == 'bess']
        self.nb = len(b); self.qb = float(b.efor.iloc[0]) if len(b) else 0.0
        grid = np.arange(len(self.h_pmf), dtype=float)
        self.h_cdf = np.cumsum(self.h_pmf)                     # P(H <= g)
        self.h_cm = np.cumsum(self.h_pmf * grid)               # sum_{h<=g} h p(h)
        yv = np.flatnonzero(self.y_pmf > 1e-15)
        self.y_val, self.y_p = yv.astype(float), self.y_pmf[yv]
        from scipy.stats import binom
        self.b_p = binom.pmf(np.arange(self.nb + 1), self.nb, 1 - self.qb) if self.nb else np.array([1.0])

    def hourly(self, nl, k, bess_dis):
        """nl: residual load to be met by hydro/gas/biogas/BESS (MW); k: hydro MW per installed MW."""
        T = len(nl); lolp = np.zeros(T); eue = np.zeros(T)
        hmax_avail = k * self.h_inst + self.y_val.max() + bess_dis
        idx = np.flatnonzero(nl > 0)
        # skip hours that cannot be short even in a 1e-12 tail
        q_lo = np.searchsorted(self.h_cdf, 1e-12)
        idx = idx[nl[idx] > k[idx] * q_lo - 1e-9]
        G = len(self.h_pmf) - 1
        j = np.arange(len(self.b_p))
        for t in idx:
            step = bess_dis[t] / self.nb if self.nb else 0.0
            a = nl[t] - self.y_val[:, None] - (j * step)[None, :]          # shortfall threshold for k*H
            w = self.y_p[:, None] * self.b_p[None, :]
            x = a / k[t]                                                    # in installed-hydro MW
            g = np.clip(np.ceil(x - 1e-9).astype(int) - 1, -1, G)          # largest grid point < x
            F = np.where(g >= 0, self.h_cdf[np.maximum(g, 0)], 0.0)
            Mh = np.where(g >= 0, self.h_cm[np.maximum(g, 0)], 0.0)
            lolp[t] = float((w * F).sum())
            eue[t] = float((w * np.maximum(0, k[t] * (x * F - Mh))).sum())
        return lolp, eue


def residual(inp, caps_gw, d=1.0, pc_mw=0.0, imports_firm=True):
    """Residual load for the dispatchable fleet (MW), the hydro scale k(t) and the BESS plan."""
    load = inp['load'] * 1000
    prof = inp['vrebase'] * 1000 + sum(caps_gw.get(t, 0.0) * d * 1000 * inp['cf'][t] for t in NEW_PROFILE)
    imp = inp['imp_cap'] * 1000 if imports_firm else 0.0
    nl0 = load - prof - imp - pc_mw
    p_b = caps_gw.get('bess', 0.0) * d * 1000
    dis, chg = bess_plan(nl0, p_b, p_b * inp['dur'], inp['eta'])
    k = inp['hydro_upper'] * 1000 / (inp['p']['hydro_operating_capacity_gw'] * 1000)
    return nl0 + chg, k, dis


def metrics_A(inp, caps_gw, d=1.0, pc_mw=0.0, unit_max=250.0, imports_firm=True, zero_for=False):
    units = fleet(inp, {t: v * d for t, v in caps_gw.items()}, unit_max)
    if zero_for:
        units['efor'] = 0.0
    cp = COPT(units)
    nl, k, dis = residual(inp, caps_gw, d, pc_mw, imports_firm)
    # available hydro = hourly upper bound x (available installed MW / installed MW on the 1-MW grid)
    k_eff = k * (inp['p']['hydro_operating_capacity_gw'] * 1000) / cp.h_inst
    lolp, eue = cp.hourly(nl, k_eff, dis)
    daily = lolp.reshape(-1, 24).max(1)
    return {'LOLE_days': float(daily.sum()), 'LOLH_h': float(lolp.sum()), 'EUE_MWh': float(eue.sum()),
            'peak_residual_mw': float(nl.max()), 'units': len(units)}, lolp


def pc_for_target(inp, caps_gw, d=1.0, unit_max=250.0, imports_firm=True, target=LOLE_TARGET):
    """Perfect capacity (MW, + added / - removed) that brings LOLE to the target (bisection)."""
    f = lambda pc: metrics_A(inp, caps_gw, d, pc, unit_max, imports_firm)[0]['LOLE_days']
    lo, hi = -20000.0, 20000.0
    if f(lo) <= target:
        return lo
    if f(hi) > target:
        return np.inf
    for _ in range(40):
        mid = (lo + hi) / 2
        if f(mid) > target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1.0:
            break
    return hi


# ----------------------------------------------------------------- method B
def outage_histories(units: pd.DataFrame, T: int, rng):
    """Two-state Markov chains, hourly steps; returns availability (units x T) in {0,1}."""
    q = units.efor.to_numpy(); r = units.mttr_h.to_numpy()
    mttf = r * (1 - q) / np.maximum(q, 1e-12)
    p_fail = np.where(q > 0, 1.0 / mttf, 0.0); p_rep = 1.0 / r
    up = np.empty((len(units), T), bool)
    s = rng.random(len(units)) >= q
    for t in range(T):
        up[:, t] = s
        u = rng.random(len(units))
        s = np.where(s, u >= p_fail, u < p_rep)
    return up


class SMC:
    """Sequential Monte Carlo with an energy-limited PyPSA dispatch (fixed capacities).

    Outage histories are drawn once per instance and reused for every evaluation (common random
    numbers), so results for different amounts of added perfect capacity are directly comparable."""

    def __init__(self, case: pp.Case, caps_gw, d=1.0, n_samples=200, seed=2026, unit_max=250.0, zero_for=False):
        self.inp = pp.lp_inputs(case)
        self.counted = {t: v * d for t, v in caps_gw.items()}
        self.c2 = pp.Case(case.name, case.year, case.scenario_config, netzero_mode=case.netzero_mode,
                          hydro_stress=case.hydro_stress, allow_shed=True, fixed_caps=self.counted)
        self.n = pp.build(self.c2, self.inp)
        # perfect capacity: always available, unlimited energy, dispatched only before load is shed
        self.n.add('Generator', 'perfect', bus='MG', p_nom=0.0, marginal_cost=0.5)
        self.units = fleet(self.inp, self.counted, unit_max)
        if zero_for:
            self.units['efor'] = 0.0
        self.base_hyd = self.n.generators_t.p_max_pu['hydro'].to_numpy().copy()
        rng = np.random.default_rng(seed)
        mw = self.units.mw.to_numpy()[:, None]
        self.av = []
        for _ in range(n_samples):
            up = outage_histories(self.units, 8760, rng)
            av = {}
            for k in ('hydro', 'gas', 'biogas', 'bess'):
                sel = (self.units.kind == k).to_numpy()
                av[k] = (mw[sel] * up[sel]).sum(0) / mw[sel].sum() if sel.any() else np.ones(8760)
            self.av.append(av)

    def run(self, perfect_mw=0.0, samples=None):
        n = self.n
        n.generators.at['perfect', 'p_nom'] = perfect_mw / 1000.0
        res = []
        for s in (range(len(self.av)) if samples is None else samples):
            av = self.av[s]
            n.generators_t.p_max_pu['hydro'] = self.base_hyd * av['hydro']
            n.generators_t.p_max_pu['gas'] = av['gas']
            n.generators_t.p_max_pu['biogas'] = 0.7 * av['biogas']
            n.storage_units_t.p_max_pu['bess'] = av['bess']
            n.storage_units_t.p_min_pu['bess'] = -av['bess']
            status, _ = n.optimize(solver_name='highs', extra_functionality=pp.extra(self.c2, self.inp),
                                   include_objective_constant=False, solver_options={'output_flag': False})
            assert status == 'ok', status
            shed = n.generators_t.p.filter(like='shed_').sum(axis=1).to_numpy()
            n._model = None                           # release the linopy model (memory) before the next sample
            gc.collect()
            lol = shed > 1e-3                         # an hour counts when more than 1 MW is unserved
            res.append({'sample': s, 'EUE_MWh': float(shed.sum() * 1000), 'LOLH_h': int(lol.sum()),
                        'LOL_days': int(lol.reshape(-1, 24).any(1).sum()), 'max_shed_mw': float(shed.max() * 1000),
                        'any_shed': bool(lol.any())})
        return pd.DataFrame(res)


def deterministic_eue(case: pp.Case, caps_gw, d=1.0):
    inp = pp.lp_inputs(case)
    c2 = pp.Case(case.name, case.year, case.scenario_config, netzero_mode=case.netzero_mode,
                 hydro_stress=case.hydro_stress, allow_shed=True, fixed_caps={t: v * d for t, v in caps_gw.items()})
    return pp.solve(c2)['shed_gwh'] * 1000


def firm_gap(smc: SMC, target=LOLE_TARGET, hi=12000.0, tol=100.0, samples=None):
    """Smallest perfect capacity (MW) with P(any unserved energy in the year) <= target. Since every year
    with a shortfall has at least one loss-of-load day, this bounds LOLE <= 0.1 day/yr from the safe side."""
    p = lambda x: smc.run(x, samples).any_shed.mean()
    if p(0.0) <= target:
        return 0.0
    if p(hi) > target:
        return np.inf
    lo = 0.0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if p(mid) > target:
            lo = mid
        else:
            hi = mid
    return hi


# ---------------------------------------------------------------------- main
def portfolio_caps(year):
    port = pd.read_csv(pp.MACRO / 'output' / f'REFRESH_{year}' / 'portfolio.csv').set_index('technology')
    return {t: max(0.0, float(v)) for t, v in port.new_surviving_capacity_gw.items()}


def deliverable_fraction():
    c = pd.read_csv(ROOT / 'hosting_capacity' / 'results' / 'portfolio_vs_hosting.csv')
    c = c[(c.siting == 'reference') & c.main & (c['mode'] == 'circle') & c.condition.isin(['HSN_caiso', 'SSN_caiso'])]
    return {int(y): float(min(1.0, g.alpha.min())) for y, g in c.groupby('year')}


PART = OUT / 'adequacy_parts'


def job_A(year):
    """Method A (COPT, ELCC) and its gate for one year; writes part files."""
    PART.mkdir(exist_ok=True)
    dfrac = deliverable_fraction()
    case = pp.refresh_cases()[0 if year == 2035 else 1]
    caps = portfolio_caps(year); inp = pp.lp_inputs(case); units = fleet(inp, caps)
    fl = units.groupby(['kind', 'cls']).agg(units=('mw', 'size'), mw=('mw', 'sum'), unit_mw_mean=('mw', 'mean'),
                                            efor=('efor', 'first'), mttr_h=('mttr_h', 'first')).reset_index().assign(year=year)
    A_rows, elcc_rows, gates = [], [], []
    for label, d, imp, umax in [('nameplate', 1.0, True, 250.0), ('deliverable only', dfrac[year], True, 250.0),
                                ('nameplate, unit max 125 MW', 1.0, True, 125.0),
                                ('nameplate, unit max 500 MW', 1.0, True, 500.0),
                                ('nameplate, no firm imports', 1.0, False, 250.0)]:
        if (label.endswith('no firm imports') and inp['imp_cap'] == 0) or (label == 'deliverable only' and d >= 1.0):
            continue
        m, _ = metrics_A(inp, caps, d, 0.0, umax, imp)
        pc0 = pc_for_target(inp, caps, d, umax, imp)
        A_rows.append({'year': year, 'counting': label, 'deliverable_fraction': d, **m,
                       'firm_capacity_gap_mw': pc0, 'peak_load_mw': float(inp['load'].max() * 1000)})
        print(year, label, {k: round(v, 3) for k, v in m.items()}, 'PC0', round(pc0), flush=True)
        if label in ('nameplate', 'deliverable only'):
            for cname, techs in CLASSES.items():
                nameplate = sum(caps.get(t, 0.0) for t in techs) * 1000
                if nameplate < 1e-3:
                    continue
                without = {t: (0.0 if t in techs else v) for t, v in caps.items()}
                elcc = pc_for_target(inp, without, d, umax, imp) - pc0
                elcc_rows.append({'year': year, 'counting': label, 'class': cname, 'nameplate_mw': nameplate,
                                  'counted_mw': nameplate * d, 'elcc_mw': elcc, 'elcc_pct_of_nameplate': 100 * elcc / nameplate})
    m0, _ = metrics_A(inp, caps, 1.0, 0.0, 250.0, True, zero_for=True)
    nl, k, dis = residual(inp, caps)
    det_short = int((nl - k * inp['p']['hydro_operating_capacity_gw'] * 1000
                     - (units[units.kind == 'gas'].mw.sum()) - 0.7 * units[units.kind == 'biogas'].mw.sum() - dis > 1.0).sum())
    gates.append({'gate': f'{year} COPT with zero EFOR = deterministic short hours', 'ok': abs(m0['LOLH_h'] - det_short) <= 1,
                  'detail': f"LOLH {m0['LOLH_h']:.0f} vs {det_short} short hours"})
    fl.to_csv(PART / f'fleet_{year}.csv', index=False)
    pd.DataFrame(A_rows).to_csv(PART / f'A_{year}.csv', index=False)
    pd.DataFrame(elcc_rows).to_csv(PART / f'elcc_{year}.csv', index=False)
    pd.DataFrame(gates).to_csv(PART / f'gatesA_{year}.csv', index=False)


def job_B(year, hydro, n_samples=100, gap_samples=50):
    """Method B (sequential Monte Carlo) and its gates for one year and hydrology; writes part files."""
    PART.mkdir(exist_ok=True)
    dfrac = deliverable_fraction()
    case = pp.refresh_cases()[0 if year == 2035 else 1]
    caps = portfolio_caps(year)
    hyd = hydro == 'dry'
    ch = pp.Case(case.name, year, case.scenario_config, netzero_mode=case.netzero_mode, hydro_stress=hyd)
    det = deterministic_eue(ch, caps)
    gates, B_rows = [], []
    if hyd:
        xc = pp.custom_lp_fixed(year, hydro_stress=True) * 1000
        gates.append({'gate': f'{year} dry no-outage EUE = custom LP fixed-case ENS', 'ok': abs(det - xc) < 1e-3 * max(1, xc),
                      'detail': f'PyPSA {det:.1f} MWh vs custom LP {xc:.1f} MWh'})
    else:
        gates.append({'gate': f'{year} normal no-outage dispatch has zero shed', 'ok': bool(det < 1e-3), 'detail': f'EUE {det:.4f} MWh'})
    z = SMC(ch, caps, 1.0, n_samples=2, zero_for=True).run(0.0)
    gates.append({'gate': f'{year} {hydro} SMC with zero EFOR = no-outage dispatch',
                  'ok': bool(np.allclose(z.EUE_MWh, det, rtol=1e-6, atol=1e-3)), 'detail': f'{z.EUE_MWh.iloc[0]:.1f} vs {det:.1f} MWh'})
    for label, d in [('nameplate', 1.0), ('deliverable only', dfrac[year])]:
        if d >= 1.0 and label == 'deliverable only':
            continue
        sm = SMC(ch, caps, d, n_samples=n_samples, seed=2026 + year + int(hyd))
        r = sm.run(0.0)
        se = lambda x: 1.96 * x.std(ddof=1) / np.sqrt(len(x))
        gap = firm_gap(sm, samples=range(min(gap_samples, n_samples)), tol=100.0)
        B_rows.append({'year': year, 'hydrology': 'dry analogue' if hyd else 'normal', 'counting': label,
                       'deliverable_fraction': d, 'samples': len(r), 'gap_samples': min(gap_samples, n_samples),
                       'no_outage_EUE_MWh': deterministic_eue(ch, caps, d),
                       'EUE_MWh': r.EUE_MWh.mean(), 'EUE_MWh_ci95': se(r.EUE_MWh),
                       'EUE_pct_of_load': 100 * r.EUE_MWh.mean() / (sm.inp['load'].sum() * 1000),
                       'p_any_shortfall': r.any_shed.mean(),
                       'LOLE_days_convention': r.LOL_days.mean(), 'LOLE_days_ci95': se(r.LOL_days),
                       'LOLH_h_convention': r.LOLH_h.mean(), 'LOLH_h_ci95': se(r.LOLH_h),
                       'max_shed_mw': r.max_shed_mw.max(), 'firm_capacity_gap_mw': gap})
        print(year, hydro, label, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in B_rows[-1].items()}, flush=True)
    pd.DataFrame(B_rows).to_csv(PART / f'B_{year}_{hydro}.csv', index=False)
    pd.DataFrame(gates).to_csv(PART / f'gatesB_{year}_{hydro}.csv', index=False)


def merge():
    cat = lambda pat: pd.concat([pd.read_csv(f) for f in sorted(PART.glob(pat))], ignore_index=True)
    cat('fleet_*.csv').to_csv(OUT / 'adequacy_fleet.csv', index=False, float_format='%.6f')
    cat('A_*.csv').sort_values(['year']).to_csv(OUT / 'adequacy_copt.csv', index=False, float_format='%.6f')
    cat('elcc_*.csv').to_csv(OUT / 'adequacy_elcc.csv', index=False, float_format='%.6f')
    B = cat('B_*.csv')
    B['h'] = B.hydrology.eq('dry analogue'); B = B.sort_values(['year', 'h', 'counting']).drop(columns='h')
    B.to_csv(OUT / 'adequacy_smc.csv', index=False, float_format='%.6f')
    g = cat('gates*.csv'); g.to_csv(OUT / 'adequacy_gates.csv', index=False)
    print(g.to_string(index=False))
    return 0 if g.ok.all() else 1


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[0] == 'A':
        job_A(int(a[1]))
    elif a[0] == 'B':
        job_B(int(a[1]), a[2], int(a[3]) if len(a) > 3 else 100, int(a[4]) if len(a) > 4 else 50)
    else:
        sys.exit(merge())
