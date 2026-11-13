#!/usr/bin/env python3
"""Deliverability of the fixed 2035/2050 portfolios on the existing MG main grid (v2).

Question (issue #3): given the LP portfolios, can the existing ≥230 kV network deliver them
under N-0 and N-1, and which reinforcements are attributable to them?

Method (DC screening, see hc_engine.py), following the CAISO policy-driven assessment flow
(ENERGY176/276 Lecture 11):
* Portfolios are fixed LP outputs. Siting is fixed: the nearest-eligible-station reference
  allocation; the regional/local MILP sitings and BESS-at-load are sensitivities.
* Study conditions:
  - off-peak deliverability: LP midday hours of peak new output (S1, after LP curtailment;
    S2, before it);
  - peak deliverability: LP evening hours of peak non-solar output (S3), and the CAISO HSN/SSN
    definitions (hours ending 19-22 at 20% exceedance with storage at 100%; hours ending 15-18
    at 50% exceedance with storage at 50%).
* Absorption (main case): all other online generation is scaled down pro rata (MG hydro and
  thermal plus boundary inflow equivalents), with the load at the target-year level (2050 load
  growth spread pro rata to existing MG load). A1 (MG only), A2 (boundary inflows only) and B
  (export at outflow boundaries) are sensitivities.
* Attribution: CAISO 5% circle. A facility row is tested only if at least one study unit has a
  DFAX >= 5% on it or a flow impact >= 5% of its rating. Pre-existing overloads inside a circle
  may not get worse. Sensitivity: the CAISO cap of at most 20 units and 1,500 MW per facility.
* Limits: N-0 normal rating, N-1 max(rate, emergency rating), LODF on 437 outages.
"""
from __future__ import annotations
import json, os, platform, sys
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')     # single-threaded BLAS: bit-stable results across runs
from pathlib import Path
import numpy as np
import pandas as pd
import scipy

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from hc_engine import Network, Rows, ROOT  # noqa: E402

NODAL = ROOT / 'MG_Step2_Congestion_Relief' / 'MG_Nodal_Connection_Optimization'
sys.path.insert(0, str(NODAL / 'code'))
import optimize_siting as ns  # noqa: E402  (deterministic candidate_table)

OUT = HERE / 'results'
OUT.mkdir(exist_ok=True)
YEARS = (2035, 2050)
MAIN = 'P_prorata'
CASES = ('P_prorata', 'A1_mg_backdown', 'A2_displace_inflow', 'B_export')
CIRCLE = 0.05              # CAISO 5% circle threshold (DFAX and flow impact / rating)
CAP_MW, CAP_UNITS = 1500.0, 20   # CAISO per-facility increase cap (sensitivity)
TOP_FRAC = 0.01            # top 1% of hours (88 h) define each LP study condition
SUBSET_LOADING = 0.90      # constraint subset: rows loaded >= 90% at full study output
SITED = ['solar_central', 'bio', 'small_hydro', 'wind_site', 'biogas', 'bess']
USED_COL = {'solar_central': 'used_new_solar_central_gw', 'solar_distributed': 'used_new_solar_distributed_gw',
            'bio': 'used_new_bio_gw', 'small_hydro': 'used_new_small_hydro_gw',
            'wind_site': 'used_new_wind_site_gw'}
OFFPEAK = ('S1_peak_delivery', 'S2_pre_curtailment')
PEAK = ('S3_evening_nonsolar', 'HSN_caiso', 'SSN_caiso')
COND_LABEL = {'S1_peak_delivery': 'off-peak S1: midday, LP-used output',
              'S2_pre_curtailment': 'off-peak S2: same hours, before LP curtailment',
              'S3_evening_nonsolar': 'peak S3: evening, BESS discharge + non-solar',
              'HSN_caiso': 'peak HSN: hours ending 19-22, 20% exceedance, storage 100%',
              'SSN_caiso': 'peak SSN: hours ending 15-18, 50% exceedance, storage 50%',
              'S1_plus_existing_pv_dg': 'S1 + existing PV/DG (sensitivity)'}


# ------------------------------------------------------------------ inputs
def portfolio_mw(year):
    p = pd.read_csv(NODAL / 'inputs' / f'portfolio_{year}.csv')
    return {r.technology: max(0.0, float(r.new_build_gw)) * 1000 for r in p.itertuples(index=False)}


def lp_factors(year, cap):
    """Per-technology output factors (fraction of new nameplate) for the LP study conditions."""
    d = pd.read_csv(NODAL / 'macro_reopt' / 'output' / f'REFRESH_{year}' / f'dispatch_{year}.csv.gz')
    frac = np.divide(d.vre_used_gw, d.available_nondispatchable_gw,
                     out=np.ones(len(d)), where=d.available_nondispatchable_gw > 1e-9)
    used = {t: d[c] * 1000 for t, c in USED_COL.items()}
    used['biogas'] = d.biogas_dispatch_gw * 1000
    used['bess'] = (d.discharge_gw - d.charge_gw) * 1000
    avail = {t: (used[t] / frac if t in USED_COL else used[t]) for t in used}
    total_new = sum(used.values())
    nonsolar = sum(used[t] for t in used if not t.startswith('solar'))
    k = int(round(TOP_FRAC * len(d)))
    conds = {'S1_peak_delivery': (np.argsort(-total_new.to_numpy())[:k], used),
             'S2_pre_curtailment': (np.argsort(-total_new.to_numpy())[:k], avail),
             'S3_evening_nonsolar': (np.argsort(-nonsolar.to_numpy())[:k], used)}
    rows = []
    for c, (hrs, src) in conds.items():
        for t in src:
            capt = cap.get(t, 0.0)
            rows.append({'year': year, 'condition': c, 'technology': t, 'new_capacity_mw': capt,
                         'factor': float(src[t].iloc[hrs].mean() / capt) if capt > 1e-6 else 0.0,
                         'hours': k, 'mean_hour_of_day': float(pd.to_datetime(d.hour_start_brt.iloc[hrs]).dt.hour.mean()),
                         'mean_lp_curtailment_gw': float(d.curtailed_nondispatchable_gw.iloc[hrs].mean())})
    ex = {'existing_central_pv_mw': float(d.used_central_pv_gw.iloc[conds['S1_peak_delivery'][0]].mean() * 1000),
          'existing_dg_mw': float(d.used_mmgd_gw.iloc[conds['S1_peak_delivery'][0]].mean() * 1000)}
    return pd.DataFrame(rows), ex


def availability_profiles():
    """Hourly availability per MW of new nameplate, exactly as the LP builds them."""
    M = NODAL / 'macro_reopt'
    f = pd.read_csv(M / 'macro' / 'processed' / 'macro_hourly_inputs.csv', parse_dates=['hour_start_brt'])
    wind = pd.read_csv(M / 'macro' / 'processed' / 'mg_wind_2019_brt_aligned.csv')['capacity_factor'].to_numpy(float)
    lo, hi = 0.0, 100.0            # optimize_macro_policy.py: scale the 2019 shape to a 0.30 mean CF
    for _ in range(50):
        m = (lo + hi) / 2
        if np.minimum(1, wind * m).mean() < 0.30:
            lo = m
        else:
            hi = m
    prof = pd.DataFrame({'solar_central': f.new_central_pv_cf, 'solar_distributed': f.new_dg_pv_cf,
                         'bio': f.new_bio_cf, 'small_hydro': f.new_small_hydro_cf,
                         'wind_site': np.minimum(1.0, wind * (lo + hi) / 2)})
    prof['hour'] = f.hour_start_brt.dt.hour
    return prof


def caiso_factors(year, cap, prof, biogas_cf=0.70):
    """CAISO HSN/SSN output levels (Lecture 11): wind/solar and other profile-based resources at
    the 20% (HSN, hours ending 19-22) or 50% (SSN, hours ending 15-18) exceedance value over all
    days; storage at 100% / 50% (the BESS fleet has >= 4 h duration); biogas at its availability."""
    rows = []
    for cond, hours_start, exceed, storage in [('HSN_caiso', (18, 19, 20, 21), 0.20, 1.0),
                                                ('SSN_caiso', (14, 15, 16, 17), 0.50, 0.5)]:
        sel = prof.hour.isin(hours_start)
        for t in ['solar_central', 'solar_distributed', 'bio', 'small_hydro', 'wind_site']:
            v = float(np.quantile(prof.loc[sel, t], 1 - exceed))     # value exceeded `exceed` of the time
            rows.append({'year': year, 'condition': cond, 'technology': t, 'new_capacity_mw': cap.get(t, 0.0),
                         'factor': v, 'hours': int(sel.sum()), 'mean_hour_of_day': float(np.mean(hours_start)),
                         'mean_lp_curtailment_gw': np.nan})
        rows.append({'year': year, 'condition': cond, 'technology': 'bess', 'new_capacity_mw': cap.get('bess', 0.0),
                     'factor': storage, 'hours': int(sel.sum()), 'mean_hour_of_day': float(np.mean(hours_start)),
                     'mean_lp_curtailment_gw': np.nan})
        rows.append({'year': year, 'condition': cond, 'technology': 'biogas', 'new_capacity_mw': cap.get('biogas', 0.0),
                     'factor': biogas_cf, 'hours': int(sel.sum()), 'mean_hour_of_day': float(np.mean(hours_start)),
                     'mean_lp_curtailment_gw': np.nan})
    return pd.DataFrame(rows)


def reference_siting(year, cap):
    """Nearest eligible station per cluster at the allocation-weight capacity (no optimisation)."""
    cand, cls = ns.candidate_table(year, {t: v / 1000 for t, v in cap.items()})
    rows = []
    for cl in cls.itertuples(index=False):
        j = np.flatnonzero(cand.cluster_id == cl.cluster_id)[0]
        r = cand.iloc[j]
        rows.append({'tech': r.tech, 'area': r.area, 'bus_id': int(r.bus_id), 'station': r.station,
                     'kv': int(r.kv), 'new_capacity_mw': float(cl.baseline_mw)})
    return pd.DataFrame(rows), cand


def milp_siting(year, scope):
    p = ROOT / 'regional_comparison' / 'results' / f'{year}_{scope}_local_consumption_chosen_sites.csv'
    s = pd.read_csv(p)
    return s[s.new_capacity_mw > 1e-6][['tech', 'area', 'bus_id', 'station', 'kv', 'new_capacity_mw']]


def coords():
    X = pd.read_csv(NODAL / 'inputs' / 'epe_ons_station_crosswalk.csv')
    X = X[X.pwf_bus_id.notna() & X.lat.notna() & X.lon.notna()]
    return {int(r.pwf_bus_id): (float(r.lat), float(r.lon)) for r in X.itertuples(index=False)}


# ---------------------------------------------------------------- analysis
def study_units(net, sites, fac, cap, bess_at_load=False):
    """Study units: one per sited (technology, area, bus) row, plus distributed PV (and BESS when
    placed at load) as load-weighted aggregates. Returns (unit patterns n x U, outputs P, metadata)."""
    cols, P, meta = [], [], []
    lw = net.load_weights()
    for r in sites.itertuples(index=False):
        if bess_at_load and r.tech == 'bess':
            continue
        e = np.zeros(net.n); e[net.idx[int(r.bus_id)]] = 1.0
        cols.append(e); P.append(r.new_capacity_mw * fac.get(r.tech, 0.0))
        meta.append({'tech': r.tech, 'area': r.area, 'bus_id': int(r.bus_id)})
    if bess_at_load and cap.get('bess', 0) > 0:
        cols.append(lw); P.append(cap['bess'] * fac.get('bess', 0.0))
        meta.append({'tech': 'bess', 'area': 'MG load buses', 'bus_id': -1})
    if cap.get('solar_distributed', 0) > 0:
        cols.append(lw); P.append(cap['solar_distributed'] * fac.get('solar_distributed', 0.0))
        meta.append({'tech': 'solar_distributed', 'area': 'MG load buses', 'bus_id': -1})
    return np.column_stack(cols), np.array(P, float), meta


def upgrade_groups(net):
    """Upgrade unit per monitored branch: a line corridor (bus pair) or a substation's
    transformation (connected component of the transformer graph)."""
    from hc_engine import STAR_KV
    nm = net.B.name.to_numpy()
    parent = list(range(net.n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for b in np.flatnonzero((net.L.kind == 'transformer').to_numpy()):
        parent[find(net.f[b])] = find(net.t[b])
    comp_label = {}
    for i in range(net.n):
        if net.kv[i] != STAR_KV and net.kv[i] >= 138:
            r = find(i)
            if r not in comp_label or net.kv[i] > comp_label[r][0]:
                comp_label[r] = (net.kv[i], nm[i][:6].rstrip('-'))
    key = {}
    for b in net.monitored:
        b = int(b)
        if net.L.kind.iloc[b] == 'line':
            a, c = sorted((int(net.L.f.iloc[b]), int(net.L.t.iloc[b])))
            key[b] = ('line', f'{nm[net.idx[a]]}-{nm[net.idx[c]]}')
        else:
            r = find(net.f[b])
            key[b] = ('transformation', comp_label.get(r, (0, nm[net.f[b]][:6]))[1] + ' substation')
    return key


def capped_g(D, P, member, cap_mw=CAP_MW, cap_units=CAP_UNITS):
    """Per-row flow change when only the highest-impact circle units, up to `cap_units` units and
    `cap_mw` MW, are raised to study output and all other new units stay at zero (a bound on the
    CAISO cap; CAISO adds a loading adder for the remainder)."""
    contrib = D * P[None, :]
    imp = np.abs(contrib) * member
    order = np.argsort(-imp, axis=1, kind='mergesort')
    sP = np.take_along_axis(np.abs(P)[None, :] * member, order, 1)
    prev = np.cumsum(sP, 1) - sP
    take = np.minimum(sP, np.clip(cap_mw - prev, 0, None))
    take[:, cap_units:] = 0.0
    frac = np.divide(take, sP, out=np.zeros_like(take), where=sP > 0)
    return (np.take_along_axis(contrib, order, 1) * frac).sum(1)


def assess(net, units, P, meta, case, groups, crd, growth_mw=0.0, mode='circle', max_upgrades=25):
    """Deliverability of the units' joint study output under one absorption case.

    mode: 'circle' (5% circle, main), 'all' (every row tested; equals PR #4's DFAX-0% rule),
    'circle_cap' (circle with the 20-unit / 1,500 MW cap). growth_mw: new MG load (pro rata to
    existing load) that absorbs injection first.
    """
    need = float(P.sum())
    w, headroom, _ = net.balancing(case)
    if need <= 1e-6:
        return {'need_mw': need, 'alpha': np.inf, 'deliverable_mw': 0.0}, [], None
    g_abs = min(growth_mw, need)
    sink = (g_abs * net.load_weights() + (need - g_abs) * w) / need
    headroom = headroom + g_abs
    r = net.rows(units @ P - sink * need)
    D = net.unit_dfax(units, sink)
    assert np.allclose(D @ P, r.g, atol=1e-6 * max(1.0, need))
    pre = np.abs(r.f0) > r.lim + 1e-6
    if mode == 'all':
        # PR #4 DFAX-0% rule: every row tested; pre-existing overloads may not get worse.
        in_circle = np.ones(len(r.f0), bool)
        active = in_circle
        member = np.broadcast_to(np.abs(P)[None, :] > 1.0, D.shape)
    else:
        in_circle, member = net.circle(D, P, r.lim, CIRCLE)
        # Pre-existing overloads are reliability needs of the base case (CAISO runs the
        # reliability assessment first and builds its fixes into the deliverability base case).
        # They are reported below as "worsened pre-existing", not counted in alpha.
        active = in_circle & ~pre
    if mode == 'circle_cap':
        r = Rows(r.f0, capped_g(D, P, member), r.lim, r.elem, r.cont)
    alpha, j = net.max_scale(r, 0.0, total=need, active=active)
    flow1 = r.f0 + r.g
    over_all = np.abs(flow1) > np.maximum(r.lim, np.abs(r.f0)) + 1e-6       # violated at scale 1, any row
    new = active & ~pre & (np.abs(flow1) > r.lim + 1e-6)
    worse = in_circle & pre & (np.abs(flow1) > np.abs(r.f0) + 1e-6)
    res = {'need_mw': need, 'alpha': alpha, 'energy_headroom_mw': headroom,
           'deliverable_mw': min(alpha * need, headroom),
           'energy_limited': bool(need > headroom + 1e-6),
           'rows_tested': int(active.sum()), 'rows_total': int(len(active)),
           'violated_rows_all': int(over_all.sum()),
           'violated_rows_excluded_by_circle': int((over_all & ~in_circle).sum()),
           'new_violation_rows': int(new.sum()), 'new_violation_elements': int(len(np.unique(r.elem[new]))),
           'worsened_preexisting_rows': int(worse.sum()),
           'worsened_preexisting_elements': int(len(np.unique(r.elem[worse]))),
           'worsened_preexisting_added_mw': float(np.max(np.abs(flow1[worse]) - np.abs(r.f0[worse]))) if worse.any() else 0.0,
           'worsened_preexisting_names': '; '.join(sorted({'{from_name}-{to_name}'.format(**net.branch_label(e))
                                                            for e in np.unique(r.elem[worse])})),
           'binding_element': '', 'binding_contingency': '', 'binding_dfax_max': np.nan,
           'n_upgrades_to_alpha1': 0}
    if j >= 0:
        e = net.branch_label(r.elem[j])
        res['binding_element'] = f"{e['from_name']}-{e['to_name']} ({e['kind']}, {e['kv']:.0f} kV)"
        res['binding_contingency'] = ('N-0' if r.cont[j] < 0 else
                                      '{from_name}-{to_name}'.format(**net.branch_label(r.cont[j])))
        res['binding_dfax_max'] = float(np.max(np.abs(D[j])))
    ups, relieved, a = [], set(), alpha
    while a < 1.0 and len(ups) < max_upgrades:
        a, jj = net.max_scale(r, 0.0, total=need, relieved=frozenset(relieved), active=active)
        if a >= 1.0 or jj < 0:
            break
        gkey = groups[int(r.elem[jj])]
        members = [b for b, k in groups.items() if k == gkey]
        m = np.isin(r.elem, members) & active
        flow_full = np.abs(r.f0[m] + r.g[m])
        over = flow_full - np.maximum(r.lim[m], np.abs(r.f0[m]))
        iw = int(np.argmax(np.round(over, 6)))
        e = int(r.elem[m][iw]); rowi = np.flatnonzero(m)[iw]
        lab = net.branch_label(e)
        circ_areas = sorted({meta[u]['area'] for u in np.flatnonzero(member[rowi])})
        km, basis = (net.line_length_km(e, crd) if gkey[0] == 'line' else (np.nan, 'n/a'))
        ups.append({'order': len(ups) + 1, 'upgrade_type': gkey[0], 'upgrade': gkey[1],
                    'n_branches_in_group': len(members), 'worst_branch': f"{lab['from_name']}-{lab['to_name']}",
                    'worst_branch_index': e, 'kv': lab['kv'], 'limit_mw': float(r.lim[m][iw]), 'alpha_before': a,
                    'max_flow_at_full_need_mw': float(flow_full[iw]),
                    'required_increase_mw': float(over[iw]),
                    'required_increase_pct': 100 * float(over[iw] / r.lim[m][iw]),
                    'worst_condition': 'N-0' if r.cont[m][iw] < 0 else 'N-1',
                    'worst_contingency': ('N-0' if r.cont[m][iw] < 0 else
                                          '{from_name}-{to_name}'.format(**net.branch_label(r.cont[m][iw]))),
                    'preexisting': bool(pre[rowi]),
                    'circle_areas': '; '.join(circ_areas), 'n_circle_areas': len(circ_areas),
                    'circle_breadth': f'{len(circ_areas)} area' + ('' if len(circ_areas) == 1 else 's'),
                    'length_km': km, 'length_basis': basis})
        relieved.update(members)
    if ups:
        res['n_upgrades_to_alpha1'] = len(ups)
        res['alpha_after_upgrades'] = net.max_scale(r, 0.0, total=need, relieved=frozenset(relieved), active=active)[0]
    res['status'] = ('energy-limited' if res['energy_limited'] else
                     'deliverable' if alpha >= 1.0 else 'needs upgrades')
    return res, ups, (r, active)


def poi_table(net, cands, cases=CASES):
    """Per-POI hosting capacity (NREL Oregon Finding 1 style): MW one POI can inject alone before
    the first attributable violation. A row is attributable when the POI's DFAX >= 5% or its flow
    impact at the row's own binding injection >= 5% of the rating."""
    pois = (cands.groupby('bus_id').agg(station=('station', 'first'), kv=('kv', 'first'),
                                        areas=('area', lambda a: ', '.join(sorted(set(a))))).reset_index())
    rows = []
    for p in pois.itertuples(index=False):
        e = np.zeros(net.n); e[net.idx[int(p.bus_id)]] = 1.0
        row = {'bus_id': int(p.bus_id), 'station': p.station, 'kv': int(p.kv), 'areas': p.areas}
        for case in cases:
            w, headroom, _ = net.balancing(case)
            for n1, suffix in ((True, ''), (False, '_n0')):
                r = net.rows(e - w, n1=n1)
                caps = Network.row_caps(r)
                pre = np.abs(r.f0) > r.lim + 1e-6
                act = ((np.abs(r.g) >= CIRCLE) |
                       (np.abs(r.g) * np.where(np.isfinite(caps), caps, 0) >= CIRCLE * r.lim)) & ~pre
                hc, j = net.max_scale(r, 0.0, total=1.0, active=act)
                row[f'hc_{case}{suffix}_mw'] = min(hc, headroom)
                if n1 and j >= 0:
                    lab = net.branch_label(r.elem[j])
                    row[f'binding_{case}'] = f"{lab['from_name']}-{lab['to_name']} ({lab['kind']})"
                    row[f'binding_{case}_contingency'] = ('N-0' if r.cont[j] < 0 else
                                                          '{from_name}-{to_name}'.format(**net.branch_label(r.cont[j])))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(f'hc_{MAIN}_mw', ascending=False)


def main():
    net = Network()
    crd = coords()
    base = net.rows(np.zeros(net.n))
    bviol = np.abs(base.f0) > base.lim + 1e-6
    base_rows = []
    for j in np.flatnonzero(bviol):
        e = net.branch_label(base.elem[j])
        base_rows.append({**e, 'contingency': 'N-0' if base.cont[j] < 0 else
                          '{from_name}-{to_name}'.format(**net.branch_label(base.cont[j])),
                          'flow_mw': abs(base.f0[j]), 'limit_mw': base.lim[j],
                          'overload_mw': abs(base.f0[j]) - base.lim[j]})
    pd.DataFrame(base_rows).to_csv(OUT / 'base_case_screen.csv', index=False, float_format='%.6f')

    checks, upgrades, facs, sub_rows = [], [], [], []
    poi = None
    groups = upgrade_groups(net)
    prof = availability_profiles()
    lp_load = {y: float(pd.read_csv(NODAL / 'macro_reopt' / 'output' / f'REFRESH_{y}' / 'annual_energy.csv').load_twh.iloc[0])
               for y in YEARS}
    pwf_mg_load = float(net.B.loc[(net.B.state == 11) & (net.B.pd > 0), 'pd'].sum())
    growth_2050 = (lp_load[2050] / lp_load[2035] - 1.0) * pwf_mg_load   # MW at PWF scale
    pairs = {}                      # (element, contingency) -> max loading, main mode, all cases
    for year in YEARS:
        cap = portfolio_mw(year)
        fac_df, existing = lp_factors(year, cap)
        fac_df = pd.concat([fac_df, caiso_factors(year, cap, prof)], ignore_index=True)
        facs.append(fac_df)
        ref_sites, cands = reference_siting(year, cap)
        ref_sites.assign(year=year).to_csv(OUT / f'reference_siting_{year}.csv', index=False, float_format='%.6f')
        sitings = {'reference': (ref_sites, False),
                   'reference_bess_at_load': (ref_sites, True),
                   'regional_12_milp': (milp_siting(year, 'Regional_12_Areas'), False),
                   'local_4_milp': (milp_siting(year, 'Local_4_North_MG'), False)}
        target_growth = growth_2050 if year == 2050 else 0.0
        variants = [(c, target_growth, 'target-year load') for c in CASES]
        if year == 2050:
            variants += [(c, 0.0, '2035 PWF load') for c in CASES]
        for cond in fac_df.condition.unique():
            fac = dict(fac_df[fac_df.condition == cond][['technology', 'factor']].itertuples(index=False))
            for sname, (sites, bess_load) in sitings.items():
                if bess_load and cap.get('bess', 0) <= 0:
                    continue
                units, P, meta = study_units(net, sites, fac, cap, bess_at_load=bess_load)
                for case, growth, load_level in variants:
                    modes = ['circle']
                    if sname == 'reference':
                        modes += ['all', 'circle_cap']
                    for mode in modes:
                        res, ups, rr = assess(net, units, P, meta, case, groups, crd, growth_mw=growth, mode=mode)
                        key = {'year': year, 'condition': cond, 'siting': sname, 'case': case,
                               'load_level': load_level, 'mode': mode,
                               'main': bool(case == MAIN and load_level == 'target-year load')}
                        checks.append({**key, **res})
                        if mode == 'circle' and sname in ('reference', 'reference_bess_at_load'):
                            upgrades += [{**key, **u} for u in ups]
                        if mode == 'circle' and rr is not None:
                            r, active = rr
                            ld = np.where(active, np.abs(r.f0 + r.g) / r.lim, 0.0)
                            for i in np.flatnonzero(ld >= SUBSET_LOADING):
                                kk = (int(r.elem[i]), int(r.cont[i]))
                                pairs[kk] = max(pairs.get(kk, 0.0), float(ld[i]))
                            # f0/lim/elem/cont are the same for every case; keep only g and the mask
                            sub_rows.append((key, P.sum(), r.g.copy(), active.copy(), res['alpha']))
        # Sensitivity: existing central PV (equal split over the PWF's UFV buses, an assumption)
        # and existing DG at load buses, added at the S1 hours.
        B = net.B
        ufv = np.flatnonzero(((B.state == 11) & (B.gen_type == 8)).to_numpy())
        fac = dict(fac_df[fac_df.condition == 'S1_peak_delivery'][['technology', 'factor']].itertuples(index=False))
        units, P, meta = study_units(net, ref_sites, fac, cap)
        e_pv = np.zeros(net.n); e_pv[ufv] = 1.0 / len(ufv)
        units = np.column_stack([units, e_pv, net.load_weights()])
        P = np.r_[P, existing['existing_central_pv_mw'], existing['existing_dg_mw']]
        meta = meta + [{'tech': 'existing_pv', 'area': 'existing UFV buses', 'bus_id': -1},
                       {'tech': 'existing_dg', 'area': 'MG load buses', 'bus_id': -1}]
        for case in CASES:
            res, _, _ = assess(net, units, P, meta, case, groups, crd, growth_mw=target_growth)
            checks.append({'year': year, 'condition': 'S1_plus_existing_pv_dg', 'siting': 'reference', 'case': case,
                           'load_level': 'target-year load', 'mode': 'circle', 'main': bool(case == MAIN),
                           **res, **existing})
        if year == 2050:
            poi = poi_table(net, cands)
            poi.to_csv(OUT / 'poi_hosting_capacity.csv', index=False, float_format='%.6f')

    # ---- constraint subset (for the production-cost run) and its validation ----
    P_sel = set(pairs)
    sub = pd.DataFrame([{'element': e, 'contingency': c, 'max_loading': v, 'group': groups[e][1],
                         **{k: net.branch_label(e)[k] for k in ('from_name', 'to_name', 'kind', 'kv')},
                         'contingency_name': 'N-0' if c < 0 else '{from_name}-{to_name}'.format(**net.branch_label(c))}
                        for (e, c), v in sorted(pairs.items())])
    sub.to_csv(OUT / 'critical_subset.csv', index=False, float_format='%.6f')

    b0 = net.rows(np.zeros(net.n))
    row_key = b0.elem.astype(np.int64) * 10000 + (b0.cont.astype(np.int64) + 1)

    def keys_of(selection):
        return np.array(sorted(e * 10000 + (c + 1) for e, c in selection), dtype=np.int64)

    def validate(selection, cases_):
        sel = np.isin(row_key, keys_of(selection))
        exact = differ = ver = 0
        for key, need, g, active, a_full in cases_:
            r = Rows(b0.f0, g, b0.lim, b0.elem, b0.cont)
            a_sub, _ = Network.max_scale(r, 0.0, total=need, active=active & sel)
            if a_full < 1:
                exact += abs(a_full - a_sub) < 1e-9
                differ += abs(a_full - a_sub) >= 1e-9
            ver += (a_full >= 1) == (a_sub >= 1)
        return exact, differ, ver
    n_lt = sum(1 for x in sub_rows if x[4] < 1)
    ex_in, df_in, ver_in = validate(P_sel, sub_rows)
    # out of sample: subset chosen from reference-siting cases only, validated on the other sitings
    pairs_ref = {}
    for key, need, g, active, a in sub_rows:
        if key['siting'] != 'reference':
            continue
        ld = np.where(active, np.abs(b0.f0 + g) / b0.lim, 0.0)
        for i in np.flatnonzero(ld >= SUBSET_LOADING):
            pairs_ref[(int(b0.elem[i]), int(b0.cont[i]))] = 1
    others = [x for x in sub_rows if x[0]['siting'] != 'reference']
    ex_out, df_out, ver_out = validate(set(pairs_ref), others)
    subset_info = {'pairs': len(P_sel), 'elements': int(sub.element.nunique()) if len(sub) else 0,
                   'groups': int(sub.group.nunique()) if len(sub) else 0,
                   'contingencies': int(sub.loc[sub.contingency >= 0, 'contingency'].nunique()) if len(sub) else 0,
                   'rows_full': int(len(b0.f0)), 'cases': len(sub_rows), 'cases_alpha_lt1': n_lt,
                   'in_sample_exact_alpha_lt1': int(ex_in), 'in_sample_differ_alpha_lt1': int(df_in),
                   'in_sample_same_verdict': int(ver_in),
                   'out_of_sample_pairs': len(pairs_ref), 'out_of_sample_cases': len(others),
                   'out_of_sample_cases_alpha_lt1': sum(1 for x in others if x[4] < 1),
                   'out_of_sample_exact_alpha_lt1': int(ex_out), 'out_of_sample_differ_alpha_lt1': int(df_out),
                   'out_of_sample_same_verdict': int(ver_out)}

    pd.concat(facs).to_csv(OUT / 'study_factors.csv', index=False, float_format='%.6f')
    chk = pd.DataFrame(checks)
    chk.to_csv(OUT / 'portfolio_vs_hosting.csv', index=False, float_format='%.6f')
    pd.DataFrame(upgrades).to_csv(OUT / 'required_upgrades.csv', index=False, float_format='%.6f')
    info = {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
            'scipy': scipy.__version__, 'monitored_branches': int(len(net.monitored)),
            'monitored_lines': int(net.n_lines), 'contingencies': int(len(net.contingencies)),
            'radial_outages_skipped': int(len(net.bridges)), 'circle_threshold': CIRCLE,
            'cap_mw': CAP_MW, 'cap_units': CAP_UNITS, 'top_hours_fraction': TOP_FRAC,
            'lp_load_twh': lp_load, 'pwf_mg_load_mw': pwf_mg_load, 'load_growth_2050_mw_at_pwf_scale': growth_2050,
            'main_case': MAIN, 'balancing_cases': {c: net.balancing(c)[2] for c in CASES},
            'energy_headroom_mw': {c: net.balancing(c)[1] for c in CASES}, 'subset': subset_info}
    (OUT / 'run_info.json').write_text(json.dumps(info, indent=2, default=float))
    write_results_md(chk, pd.DataFrame(upgrades), poi, pd.DataFrame(base_rows), pd.concat(facs), info)
    view = chk[(chk.siting == 'reference') & (chk['mode'] == 'circle') & chk.main]
    print(view[['year', 'condition', 'case', 'need_mw', 'alpha', 'deliverable_mw', 'n_upgrades_to_alpha1',
                'status', 'binding_element']].round(2).to_string())
    print(json.dumps(subset_info, indent=1))


# ------------------------------------------------------------------ report
def md(df, fmt=None):
    fmt = fmt or {}
    cols = list(df.columns)
    head = [str(c) for c in cols]
    # Right-align numeric columns only (pandas 3 stores text as 'str', not object).
    out = ['| ' + ' | '.join(head) + ' |',
           '|' + '|'.join('---:' if pd.api.types.is_numeric_dtype(df[c]) else '---' for c in cols) + '|']
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append('—' if not np.isfinite(v) else fmt.get(c, '{:,.0f}').format(v))
            else:
                cells.append(str(v))
        out.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(out)


def write_results_md(chk, ups, poi, base, facs, info):
    h = info['energy_headroom_mw']
    ref = chk[(chk.siting == 'reference') & (chk['mode'] == 'circle') & (chk.condition != 'S1_plus_existing_pv_dg')]
    main = ref[ref.main].copy()
    t = main[['year', 'condition', 'need_mw', 'energy_headroom_mw', 'alpha', 'deliverable_mw', 'status',
              'n_upgrades_to_alpha1', 'new_violation_elements', 'worsened_preexisting_elements',
              'violated_rows_excluded_by_circle', 'binding_element', 'binding_contingency']].copy()
    t['condition'] = t.condition.map(COND_LABEL); t['year'] = t['year'].astype(str)
    t.columns = ['year', 'condition', 'need MW', 'energy headroom MW ↑', 'α ↑', 'deliverable MW ↑', 'status',
                 'upgrade groups ↓', 'new violations (elements) ↓', 'worsened pre-existing (elements) ↓',
                 'violated rows outside the circle', 'first binding element', 'under']
    ab = ref.copy()
    ab['variant'] = ab.case + np.where(ab.load_level == '2035 PWF load', ' (2035 load)', '')
    ab = ab.pivot_table(index=['year', 'condition'], columns='variant', values='alpha').reset_index()
    ab['year'] = ab['year'].astype(str); ab['condition'] = ab.condition.map(COND_LABEL)
    rule = chk[(chk.siting == 'reference') & chk.main & (chk.condition != 'S1_plus_existing_pv_dg')]
    rule = rule.pivot_table(index=['year', 'condition'], columns='mode', values='alpha').reset_index()
    rule = rule.rename(columns={'all': 'every row tested', 'circle': '5% circle (main)',
                                'circle_cap': '5% circle + 20-unit/1,500 MW cap'})
    rule['year'] = rule['year'].astype(str); rule['condition'] = rule.condition.map(COND_LABEL)
    sens = chk[(chk['mode'] == 'circle') & chk.main & (chk.condition != 'S1_plus_existing_pv_dg')]
    sens = sens.pivot_table(index=['year', 'condition'], columns='siting', values='alpha').reset_index()
    sens['year'] = sens['year'].astype(str); sens['condition'] = sens.condition.map(COND_LABEL)
    u = ups[(ups.siting == 'reference') & ups.main] if len(ups) else ups
    if len(u):
        u = u[['year', 'condition', 'order', 'upgrade', 'kv', 'limit_mw', 'required_increase_mw',
               'required_increase_pct', 'worst_condition', 'preexisting', 'circle_breadth', 'circle_areas', 'length_km']].copy()
        u['year'] = u['year'].astype(str); u['condition'] = u.condition.map(COND_LABEL)
        u['preexisting'] = np.where(u.preexisting, 'worsened pre-existing', 'new')
        u.columns = ['year', 'condition', '#', 'upgrade', 'kV', 'limit MW', 'needed +MW ↓', 'needed +% ↓',
                     'worst', 'violation type', 'circle breadth', 'areas whose units are in the circle', 'length km']
    ex = chk[chk.condition == 'S1_plus_existing_pv_dg'][['year', 'case', 'need_mw', 'alpha', 'deliverable_mw',
                                                          'status', 'n_upgrades_to_alpha1']].copy()
    ex['year'] = ex['year'].astype(str)
    ex.columns = ['year', 'absorption', 'need MW', 'α ↑', 'deliverable MW ↑', 'status', 'upgrade groups ↓']
    fac = facs.pivot_table(index=['year', 'technology'], columns='condition', values='factor').reset_index()
    fac['year'] = fac['year'].astype(str)
    p = poi[['station', 'kv', 'areas', f'hc_{MAIN}_mw', 'hc_A1_mg_backdown_mw', 'hc_A2_displace_inflow_mw',
             'hc_B_export_mw', f'hc_{MAIN}_n0_mw', f'binding_{MAIN}', f'binding_{MAIN}_contingency']].copy()
    p['kv'] = p['kv'].astype(str)
    p.columns = ['POI', 'kV', 'candidate areas', 'pro rata MW ↑', 'A1 MW ↑', 'A2 MW ↑', 'B MW ↑',
                 'pro rata N-0 only MW ↑', 'pro-rata binding element', 'under']
    s = info['subset']
    lines = [
        '# Deliverability of the fixed portfolios on the existing MG main grid',
        '',
        f"Generated by `hosting_capacity/run_hosting.py` (Python {info['python']}, pandas {info['pandas']}, "
        f"numpy {info['numpy']}, scipy {info['scipy']}). Do not edit by hand. Question and plan: issue #3.",
        '',
        '↑ = higher is better, ↓ = lower is better; columns without an arrow are sizes or labels. '
        'α = largest scaling of the portfolio injection pattern that meets every attributable N-0 and N-1 limit '
        '(α ≥ 1: deliverable as sited). Deliverable MW = min(α × need, energy headroom).',
        '',
        '## Setup',
        '',
        f"- Network: EPE 2035 PWF, MG-centred equivalent (1,489 buses, 1,890 branches). Incremental DC flows anchored on "
        f"the AC P flows; LODF for N-1 (`check_engine.py`, `check_pypsa.py`). Scope: main grid only — "
        f"{info['monitored_branches']} monitored branches = {info['monitored_lines']} lines ≥230 kV + "
        f"{info['monitored_branches'] - info['monitored_lines']} network-transformer branches; {info['contingencies']} "
        f"contingencies ({info['radial_outages_skipped']} radial outages skipped). Limits: normal rating (N-0), "
        f"max(normal, emergency) (N-1).",
        f"- Base case without new resources: {int((base.contingency == 'N-0').sum()) if len(base) else 0} N-0 and "
        f"{int((base.contingency != 'N-0').sum()) if len(base) else 0} N-1 violations"
        + (': ' + '; '.join(f"{r.from_name}-{r.to_name} +{r.overload_mw:.1f} MW for loss of {r.contingency}"
                            for r in base.itertuples()) if len(base) else '') + '.',
        f"- Absorption, main case (CAISO convention): **{info['balancing_cases']['P_prorata']}**, headroom "
        f"{h['P_prorata']:,.0f} MW, with the load at the target-year level (2050: +{info['load_growth_2050_mw_at_pwf_scale']:,.0f} MW "
        f"at PWF scale, pro rata to existing MG load). Sensitivities: A1 {info['balancing_cases']['A1_mg_backdown']} "
        f"({h['A1_mg_backdown']:,.0f} MW); A2 {info['balancing_cases']['A2_displace_inflow']} ({h['A2_displace_inflow']:,.0f} MW); "
        f"B {info['balancing_cases']['B_export']} (network-limited only; it scales every existing outflow tie in proportion).",
        f"- Attribution: CAISO 5% circle — a facility row is tested only if a study unit (one per sited technology and area; "
        f"distributed PV as a load-weighted aggregate) has DFAX ≥ {info['circle_threshold']:.0%} on it or a flow impact at "
        f"study output ≥ {info['circle_threshold']:.0%} of its rating. Pre-existing overloads inside a circle may not get worse. "
        f"Sensitivity: at most {info['cap_units']} units and {info['cap_mw']:,.0f} MW raised per facility (others at zero, a bound).",
        f"- Study conditions: off-peak deliverability = LP midday hours (S1, S2; top {info['top_hours_fraction']:.0%} of hours); "
        f"peak deliverability = LP evening hours (S3) and the CAISO HSN/SSN definitions (exceedance values over all days).",
        '',
        '## Study factors (fraction of new nameplate)',
        '',
        md(fac, {c: '{:.3f}' for c in fac.columns}),
        '',
        '## Main result: portfolio need vs deliverability (reference siting, pro-rata absorption, target-year load)',
        '',
        md(t, {'α ↑': '{:.2f}'}),
        '',
        '## Attributable reinforcements to reach α = 1 (main case, reference siting)',
        '',
        'Greedy: relieve the first binding upgrade group (line corridor or substation transformation), re-solve, repeat. '
        'Ratings are relaxed only; impedances are unchanged. Costs and alternatives are in the mitigation phase. '
        '"circle breadth" counts the candidate areas with a unit in the facility\'s 5% circle (a descriptor, not a distance: '
        'a GW-scale unit can be in the circle of a distant facility).',
        '',
        md(u, {'needed +% ↓': '{:.1f}', 'length km': '{:.0f}'}) if len(u) else '_None needed._',
        '',
        '## Sensitivity: absorption case (α ↑, reference siting, 5% circle)',
        '',
        md(ab, {c: '{:.2f}' for c in ab.columns}),
        '',
        '## Sensitivity: attribution rule (α ↑, main case, reference siting)',
        '',
        md(rule, {c: '{:.2f}' for c in rule.columns}),
        '',
        '## Sensitivity: siting (α ↑, main case, 5% circle)',
        '',
        md(sens, {c: '{:.2f}' for c in sens.columns}),
        '',
        '## Sensitivity: midday with existing PV and DG added',
        '',
        'The PWF snapshot has all 106 existing PV buses at 0 MW (an evening case). Here the LP\'s existing central PV at '
        'the S1 hours is split equally across those buses (an assumption: plant sizes are not in the data) and existing '
        'DG is added at load buses.',
        '',
        md(ex, {'α ↑': '{:.2f}'}),
        '',
        '## Per-POI hosting capacity (2050 candidate stations, N-0 + N-1, injected MW, 5% circle)',
        '',
        md(p),
        '',
        '## Constraint subset for the production-cost run',
        '',
        f"Rows (element, contingency) loaded ≥ {SUBSET_LOADING:.0%} of their limit at full study output in any of the "
        f"{s['cases']} circle-mode cases: {s['pairs']:,} of {s['rows_full']:,} rows ({100 * s['pairs'] / s['rows_full']:.1f}%), "
        f"{s['elements']} elements in {s['groups']} groups. In-sample: α identical in {s['in_sample_exact_alpha_lt1']} of "
        f"{s['cases_alpha_lt1']} cases with α < 1 and the same deliverable/not verdict in {s['in_sample_same_verdict']} of "
        f"{s['cases']} cases. Out of sample (subset from the reference siting only, {s['out_of_sample_pairs']:,} rows, tested on "
        f"the other {s['out_of_sample_cases']} cases): α identical in {s['out_of_sample_exact_alpha_lt1']} of "
        f"{s['out_of_sample_cases_alpha_lt1']} cases with α < 1, same verdict in {s['out_of_sample_same_verdict']} of "
        f"{s['out_of_sample_cases']}. Any solution built on the subset is re-checked on all rows.",
        '',
        '## Not covered here',
        '',
        '- **No midday base case.** The PWF is an evening snapshot (existing PV at 0 MW). The main tables add only the NEW '
        'resources to it; midday conclusions are conditional on the base case (see the existing-PV sensitivity).',
        '- 138 kV and below are out of scope (main grid only). AC voltage, stability, N-2/P7 and the annual curtailment '
        'criterion are evaluated in later phases. The 2050 topology is the frozen 2035 PWF; one LP weather year.',
        '- Upgrades relax ratings only; real reinforcements change impedances.',
    ]
    (HERE / 'RESULTS.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
