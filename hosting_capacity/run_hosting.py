#!/usr/bin/env python3
"""Hosting capacity of the existing (2035 PWF) MG grid vs the fixed 2035/2050 portfolios.

Question (issue #3): given the LP portfolios, can the existing network deliver them under
N-0 and N-1, and if not, are the needed fixes minor or major?

Method (DC screening, see hc_engine.py):
* Portfolios are fixed LP outputs. Siting is fixed: the nearest-eligible-station reference
  allocation (independent of any congestion convention); the regional/local MILP sitings
  are sensitivities.
* Study conditions come from the LP 8,760-h dispatch (CAISO HSN/SSN-style factors), using
  the output the LP actually USES after its own curtailment.
* The injected MW must be absorbed somewhere. Three explicit cases replace the old
  load-proportional convention: A1 MG hydro/thermal back-down, A2 displaced boundary
  inflows (incl. the Estreito HVDC infeed), B export to the rest of the SIN.
* Limits: N-0 normal rating, N-1 max(rate, emergency rating), LODF on 437 outages.
  Pre-existing violations only count if worsened with distribution factor >= DFAX_CUT.
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
from hc_engine import Network, ROOT  # noqa: E402

NODAL = ROOT / 'MG_Step2_Congestion_Relief' / 'MG_Nodal_Connection_Optimization'
sys.path.insert(0, str(NODAL / 'code'))
import optimize_siting as ns  # noqa: E402  (deterministic candidate_table)

OUT = HERE / 'results'
OUT.mkdir(exist_ok=True)
YEARS = (2035, 2050)
CASES = ('A1_mg_backdown', 'A2_displace_inflow', 'B_export')
DFAX_CUT = 0.05            # provisional (issue #3); 0.0 reported as a strict comparison
TOP_FRAC = 0.01            # top 1% of hours (88 h) define each study condition
MINOR_LINE_KM = 50.0       # provisional: a new parallel circuit up to this length is "minor"
MINOR_UPRATE = 0.25        # provisional: a line needing <= +25% of its rating is an uprate/reconductor
MAX_MINOR_ELEMENTS = 10    # provisional: more upgrade groups than this is "major" regardless
SITED = ['solar_central', 'bio', 'small_hydro', 'wind_site', 'biogas', 'bess']
USED_COL = {'solar_central': 'used_new_solar_central_gw', 'solar_distributed': 'used_new_solar_distributed_gw',
            'bio': 'used_new_bio_gw', 'small_hydro': 'used_new_small_hydro_gw',
            'wind_site': 'used_new_wind_site_gw'}


# ------------------------------------------------------------------ inputs
def portfolio_mw(year):
    p = pd.read_csv(NODAL / 'inputs' / f'portfolio_{year}.csv')
    return {r.technology: max(0.0, float(r.new_build_gw)) * 1000 for r in p.itertuples(index=False)}


def lp_factors(year, cap):
    """Per-technology output factors (fraction of new nameplate) for three study conditions."""
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
    # Existing PV/DG output at the same hours (used only in the existing-PV sensitivity).
    ex = {'existing_central_pv_mw': float(d.used_central_pv_gw.iloc[conds['S1_peak_delivery'][0]].mean() * 1000),
          'existing_dg_mw': float(d.used_mmgd_gw.iloc[conds['S1_peak_delivery'][0]].mean() * 1000)}
    return pd.DataFrame(rows), ex


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
def injection(net, sites, fac, cap, extra=None, bess_at_load=False):
    v = np.zeros(net.n)
    lw = net.load_weights()
    for r in sites.itertuples(index=False):
        if bess_at_load and r.tech == 'bess':
            continue
        v[net.idx[int(r.bus_id)]] += r.new_capacity_mw * fac.get(r.tech, 0.0)
    if bess_at_load:
        v += lw * cap.get('bess', 0.0) * fac.get('bess', 0.0)
    v += lw * cap.get('solar_distributed', 0.0) * fac.get('solar_distributed', 0.0)
    if extra is not None:
        v += extra
    return v


def upgrade_groups(net):
    """Upgrade unit per monitored branch: a line corridor (bus pair) or a substation's
    transformation (all transformer branches whose real-kV end shares the station prefix)."""
    from hc_engine import STAR_KV
    nm = net.B.name.to_numpy()
    # Substations = connected components of the transformer graph (real + star buses).
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


def assess(net, inj, case, dfax, crd, groups, growth_mw=0.0, max_upgrades=25):
    """Deliverability of an injection pattern under one absorption case.

    growth_mw: MW of new MG load (at load buses, pro rata) that absorbs injection first
    (2050 load-growth sensitivity); the remainder goes to `case`.
    """
    need = float(inj.sum())
    w, headroom, _ = net.balancing(case)
    if need <= 1e-6:
        return {'need_mw': need, 'alpha': np.inf, 'deliverable_mw': 0.0}, []
    g_abs = min(growth_mw, need)
    w = (g_abs * net.load_weights() + (need - g_abs) * w) / need
    headroom = headroom + g_abs
    dp = inj - w * need
    r = net.rows(dp)
    alpha, j = net.max_scale(r, dfax, total=need)
    res = {'need_mw': need, 'alpha': alpha, 'energy_headroom_mw': headroom,
           'deliverable_mw': min(alpha * need, headroom),
           'binding_element': '', 'binding_contingency': '', 'n_upgrades_to_alpha1': 0}
    if j >= 0:
        e = net.branch_label(r.elem[j])
        res['binding_element'] = f"{e['from_name']}-{e['to_name']} ({e['kind']}, {e['kv']:.0f} kV)"
        if r.cont[j] >= 0:
            c = net.branch_label(r.cont[j])
            res['binding_contingency'] = f"{c['from_name']}-{c['to_name']}"
        else:
            res['binding_contingency'] = 'N-0'
    ups = []
    relieved = set()
    a = alpha
    while a < 1.0 and len(ups) < max_upgrades:
        a, j = net.max_scale(r, dfax, total=need, relieved=frozenset(relieved))
        if a >= 1.0 or j < 0:
            break
        gkey = groups[int(r.elem[j])]
        members = [b for b, k in groups.items() if k == gkey]
        m = np.isin(r.elem, members)
        flow_full = np.abs(r.f0[m] + r.g[m])               # at the full need (scale 1)
        over = flow_full - np.maximum(r.lim[m], np.abs(r.f0[m]))
        iw = int(np.argmax(np.round(over, 6)))
        e = int(r.elem[m][iw])                             # worst member
        lab = net.branch_label(e)
        inc_pct = float(over[iw] / r.lim[m][iw])
        km, basis = (net.line_length_km(e, crd) if gkey[0] == 'line' else (np.nan, 'n/a'))
        if gkey[0] == 'line':
            klass = ('minor: uprate / reconductor in existing corridor' if inc_pct <= MINOR_UPRATE else
                     'unclassified (length unknown)' if not np.isfinite(km) else
                     'minor: short parallel circuit in existing corridor' if km <= MINOR_LINE_KM else
                     'major: new long circuit')
        else:
            klass = 'minor: add transformer bank at existing substation'
        ups.append({'order': len(ups) + 1, 'upgrade_type': gkey[0], 'upgrade': gkey[1],
                    'n_branches_in_group': len(members), 'worst_branch': f"{lab['from_name']}-{lab['to_name']}",
                    'kv': lab['kv'], 'limit_mw': float(r.lim[m][iw]), 'alpha_before': a,
                    'max_flow_at_full_need_mw': float(flow_full[iw]),
                    'required_increase_mw': float(over[iw]), 'required_increase_pct': 100 * inc_pct,
                    'worst_condition': 'N-0' if r.cont[m][iw] < 0 else 'N-1',
                    'length_km': km, 'length_basis': basis, 'classification': klass})
        relieved.update(members)
    if ups:
        res['n_upgrades_to_alpha1'] = len(ups)
        res['alpha_after_upgrades'] = net.max_scale(r, dfax, total=need, relieved=frozenset(relieved))[0]
    # Verdict (thresholds provisional, issue #3).
    if need > headroom + 1e-6:
        verdict = 'energy-limited: cannot be absorbed this way'
    elif alpha >= 1.0:
        verdict = 'no action'
    elif ups and res.get('alpha_after_upgrades', 0) >= 1.0 and len(ups) <= MAX_MINOR_ELEMENTS and \
            all(u['classification'].startswith('minor') for u in ups):
        verdict = 'minor'
    else:
        verdict = 'major'
    res['verdict'] = verdict
    return res, ups


def poi_table(net, cands, dfax):
    pois = (cands.groupby('bus_id').agg(station=('station', 'first'), kv=('kv', 'first'),
                                        areas=('area', lambda a: ', '.join(sorted(set(a))))).reset_index())
    rows = []
    for p in pois.itertuples(index=False):
        e = np.zeros(net.n); e[net.idx[int(p.bus_id)]] = 1.0
        row = {'bus_id': int(p.bus_id), 'station': p.station, 'kv': int(p.kv), 'areas': p.areas}
        for case in CASES:
            w, headroom, _ = net.balancing(case)
            r = net.rows(e - w)
            hc, j = net.max_scale(r, dfax, total=1.0)
            row[f'hc_{case}_mw'] = min(hc, headroom)
            hc0, _ = net.max_scale(net.rows(e - w, n1=False), dfax, total=1.0)
            row[f'hc_{case}_n0_mw'] = min(hc0, headroom)
            if case == 'B_export' and j >= 0:
                lab = net.branch_label(r.elem[j])
                row['binding_B'] = f"{lab['from_name']}-{lab['to_name']} ({lab['kind']})"
                row['binding_B_contingency'] = ('N-0' if r.cont[j] < 0 else
                                                '{from_name}-{to_name}'.format(**net.branch_label(r.cont[j])))
        rows.append(row)
    return pd.DataFrame(rows).sort_values('hc_B_export_mw', ascending=False)


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

    checks, upgrades, facs = [], [], []
    poi = None
    groups = upgrade_groups(net)
    lp_load = {y: float(pd.read_csv(NODAL / 'macro_reopt' / 'output' / f'REFRESH_{y}' / 'annual_energy.csv').load_twh.iloc[0])
               for y in YEARS}
    pwf_mg_load = float(net.B.loc[(net.B.state == 11) & (net.B.pd > 0), 'pd'].sum())
    growth_2050 = (lp_load[2050] / lp_load[2035] - 1.0) * pwf_mg_load   # MW at PWF scale
    for year in YEARS:
        cap = portfolio_mw(year)
        fac_df, existing = lp_factors(year, cap)
        facs.append(fac_df)
        ref_sites, cands = reference_siting(year, cap)
        ref_sites.assign(year=year).to_csv(OUT / f'reference_siting_{year}.csv', index=False, float_format='%.6f')
        sitings = {'reference': (ref_sites, False),
                   'reference_bess_at_load': (ref_sites, True),
                   'regional_12_milp': (milp_siting(year, 'Regional_12_Areas'), False),
                   'local_4_milp': (milp_siting(year, 'Local_4_North_MG'), False)}
        variants = [(c, 0.0) for c in CASES] + ([(c + '+load2050', growth_2050) for c in CASES] if year == 2050 else [])
        for cond in fac_df.condition.unique():
            fac = dict(fac_df[fac_df.condition == cond][['technology', 'factor']].itertuples(index=False))
            for sname, (sites, bess_load) in sitings.items():
                if bess_load and cap.get('bess', 0) <= 0:
                    continue
                for vname, growth in variants:
                    case = vname.split('+')[0]
                    for dfax in ((DFAX_CUT, 0.0) if sname == 'reference' and growth == 0 else (DFAX_CUT,)):
                        res, ups = assess(net, injection(net, sites, fac, cap, bess_at_load=bess_load), case,
                                          dfax, crd, groups, growth_mw=growth)
                        key = {'year': year, 'condition': cond, 'siting': sname, 'case': vname, 'dfax_cut': dfax}
                        checks.append({**key, **res})
                        if dfax == DFAX_CUT and sname in ('reference', 'reference_bess_at_load'):
                            upgrades += [{**key, **u} for u in ups]
        # Sensitivity: existing central PV (equal split over the PWF's UFV buses, an assumption)
        # and existing DG at load buses, added at the S1 hours.
        B = net.B
        ufv = np.flatnonzero(((B.state == 11) & (B.gen_type == 8)).to_numpy())
        extra = np.zeros(net.n)
        extra[ufv] += existing['existing_central_pv_mw'] / len(ufv)
        extra += net.load_weights() * existing['existing_dg_mw']
        fac = dict(fac_df[fac_df.condition == 'S1_peak_delivery'][['technology', 'factor']].itertuples(index=False))
        for case in CASES:
            res, _ = assess(net, injection(net, ref_sites, fac, cap, extra), case, DFAX_CUT, crd, groups)
            checks.append({'year': year, 'condition': 'S1_plus_existing_pv_dg', 'siting': 'reference',
                           'case': case, 'dfax_cut': DFAX_CUT, **res, **existing})
        if year == 2050:
            poi = poi_table(net, cands, DFAX_CUT)
            poi.to_csv(OUT / 'poi_hosting_capacity.csv', index=False, float_format='%.6f')
    pd.concat(facs).to_csv(OUT / 'lp_study_factors.csv', index=False, float_format='%.6f')
    chk = pd.DataFrame(checks)
    chk.to_csv(OUT / 'portfolio_vs_hosting.csv', index=False, float_format='%.6f')
    pd.DataFrame(upgrades).to_csv(OUT / 'required_upgrades.csv', index=False, float_format='%.6f')
    info = {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
            'scipy': scipy.__version__, 'monitored_branches': int(len(net.monitored)),
            'monitored_lines': int(net.n_lines), 'contingencies': int(len(net.contingencies)),
            'radial_outages_skipped': int(len(net.bridges)), 'dfax_cut': DFAX_CUT, 'top_hours_fraction': TOP_FRAC,
            'minor_line_km': MINOR_LINE_KM, 'minor_uprate_fraction': MINOR_UPRATE,
            'max_minor_upgrade_groups': MAX_MINOR_ELEMENTS,
            'lp_load_twh': lp_load, 'pwf_mg_load_mw': pwf_mg_load, 'load_growth_2050_mw_at_pwf_scale': growth_2050,
            'balancing_cases': {c: net.balancing(c)[2] for c in CASES},
            'energy_headroom_mw': {c: net.balancing(c)[1] for c in CASES}}
    (OUT / 'run_info.json').write_text(json.dumps(info, indent=2, default=float))
    write_results_md(chk, pd.DataFrame(upgrades), poi, pd.DataFrame(base_rows), pd.concat(facs), info)
    print(chk[(chk.siting == 'reference') & (chk.dfax_cut == DFAX_CUT)][
        ['year', 'condition', 'case', 'need_mw', 'alpha', 'deliverable_mw', 'n_upgrades_to_alpha1', 'verdict',
         'binding_element']].round(2).to_string())


# ------------------------------------------------------------------ report
def md(df, fmt=None):
    fmt = fmt or {}
    cols = list(df.columns)
    head = [f'DFAX {c:.0%}' if isinstance(c, float) else str(c) for c in cols]
    out = ['| ' + ' | '.join(head) + ' |', '|' + '|'.join('---' if df[c].dtype == object else '---:' for c in cols) + '|']
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append('—' if not np.isfinite(v) else fmt.get(c, '{:,.0f}').format(v))
            else:
                cells.append(str(v))
        out.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(out)


COND_LABEL = {'S1_peak_delivery': 'S1 midday peak delivery (LP-used output)',
              'S2_pre_curtailment': 'S2 same hours, before LP curtailment',
              'S3_evening_nonsolar': 'S3 evening, BESS discharge + non-solar',
              'S1_plus_existing_pv_dg': 'S1 + existing PV/DG (sensitivity)'}


def write_results_md(chk, ups, poi, base, facs, info):
    ref = chk[(chk.siting == 'reference') & (chk.dfax_cut == DFAX_CUT) & (chk.condition != 'S1_plus_existing_pv_dg')]
    t = ref[['year', 'condition', 'case', 'need_mw', 'energy_headroom_mw', 'alpha', 'deliverable_mw',
             'n_upgrades_to_alpha1', 'verdict', 'binding_element', 'binding_contingency']].copy()
    t['condition'] = t.condition.map(COND_LABEL)
    t.columns = ['year', 'condition', 'absorption', 'need MW', 'energy headroom MW ↑', 'α ↑', 'deliverable MW ↑',
                 'upgrade groups ↓', 'verdict', 'first binding element', 'under']
    t['year'] = t['year'].astype(str)
    fac = facs.pivot_table(index=['year', 'technology'], columns='condition', values='factor').reset_index()
    fac['year'] = fac['year'].astype(str)
    u = ups[(ups.siting == 'reference') & (~ups.case.str.contains('load2050'))] if len(ups) else ups
    u = u[['year', 'condition', 'case', 'order', 'upgrade', 'kv', 'limit_mw', 'required_increase_mw',
           'required_increase_pct', 'worst_condition', 'length_km', 'classification']].copy()
    u['year'] = u['year'].astype(str); u['condition'] = u.condition.map(COND_LABEL)
    u.columns = ['year', 'condition', 'absorption', '#', 'upgrade', 'kV', 'limit MW', 'needed +MW ↓', 'needed +% ↓',
                 'worst', 'length km', 'class']
    sens = chk[(chk.dfax_cut == DFAX_CUT) & (chk.condition != 'S1_plus_existing_pv_dg')].pivot_table(
        index=['year', 'condition', 'case'], columns='siting', values='alpha').reset_index()
    sens['year'] = sens['year'].astype(str); sens['condition'] = sens.condition.map(COND_LABEL)
    d0 = chk[(chk.siting == 'reference') & ~chk.case.str.contains('load2050') & (chk.condition != 'S1_plus_existing_pv_dg')]
    d0 = d0.pivot_table(index=['year', 'condition', 'case'], columns='dfax_cut', values='alpha').reset_index()
    d0 = d0[(d0[0.0] - d0[DFAX_CUT]).abs() > 1e-9].copy()
    d0['year'] = d0['year'].astype(str); d0['condition'] = d0.condition.map(COND_LABEL)
    ex = chk[chk.condition == 'S1_plus_existing_pv_dg'][['year', 'case', 'need_mw', 'alpha', 'deliverable_mw',
                                                         'n_upgrades_to_alpha1', 'verdict']].copy()
    ex['year'] = ex['year'].astype(str)
    ex.columns = ['year', 'absorption', 'need MW', 'α ↑', 'deliverable MW ↑', 'upgrade groups ↓', 'verdict']
    p = poi[['station', 'kv', 'areas', 'hc_A1_mg_backdown_mw', 'hc_A2_displace_inflow_mw', 'hc_B_export_mw',
             'hc_B_export_n0_mw', 'binding_B', 'binding_B_contingency']].copy()
    p['kv'] = p['kv'].astype(str)
    p.columns = ['POI', 'kV', 'candidate areas', 'A1 MW ↑', 'A2 MW ↑', 'B MW ↑', 'B N-0 only MW ↑',
                 'B binding element', 'under']
    h = info['energy_headroom_mw']
    lines = [
        '# Hosting capacity of the existing MG grid vs the fixed portfolios',
        '',
        f"Generated by `hosting_capacity/run_hosting.py` (Python {info['python']}, pandas {info['pandas']}, "
        f"numpy {info['numpy']}, scipy {info['scipy']}). Do not edit by hand. Question and plan: issue #3.",
        '',
        '↑ = higher is better, ↓ = lower is better; columns without an arrow are sizes. '
        'α = largest scaling of the portfolio injection pattern that meets every N-0 and N-1 limit '
        '(α ≥ 1 means the portfolio is deliverable as sited). Deliverable MW = min(α × need, energy headroom).',
        '',
        '## Setup',
        '',
        f"- Network: EPE 2035 PWF, MG-centred equivalent (1,489 buses, 1,890 branches). Incremental DC flows "
        f"anchored on the AC P flows; LODF for N-1 (verified against explicit outage re-solves, `check_engine.py`).",
        f"- Monitored: {info['monitored_branches']} branches = {info['monitored_lines']} lines (the original 283-line set) "
        f"+ {info['monitored_branches'] - info['monitored_lines']} network-transformer branches (windings ≥138 kV, one ≥230 kV; "
        f"generator step-ups excluded). Limits: normal rating (N-0), max(normal, emergency) (N-1).",
        f"- Contingencies: {info['contingencies']} (monitored lines, two-winding transformers, and the HV leg of each "
        f"three-winding unit); {info['radial_outages_skipped']} radial outages skipped.",
        f"- Base case without new resources: {int((base.contingency == 'N-0').sum()) if len(base) else 0} N-0 and "
        f"{int((base.contingency != 'N-0').sum()) if len(base) else 0} N-1 violations"
        + (': ' + '; '.join(f"{r.from_name}-{r.to_name} +{r.overload_mw:.1f} MW for loss of {r.contingency}"
                            for r in base.itertuples()) if len(base) else '') + '. '
        f"Pre-existing violations only count against the new resources when worsened with a distribution factor "
        f"≥ {info['dfax_cut']:.0%}.",
        f"- Absorption cases (the old load-proportional convention is retired): "
        f"**A1** {info['balancing_cases']['A1_mg_backdown']}, headroom {h['A1_mg_backdown']:,.0f} MW; "
        f"**A2** {info['balancing_cases']['A2_displace_inflow']}, headroom {h['A2_displace_inflow']:,.0f} MW; "
        f"**B** {info['balancing_cases']['B_export']}, network-limited only. "
        f"`+load2050` variants let 2050 MG load growth ({info['load_growth_2050_mw_at_pwf_scale']:,.0f} MW at PWF scale, "
        f"from the LP's {info['lp_load_twh']['2035'] if '2035' in info['lp_load_twh'] else info['lp_load_twh'][2035]:.1f} → "
        f"{info['lp_load_twh']['2050'] if '2050' in info['lp_load_twh'] else info['lp_load_twh'][2050]:.1f} TWh) absorb injection first.",
        f"- Siting: nearest eligible station at the allocation weights (`reference_siting_YYYY.csv`); MILP sitings and "
        f"BESS-at-load are sensitivities. Study conditions: mean LP output over the top {info['top_hours_fraction']:.0%} of hours.",
        f"- Provisional minor/major thresholds (issue #3): a line needing ≤ +{info['minor_uprate_fraction']:.0%} is an uprate/"
        f"reconductor, otherwise a parallel circuit ≤ {info['minor_line_km']:.0f} km is minor and longer is major; adding "
        f"transformer banks at an existing substation is minor; more than {info['max_minor_upgrade_groups']} upgrade groups "
        f"is major; a case whose need exceeds its energy headroom is energy-limited.",
        '',
        '## LP study factors (fraction of new nameplate)',
        '',
        md(fac, {c: '{:.3f}' for c in fac.columns}),
        '',
        '## Portfolio need vs hosting capacity (reference siting)',
        '',
        md(t, {'α ↑': '{:.2f}'}),
        '',
        '## Upgrades needed to reach α = 1 (reference siting)',
        '',
        'Greedy: relieve the first binding upgrade group (line corridor or substation transformation), re-solve, repeat. '
        'Ratings are relaxed only; impedances are unchanged.',
        '',
        md(u, {'needed +% ↓': '{:.1f}', 'length km': '{:.0f}'}) if len(u) else '_None needed._',
        '',
        '## Sensitivity: siting (α ↑)',
        '',
        md(sens, {c: '{:.2f}' for c in sens.columns}),
        '',
        f'## Sensitivity: distribution-factor cut-off (α ↑ at 0% vs {DFAX_CUT:.0%}; only rows that differ)',
        '',
        md(d0, {c: '{:.2f}' for c in d0.columns}) if len(d0) else '_No differences._',
        '',
        '## Sensitivity: midday with existing PV and DG added',
        '',
        'The PWF snapshot has all 106 existing PV buses at 0 MW (an evening case). Here the LP\'s existing central PV at '
        'the S1 hours is split equally across those buses (an assumption: plant sizes are not in the data) and existing '
        'DG is added at load buses.',
        '',
        md(ex, {'α ↑': '{:.2f}'}),
        '',
        '## Per-POI hosting capacity (2050 candidate stations, N-0 + N-1, injected MW)',
        '',
        'In the style of NREL\'s Oregon offshore-wind Finding 1: the extra MW one POI can inject alone before the first new '
        'or worsened violation. Capped at the case\'s energy headroom.',
        '',
        md(p),
        '',
        '## Not covered',
        '',
        '- AC voltage/reactive limits, stability, N-2/N-1-1; the 2050 topology is the frozen 2035 PWF; one LP weather year.',
        '- One network snapshot: the annual extra-curtailment criterion (≤5% in issue #3) is not evaluated; '
        'gaps are MW at the study hours.',
        '- Upgrades relax ratings only; real reinforcements change impedances and costs are not estimated here.',
    ]
    (HERE / 'RESULTS.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
