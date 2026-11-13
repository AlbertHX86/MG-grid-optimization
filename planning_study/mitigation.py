#!/usr/bin/env python3
"""Phase 3: mitigation alternatives and costs for the attributable reinforcements (3f) and the
BESS-location experiment (2c).

For every upgrade group of the phase-1 main case (reference siting; and BESS at load buses) the
alternatives follow the structure of the PG&E cluster-study slides (Lecture 12): status quo,
limiting-element removal / re-rate, reconductoring, transformer addition or new circuit, storage
(here: siting the planned BESS at load), grid-enhancing technologies and RAS. Costed alternatives use
published EPE/ANEEL unit costs (planning_study/data/unit_costs.csv); the others get a disposition.

Sizing assumes identical parallel elements share flow equally: k new elements on a group of N raise
its capability by (N+k)/N. The chosen plan is then verified on a network that actually contains the
new elements (PTDF/LODF rebuilt, base flows re-anchored); if any study condition still binds, the
binding group gets one more element and the check repeats.
"""
from __future__ import annotations
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'hosting_capacity'))
import run_hosting as rh  # noqa: E402
from hc_engine import Network, STAR_KV  # noqa: E402

OUT = HERE / 'results'
UC = pd.read_csv(HERE / 'data' / 'unit_costs.csv')
COSTS = json.loads((rh.NODAL / 'inputs' / 'cost_assumptions.json').read_text())
ROUTE = COSTS['route_multiplier_base']                       # 1.25, the project's routing factor
CRF = 0.08 * 1.08 ** 30 / (1.08 ** 30 - 1)                   # 8 %, 30 years (cost_assumptions.json)
OM = COSTS['annual_o_and_m_fraction_assumed']                # 1.5 % of capex per year
PEAK_OFF = ['S2_pre_curtailment', 'S3_evening_nonsolar', 'HSN_caiso', 'SSN_caiso', 'S1_peak_delivery']


def uc(item, kv=None):
    r = UC[UC.item.str.startswith(item) & ((UC.voltage_kv.astype(str) == str(kv)) if kv is not None else True)]
    assert len(r) >= 1, (item, kv)
    return float(r.value_thousand_brl.iloc[0]) / 1000.0     # million BRL


TX_MVA = {'500/345': (uc('ATF single-phase unit 500/345'), 500.0, 'EPE-DEE-NT-047-2022'),
          '500/230': (uc('ATF single-phase unit 500/230'), 100.0, 'EPE-DEE-RE-071-2025'),
          '345/138': (uc('ATF single-phase unit 345/138'), 100.0, 'EPE-DEE-NT-029-2023')}
TX_PROXY = {'345/230': '345/138', '500/138': '500/230', '230/138': '345/138'}
CT = {500: uc('CT transformer connection bay 500'), 345: uc('CT transformer connection bay 345'),
      230: uc('CT transformer connection bay 230'), 138: uc('CT transformer connection bay 138')}
LINE_KM = {230: uc('New line 230'), 345: uc('New line 345'), 500: uc('New line 500')}
EL = {230: uc('Line entry bay 230'), 345: uc('Line entry bay 345'), 500: uc('Line entry bay 500')}
RECOND = (float(UC[UC.item.str.startswith('Reconductoring')].value_thousand_brl.min()),
          float(UC[UC.item.str.startswith('Reconductoring')].value_thousand_brl.max()))


def tx_class(net, b):
    """(HV, LV) winding kV of the transformer unit a monitored branch belongs to."""
    s = net.unit_of_leg.get(int(b))
    if s is None:
        k = sorted({net.kv[net.f[b]], net.kv[net.t[b]]}, reverse=True)
        return int(k[0]), int(k[1])
    legs = [l for l, ss in net.unit_of_leg.items() if ss == s]
    kvs = sorted({int(net.kv[net.t[l]] if net.f[l] == s else net.kv[net.f[l]]) for l in legs}, reverse=True)
    kvs = [k for k in kvs if k >= 138]
    return kvs[0], kvs[1]


def group_alternatives(net, groups, crd, key, inc_frac, inc_mw, worst_b):
    """Alternatives for one upgrade group needing +inc_frac on its worst element."""
    members = [b for b, k in groups.items() if k == key]
    rows = []
    if key[0] == 'line':
        N = len(members)
        kv = int(round(net.branch_label(worst_b)['kv']))
        km, basis = net.line_length_km(worst_b, crd)
        route = km * ROUTE
        c_new = route * LINE_KM.get(kv, np.nan) + 2 * EL.get(kv, np.nan)
        k = max(1, math.ceil(N * inc_frac - 1e-9))
        rows.append({'alternative': 'new parallel circuit(s)', 'costed': True, 'n_elements': k,
                     'capex_mbrl': k * c_new, 'capex_low_mbrl': k * c_new, 'capex_high_mbrl': k * c_new,
                     'basis': f'{k} x ({route:.1f} km routed [{basis}, x{ROUTE}] x R${LINE_KM.get(kv, np.nan):.3f}M/km + 2 line bays)',
                     'feasible': True})
        lo, hi = N * RECOND[0] * c_new, N * RECOND[1] * c_new
        rows.append({'alternative': 'reconductor existing circuit(s) (HTLS)', 'costed': True, 'n_elements': N,
                     'capex_mbrl': (lo + hi) / 2, 'capex_low_mbrl': lo, 'capex_high_mbrl': hi,
                     'basis': f'{N} circuit(s) x {RECOND[0]:.0%}-{RECOND[1]:.1%} of a new circuit (EPE-DEE-RE-071-2025 Table 14-39)',
                     'feasible': inc_frac <= 1.0})
        rows.append({'alternative': 'dynamic line rating (GET)', 'costed': False, 'n_elements': N, 'capex_mbrl': np.nan,
                     'capex_low_mbrl': np.nan, 'capex_high_mbrl': np.nan, 'feasible': inc_frac <= 0.10,
                     'basis': 'candidate where the need is small (<= 10 %); needs weather-based ratings (Lecture 6); not costed'})
        meta = {'n_existing': N, 'kv': kv, 'length_km': km, 'length_basis': basis, 'class': f'{kv} kV line'}
    else:
        units = {net.unit_of_leg.get(int(b), ('2w', int(b))) for b in members}
        N = len(units)
        hv, lv = tx_class(net, worst_b)
        cls = f'{hv}/{lv}'
        src_cls = TX_PROXY.get(cls, cls)
        unit_cost, unit_mva, src = TX_MVA[src_cls]
        mva = float(net.rate[worst_b])
        c_bank = mva * unit_cost / unit_mva + CT.get(hv, np.nan) + CT.get(lv, np.nan)
        k = max(1, math.ceil(N * inc_frac - 1e-9))
        rows.append({'alternative': 'add transformer bank(s)', 'costed': True, 'n_elements': k,
                     'capex_mbrl': k * c_bank, 'capex_low_mbrl': k * c_bank, 'capex_high_mbrl': k * c_bank,
                     'basis': f'{k} x ({mva:.0f} MVA x R${unit_cost / unit_mva * 1000:.1f}k/MVA [{src_cls}'
                              f'{" proxy for " + cls if src_cls != cls else ""}, {src}] + CT {hv} kV + CT {lv} kV)',
                     'feasible': True})
        meta = {'n_existing': N, 'kv': hv, 'length_km': np.nan, 'length_basis': 'n/a', 'class': f'{cls} kV transformation',
                'bank_mva': mva, 'proxy_cost_class': src_cls != cls}
    rows.append({'alternative': 'limiting-element removal / re-rate', 'costed': False, 'n_elements': np.nan,
                 'capex_mbrl': np.nan, 'capex_low_mbrl': np.nan, 'capex_high_mbrl': np.nan, 'feasible': inc_frac <= 0.10,
                 'basis': 'possible only if terminal equipment, not the conductor/transformer, sets the rating; '
                          'equipment data not in the package (PG&E LC25 example: US$0.5 M, Lecture 12)'})
    rows.append({'alternative': 'RAS / operational curtailment', 'costed': False, 'n_elements': np.nan,
                 'capex_mbrl': np.nan, 'capex_low_mbrl': np.nan, 'capex_high_mbrl': np.nan, 'feasible': True,
                 'basis': 'post-contingency generation tripping or curtailment; energy cost valued in the production-cost phase'})
    for r in rows:
        r.update({'group_type': key[0], 'group': key[1], 'required_increase_pct': 100 * inc_frac,
                  'required_increase_mw': inc_mw, **meta})
    return rows


def extra_branches(net, plan):
    """PWF-format branch rows for the new elements of a plan (transformers as HV-LV equivalents)."""
    L = net.L
    rows = []
    for p in plan:
        b = int(p['worst_branch_index'])
        if p['alternative'].startswith('reconductor'):
            continue
        for i in range(int(p['n_elements'])):
            if p['group_type'] == 'line' or net.unit_of_leg.get(b) is None:
                r = L.iloc[b].copy(); r['c'] = 90 + len(rows)
            else:
                s = net.unit_of_leg[b]
                legs = [l for l, ss in net.unit_of_leg.items() if ss == s]
                hv, lv = tx_class(net, b)
                bus_of = {int(net.kv[net.t[l]] if net.f[l] == s else net.kv[net.f[l]]):
                          int(L.t.iloc[l] if net.f[l] == s else L.f.iloc[l]) for l in legs}
                xeq = sum(float(L.x.iloc[l] * L.tap.iloc[l]) for l in legs
                          if int(net.kv[net.t[l]] if net.f[l] == s else net.kv[net.f[l]]) in (hv, lv))
                r = L.iloc[b].copy()
                r['f'], r['t'], r['x'], r['tap'], r['shift'], r['c'] = bus_of[hv], bus_of[lv], xeq, 1.0, 0.0, 90 + len(rows)
            rows.append(r)
    return pd.DataFrame(rows) if rows else None


def apply_reconductor(net, groups, plan):
    for p in plan:
        if not p['alternative'].startswith('reconductor'):
            continue
        members = [b for b, k in groups.items() if k == (p['group_type'], p['group'])]
        f = 1 + p['required_increase_pct'] / 100 * 1.0001
        for b in members:
            net.rate[b] *= f; net.emrate[b] *= f


def verify(year, plan, siting, conditions, max_iter=15):
    """Rebuild the network with the plan, re-run the main case; add elements until alpha >= 1."""
    cap = rh.portfolio_mw(year)
    fac_df, _ = rh.lp_factors(year, cap)
    fac_df = pd.concat([fac_df, rh.caiso_factors(year, cap, rh.availability_profiles())], ignore_index=True)
    sites, _ = rh.reference_siting(year, cap)
    lp_load = {y: float(pd.read_csv(rh.NODAL / 'macro_reopt' / 'output' / f'REFRESH_{y}' / 'annual_energy.csv').load_twh.iloc[0])
               for y in (2035, 2050)}
    log = []
    for it in range(max_iter):
        net0 = Network()
        net = Network(extra=extra_branches(net0, plan))
        groups = rh.upgrade_groups(net)
        apply_reconductor(net, groups, plan)
        crd = rh.coords()
        pwf_load = float(net.B.loc[(net.B.state == 11) & (net.B.pd > 0), 'pd'].sum())
        growth = (lp_load[year] / lp_load[2035] - 1.0) * pwf_load
        worst = None
        res_all = {}
        for cond in conditions:
            fac = dict(fac_df[fac_df.condition == cond][['technology', 'factor']].itertuples(index=False))
            units, P, meta = rh.study_units(net, sites, fac, cap, bess_at_load=(siting == 'reference_bess_at_load'))
            res, ups, _ = rh.assess(net, units, P, meta, rh.MAIN, groups, crd, growth_mw=growth)
            res_all[cond] = res['alpha']
            if res['alpha'] < 1 and (worst is None or res['alpha'] < worst[0]):
                worst = (res['alpha'], cond, ups[0] if ups else None)
        log.append({'iteration': it, 'siting': siting, 'n_new_elements': int(sum(p['n_elements'] for p in plan
                                                                                 if not p['alternative'].startswith('reconductor'))),
                    **{f'alpha_{c}': a for c, a in res_all.items()}})
        if worst is None:
            return plan, log, True
        u = worst[2]
        key = (u['upgrade_type'], u['upgrade'])
        hit = [p for p in plan if (p['group_type'], p['group']) == key]
        if hit:
            p = hit[0]
            if p['alternative'].startswith('reconductor'):
                p['required_increase_pct'] += u['required_increase_pct'] + 1.0
            else:
                p['n_elements'] += 1
            p['added_in_verification'] = p.get('added_in_verification', 0) + 1
        else:
            alts = group_alternatives(net0, rh.upgrade_groups(net0), crd, key, u['required_increase_pct'] / 100,
                                      u['required_increase_mw'], u['worst_branch_index'])
            best = choose(alts)
            best['worst_branch_index'] = u['worst_branch_index']; best['added_in_verification'] = 1
            plan.append(best)
    return plan, log, False


def choose(alts):
    c = [a for a in alts if a['costed'] and a['feasible']]
    return dict(min(c, key=lambda a: a['capex_mbrl']))


def main():
    ups = pd.read_csv(ROOT / 'hosting_capacity' / 'results' / 'required_upgrades.csv')
    chk = pd.read_csv(ROOT / 'hosting_capacity' / 'results' / 'portfolio_vs_hosting.csv')
    net = Network(); groups = rh.upgrade_groups(net); crd = rh.coords()
    alt_rows, plans, ver_logs, totals = [], {}, [], []
    for siting in ('reference', 'reference_bess_at_load'):
        u = ups[(ups.siting == siting) & ups.main & (ups.year == 2050)]
        conds = sorted(chk[(chk.siting == siting) & chk.main & (chk['mode'] == 'circle') & (chk.year == 2050)
                           & chk.condition.isin(PEAK_OFF)].condition.unique())
        design = (u.sort_values('required_increase_pct', ascending=False)
                  .groupby(['upgrade_type', 'upgrade'], as_index=False).first())
        plan = []
        for d in design.itertuples(index=False):
            key = (d.upgrade_type, d.upgrade)
            alts = group_alternatives(net, groups, crd, key, d.required_increase_pct / 100, d.required_increase_mw,
                                      int(d.worst_branch_index))
            for a in alts:
                alt_rows.append({'siting': siting, 'design_condition': d.condition, 'violation': 'worsened pre-existing'
                                 if d.preexisting else 'new', 'circle_areas': d.circle_areas, **a})
            best = choose(alts); best['worst_branch_index'] = int(d.worst_branch_index); best['design_condition'] = d.condition
            plan.append(best)
        ratings_only_cost = sum(p['capex_mbrl'] for p in plan)
        plan, log, ok = verify(2050, [dict(p) for p in plan], siting, conds)
        ver_logs += log
        plans[siting] = plan
        for p in plan:   # recompute cost after verification additions
            if p.get('added_in_verification') and not p['alternative'].startswith('reconductor'):
                per = p['capex_low_mbrl'] / max(1, p['n_elements'] - p['added_in_verification'])
                p['capex_mbrl'] = p['capex_low_mbrl'] = p['capex_high_mbrl'] = per * p['n_elements']
        cost = sum(p['capex_mbrl'] for p in plan)
        for p in plan:   # small line needs: re-rate / limiting-element / DLR are plausible but cannot be costed
            p['non_wires_candidate'] = bool(p['group_type'] == 'line' and p['required_increase_pct'] <= 10.0)
        lower = sum(p['capex_mbrl'] for p in plan if not p['non_wires_candidate'])
        totals.append({'siting': siting, 'groups': len(plan), 'new_elements': int(sum(p['n_elements'] for p in plan
                       if not p['alternative'].startswith('reconductor'))),
                       'reconductored_corridors': int(sum(1 for p in plan if p['alternative'].startswith('reconductor'))),
                       'capex_ratings_only_mbrl': ratings_only_cost, 'capex_verified_mbrl': cost,
                       'capex_low_mbrl': sum(p['capex_low_mbrl'] for p in plan),
                       'capex_high_mbrl': sum(p['capex_high_mbrl'] for p in plan),
                       'annual_revenue_requirement_mbrl': cost * (CRF + OM), 'verified_alpha_ge_1': ok,
                       'capex_excluding_small_line_needs_mbrl': lower,
                       'annual_revenue_requirement_excl_small_mbrl': lower * (CRF + OM),
                       'small_line_needs': int(sum(p['non_wires_candidate'] for p in plan)),
                       'iterations': len(log)})
        print(siting, totals[-1], flush=True)
    pd.DataFrame(alt_rows).to_csv(OUT / 'mitigation_alternatives.csv', index=False, float_format='%.6f')
    pd.DataFrame([{**p, 'siting': s} for s, pl in plans.items() for p in pl]).to_csv(OUT / 'mitigation_plan.csv',
                                                                                      index=False, float_format='%.6f')
    pd.DataFrame(ver_logs).to_csv(OUT / 'mitigation_verification.csv', index=False, float_format='%.6f')
    T = pd.DataFrame(totals); T.to_csv(OUT / 'mitigation_totals.csv', index=False, float_format='%.6f')
    # worsened pre-existing reliability need (São Gonçalo do Pará), reported separately
    pre = chk[(chk.siting == 'reference') & chk.main & (chk['mode'] == 'circle') & (chk.worsened_preexisting_rows > 0)]
    base = pd.read_csv(ROOT / 'hosting_capacity' / 'results' / 'base_case_screen.csv')
    pre_rows = []
    for b in base.itertuples(index=False):
        name = f'{b.from_name}-{b.to_name}'
        worst_add = float(pre[pre.worsened_preexisting_names.fillna('').str.contains(name, regex=False)]
                          .worsened_preexisting_added_mw.max() or 0.0)
        e = int(net.L.index[(net.L.f == b.from_bus) & (net.L.t == b.to_bus) & (net.L.c == b.circuit)][0])
        key = groups[e]
        base_need = b.overload_mw / b.limit_mw
        with_pf = (b.overload_mw + worst_add) / b.limit_mw
        a0 = choose(group_alternatives(net, groups, crd, key, base_need, b.overload_mw, e))
        a1 = choose(group_alternatives(net, groups, crd, key, with_pf, b.overload_mw + worst_add, e))
        pre_rows.append({'element': name, 'contingency': b.contingency, 'limit_mw': b.limit_mw,
                         'base_overload_mw': b.overload_mw, 'portfolio_adds_mw': worst_add,
                         'base_need_pct': 100 * base_need, 'need_with_portfolio_pct': 100 * with_pf,
                         'alternative': a0['alternative'], 'banks_base': a0['n_elements'], 'banks_with_portfolio': a1['n_elements'],
                         'capex_base_mbrl': a0['capex_mbrl'], 'capex_with_portfolio_mbrl': a1['capex_mbrl'],
                         'capex_attributable_increment_mbrl': a1['capex_mbrl'] - a0['capex_mbrl']})
    pd.DataFrame(pre_rows).to_csv(OUT / 'mitigation_preexisting.csv', index=False, float_format='%.6f')
    print(pd.DataFrame(pre_rows).round(2).to_string())
    ok = bool(T.verified_alpha_ge_1.all())
    print('GATE every main-case condition deliverable with the verified plan:', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
