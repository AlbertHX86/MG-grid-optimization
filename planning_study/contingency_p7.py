#!/usr/bin/env python3
"""Phase 5 (optional check): NERC TPL P7 common-structure contingencies.

Every pair of parallel circuits between the same two buses (both in the N-1 contingency set) is
treated as sharing a structure (conservative: tower sharing is not in the data) and is taken out
together. Post-contingency flows use multiple-outage distribution factors,
    f_M^post = f_M + L_MS (I - L_SS^0)^-1 f_S,
verified against explicit DC re-solves. Deliverability of the main case is re-evaluated with the
P7 rows added (5% circle, emergency ratings), for the reference siting and for BESS at load buses,
on the existing grid and (2050) on the network with the verified phase-3 plan, which is sized for P1 only.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'hosting_capacity'))
sys.path.insert(0, str(HERE))
import run_hosting as rh  # noqa: E402
from hc_engine import Network, Rows  # noqa: E402

OUT = HERE / 'results'


def p7_pairs(net, exclude=frozenset()):
    K = set(int(k) for k in net.contingencies) - set(exclude)
    L = net.L
    df = pd.DataFrame({'b': np.arange(net.m), 'f': L.f, 't': L.t, 'kind': L.kind})
    df = df[df.b.isin(K) & (df.kind == 'line')]
    df['key'] = [tuple(sorted(x)) for x in zip(df.f, df.t)]
    pairs = []
    for _, g in df.groupby('key'):
        b = sorted(g.b)
        pairs += [(b[i], b[j]) for i in range(len(b)) for j in range(i + 1, len(b))]
    out = []
    for k1, k2 in pairs:
        S = [k1, k2]
        Lss = np.array([[0.0, lodf(net, k1, k2)], [lodf(net, k2, k1), 0.0]])
        A = np.eye(2) - Lss
        if abs(np.linalg.det(A)) < 1e-6:
            continue                                   # double outage islands part of the system
        out.append((k1, k2, np.linalg.inv(A)))
    return out


def lodf(net, l, k):
    P = net.PTDF
    den = 1 - (P[k, net.f[k]] - P[k, net.t[k]])
    return (P[l, net.f[k]] - P[l, net.t[k]]) / den


def p7_matrices(net, pairs):
    M = net.monitored
    P = net.PTDF
    mats = []
    for k1, k2, inv in pairs:
        cols = []
        for k in (k1, k2):
            den = 1 - (P[k, net.f[k]] - P[k, net.t[k]])
            cols.append((P[M, net.f[k]] - P[M, net.t[k]]) / den)
        LMS = np.column_stack(cols)
        keep = ~np.isin(M, [k1, k2])
        mats.append((k1, k2, LMS @ inv, keep))
    return mats


def rows_p7(net, mats, vec_all):
    """vec_all: branch-flow vector (m,) of a pattern (or base flows); returns post-contingency values on all P7 rows."""
    out = []
    for k1, k2, T, keep in mats:
        v = vec_all[net.monitored] + T @ vec_all[[k1, k2]]
        out.append(v[keep])
    return np.concatenate(out)


def gate(net, pairs, n_test=6, seed=0):
    rng = np.random.default_rng(seed)
    dp = rng.normal(size=net.n); dp -= dp.mean()
    f_pre = net.PTDF @ dp
    worst = 0.0
    for i in rng.choice(len(pairs), min(n_test, len(pairs)), replace=False):
        k1, k2, inv = pairs[i]
        keep_b = ~np.isin(np.arange(net.m), [k1, k2])
        A2, w2 = net.A[keep_b], net.w[keep_b]
        Bb = A2.T @ (w2[:, None] * A2)
        ref = net.idx[585]; free = np.array([j for j in range(net.n) if j != ref])
        th = np.zeros(net.n); th[free] = np.linalg.solve(Bb[np.ix_(free, free)], dp[free])
        f_post = np.zeros(net.m); f_post[keep_b] = w2 * (A2 @ th)
        mats = p7_matrices(net, [pairs[i]])
        pred = rows_p7(net, mats, f_pre)
        keep = mats[0][3]
        worst = max(worst, float(np.max(np.abs(pred - f_post[net.monitored][keep]))))
    return worst


def p7_setup(net, exclude=frozenset()):
    pairs = p7_pairs(net, exclude)
    mats = p7_matrices(net, pairs)
    lim = np.concatenate([net.emrate[net.monitored][keep] for _, _, _, keep in mats])
    elem = np.concatenate([net.monitored[keep] for _, _, _, keep in mats])
    cont = np.concatenate([np.full(keep.sum(), k1) for k1, _, _, keep in mats])
    f0 = rows_p7(net, mats, net.F)
    return pairs, mats, lim, elem, cont, f0, np.abs(f0) > lim + 1e-6


def phase3_plan(siting):
    P = pd.read_csv(OUT / 'mitigation_plan.csv')
    return P[P.siting == siting].to_dict('records')


def plan_network(plan):
    """The network with a plan (new elements + reconductoring). Returns (net, exclude): `exclude` holds the new
    branches, which are built on their own structures and so are never part of a P7 pair (they stay monitored)."""
    import mitigation as mit
    net0 = Network()
    net = Network(extra=mit.extra_branches(net0, plan))
    mit.apply_reconductor(net, rh.upgrade_groups(net), plan)
    return net, frozenset(range(net.m_base, net.m))


def evaluate(net, year, siting, network_label, a1_of, cap, fac_df, sites, growth, groups, crd, exclude=frozenset()):
    pairs, mats, lim_p7, elem_p7, cont_p7, f0_p7, pre = p7_setup(net, exclude)
    rows = []
    for cond in fac_df.condition.unique():
        fac = dict(fac_df[fac_df.condition == cond][['technology', 'factor']].itertuples(index=False))
        units, P, meta = rh.study_units(net, sites, fac, cap, bess_at_load=(siting == 'reference_bess_at_load'))
        need = P.sum()
        w, _, _ = net.balancing(rh.MAIN)
        g_abs = min(growth, need)
        sink = (g_abs * net.load_weights() + (need - g_abs) * w) / need
        g_all = net.PTDF @ (units @ P - sink * need)
        Gu = net.PTDF @ (units - sink[:, None])                    # m x U
        g_p7 = rows_p7(net, mats, g_all)
        D_p7 = np.column_stack([rows_p7(net, mats, Gu[:, u]) for u in range(Gu.shape[1])])
        act, member = net.circle(D_p7, P, lim_p7, rh.CIRCLE)
        act &= ~pre
        r7 = Rows(f0_p7, g_p7, lim_p7, elem_p7, cont_p7)
        a7, j = net.max_scale(r7, 0.0, total=need, active=act)
        a1 = a1_of(cond, units, P, meta)
        lab, bind = '', {}
        if j >= 0:
            e = net.branch_label(elem_p7[j]); c1 = net.branch_label(cont_p7[j])
            lab = f"{e['from_name']}-{e['to_name']} for loss of both {c1['from_name']}-{c1['to_name']} circuits"
            key = groups.get(int(elem_p7[j]))
            bind = {'bind_elem': int(elem_p7[j]), 'bind_cont': int(cont_p7[j]), 'bind_lim_mw': float(lim_p7[j]),
                    'bind_flow_full_need_mw': float(abs(f0_p7[j] + g_p7[j])), 'bind_group': key,
                    'bind_pair_in_group': groups.get(int(cont_p7[j])) == key,
                    'bind_group_size': sum(1 for k in groups.values() if k == key)}
        rows.append({'year': year, 'siting': siting, 'network': network_label, 'condition': cond, 'alpha_P1': a1,
                     'alpha_P7_rows': a7, 'alpha_P1_P7': min(a1, a7), 'P7_binds_first': a7 < a1, 'binding_P7': lab,
                     'p7_pairs': len(pairs), 'p7_rows_tested': int(act.sum()), **bind})
        print(rows[-1], flush=True)
    return rows, pairs, pre


def on_plan(plan, year, siting, label, cap, fac_df, sites, growth, crd):
    pnet, exclude = plan_network(plan); pgroups = rh.upgrade_groups(pnet)

    def a1_plan(cond, units, P, meta):
        res, _, _ = rh.assess(pnet, units, P, meta, rh.MAIN, pgroups, crd, growth_mw=growth)
        return float(res['alpha'])
    return evaluate(pnet, year, siting, label, a1_plan, cap, fac_df, sites, growth, pgroups, crd, exclude)[0]


def p7_remedy(plan, rows, year, siting, cap, fac_df, sites, growth, crd, max_iter=8):
    """If P7 still binds below alpha = 1 on the plan network, add the cheapest costed alternative for the binding
    P7 element and re-verify (P1 and P7), as in phase 3. New circuits are on their own structures (not paired in P7).
    Sizing: the circuits that survive the double outage share the flow equally with the new ones."""
    import mitigation as mit
    net0 = Network(); groups0 = rh.upgrade_groups(net0)
    plan = [dict(p) for p in plan]; added = []; log = []
    for it in range(max_iter + 1):
        worst = min(rows, key=lambda r: r['alpha_P1_P7'])
        log.append({'siting': siting, 'iteration': it, 'n_remedy_elements': int(sum(a['n_elements'] for a in added)),
                    'alpha_P1_P7_min': worst['alpha_P1_P7'], 'worst_condition': worst['condition'],
                    **{f"alpha_P1_P7_{r['condition']}": r['alpha_P1_P7'] for r in rows}})
        print(log[-1], flush=True)
        if worst['alpha_P1_P7'] >= 1.0 - 1e-6 or it == max_iter:
            break
        if worst['alpha_P1'] < 1.0 - 1e-6 or 'bind_elem' not in worst:
            raise RuntimeError('P1 must already hold on the plan network')
        key = worst['bind_group']
        b = min(x for x, k in groups0.items() if k == key)          # an existing member: kV and route length
        n_surv = worst['bind_group_size'] - (2 if worst['bind_pair_in_group'] else 0)
        inc = worst['bind_flow_full_need_mw'] / worst['bind_lim_mw'] - 1.0
        hit = [a for a in added if (a['group_type'], a['group']) == key]
        if hit:
            a = hit[0]
            if a['alternative'].startswith('reconductor'):
                a['required_increase_pct'] += 100 * inc + 1.0
            else:
                a['n_elements'] += 1
                a['capex_mbrl'] = a['capex_low_mbrl'] = a['capex_high_mbrl'] = a['n_elements'] * a['_per_element']
            a['added_in_verification'] = a.get('added_in_verification', 0) + 1
        else:
            alts = mit.group_alternatives(net0, groups0, crd, key, inc, inc * worst['bind_lim_mw'], b)
            new = next(a for a in alts if a['alternative'].startswith('new parallel'))
            per = new['capex_mbrl'] / new['n_elements']
            k = max(1, int(np.ceil(n_surv * inc - 1e-9)))
            new.update({'n_elements': k, 'capex_mbrl': k * per, 'capex_low_mbrl': k * per, 'capex_high_mbrl': k * per,
                        '_per_element': per, 'basis': 'n x ' + new['basis'].split(' x ', 1)[1] + ' on its own structure'})
            rec = next(a for a in alts if a['alternative'].startswith('reconductor'))
            best = min([a for a in (new, rec) if a['feasible']], key=lambda a: a['capex_mbrl'])
            best.update({'worst_branch_index': b, 'p7_remedy': True, 'design_condition': worst['condition'],
                         'outaged_pair': worst['binding_P7'].split('both ')[-1], 'surviving_circuits': n_surv,
                         'flow_full_need_mw': worst['bind_flow_full_need_mw'], 'limit_mw': worst['bind_lim_mw']})
            added.append(best)
        rows = on_plan(plan + added, year, siting, 'with plan + P7 remedy', cap, fac_df, sites, growth, crd)
    out = [{**{k: v for k, v in a.items() if not k.startswith('_')}, 'siting': siting} for a in added]
    rows = [{k: v for k, v in r.items() if k != 'bind_group'} for r in rows]
    return out, log, (rows if added else [])


def main():
    net = Network(); groups = rh.upgrade_groups(net); crd = rh.coords()
    pairs = p7_pairs(net)
    err = gate(net, pairs)
    print(f'P7 pairs {len(pairs)}; multiple-outage formula vs explicit re-solve: max |diff| {err:.2e} MW', flush=True)
    rows = []
    lp_load = {y: float(pd.read_csv(rh.NODAL / 'macro_reopt' / 'output' / f'REFRESH_{y}' / 'annual_energy.csv').load_twh.iloc[0])
               for y in (2035, 2050)}
    pwf_load = float(net.B.loc[(net.B.state == 11) & (net.B.pd > 0), 'pd'].sum())
    chk = pd.read_csv(ROOT / 'hosting_capacity' / 'results' / 'portfolio_vs_hosting.csv')
    prof = rh.availability_profiles()
    pre_count = None
    rem_rows, rem_log = [], []
    for year in (2035, 2050):
        cap = rh.portfolio_mw(year)
        fac_df, _ = rh.lp_factors(year, cap)
        fac_df = pd.concat([fac_df, rh.caiso_factors(year, cap, prof)], ignore_index=True)
        sites, _ = rh.reference_siting(year, cap)
        growth = (lp_load[year] / lp_load[2035] - 1.0) * pwf_load if year == 2050 else 0.0
        for siting in ('reference', 'reference_bess_at_load'):
            if siting == 'reference_bess_at_load' and cap.get('bess', 0) <= 0:
                continue

            def a1_existing(cond, *_):
                b = chk[(chk.year == year) & (chk.condition == cond) & (chk.siting == siting) & chk.main & (chk['mode'] == 'circle')]
                return float(b.alpha.iloc[0])
            r, _, pre = evaluate(net, year, siting, 'existing grid', a1_existing, cap, fac_df, sites, growth, groups, crd)
            rows += [{k: v for k, v in x.items() if k != 'bind_group'} for x in r]
            if pre_count is None:
                pre_count = int(pre.sum())
            if year == 2050:                               # the same check on the network with the verified P1 plan
                plan = phase3_plan(siting)
                r = on_plan(plan, year, siting, 'with verified plan', cap, fac_df, sites, growth, crd)
                rows += [{k: v for k, v in x.items() if k != 'bind_group'} for x in r]
                rem, log, fin = p7_remedy(plan, r, year, siting, cap, fac_df, sites, growth, crd)
                rem_rows += rem; rem_log += log; rows += fin
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'contingency_p7.csv', index=False, float_format='%.6f')
    pd.DataFrame([{'p7_pairs': len(pairs), 'gate_max_abs_diff_mw': err, 'preexisting_p7_rows': pre_count}]).to_csv(
        OUT / 'contingency_p7_info.csv', index=False)
    pd.DataFrame(rem_rows).to_csv(OUT / 'contingency_p7_remedy.csv', index=False, float_format='%.6f')
    L = pd.DataFrame(rem_log); L.to_csv(OUT / 'contingency_p7_remedy_verification.csv', index=False, float_format='%.6f')
    ok = err < 1e-6
    plan_p1 = df[df.network == 'with verified plan'].alpha_P1.min()
    ok_plan = plan_p1 >= 1.0 - 1e-6
    ok_rem = bool(L.empty or L.groupby('siting').last().alpha_P1_P7_min.min() >= 1.0 - 1e-6)
    print('GATE P7 formula vs explicit re-solve:', 'PASS' if ok else 'FAIL')
    print(f'GATE P1 alpha on the plan networks reproduces the phase-3 verification (min {plan_p1:.4f} >= 1):',
          'PASS' if ok_plan else 'FAIL')
    print('GATE every 2050 condition deliverable under P1+P7 with the plan and the P7 remedy:', 'PASS' if ok_rem else 'FAIL')
    return 0 if ok and ok_plan and ok_rem else 1


if __name__ == '__main__':
    sys.exit(main())
