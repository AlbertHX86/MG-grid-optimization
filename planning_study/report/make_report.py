#!/usr/bin/env python3
"""Builds every table (tables/*.tex), number macro (numbers.tex) and figure (figures/*.pdf) of the
report from the result CSVs, so no number in the PDF is typed by hand."""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PS = HERE.parent
ROOT = PS.parent
R1 = ROOT / 'hosting_capacity' / 'results'
R2 = PS / 'results'
TAB = HERE / 'tables'; FIG = HERE / 'figures'
TAB.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)

# categorical slots 1-4 of the validated reference palette (light mode), fixed order
C1, C2, C3, C4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#dedcd6'
plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
                     'font.family': 'serif', 'pdf.fonttype': 42})

NAMES = {'IRAPE': 'Irapé', 'MCLAR2': 'Montes Claros 2', 'MCLAR1': 'Montes Claros 1', 'JANAU3': 'Janaúba 3',
         'JANAU6': 'Janaúba 6', 'SGPARA': 'São Gonçalo do Pará', 'JAGUAR': 'Jaguara', 'JAGUA': 'Jaguara',
         'ESTREI': 'Estreito', 'MARIMB': 'Marimbondo', 'MARIM2': 'Marimbondo 2', 'LEOPO2': 'Leopoldina 2',
         'SDUMO2': 'Santos Dumont 2', 'IPATI1': 'Ipatinga 1', 'MESQUI': 'Mesquita', 'TIMOT2': 'Timóteo 2',
         'TIMOTE': 'Timóteo', 'VGRAND': 'Volta Grande', 'LCBARR': 'L.C. Barreto', 'VPALMA': 'Várzea da Palma',
         'PCOLOM': 'Porto Colômbia', 'ITAJU3': 'Itajubá 3', 'C.PAUL': 'Cachoeira Paulista', 'ITABI2': 'Itabira 2',
         'ITABI5': 'Itabira 5', 'PIRAP2': 'Pirapora 2', 'BURIT3': 'Buritizeiro 3', 'PARACA': 'Paracatu 4',
         'EMBORC': 'Emborcação', 'NPONT3': 'Nova Ponte 3', 'NPONTE': 'Nova Ponte', 'ARINO2': 'Arinos 2',
         'JAIBA': 'Jaíba', 'CVERED': 'CVERED', 'AUROR3': 'Aurora 3', 'ITUMBI': 'Itumbiara', 'RVERDN': 'Rio Verde Norte',
         'ARARAQ': 'Araraquara'}


def esc(s):
    s = str(s)
    for a, b in [('\\', r'\textbackslash{}'), ('&', r'\&'), ('%', r'\%'), ('$', r'\$'), ('#', r'\#'), ('_', r'\_'),
                 ('{', r'\{'), ('}', r'\}'), ('~', r'\textasciitilde{}'), ('^', r'\textasciicircum{}')]:
        s = s.replace(a, b)
    return s


STATION = {'PIRAPORA 2': 'Pirapora 2', 'PARACATU 4': 'Paracatu 4', 'JANAUBA 6': 'Janaúba 6', 'JANAUBA 3': 'Janaúba 3',
           'NOVA PONTE 3': 'Nova Ponte 3', 'JAIBA': 'Jaíba', 'EMBORCACAO': 'Emborcação', 'MARIMBONDO II': 'Marimbondo II',
           'ARINOS 2': 'Arinos 2', 'P. JUSCELINO': 'Presidente Juscelino', 'SAO SIMAO-SE': 'São Simão', 'MUTUM': 'Mutum',
           'MONTESCLAROS2': 'Montes Claros 2', 'JAGUARA-500': 'Jaguara', 'VOLTA GRANDE-SE': 'Volta Grande',
           'MONTE ALEGRE2': 'Monte Alegre 2', 'VARZEA PALMA1': 'Várzea da Palma 1', 'VARZEA PALMA 4': 'Várzea da Palma 4',
           'P. COLOMBIA': 'Porto Colômbia', 'MARIMBONDO': 'Marimbondo', 'ITUMBIARA': 'Itumbiara', 'IRAPE-SE': 'Irapé',
           'CONSE. PENA': 'Conselheiro Pena', 'AIMORES-SE': 'Aimorés'}


def station(name):
    return STATION.get(str(name).strip(), str(name).title())


def pretty(code):
    """'IRAPE--MG345-IRAPEB-MG000' -> 'Irapé 345 kV ...'; substation/corridor labels to readable names."""
    code = str(code)
    m = re.match(r'^([A-Z0-9.]+?)-*\s*substation$', code)
    if m:
        k = m.group(1).rstrip('-')
        return NAMES.get(k, k) + ' transformation'
    parts = re.findall(r'([A-Z0-9.]+?)-*(?:MG|SP|GO)(\d{3})', code)
    if len(parts) >= 2:
        (a, va), (b, vb) = parts[0], parts[1]
        a = NAMES.get(a.rstrip('-'), a.rstrip('-')); b = NAMES.get(b.rstrip('-'), b.rstrip('-'))
        if vb == '000':
            return f'{a} {int(va)} kV transformer'
        return f'{a}–{b} {int(va)} kV' if va == vb else f'{a} {int(va)}/{int(vb)} kV'
    return code


def sci(x):
    """LaTeX scientific notation for small diagnostics, e.g. 2\times10^{-15}."""
    x = float(x)
    if x == 0:
        return '0'
    e = int(np.floor(np.log10(abs(x)))); m = x / 10 ** e
    return rf'{m:.0f}\times10^{{{e}}}'


def fmt(v, d=0):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return '—'
    if abs(v) < 0.5 * 10 ** (-d):
        v = 0.0                                     # no negative zero
    return f'{v:,.{d}f}'


def table(name, df, cols, caption, label, align=None, note=None, size=r'\small'):
    """cols: list of (column, header, decimals or None for text)."""
    al = align or ''.join('l' if d is None else 'r' for _, _, d in cols)
    lines = [r'\begin{table}[!htbp]', r'\centering', size, rf'\caption{{{caption}}}', rf'\label{{{label}}}',
             r'\begin{adjustbox}{max width=\linewidth}', rf'\begin{{tabular}}{{{al}}}', r'\toprule', ' & '.join(h for _, h, _ in cols) + r' \\', r'\midrule']
    for _, r in df.iterrows():
        cells = []
        for c, _, d in cols:
            v = r[c]
            if c == 'year':
                cells.append(str(int(v)))
            else:
                cells.append(esc(v) if d is None else (fmt(float(v), d) if pd.notna(v) else '—'))
        lines.append(' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{adjustbox}']
    if note:
        lines.append(rf'\par\vspace{{2pt}}\begin{{minipage}}{{0.95\linewidth}}\footnotesize {note}\end{{minipage}}')
    lines.append(r'\end{table}')
    (TAB / f'{name}.tex').write_text('\n'.join(lines) + '\n')


NUM = {}


DIG = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']


def num(key, value, d=0, unit=''):
    """LaTeX macro names cannot hold digits: 2035 -> A, 2050 -> B, other digits spelled out."""
    key = str(key).replace('2035', 'A').replace('2050', 'B')
    key = re.sub(r'\d', lambda m: DIG[int(m.group())], key)
    key = re.sub(r'[^A-Za-z]', '', key)
    assert key not in NUM, f'duplicate number macro {key}'
    NUM[key] = (fmt(value, d) if isinstance(value, (int, float, np.floating, np.integer)) else str(value)) + unit


COND = {'S1_peak_delivery': 'Off-peak S1 (midday, LP-used)', 'S2_pre_curtailment': 'Off-peak S2 (midday, pre-curtailment)',
        'S3_evening_nonsolar': 'Peak S3 (evening, BESS)', 'HSN_caiso': 'Peak HSN (CAISO)', 'SSN_caiso': 'Peak SSN (CAISO)',
        'S1_plus_existing_pv_dg': 'S1 + existing PV/DG'}
SHORT = {'S1_peak_delivery': 'S1 midday (LP-used)', 'S2_pre_curtailment': 'S2 midday (pre-curt.)',
         'S3_evening_nonsolar': 'S3 evening (BESS)', 'HSN_caiso': 'HSN (CAISO)', 'SSN_caiso': 'SSN (CAISO)'}
ORDER = ['S1_peak_delivery', 'S2_pre_curtailment', 'S3_evening_nonsolar', 'HSN_caiso', 'SSN_caiso']


def deliverability():
    chk = pd.read_csv(R1 / 'portfolio_vs_hosting.csv')
    info = json.loads((R1 / 'run_info.json').read_text())
    num('monitored', info['monitored_branches']); num('monlines', info['monitored_lines'])
    num('montx', info['monitored_branches'] - info['monitored_lines']); num('ncont', info['contingencies'])
    num('nbridges', info['radial_outages_skipped']); num('headroomP', info['energy_headroom_mw']['P_prorata'])
    num('headroomAone', info['energy_headroom_mw']['A1_mg_backdown']); num('headroomAtwo', info['energy_headroom_mw']['A2_displace_inflow'])
    num('growthMW', info['load_growth_2050_mw_at_pwf_scale'])
    s = info['subset']
    for k in ('pairs', 'elements', 'groups', 'rows_full', 'cases', 'cases_alpha_lt1', 'in_sample_exact_alpha_lt1',
              'out_of_sample_cases', 'out_of_sample_cases_alpha_lt1', 'out_of_sample_exact_alpha_lt1', 'in_sample_same_verdict',
              'out_of_sample_same_verdict', 'out_of_sample_pairs'):
        num('subset' + k, s[k])
    num('subsetpct', 100 * s['pairs'] / s['rows_full'], 1)
    m = chk[(chk.siting == 'reference') & chk.main & (chk['mode'] == 'circle') & chk.condition.isin(ORDER)].copy()
    m['cond'] = m.condition.map(SHORT); m['ord'] = m.condition.map({c: i for i, c in enumerate(ORDER)})
    m = m.sort_values(['year', 'ord'])
    m['binding'] = m.binding_element.map(lambda x: pretty(str(x).split(' (')[0]))
    m['pct_tested'] = 100 * m.rows_tested / m.rows_total
    table('deliv_main', m, [('year', 'Year', 0), ('cond', 'Study condition', None), ('need_mw', 'Need MW', 0),
                            ('alpha', r'$\alpha$ $\uparrow$', 2), ('n_upgrades_to_alpha1', r'Groups $\downarrow$', 0),
                            ('new_violation_elements', r'New viol. $\downarrow$', 0),
                            ('worsened_preexisting_added_mw', r'Pre-ex. +MW $\downarrow$', 0),
                            ('binding', 'First binding element', None)],
          r'Deliverability of the fixed portfolios, reference siting, main case (pro-rata absorption, target-year load, 5\% circle). '
          r'$\alpha$: largest scaling of the joint study output meeting every attributable N-0/N-1 limit ($\alpha\ge1$: deliverable).',
          'tab:deliv', size=r'\footnotesize')
    num('pcttestedmin', m.pct_tested.min()); num('pcttestedmax', m.pct_tested.max())
    for _, r in m.iterrows():
        num(f"alpha{r.year}{r.condition}", r.alpha, 2); num(f"need{r.year}{r.condition}", r.need_mw)
        num(f"ups{r.year}{r.condition}", r.n_upgrades_to_alpha1)
    num('preworstadd', m.worsened_preexisting_added_mw.max())
    ref = chk[(chk.siting == 'reference') & (chk['mode'] == 'circle') & chk.condition.isin(ORDER)].copy()
    ref['variant'] = ref.case.map({'P_prorata': 'P', 'A1_mg_backdown': 'A1', 'A2_displace_inflow': 'A2', 'B_export': 'B'}) + \
        np.where(ref.load_level == '2035 PWF load', '0', '')
    ab = ref.pivot_table(index=['year', 'condition'], columns='variant', values='alpha').reset_index()
    ab['cond'] = ab.condition.map(COND); ab['ord'] = ab.condition.map({c: i for i, c in enumerate(ORDER)}); ab = ab.sort_values(['year', 'ord'])
    cols = [('year', 'Year', 0), ('cond', 'Condition', None), ('P', r'Pro rata', 2), ('A1', 'A1 MG only', 2),
            ('A2', 'A2 inflows', 2), ('B', 'B export', 2)]
    cols += [(c, h, 2) for c, h in [('P0', 'Pro rata, 2035 load'), ('B0', 'B, 2035 load')] if c in ab]
    table('deliv_absorption', ab, cols, r'Sensitivity of $\alpha$ $\uparrow$ to the absorption convention (reference siting, 5\% circle). '
          r'For 2050 the main case includes target-year load growth; \enquote{2035 load} omits it.', 'tab:absorption', size=r'\scriptsize')
    ru = chk[(chk.siting == 'reference') & chk.main & chk.condition.isin(ORDER)].pivot_table(
        index=['year', 'condition'], columns='mode', values='alpha').reset_index()
    ru['cond'] = ru.condition.map(COND); ru['ord'] = ru.condition.map({c: i for i, c in enumerate(ORDER)}); ru = ru.sort_values(['year', 'ord'])
    table('deliv_rule', ru, [('year', 'Year', 0), ('cond', 'Condition', None), ('all', r'Every row tested', 2),
                             ('circle', r'5\% circle (main)', 2), ('circle_cap', r'Circle + 20 unit / 1,500 MW cap', 2)],
          r'Sensitivity of $\alpha$ $\uparrow$ to the attribution rule. \enquote{Every row tested} also forbids any worsening of the one pre-existing '
          r'overload, which drives $\alpha$ to zero; the cap variant raises only the highest-impact circle units (a bound).', 'tab:rule', size=r'\scriptsize')
    si = chk[(chk['mode'] == 'circle') & chk.main & chk.condition.isin(ORDER) & (chk.year == 2050)].pivot_table(
        index=['condition'], columns='siting', values='alpha').reset_index()
    si['cond'] = si.condition.map(COND); si['ord'] = si.condition.map({c: i for i, c in enumerate(ORDER)}); si = si.sort_values('ord')
    table('deliv_siting', si, [('cond', 'Condition (2050)', None), ('reference', 'Reference', 2),
                               ('reference_bess_at_load', 'Reference, BESS at load', 2),
                               ('regional_12_milp', 'Regional-12 MILP', 2), ('local_4_milp', 'Local-4 MILP', 2)],
          r'Sensitivity of 2050 $\alpha$ $\uparrow$ to siting (main case).', 'tab:siting', size=r'\small')
    for _, r in si.iterrows():
        num(f"alphaBL{r.condition}", r.reference_bess_at_load, 2)
    ex = chk[chk.condition == 'S1_plus_existing_pv_dg'].copy()
    ex['case'] = ex.case.map({'P_prorata': 'Pro rata', 'A1_mg_backdown': 'A1 MG only', 'A2_displace_inflow': 'A2 inflows', 'B_export': 'B export'})
    table('deliv_existingpv', ex, [('year', 'Year', 0), ('case', 'Absorption', None), ('need_mw', 'Need (MW)', 0),
                                    ('alpha', r'$\alpha$ $\uparrow$', 2), ('status', 'Status', None)],
          r'Midday sensitivity with existing central PV (split equally over the 106 UFV buses) and existing DG added.', 'tab:existingpv')
    f = pd.read_csv(R1 / 'study_factors.csv')
    pv = f.pivot_table(index=['year', 'technology'], columns='condition', values='factor').reset_index()
    pv['technology'] = pv.technology.str.replace('_', ' ')
    table('study_factors', pv, [('year', 'Year', 0), ('technology', 'Technology', None)] +
          [(c, COND[c].split(' (')[0], 2) for c in ORDER],
          'Study output levels (fraction of new nameplate). S1–S3: mean of the top 1\\% LP hours; HSN/SSN: CAISO exceedance definitions.',
          'tab:factors', size=r'\scriptsize')
    p = pd.read_csv(R1 / 'poi_hosting_capacity.csv').sort_values('hc_P_prorata_mw', ascending=False)
    num('poitop', p.hc_P_prorata_mw.max()); num('poitopname', station(p.station.iloc[0]))
    num('npoi', len(p)); num('poiabovefive', int((p.hc_P_prorata_mw >= 5000).sum()))
    return chk, p


def figures_deliverability(chk, poi):
    p = poi.sort_values('hc_P_prorata_mw').copy()
    lab = [f"{station(s)} {k} kV" for s, k in zip(p.station, p.kv)]
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    y = np.arange(len(p))
    ax.barh(y, p.hc_P_prorata_mw / 1000, height=0.62, color=C1, edgecolor='white', linewidth=0.8)
    for yi, v in zip(y, p.hc_P_prorata_mw / 1000):
        ax.text(v + 0.08, yi, f'{v:.1f}', va='center', fontsize=6.5, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=6.5)
    ax.set_xlabel('Hosting capacity (GW), pro-rata absorption, N-0 and N-1, 5% circle')
    ax.grid(axis='y', visible=False)
    fig.tight_layout(); fig.savefig(FIG / 'poi_hosting.pdf'); plt.close(fig)
    si = chk[(chk['mode'] == 'circle') & chk.main & chk.condition.isin(ORDER) & (chk.year == 2050)]
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    sits = [('reference', 'Reference siting', C1), ('reference_bess_at_load', 'BESS at load buses', C2),
            ('regional_12_milp', 'Regional-12 MILP', C3), ('local_4_milp', 'Local-4 MILP', C4)]
    x = np.arange(len(ORDER)); w = 0.2
    for i, (s, l, c) in enumerate(sits):
        v = [float(si[(si.siting == s) & (si.condition == cnd)].alpha.iloc[0]) for cnd in ORDER]
        ax.bar(x + (i - 1.5) * w, v, width=w * 0.9, color=c, label=l, edgecolor='white', linewidth=0.6)
        for xi, vi in zip(x + (i - 1.5) * w, v):
            ax.text(xi, vi + 0.03, f'{vi:.2f}', ha='center', fontsize=5.6, color=INK2, rotation=90)
    ax.axhline(1.0, color=INK, lw=0.9, ls='--'); ax.text(len(ORDER) - 0.45, 1.03, r'$\alpha$ = 1', fontsize=7, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(['Off-peak S1', 'Off-peak S2', 'Peak S3', 'Peak HSN', 'Peak SSN'], fontsize=7)
    ax.set_ylabel(r'$\alpha$ (higher is better)'); ax.set_ylim(0, 2.25)
    ax.legend(ncol=4, fontsize=6.6, frameon=False, loc='upper left', bbox_to_anchor=(0, 1.16))
    ax.grid(axis='x', visible=False)
    fig.tight_layout(); fig.savefig(FIG / 'alpha_2050.pdf'); plt.close(fig)


def portfolio():
    d = pd.read_csv(R2 / 'portfolio_pypsa.csv')
    g = d[d.case.str.startswith('REFRESH')]
    for _, r in g.iterrows():
        num(f'obj{r.year}', r.objective_mbrl, 2); num(f'objlp{r.year}', r.lp_objective_mbrl, 2)
        num(f'relerr{r.year}', sci(r.objective_rel_diff))
    x = d[d.case.str.endswith('hydro_dry')]
    for _, r in x.iterrows():
        num(f'objdry{r.year}', r.objective_mbrl, 2)
        num(f'relerrdry{r.year}', sci(r.objective_rel_diff))
    lab = {'REFRESH_2035': 'Base (LP record)', 'REFRESH_2050': 'Base (LP record)'}
    d['label'] = d.case.map(lambda c: lab.get(c, c[5:].replace('_', ' ').replace('plus6', '+6%').replace('minus6', '-6%')
                                       .replace('plus50', '+50%').replace('minus25', '-25%').replace('param', '(param.)')))
    d['solar'] = d.cap_solar_central_gw + d.cap_solar_distributed_gw
    d['other'] = d.cap_bio_gw + d.cap_small_hydro_gw + d.cap_biogas_gw
    oth = d.groupby('year').other.agg(['min', 'max'])
    assert ((oth['max'] - oth['min']) < 5e-4).all(), oth                 # constant per year -> stated in the caption
    table('portfolio_sens', d, [('year', 'Year', 0), ('label', 'Portfolio', None), ('objective_mbrl', r'Cost (R\$ M) $\downarrow$', 0),
                                ('cap_solar_central_gw', 'PV', 2), ('cap_solar_distributed_gw', 'DG PV', 2),
                                ('cap_wind_site_gw', 'Wind', 2), ('cap_bess_gw', 'BESS', 2),
                                ('imports_twh', 'Imports', 2), ('curtailment_twh', 'Curtailed', 2)],
          r'Base and sensitivity portfolios (PyPSA): present-value cost, new capacity (GW) and annual energy (TWh). Sensitivities follow '
          r'the CAISO/PJM list (Lectures 9--10): load $\pm$6\%, dry and wet hydrology, gas price. \enquote{Wet} is a labelled parametric '
          r'+10\% monthly hydro energy (no wet-year data); curtailment is not unique in the LP (zero-cost resources). Biomass + small '
          r'hydro + biogas is ' + f"{oth.loc[2035, 'min']:.2f}" + r'~GW (2035) and ' + f"{oth.loc[2050, 'min']:.2f}" +
          r'~GW (2050) in every case.', 'tab:portfolios', size=r'\footnotesize')
    s = pd.read_csv(R2 / 'sensitivity_portfolio_deliverability.csv')
    s = s[s.portfolio.str.startswith('20')].copy()
    s['label'] = s.portfolio.str[5:].map(lambda c: c.replace('_', ' ').replace('plus6', '+6%').replace('minus6', '-6%')
                                          .replace('plus50', '+50%').replace('minus25', '-25%').replace('param', '(param.)'))
    s['cond'] = s.condition.map({'HSN_caiso': 'HSN', 'SSN_caiso': 'SSN'})
    s = s[s.year == 2050]
    table('sens_deliv', s, [('label', 'Sensitivity portfolio (2050)', None), ('cond', 'Cond.', None), ('need_mw', 'Need (MW)', 0),
                            ('alpha', r'$\alpha$ $\uparrow$', 2), ('n_upgrades_to_alpha1', r'Upgrade groups $\downarrow$', 0)],
          r'Peak deliverability of the 2050 sensitivity portfolios (reference siting by allocation weights, main case). '
          r'The 2035 sensitivity portfolios equal the base portfolio (policy floors bind) and stay deliverable ($\alpha\ge3.45$).',
          'tab:sensdeliv', size=r'\scriptsize')
    m = pd.read_csv(R2 / 'matching_2050.csv')
    table('matching', m, [('definition', '2050 definition', None), ('objective_mbrl', r'Cost (R\$ M PV) $\downarrow$', 0),
                          ('cost_vs_H0_pct', r'vs H0 (\%) $\downarrow$', 1), ('cap_solar_central_gw', 'Central PV', 2),
                          ('cap_wind_site_gw', 'Wind', 2), ('cap_bess_gw', 'BESS', 2), ('imports_twh', 'Imports (TWh)', 1),
                          ('exports_twh', 'Exports (TWh)', 1)],
          r'Hourly versus annual clean matching in 2050 (capacities in GW; gas is zero in every case).', 'tab:matching', size=r'\scriptsize')
    for _, r in m.iterrows():
        num('match' + r.definition.split()[0], r.cost_vs_H0_pct, 1)
        num('matchabs' + r.definition.split()[0], abs(r.cost_vs_H0_pct), 1)
    m2 = m.set_index(m.definition.str.split().str[0])
    num('matchAimports', m2.loc['A1', 'imports_twh'], 1); num('matchAexports', m2.loc['A1', 'exports_twh'], 1)
    num('matchHexports', m2.loc['H1', 'exports_twh'], 1)


def adequacy():
    g = pd.read_csv(PS / 'data' / 'gads_class_averages.csv')
    g['cls'] = g.gads_category.str.title()
    table('gads', g[g.model_class != 'hydro_all'], [('cls', 'NERC GADS category', None), ('units', 'Units', 0), ('unit_years', 'Unit-years', 0),
                         ('EFORd_pct', r'EFORd (\%)', 2), ('mean_forced_outage_h', 'Mean outage (h)', 0)],
          r'Forced-outage data: NERC GADS Generating Unit Statistical Brochure 4, 2016--2020, all units reporting. Mapping: reservoir hydro '
          r'$\to$ Hydro 30+ MW (one plant unit below 30 MW $\to$ Hydro 1--29 MW); gas $\to$ combined cycle; biogas engines $\to$ diesel; '
          r'BESS $\to$ pumped storage (no battery class in GADS). Mean outage = forced-outage hours / forced outages.', 'tab:gads', size=r'\small')
    a = pd.read_csv(R2 / 'adequacy_copt.csv')
    table('ra_copt', a, [('year', 'Year', 0), ('counting', 'Counting', None), ('LOLE_days', r'LOLE (d/yr) $\downarrow$', 3),
                         ('LOLH_h', r'LOLH (h/yr) $\downarrow$', 2), ('EUE_MWh', r'EUE (MWh/yr) $\downarrow$', 0),
                         ('firm_capacity_gap_mw', r'Firm gap to 0.1 d/yr (MW) $\downarrow$', 0)],
          r'Method A (COPT convolution, capacity only). Firm gap: perfect capacity to add ($+$) or that could be removed ($-$) for LOLE = 0.1 '
          r'day/yr. \enquote{Deliverable only} scales every new resource by the peak deliverable fraction $\min(1,\alpha_{HSN},\alpha_{SSN})$.',
          'tab:copt', size=r'\scriptsize')
    e = pd.read_csv(R2 / 'adequacy_elcc.csv')
    table('ra_elcc', e, [('year', 'Year', 0), ('counting', 'Counting', None), ('class', 'Resource class', None),
                         ('nameplate_mw', 'Nameplate (MW)', 0), ('elcc_mw', r'ELCC (MW) $\uparrow$', 0),
                         ('elcc_pct_of_nameplate', r'ELCC (\% of nameplate) $\uparrow$', 1)],
          r'Effective load-carrying capability by the CAISO calibration (Lecture 11, method A).', 'tab:elcc', size=r'\scriptsize')
    b = pd.read_csv(R2 / 'adequacy_smc.csv')
    GAPRES = 12000.0 / 2 ** 7               # adequacy.firm_gap(hi=12000, tol=100): bisection stops at 93.75 MW
    assert np.allclose(np.round(b.firm_capacity_gap_mw / GAPRES), b.firm_capacity_gap_mw / GAPRES)
    num('gapres', GAPRES)
    b['o'] = b.counting.ne('nameplate'); b['h'] = b.hydrology.ne('normal')
    b = b.sort_values(['year', 'h', 'o']).drop(columns=['o', 'h'])
    b['hydrology'] = b.hydrology.str.replace(' analogue', '', regex=False)
    table('ra_smc', b, [('year', 'Year', 0), ('hydrology', 'Hydrology', None), ('counting', 'Counting', None),
                        ('no_outage_EUE_MWh', r'No-outage EUE $\downarrow$', 0), ('EUE_MWh', r'EUE $\downarrow$', 0),
                        ('EUE_pct_of_load', r'EUE (\%) $\downarrow$', 3), ('p_any_shortfall', r'P(any) $\downarrow$', 2),
                        ('LOLE_days_convention', r'LOLE (d/yr)$^\dagger$ $\downarrow$', 1), ('firm_capacity_gap_mw', r'Firm gap (MW) $\downarrow$', 0)],
          r'Method B (sequential Monte Carlo, energy-limited PyPSA dispatch, ' + str(int(b.samples.max())) + r' outage histories per case; '
          r'dry = dry-hydrology analogue). EUE in MWh/yr and \% of load; P(any): probability of any shortfall in the year. '
          r'P(any shortfall) bounds LOLE from below (each year with a shortfall has $\ge$1 loss-of-load day). Firm gap: perfect capacity '
          r'for P(any shortfall) $\le$ 0.1, by bisection on 0--12,000 MW to ' + f'{GAPRES:.2f}' + r'~MW resolution (each value is the upper end of '
          r'its interval). \enquote{Deliverable only} scales the output of new resources by $d$ in every hour, a loose upper '
          r'bound in an energy dispatch (midday output is deliverable). $^\dagger$Depends on how energy deficits are spread over hours '
          r'(rotating 5\% tranches).',
          'tab:smc', size=r'\scriptsize')
    for _, r in b.iterrows():
        k = f"{r.year}{'dry' if r.hydrology.startswith('dry') else 'normal'}{'del' if r.counting.startswith('deliv') else 'name'}"
        num('eue' + k, r.EUE_MWh / 1000, 1); num('gap' + k, r.firm_capacity_gap_mw); num('pany' + k, r.p_any_shortfall, 2)
        num('euepct' + k, r.EUE_pct_of_load, 2)
    for _, r in a.iterrows():
        k = f"{r.year}{r.counting}"
        num('pcgap' + k, r.firm_capacity_gap_mw); num('lole' + k, r.LOLE_days, 2)
    num('dfrac', b.deliverable_fraction.min(), 2)
    assert (b[(b.year == 2050) & (b.counting == 'nameplate')].p_any_shortfall == 1.0).all()   # text: 'every simulated year'
    assert (b[b.year == 2035].EUE_MWh == 0).all() and (a[a.year == 2035].query('counting != "nameplate, no firm imports"').EUE_MWh == 0).all()  # text: 2035 no unserved energy
    a35 = a[(a.year == 2035) & (a.counting == 'nameplate')].iloc[0]
    num('pcsurplusA', -a35.firm_capacity_gap_mw)


def mitigation():
    alt = pd.read_csv(R2 / 'mitigation_alternatives.csv')
    plan = pd.read_csv(R2 / 'mitigation_plan.csv'); T = pd.read_csv(R2 / 'mitigation_totals.csv').set_index('siting')
    pre = pd.read_csv(R2 / 'mitigation_preexisting.csv')
    plan['grp'] = plan.group.map(pretty)
    plan['alt'] = plan.alternative.map({'new parallel circuit(s)': 'new circuit', 'reconductor existing circuit(s) (HTLS)': 'reconductor (HTLS)',
                                        'add transformer bank(s)': 'add bank(s)'})
    plan['grp'] = plan.grp.str.replace(' transformation', '', regex=False)
    plan['grp'] = np.where(plan.group_type == 'line', plan.grp,
                           plan.grp + ' ' + plan['class'].str.replace(' transformation', '', regex=False) + ' transf.')
    plan['flag'] = np.where(plan.non_wires_candidate, 'yes', '')
    plan['sit'] = plan.siting.map({'reference': 'Reference', 'reference_bess_at_load': 'BESS at load'})
    table('mit_plan', plan, [('sit', 'Siting', None), ('grp', 'Upgrade group', None),
                             ('required_increase_pct', r'Need (\%)', 1), ('n_existing', 'Existing', 0), ('alt', 'Least-cost alternative', None),
                             ('n_elements', 'Elements', 0), ('capex_mbrl', r'Capex (R\$ M) $\downarrow$', 1),
                             ('flag', r'Non-wires', None)],
          r'Attributable reinforcement plan (2050, main case), verified on a network that contains the new elements. Need: largest '
          r'required increase over the study conditions. Non-wires candidate: line need $\le$10\%, where re-rating, limiting-element removal '
          r'or dynamic line rating may suffice (not costable with the data). Existing: parallel elements in the group.', 'tab:plan', size=r'\footnotesize')
    for s in ('reference', 'reference_bess_at_load'):
        k = 'R' if s == 'reference' else 'L'
        num('capex' + k, T.loc[s, 'capex_verified_mbrl']); num('capexlow' + k, T.loc[s, 'capex_low_mbrl'])
        num('capexhigh' + k, T.loc[s, 'capex_high_mbrl']); num('capexexcl' + k, T.loc[s, 'capex_excluding_small_line_needs_mbrl'])
        num('arr' + k, T.loc[s, 'annual_revenue_requirement_mbrl'], 1); num('arrexcl' + k, T.loc[s, 'annual_revenue_requirement_excl_small_mbrl'], 1)
        num('groups' + k, T.loc[s, 'groups'])
    r = pre.iloc[0]
    num('prebase', r.base_overload_mw, 1); num('preadd', r.portfolio_adds_mw, 1); num('prebanks', r.banks_with_portfolio)
    num('precapexinc', r.capex_attributable_increment_mbrl, 1); num('precapexbase', r.capex_base_mbrl, 1)
    v = pd.read_csv(R2 / 'mitigation_verification.csv')
    num('veralphamin', v[[c for c in v.columns if c.startswith('alpha_')]].min(axis=1).min(), 3)
    uc = pd.read_csv(PS / 'data' / 'unit_costs.csv')
    uc['val'] = uc.value_thousand_brl / 1000
    uc.loc[uc.item.str.startswith('Reconductoring'), 'val'] = uc.value_thousand_brl
    table('unit_costs', uc, [('item', 'Item', None), ('unit', 'Unit', None), ('val', 'Value (R\\$ M or ratio)', 3),
                             ('source', 'Source', None), ('price_basis', 'Price basis', None)],
          r'Unit costs used for the costed alternatives (published EPE/ANEEL values; price bases differ and are not indexed).',
          'tab:unitcosts', align=r'>{\raggedright\arraybackslash}p{3.6cm}>{\raggedright\arraybackslash}p{1.7cm}r>{\raggedright\arraybackslash}p{5.4cm}>{\raggedright\arraybackslash}p{2.6cm}', size=r'\scriptsize')


def regional():
    f = pd.read_csv(ROOT / 'regional_comparison' / 'results' / 'FINAL_COMPARISON.csv')
    f = f[f.scenario_set == 'local_consumption']
    for _, r in f.iterrows():
        num(f"poicost{r.year}{'L' if r.scope.startswith('Local') else 'R'}", r.optimized_connection_cost_million_brl, 1)
    f['scope'] = f.scope.str.replace('_', ' ')
    table('regional', f, [('year', 'Year', 0), ('scope', 'Candidate set', None),
                          ('optimized_connection_cost_million_brl', r'POI connection cost (R\$ M) $\downarrow$', 1)],
          r'Busbar-mapping alternatives: optimised POI connection cost (the POI half of interconnection cost; network upgrades in '
          r'Section~\ref{sec:mitigation}).', 'tab:regional')


def p7_ac():
    p = pd.read_csv(R2 / 'contingency_p7.csv'); info = pd.read_csv(R2 / 'contingency_p7_info.csv').iloc[0]
    num('pairs', info.p7_pairs); num('pgate', sci(info.gate_max_abs_diff_mw))
    p['cond'] = p.condition.map(COND); p['sit'] = p.siting.map({'reference': 'Reference', 'reference_bess_at_load': 'BESS at load'})
    p['bind'] = p.binding_P7.map(lambda s: pretty(s.split(' for loss')[0]) if isinstance(s, str) and s else '—')
    p['lost'] = p.binding_P7.map(lambda s: pretty(s.split('both ')[-1].split(' circuits')[0]) if isinstance(s, str) and s else '—')
    p['cond'] = p.condition.map(SHORT)
    p['ord'] = p.condition.map({c: i for i, c in enumerate(ORDER)})
    lo35 = np.floor(10 * p[p.year == 2035].alpha_P7_rows.min()) / 10
    q = p[p.year == 2050]
    NETS = {'existing grid': 'ex', 'with verified plan': 'plan', 'with plan + P7 remedy': 'rem'}
    w = q.pivot_table(index=['siting', 'sit', 'condition', 'cond', 'ord'], columns='network', values='alpha_P1_P7').rename(columns=NETS)
    e = q[q.network == 'existing grid'].set_index(['siting', 'condition'])
    w = w.reset_index()
    w['a1'] = [e.loc[(r.siting, r.condition), 'alpha_P1'] for r in w.itertuples()]
    w['bind'] = [e.loc[(r.siting, r.condition), 'bind'] for r in w.itertuples()]
    w['lost'] = [e.loc[(r.siting, r.condition), 'lost'] for r in w.itertuples()]
    w = w.sort_values(['sit', 'ord'], ascending=[False, True])
    for (sit, net_), g in q.groupby(['siting', 'network']):
        k = ('ref' if sit == 'reference' else 'bal') + NETS[net_]
        num(f'psevenmin{k}', g.alpha_P1_P7.min(), 2)
        num(f'psevenbinds{k}', int(g.P7_binds_first.sum()))
    table('p7', w, [('sit', 'Siting', None), ('cond', 'Condition', None), ('a1', r'P1 $\uparrow$', 2),
                    ('ex', r'P1+P7 $\uparrow$', 2), ('plan', r'Plan $\uparrow$', 2), ('rem', r'Plan + remedy $\uparrow$', 2),
                    ('bind', 'Binding P7 element (existing grid)', None)],
          r'NERC TPL P7 check (2050), $\alpha$ by network. P1: existing grid, P1 rows only; P1+P7: existing grid with P7 rows; '
          r'Plan: verified phase-3 plan (sized for P1), P1+P7; Plan + remedy: with the P7 remedy circuits (--- = none needed). '
          r'Every pair of existing parallel circuits on one corridor is assumed to share a structure (conservative); new circuits '
          r'are on their own structures. 2035 stays deliverable with P7 ($\alpha\ge' + f'{lo35:.1f}' + r'$ on P7 rows).', 'tab:p7', size=r'\scriptsize')
    rem = pd.read_csv(R2 / 'contingency_p7_remedy.csv')
    T = pd.read_csv(R2 / 'mitigation_totals.csv').set_index('siting')
    B = pd.read_csv(R2 / 'bcr.csv')
    fac = T.loc['reference', 'annual_revenue_requirement_mbrl'] / T.loc['reference', 'capex_verified_mbrl']
    rr = rem[rem.siting == 'reference']
    num('premn', int(rr.n_elements.sum())); num('premcapex', rr.capex_mbrl.sum(), 1); num('premarr', fac * rr.capex_mbrl.sum(), 1)
    for i, r in enumerate(rr.itertuples()):              # in the order the remedy loop added them
        tag = ['one', 'two', 'three', 'four'][i]
        num(f'premname{tag}', pretty(r.group)); num(f'premcapex{tag}', r.capex_mbrl, 1)
        num(f'premkm{tag}', r.length_km * 1.25, 1); num(f'preminc{tag}', r.required_increase_pct, 0)
    num('premplancapex', T.loc['reference', 'capex_verified_mbrl'] + rr.capex_mbrl.sum(), 1)
    b0 = B[(B.measure == 'reinforcement plan, reference siting') & (B.sensitivity == 'base')].iloc[0]
    num('prembcr', b0.annual_benefit_mbrl / (b0.arr_full_plan_mbrl + fac * rr.capex_mbrl.sum()), 1)
    assert len(rr) <= 4 and (rem.siting == 'reference_bess_at_load').sum() == 0
    g = pd.read_csv(R2 / 'ac_gate.csv').iloc[0]
    num('acmed', g.median_err_mw, 3); num('acshare', 100 * g.share_within_2mw_2pct, 1); num('acvmed', g.median_v_err_pu, 4)
    num('acpninefive', g.p95_err_mw, 1)
    v = pd.read_csv(R2 / 'ac_voltage_screen.csv')
    v['cond'] = v.condition.map(COND); v['sit'] = v.siting.map({'reference': 'Reference', 'reference_bess_at_load': 'BESS at load'})
    v['ord'] = v.condition.map({c: i for i, c in enumerate(ORDER)}); v = v.sort_values(['year', 'sit', 'ord'])
    v['tight'] = v.tightest_bus.map(lambda c: f"{NAMES.get(c.split('-')[0].rstrip('-'), c.split('-')[0])} {int(c[-3:])} kV")
    v['cond'] = v.condition.map(SHORT)
    table('ac', v, [('year', 'Year', 0), ('sit', 'Siting', None), ('cond', 'Condition', None), ('min_vm_mg_230plus', r'$V_{min}$ (pu)', 3),
                    ('max_vm_mg_230plus', r'$V_{max}$ (pu)', 3), ('new_below_limit', r'New low $\downarrow$', 0),
                    ('new_above_limit', r'New high $\downarrow$', 0), ('min_margin_pu', r'Min. margin (pu) $\uparrow$', 3),
                    ('tight', 'Tightest bus', None)],
          r'AC voltage screen (PyPSA Newton--Raphson, N-0, MG buses $\ge$230 kV, bus-specific normal limits of the PWF). The base case '
          r'already has ' + str(int(v.base_above_limit.max())) + r' buses above their limit. Generator reactive limits are not enforced.',
          'tab:ac', size=r'\scriptsize')


def pcm():
    if not (R2 / 'pcm_scenarios.csv').exists():
        return
    S = pd.read_csv(R2 / 'pcm_scenarios.csv'); C = pd.read_csv(R2 / 'pcm_system_cost.csv', index_col=0)
    S = S.merge(C[['production_mbrl_yr', 'unused_renewable_twh', 'system_cost_mbrl_yr', 'congestion_cost_vs_copperplate_mbrl_yr']],
                left_on='scenario', right_index=True)
    lab = {'2035_copperplate': '2035, no network limits', '2035_existing_grid': '2035, existing grid',
           '2050_copperplate': '2050, no network limits', '2050_existing_grid': '2050, existing grid',
           '2050_with_plan': '2050, grid + plan', '2050_bess_at_load_existing_grid': '2050 BESS at load, existing grid',
           '2050_bess_at_load_with_plan': '2050 BESS at load, grid + plan'}
    PP = pd.read_csv(R2 / 'portfolio_pypsa.csv').set_index('case')
    LOAD_TWH = {2035: float(PP.loc['REFRESH_2035', 'load_twh']), 2050: float(PP.loc['REFRESH_2050', 'load_twh'])}
    S['label'] = S.scenario.map(lab)
    S['ens_gwh'] = S.ens_mwh / 1000
    # congestion effect on total unused zero-cost energy: the VRE/hydro split is not unique (degenerate LP), the total is
    S['avail_all_twh'] = S.profile_available_twh + S.hydro_twh + S.hydro_spill_twh
    cp_unused = S[S.scenario.str.endswith('copperplate')].set_index('year').unused_renewable_twh
    S['cong_curt_pct'] = 100 * (S.unused_renewable_twh - S.year.map(cp_unused)) / S.avail_all_twh
    S['unused_pct'] = 100 * S.unused_renewable_twh / S.avail_all_twh
    table('pcm', S, [('label', 'Scenario', None), ('production_mbrl_yr', r'Production $\downarrow$', 1),
                     ('ens_gwh', r'Unserved GWh $\downarrow$', 1), ('unused_renewable_twh', r'Unused TWh $\downarrow$', 2),
                     ('unused_pct', r'Unused \% $\downarrow$', 1), ('cong_curt_pct', r'Cong. curt. \% $\downarrow$', 2),
                     ('system_cost_mbrl_yr', r'System cost $\downarrow$', 0),
                     ('congestion_cost_vs_copperplate_mbrl_yr', r'Congestion $\downarrow$', 0),
                     ('flowgate_rows_used', 'Rows', 0)],
          r'Network-constrained 8,760-h production cost (PyPSA, monthly, lazy flowgates on all N-0/N-1 rows). System cost = production '
          r'+ unserved energy at R\$ 8,327.76/MWh + unused zero-cost energy (curtailment of profile-based resources and unused hydro budget) '
          r'at R\$ 209.47/MWh. Unused \%: share of the available profile-based and hydro energy; Cong. curt.: unused energy above the '
          r'same year without network limits, in the same units.',
          'tab:pcm', size=r'\footnotesize', note=r'Costs in R\$ M/yr; Rows: flowgate rows added by the lazy check.')
    for _, r in S.iterrows():
        k = re.sub('[^A-Za-z]', '', r.scenario.replace('2035', 'a').replace('2050', 'b'))
        num('ens' + k, r.ens_gwh, 1); num('syscost' + k, r.system_cost_mbrl_yr); num('cong' + k, r.congestion_cost_vs_copperplate_mbrl_yr)
        num('unused' + k, r.unused_renewable_twh, 2); num('rows' + k, r.flowgate_rows_used)
        num('unusedpct' + k, r.unused_pct, 1); num('congcurtpct' + k, r.cong_curt_pct, 2)
        num('enspct' + k, 100 * r.ens_mwh / (LOAD_TWH[int(r.year)] * 1e6), 2)
    B = pd.read_csv(R2 / 'bcr.csv')
    B['bcr_lo'] = B.bcr_full_plan; B['bcr_hi'] = B.bcr_excl_small
    bb = B[~B.measure.str.startswith('site')]
    base_b = bb[bb.sensitivity == 'base'].set_index('measure').bcr_full_plan
    curt = bb[bb.sensitivity.str.startswith('unused')]
    curt_rel = float((curt.bcr_full_plan / curt.measure.map(base_b) - 1).abs().max() * 100)
    Bt = B[~B.measure.str.startswith('site') & B.sensitivity.isin(['base', 'VOLL -50%', 'VOLL +50%'])].copy()
    Bt['measure'] = Bt.measure.map({'reinforcement plan, reference siting': 'Plan, reference siting',
                                    'reinforcement plan, BESS at load': 'Plan, BESS at load'})
    table('bcr', Bt, [('measure', 'Measure', None), ('sensitivity', 'Valuation', None),
                      ('annual_benefit_mbrl', r'Benefit (R\$ M/yr) $\uparrow$', 0),
                      ('arr_full_plan_mbrl', r'Cost, full plan (R\$ M/yr)', 1), ('bcr_full_plan', r'BCR $\uparrow$', 1),
                      ('arr_excl_small_mbrl', r'Cost, excl. small needs', 1), ('bcr_excl_small', r'BCR $\uparrow$', 1)],
          r'Benefit-cost ratios (annual benefit / annual revenue requirement at CRF(8\%, 30 yr) + 1.5\% O\&M). BCR $>$ 1 justifies the '
          r'measure (CAISO/PJM economic criterion). Valuing unused zero-cost energy $\pm$50\% changes the BCRs by at most '
          + f'{curt_rel:.1f}' + r'\%.',
          'tab:bcr', size=r'\footnotesize')
    base = B[B.sensitivity == 'base'].set_index('measure')
    num('bcrRfull', base.loc['reinforcement plan, reference siting', 'bcr_full_plan'], 1)
    num('bcrRexcl', base.loc['reinforcement plan, reference siting', 'bcr_excl_small'], 1)
    num('benR', base.loc['reinforcement plan, reference siting', 'annual_benefit_mbrl'])
    num('bcrL', base.loc['reinforcement plan, BESS at load', 'bcr_full_plan'], 1)
    num('benL', base.loc['reinforcement plan, BESS at load', 'annual_benefit_mbrl'])
    num('benreloc', base.loc['site the planned BESS at load buses (no grid plan)', 'annual_benefit_mbrl'])
    num('bcrRmin', B[B.measure == 'reinforcement plan, reference siting'].bcr_full_plan.min(), 1)
    M = pd.read_csv(R2 / 'pcm_monthly.csv')
    MON = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    bud = M[M.scenario == '2050_copperplate'].set_index('month').hydro_budget_gwh
    mmin = int(bud.idxmin())
    for s_ in ('2050_with_plan', '2050_bess_at_load_existing_grid', '2050_bess_at_load_with_plan'):
        short = M[(M.scenario == s_) & (M.ens_gwh > 1e-3)].month.tolist()
        assert short == [mmin], (s_, short, mmin)       # the text says: only in the month with the smallest hydro budget
    num('budgetminmonth', MON[mmin - 1]); num('budgetmin', bud.min()); num('budgetothermin', bud.drop(mmin).min())
    num('budgetothermax', bud.drop(mmin).max())
    FG = pd.read_csv(R2 / 'pcm_flowgates.csv')
    top = FG[FG.scenario == '2050_bess_at_load_with_plan'].element.value_counts()
    for i, tag in enumerate(['one', 'two', 'three']):
        num(f'fgbalplan{tag}', pretty(top.index[i])); num(f'fgbalplanrows{tag}', int(top.iloc[i]))
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    sc = [('2050_existing_grid', 'Reference siting, existing grid', C1, '-'), ('2050_with_plan', 'Reference siting, grid + plan', C1, '--'),
          ('2050_bess_at_load_existing_grid', 'BESS at load, existing grid', C2, '-'), ('2050_bess_at_load_with_plan', 'BESS at load, grid + plan', C2, '--')]
    for s, l, c, ls in sc:
        mm = M[M.scenario == s]
        if len(mm):
            ax.plot(mm.month, mm.ens_gwh, color=c, ls=ls, lw=2, marker='o', ms=4, label=l)
    ax.set_xticks(range(1, 13)); ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    ax.set_ylabel('Unserved energy (GWh)\n(lower is better)'); ax.set_xlabel('Month (2050)')
    ax.legend(fontsize=6.6, frameon=False, ncol=2, loc='lower left', bbox_to_anchor=(0, 1.0))
    fig.tight_layout(); fig.savefig(FIG / 'pcm_ens.pdf'); plt.close(fig)


def main():
    chk, poi = deliverability(); figures_deliverability(chk, poi)
    portfolio(); adequacy(); mitigation(); regional(); p7_ac(); pcm()
    (HERE / 'numbers.tex').write_text('\n'.join(rf'\newcommand{{\N{k}}}{{{v}}}' for k, v in sorted(NUM.items())) + '\n')
    print(len(NUM), 'numbers;', len(list(TAB.glob('*.tex'))), 'tables;', len(list(FIG.glob('*.pdf'))), 'figures')


if __name__ == '__main__':
    main()
