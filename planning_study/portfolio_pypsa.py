#!/usr/bin/env python3
"""Phase 2a: the single-year portfolio LP of optimize_macro_policy.py rebuilt in PyPSA.

The custom LP (scipy/HiGHS) stays the record for the base portfolios; this PyPSA model must
reproduce its objective and capacity mix (gate), and is then used for the CAISO/PJM-style
sensitivity portfolios (load ±6 %, dry and wet hydro, gas price +50 %/−25 %) and, later, for the
annual-vs-hourly matching check. Units: GW, GWh, million constant-2024 BRL (as in the custom LP).

Mapping (one bus, 8,760 hourly snapshots, objective weighting = the LP's discounted 5-year block weight):
* new technologies -> extendable Generators (p_max_pu = LP availability profile) with the LP's
  capital coefficient (PV capital + PV fixed O&M − discounted salvage); policy floors -> p_nom_min;
  resource envelope / scenario caps -> p_nom_max;
* existing non-dispatchable supply (central PV, MMGD, observed biomass, mixed thermal) -> one
  fixed Generator whose availability is the LP's `vrebase`;
* reservoir hydro -> Generator with the hourly upper bound, plus custom monthly-energy and cyclic
  ramp constraints; gas and imports -> Generators with the LP's costs and bounds;
* BESS -> extendable StorageUnit (fixed duration, sqrt round-trip efficiency, cyclic SOC) plus the
  LP's single-converter constraint (charge + discharge <= power) and 0.001 tie-break costs.
"""
from __future__ import annotations
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import pandas as pd
import pypsa

logging.getLogger('pypsa').setLevel(logging.ERROR)
logging.getLogger('linopy').setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parents[1]
MACRO = ROOT / 'MG_Step2_Congestion_Relief' / 'MG_Nodal_Connection_Optimization' / 'macro_reopt'
OUT = Path(__file__).resolve().parent / 'results'
OUT.mkdir(exist_ok=True)
TECH = ['solar_central', 'solar_distributed', 'bio', 'small_hydro', 'wind', 'wind_site', 'biogas', 'bess']
CF_TECH = ['solar_central', 'solar_distributed', 'bio', 'small_hydro', 'wind', 'wind_site']
SHED_TRANCHES = 20


@dataclass
class Case:
    name: str
    year: int
    scenario_config: str | None
    netzero_mode: str = 'none'
    load_scale: float = 1.0          # sensitivity: multiplies hourly load
    hydro_stress: bool = False       # LP's dry analogue (monthly ratios; also scales new small hydro)
    hydro_energy_scale: float = 1.0  # parametric wet case: scales monthly hydro energy budgets
    gas_price_scale: float = 1.0
    import_scale: float = 1.0
    price_scale: float = 1.0
    allow_shed: bool = False
    annual_clean_matching: bool = False   # phase 5 option (see annual_matching())
    fixed_caps: dict | None = None        # fixed new capacities (GW) -> dispatch-only run
    notes: str = ''


def lp_inputs(case: Case) -> dict:
    """Every coefficient exactly as optimize_macro_policy.solve_case(only_year=case.year) builds it."""
    p = json.loads((MACRO / 'macro' / 'processed' / 'macro_parameters.json').read_text())
    cfg = json.loads((MACRO / 'optimization' / case.scenario_config).read_text()) if case.scenario_config else {}
    p['technology']['wind_site'] = dict(p['technology']['wind'])
    p['technology']['biogas'] = {'life_years': 20., 'capex_brl_kw': float(cfg.get('biogas_capex_brl_kw', 7200.)),
                                 'fixed_cost_brl_kw_year': float(cfg.get('biogas_fixed_brl_kw_year', 300.)),
                                 'construction_interest_fraction': 0.064}
    p['base_capacity_gw']['wind_site'] = 0.; p['base_capacity_gw']['biogas'] = 0.
    dur = float(cfg.get('bess_duration_hours', p['bess_duration_hours']))
    f = pd.read_csv(MACRO / 'macro' / 'processed' / 'macro_hourly_inputs.csv', parse_dates=['hour_start_brt'])
    h = len(f); assert h == 8760
    wind = pd.read_csv(MACRO / 'macro' / 'processed' / 'mg_wind_2019_brt_aligned.csv')['capacity_factor'].to_numpy(float)
    target = float(cfg.get('wind_site_mean_cf', 0.30))
    lo, hi = 0., 100.
    for _ in range(50):
        m = (lo + hi) / 2
        if np.minimum(1, wind * m).mean() < target:
            lo = m
        else:
            hi = m
    cand_wind = np.minimum(1., wind * (lo + hi) / 2)
    y = case.year; start = y - 4; rate = p['discount_rate']
    w = sum((1 + rate) ** (-(z - 2025)) for z in range(y - 4, y + 1))
    month = f.hour_start_brt.dt.month.to_numpy()
    load_factor = (1 + p['low_demand_growth']) ** (y - 2025)
    load = f.load_mw.to_numpy() / 1000 * load_factor * case.load_scale
    retirement = max(0, (2050 - y) / 25) if case.netzero_mode != 'none' else 1.
    vrebase = (f.existing_solar_central_mw + f.existing_mmgd_mw + f.existing_biomass_observed_mw).to_numpy() / 1000 \
        + f.existing_mixed_thermal_mw.to_numpy() / 1000 * retirement
    cf = {'solar_central': f.new_central_pv_cf.to_numpy(), 'solar_distributed': f.new_dg_pv_cf.to_numpy(),
          'bio': f.new_bio_cf.to_numpy(), 'small_hydro': f.new_small_hydro_cf.to_numpy(), 'wind': wind,
          'wind_site': cand_wind}
    ratio = np.array([p['hydro_stress_month_ratio'][str(int(mm))] for mm in month]) if case.hydro_stress else np.ones(h)
    if case.hydro_stress:
        cf['small_hydro'] = cf['small_hydro'] * ratio
    carbon_price = float(cfg.get('carbon_price_brl_tco2', 0.)); gas_factor = float(cfg.get('gas_tco2_per_mwh', 0.40))
    fiscal_rate = float(cfg.get('eligible_purchase_tax_relief', 0.0))
    fiscal_share = float(cfg.get('eligible_purchase_share_of_capex', 0.0))
    fiscal_tech = set(cfg.get('fiscal_eligible_technologies', []))
    bess_energy_cost = float(cfg.get('additional_bess_energy_cost_brl_kw_h', 450.))
    capcoef, floor, ceil = {}, {}, {}
    for t in TECH:
        tp = p['technology'][t]; life = tp['life_years']; used = min(life, 2051 - start)
        remaining = max(0, life - used) / life
        base_capex = tp['capex_brl_kw'] + (dur - 4) * bess_energy_cost if t == 'bess' else tp['capex_brl_kw']
        base_capex += float(cfg.get('interconnection_addon_brl_kw_by_tech', {}).get(t, 0.))
        capital_before_tax = base_capex * (1 + tp['construction_interest_fraction'])
        capital = capital_before_tax * (1 - (fiscal_rate * fiscal_share if t in fiscal_tech else 0.))
        pv_capital = capital / (1 + rate) ** (start - 2026)
        om = sum(tp['fixed_cost_brl_kw_year'] / (1 + rate) ** (z - 2025) for z in range(start, min(2051, int(start + life))))
        capcoef[t] = pv_capital + om - capital * remaining / (1 + rate) ** 25
        floor[t] = 0.0; ceil[t] = np.inf
    ceil['wind_site'] = max(0., float(cfg.get('wind_site_build_cap_gw', 0.)))
    ceil['biogas'] = max(0., float(cfg.get('biogas_build_cap_gw', 0.)))
    for t, v in cfg.get('block_new_capacity_caps_gw', {}).items():
        ceil[t] = min(ceil[t], float(v))
    for t in ['solar_central', 'solar_distributed', 'bio', 'small_hydro']:
        tgt = p['policy_2030_gw'][t] + (p['policy_2050_gw'][t] - p['policy_2030_gw'][t]) * (y - 2030) / 20
        floor[t] = max(0.0, tgt - p['base_capacity_gw'][t])
    for t in ['bio', 'small_hydro']:
        c = max(p['policy_2030_gw'][t], p['policy_2050_gw'][t])
        ceil[t] = min(ceil[t], c - p['base_capacity_gw'][t])
    energy = {}
    obs = f.observed_hydro_mw.to_numpy() / 1000
    for mm in range(1, 13):
        sel = month == mm
        energy[mm] = float((obs[sel] * ratio[sel]).sum()) * case.hydro_energy_scale
    gas_cap = 0. if case.netzero_mode != 'none' else p['gas_operating_capacity_gw'] * retirement
    imp_cap = 0. if (case.netzero_mode == 'local_zero_2050' and y == 2050) else p['import_bound_gw'] * case.import_scale
    return dict(p=p, cfg=cfg, f=f, w=w, load=load, vrebase=vrebase, cf=cf, capcoef=capcoef, floor=floor, ceil=ceil,
                month=month, energy=energy, hydro_upper=f.hydro_hourly_upper_mw.to_numpy() / 1000,
                ramp=p['hydro_hourly_ramp_gw'], gas_cap=gas_cap, imp_cap=imp_cap, dur=dur,
                eta=np.sqrt(p['bess_roundtrip_efficiency']),
                gas_mc=(p['gas_dispatch_cost_brl_mwh'] * case.gas_price_scale + carbon_price * gas_factor) / 1000,
                imp_mc=p['import_price_brl_mwh'] * case.price_scale / 1000,
                biogas_cf=float(cfg.get('biogas_availability_factor', 0.70)),
                biogas_mc=float(cfg.get('biogas_fuel_cost_brl_mwh', 250.)) / 1000)


def build(case: Case, d: dict) -> pypsa.Network:
    n = pypsa.Network()
    sn = pd.RangeIndex(8760, name='snapshot')
    n.set_snapshots(sn)
    n.snapshot_weightings.loc[:, 'objective'] = d['w']
    n.add('Bus', 'MG')
    n.add('Load', 'load', bus='MG', p_set=pd.Series(d['load'], sn))
    vb = d['vrebase']; vmax = float(vb.max())
    n.add('Generator', 'existing_nondispatchable', bus='MG', p_nom=vmax, p_max_pu=pd.Series(vb / vmax, sn), marginal_cost=0.0)
    hu = d['hydro_upper']; hmax = float(hu.max())
    n.add('Generator', 'hydro', bus='MG', p_nom=hmax, p_max_pu=pd.Series(hu / hmax, sn), marginal_cost=0.0)
    n.add('Generator', 'gas', bus='MG', p_nom=d['gas_cap'], marginal_cost=d['gas_mc'])
    n.add('Generator', 'imports', bus='MG', p_nom=d['imp_cap'], marginal_cost=d['imp_mc'])
    if case.allow_shed:
        # Unserved energy at the LP's penalty (1000 per GWh, weighted). With energy-limited hydro the
        # hourly placement of a monthly deficit is otherwise arbitrary (linear cost), so it is split into
        # 20 tranches of 5 % of each hour's load with marginally rising cost: shortfalls are shared
        # across hours like rotating outages, and the total unserved energy is unchanged.
        lm = float(d['load'].max())
        for i in range(SHED_TRANCHES):
            n.add('Generator', f'shed_{i:02d}', bus='MG', p_nom=lm, p_max_pu=pd.Series(d['load'] / lm / SHED_TRANCHES, sn),
                  marginal_cost=1000.0 * (1 + 1e-4 * i))
    fixed = case.fixed_caps
    for t in CF_TECH + ['biogas']:
        prof = pd.Series(d['cf'][t], sn) if t in CF_TECH else d['biogas_cf']
        mc = d['biogas_mc'] if t == 'biogas' else 0.0
        if fixed is not None:
            n.add('Generator', t, bus='MG', p_nom=float(fixed.get(t, 0.0)), p_max_pu=prof, marginal_cost=mc)
        else:
            n.add('Generator', t, bus='MG', p_nom_extendable=True, p_nom_min=d['floor'][t],
                  p_nom_max=d['ceil'][t], capital_cost=d['capcoef'][t], p_max_pu=prof, marginal_cost=mc)
    kw = dict(bus='MG', max_hours=d['dur'], efficiency_store=d['eta'], efficiency_dispatch=d['eta'],
              cyclic_state_of_charge=True, marginal_cost=0.001)
    if fixed is not None:
        n.add('StorageUnit', 'bess', p_nom=float(fixed.get('bess', 0.0)), **kw)
    else:
        n.add('StorageUnit', 'bess', p_nom_extendable=True, p_nom_min=d['floor']['bess'], p_nom_max=d['ceil']['bess'],
              capital_cost=d['capcoef']['bess'], **kw)
    return n


def extra(case: Case, d: dict):
    def ef(n, sns):
        m = n.model
        p = m.variables['Generator-p']
        hyd = p.sel(name='hydro')
        mon = pd.Series(d['month'], index=n.snapshots)
        for mm in range(1, 13):
            idx = mon.index[mon.values == mm]
            m.add_constraints(hyd.sel(snapshot=idx).sum() <= d['energy'][mm], name=f'hydro_energy_m{mm}')
        prev = hyd.roll(snapshot=1)                      # cyclic, like np.roll in the LP
        m.add_constraints(hyd - prev <= d['ramp'], name='hydro_ramp_up')
        m.add_constraints(prev - hyd <= d['ramp'], name='hydro_ramp_dn')
        ps = m.variables['StorageUnit-p_store'].sel(name='bess')
        pd_ = m.variables['StorageUnit-p_dispatch'].sel(name='bess')
        if case.fixed_caps is None:
            pn = m.variables['StorageUnit-p_nom'].sel(name='bess')
            m.add_constraints(ps + pd_ - pn <= 0, name='bess_single_converter')
        else:
            m.add_constraints(ps + pd_ <= float(case.fixed_caps.get('bess', 0.0)), name='bess_single_converter')
        # charge tie-break (the LP prices both charge and discharge at 0.001 per GWh, weighted)
        m.add_objective(m.objective.expression + (d['w'] * 0.001) * ps.sum(), overwrite=True)
        if case.annual_clean_matching:
            annual_matching(n, m, d)
    return ef


def annual_matching(n, m, d):
    """Annual (not hourly) clean matching for the 2050 'fossil-free' definition (phase 5):
    imports and gas are allowed hour by hour, but annual local clean supply (all MG resources
    except gas and imports) must cover annual load plus storage losses. Defined in phase 5."""
    p = m.variables['Generator-p']
    clean = [g for g in n.generators.index if g not in ('gas', 'imports') and not g.startswith('shed')]
    ps = m.variables['StorageUnit-p_store'].sel(name='bess'); pdis = m.variables['StorageUnit-p_dispatch'].sel(name='bess')
    lhs = p.sel(name=clean).sum() + pdis.sum() - ps.sum()
    m.add_constraints(lhs >= float(d['load'].sum()), name='annual_clean_matching')


def solve(case: Case) -> dict:
    d = lp_inputs(case)
    n = build(case, d)
    t0 = time.time()
    status, cond = n.optimize(solver_name='highs', extra_functionality=extra(case, d),
                              include_objective_constant=False,
                              solver_options={'output_flag': False, 'primal_feasibility_tolerance': 1e-7,
                                              'dual_feasibility_tolerance': 1e-7})
    el = time.time() - t0
    out = {'case': case.name, 'year': case.year, 'status': status, 'condition': cond, 'seconds': el,
           'objective_mbrl': float(n.objective) if status == 'ok' else np.nan, 'notes': case.notes}
    if status != 'ok':
        return out
    g = n.generators_t.p
    caps = {t: float(n.generators.at[t, 'p_nom_opt']) for t in CF_TECH + ['biogas']}
    caps['bess'] = float(n.storage_units.at['bess', 'p_nom_opt'])
    out.update({f'cap_{t}_gw': caps[t] for t in TECH})
    avail = d['vrebase'] + sum(caps[t] * d['cf'][t] for t in CF_TECH)
    used = g[['existing_nondispatchable'] + CF_TECH].sum(axis=1).to_numpy()
    su = n.storage_units_t
    capital = sum(caps[t] * d['capcoef'][t] for t in TECH)
    out.update({'capital_and_fixed_cost_pv_mbrl': capital, 'operating_cost_pv_mbrl': out['objective_mbrl'] - capital,
                'load_twh': float(d['load'].sum() / 1000), 'hydro_twh': float(g.hydro.sum() / 1000),
                'gas_twh': float(g.gas.sum() / 1000), 'imports_twh': float(g.imports.sum() / 1000),
                'biogas_twh': float(g.biogas.sum() / 1000),
                'nonreservoir_used_twh': float(used.sum() / 1000),
                'curtailment_twh': float(np.maximum(0, avail - used).sum() / 1000),
                'bess_charge_twh': float(su.p_store['bess'].sum() / 1000),
                'bess_discharge_twh': float(su.p_dispatch['bess'].sum() / 1000),
                'shed_gwh': float(g.filter(like='shed_').sum().sum()),
                'peak_load_gw': float(d['load'].max())})
    out['_network'] = n
    out['_inputs'] = d
    return out


def custom_lp(year, hydro_stress=False):
    """Objective of optimize_macro_policy.py for one year, run in a temporary copy of macro_reopt."""
    import shutil, subprocess, tempfile
    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / 'macro_reopt'
        shutil.copytree(MACRO / 'macro', dst / 'macro')
        shutil.copytree(MACRO / 'optimization', dst / 'optimization')
        cfg = 'scenario_addon_2035.json' if year == 2035 else 'scenario_addon_2050.json'
        cmd = [sys.executable, str(dst / 'optimization' / 'optimize_macro_policy.py'), '--name', 'XCHECK',
               '--year', str(year), '--scenario-config', str(dst / 'optimization' / cfg), '--solver', 'highs-ds']
        if year == 2050:
            cmd += ['--netzero-mode', 'local_zero_2050']
        if hydro_stress:
            cmd += ['--hydro-stress']
        subprocess.run(cmd, cwd=dst, check=True, capture_output=True)
        return float(json.loads((dst / 'output' / 'XCHECK' / 'solver.json').read_text())['objective'])


def custom_lp_fixed(year, hydro_stress=False):
    """Minimum unserved energy (GWh) of the saved REFRESH portfolio, from the custom LP's fixed-case mode."""
    import shutil, subprocess, tempfile
    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / 'macro_reopt'
        shutil.copytree(MACRO / 'macro', dst / 'macro')
        shutil.copytree(MACRO / 'optimization', dst / 'optimization')
        (dst / 'output' / f'REFRESH_{year}').mkdir(parents=True)
        shutil.copy(MACRO / 'output' / f'REFRESH_{year}' / 'portfolio.csv', dst / 'output' / f'REFRESH_{year}' / 'portfolio.csv')
        cfg = 'scenario_addon_2035.json' if year == 2035 else 'scenario_addon_2050.json'
        cmd = [sys.executable, str(dst / 'optimization' / 'optimize_macro_policy.py'), '--name', 'FX', '--year', str(year),
               '--scenario-config', str(dst / 'optimization' / cfg), '--solver', 'highs-ds', '--fixed-case', f'REFRESH_{year}']
        if year == 2050:
            cmd += ['--netzero-mode', 'local_zero_2050']
        if hydro_stress:
            cmd += ['--hydro-stress']
        subprocess.run(cmd, cwd=dst, check=True, capture_output=True)
        return float(json.loads((dst / 'output' / 'FX' / 'solver.json').read_text())['objective'])


def refresh_cases():
    return [Case('REFRESH_2035', 2035, 'scenario_addon_2035.json'),
            Case('REFRESH_2050', 2050, 'scenario_addon_2050.json', netzero_mode='local_zero_2050')]


def sensitivity_cases():
    out = []
    for base in refresh_cases():
        kw = dict(year=base.year, scenario_config=base.scenario_config, netzero_mode=base.netzero_mode)
        out += [Case(f'{base.year}_load_plus6', load_scale=1.06, notes='CAISO load +6%', **kw),
                Case(f'{base.year}_load_minus6', load_scale=0.94, notes='CAISO load -6%', **kw),
                Case(f'{base.year}_hydro_dry', hydro_stress=True, notes="LP's dry-hydro analogue (monthly ratios)", **kw),
                Case(f'{base.year}_hydro_wet_param', hydro_energy_scale=1.10,
                     notes='parametric +10% monthly hydro energy (no wet-year data in the package)', **kw),
                Case(f'{base.year}_gas_plus50', gas_price_scale=1.5, notes='CAISO gas price +50%', **kw),
                Case(f'{base.year}_gas_minus25', gas_price_scale=0.75, notes='CAISO gas price -25%', **kw)]
    return out


def main():
    rows = []
    # Gate: reproduce the custom LP's saved REFRESH runs
    for c in refresh_cases():
        r = solve(c)
        rec = json.loads((MACRO / 'output' / c.name / 'solver.json').read_text())
        port = pd.read_csv(MACRO / 'output' / c.name / 'portfolio.csv').set_index('technology').new_build_gw
        ann = pd.read_csv(MACRO / 'output' / c.name / 'annual_energy.csv').iloc[0]
        r['lp_objective_mbrl'] = rec['objective']
        r['objective_rel_diff'] = abs(r['objective_mbrl'] - rec['objective']) / rec['objective']
        r['max_cap_diff_gw'] = max(abs(r[f'cap_{t}_gw'] - max(0.0, float(port[t]))) for t in TECH)
        for k in ['hydro_twh', 'imports_twh', 'curtailment_twh', 'bess_discharge_twh', 'biogas_twh']:
            r[f'lp_{k}'] = float(ann[k])
        rows.append(r)
        print(f"{c.name}: PyPSA {r['objective_mbrl']:.6f} vs LP {rec['objective']:.6f} "
              f"(rel {r['objective_rel_diff']:.1e}); max capacity diff {r['max_cap_diff_gw']:.2e} GW; {r['seconds']:.0f} s",
              flush=True)
    for c in sensitivity_cases():
        r = solve(c)
        rows.append(r)
        print(f"{c.name}: {r['status']} obj {r.get('objective_mbrl', np.nan):.2f}; "
              + ', '.join(f"{t} {r.get(f'cap_{t}_gw', np.nan):.3f}" for t in TECH), flush=True)
    # Cross-check beyond the base runs: re-solve the dry-hydro cases with the custom LP itself
    # (in a temporary copy, so the repository outputs are untouched) and compare.
    for r in rows:
        if r['case'] in ('2035_hydro_dry', '2050_hydro_dry'):
            lp = custom_lp(r['year'], hydro_stress=True)
            r['lp_objective_mbrl'] = lp
            r['objective_rel_diff'] = abs(r['objective_mbrl'] - lp) / lp
            print(f"cross-check {r['case']}: PyPSA {r['objective_mbrl']:.6f} vs custom LP {lp:.6f} "
                  f"(rel {r['objective_rel_diff']:.1e})", flush=True)
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith('_')} for r in rows])
    df.to_csv(OUT / 'portfolio_pypsa.csv', index=False, float_format='%.9f')
    gate = df[df.case.str.startswith('REFRESH')]
    ok = bool((gate.objective_rel_diff < 1e-6).all() and (gate.max_cap_diff_gw < 1e-4).all())
    xc = df[df.case.str.endswith('hydro_dry')]
    ok = ok and bool((xc.objective_rel_diff < 1e-6).all())
    print('GATE', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
