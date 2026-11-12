#!/usr/bin/env python3
"""Phase 4b: benefit-cost ratios of the attributable reinforcements and of siting the BESS at load.

Annual system cost of a production-cost scenario (valuation fixed before the runs, see pcm.py):
    production cost (imports, gas, biogas fuel, BESS tie-break)
  + unserved energy x Custo do Déficit 2025 (CCEE, R$ 8,327.76/MWh)
  + unused renewable energy (new and existing VRE curtailment + unused monthly hydro budget)
    x 2025 SE CMO (R$ 209.47/MWh), its opportunity value in the SIN.
Benefit of a measure = cost without it - cost with it (same year). The cost of a reinforcement plan is
its annual revenue requirement, capex x (CRF(8 %, 30 y) + 1.5 % O&M) (cost_assumptions.json), for the
full costed plan (upper) and with small line needs handled by uncosted non-wires options (lower).
BCR > 1 is the CAISO/PJM economic criterion (Lectures 9-10). Benefits are assumed constant over the
asset life at their 2050 value (the 2035 portfolio has no congestion, see below).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results'
VOLL, P_CURT = 8327.76, 209.47


def system_cost(s, voll=VOLL, pc=P_CURT):
    unused = s.curt_new_twh + s.curt_existing_twh + s.hydro_spill_twh
    return s.production_cost_mbrl_yr + s.ens_mwh * voll / 1e6 + unused * 1e6 * pc / 1e6


def main():
    S = pd.read_csv(OUT / 'pcm_scenarios.csv').set_index('scenario')
    T = pd.read_csv(OUT / 'mitigation_totals.csv').set_index('siting')
    rows = []
    for vl, vf in (('base', 1.0), ('VOLL -50%', 0.5), ('VOLL +50%', 1.5)):
        for cl, cf in (('base', 1.0), ('unused energy -50%', 0.5), ('unused energy +50%', 1.5)):
            if vl != 'base' and cl != 'base':
                continue
            c = {k: system_cost(S.loc[k], VOLL * vf, P_CURT * cf) for k in S.index}
            sens = 'base' if vl == cl == 'base' else (vl if vl != 'base' else cl)
            for label, without, with_, siting in [
                    ('reinforcement plan, reference siting', '2050_existing_grid', '2050_with_plan', 'reference'),
                    ('reinforcement plan, BESS at load', '2050_bess_at_load_existing_grid', '2050_bess_at_load_with_plan',
                     'reference_bess_at_load')]:
                ben = c[without] - c[with_]
                arr_hi = T.loc[siting, 'annual_revenue_requirement_mbrl']
                arr_lo = T.loc[siting, 'annual_revenue_requirement_excl_small_mbrl']
                rows.append({'measure': label, 'sensitivity': sens, 'annual_benefit_mbrl': ben,
                             'arr_full_plan_mbrl': arr_hi, 'arr_excl_small_mbrl': arr_lo,
                             'bcr_full_plan': ben / arr_hi, 'bcr_excl_small': ben / arr_lo if arr_lo > 0 else np.nan})
            ben = c['2050_existing_grid'] - c['2050_bess_at_load_existing_grid']
            avoided = T.loc['reference', 'annual_revenue_requirement_mbrl'] - T.loc['reference_bess_at_load', 'annual_revenue_requirement_mbrl']
            rows.append({'measure': 'site the planned BESS at load buses (no grid plan)', 'sensitivity': sens,
                         'annual_benefit_mbrl': ben, 'arr_full_plan_mbrl': 0.0, 'arr_excl_small_mbrl': 0.0,
                         'bcr_full_plan': np.inf, 'bcr_excl_small': np.inf, 'avoided_plan_arr_mbrl': avoided})
    B = pd.DataFrame(rows)
    B.to_csv(OUT / 'bcr.csv', index=False, float_format='%.6f')
    cost = pd.DataFrame({k: {'production_mbrl_yr': S.loc[k, 'production_cost_mbrl_yr'], 'ens_mwh': S.loc[k, 'ens_mwh'],
                             'unused_renewable_twh': S.loc[k, 'curt_new_twh'] + S.loc[k, 'curt_existing_twh'] + S.loc[k, 'hydro_spill_twh'],
                             'system_cost_mbrl_yr': system_cost(S.loc[k])} for k in S.index}).T
    cost['congestion_cost_vs_copperplate_mbrl_yr'] = np.nan
    for y in (2035, 2050):
        cp = cost.loc[f'{y}_copperplate', 'system_cost_mbrl_yr']
        for k in cost.index:
            if k.startswith(str(y)):
                cost.loc[k, 'congestion_cost_vs_copperplate_mbrl_yr'] = cost.loc[k, 'system_cost_mbrl_yr'] - cp
    cost.to_csv(OUT / 'pcm_system_cost.csv', float_format='%.6f')
    print(cost.round(3).to_string()); print(B.round(3).to_string())
    return 0


if __name__ == '__main__':
    sys.exit(main())
