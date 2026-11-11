from pathlib import Path
import json,pandas as pd
R=Path(__file__).resolve().parents[1]
records=[]
for year in [2035,2050]:
 for penalty in [.1,1,3]:
  f=R/'outputs'/f'{year}_ROBUST_distributed_MG_load_border_585_PV35to90_pen{f"{penalty:.1f}".replace(".","p")}_summary.json'
  a=json.loads(f.read_text())
  records.append({'year':year,'penalty_mbrl_per_mw_ASSUMED':penalty,
  'baseline_connection_billion_brl':a['reference_connection_nominal_mbrl']/1000,
  'optimized_connection_billion_brl':a['optimized_connection_nominal_mbrl']/1000,
  'connection_saving_billion_brl':(a['reference_connection_nominal_mbrl']-a['optimized_connection_nominal_mbrl'])/1000,
  'saving_percent':100*(1-a['optimized_connection_nominal_mbrl']/a['reference_connection_nominal_mbrl']),
  'baseline_worstcase_uplift_MW_NOT_REAL_LINES':a['reference_worstcase_uplift_mw'],
  'optimized_worstcase_uplift_MW_NOT_REAL_LINES':a['optimized_worstcase_uplift_mw'],
  'model_MIP_gap':a['mip_gap'],'model_constraint_error_MW':a['max_constraint_violation']})
pd.DataFrame(records).to_csv(R/'outputs/PROJECT_RESULTS_SUMMARY.csv',index=False)
