#!/usr/bin/env python3
"""Reproduces one two-pass location-aware optimization from archived inputs.

1. Reference portfolio (old hourly optimum) -> nodal PTDF MIP -> average adders
2. Hourly portfolio LP with indicative connection adders (once)
3. Common siting under 6 PV/balance contingencies -> audits and figures
The process is sequential and is NOT a globally optimal joint dispatch/network MILP.
"""
from pathlib import Path
import shutil,subprocess,sys,json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];I=ROOT/'inputs';O=ROOT/'outputs';M=ROOT/'macro_reopt'
sys.path.insert(0,str(ROOT/'code'))
from optimize_siting import solve_robust

def macro(year,scenario_file):
 command=[sys.executable,str(M/'optimization/optimize_macro_policy.py'),'--name',f'REFRESH_{year}','--year',str(year),
   '--scenario-config',str(scenario_file),'--solver','highs-ds']
 if year==2050:command+=['--netzero-mode','local_zero_2050']
 with open(O/f'macro_refresh_{year}.log','w') as log:
  subprocess.run(command,cwd=M,stdout=log,stderr=subprocess.STDOUT,check=True)
 rec=json.loads((M/f'output/REFRESH_{year}/solver.json').read_text())
 assert rec['success'] and rec['max_balance_residual_gw']<1e-6
 return rec

for year in (2035,2050):
 shutil.copy2(I/f'portfolio_{year}_initial.csv',I/f'portfolio_{year}.csv')
 print('STAGE 1:',year,'reference siting',flush=True)
 solve_robust(year,('distributed_MG_load','border_585'),(.35,.70,.90),1.)
 import glob
 globpath=O/f'{year}_ROBUST_distributed_MG_load_border_585_PV35to90_pen1p0_selected.csv'
 x=pd.read_csv(globpath)
 sums=x.groupby('tech').agg(MW=('new_capacity_mw','sum'),cost_mbrl=('connection_cost_mbrl','sum'))
 add={t:round(float(v),3) for t,v in (1000*sums.cost_mbrl/sums.MW).to_dict().items()}
 add['solar_distributed']=100
 cfg={} if year==2035 else json.loads((M/'optimization/scenario_base_2050.json').read_text())
 cfg['interconnection_addon_brl_kw_by_tech']=add
 configfile=M/f'optimization/scenario_addon_{year}.json'
 configfile.write_text(json.dumps(cfg,indent=2))
 print('STAGE 2:',year,'refresh hourly portfolio (indicative average connection CAPEX)',flush=True)
 macro(year,configfile)
 shutil.copy2(M/f'output/REFRESH_{year}/portfolio.csv',I/f'portfolio_{year}.csv')
 print('STAGE 3:',year,'common 6-case nodal reinforcement',flush=True)
 for penalty in (0.1,1.0,3.0):solve_robust(year,('distributed_MG_load','border_585'),(.35,.70,.90),penalty)
print('STEP 4: Audit and visualization',flush=True)
subprocess.run([sys.executable,str(ROOT/'code/audit_and_plot.py')],check=True)
print('SUCCESS — numerical study reproduced; engineering feasibility NOT certified.')
