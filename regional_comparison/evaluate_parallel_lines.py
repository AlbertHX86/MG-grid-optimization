#!/usr/bin/env python3
"""Physical parallel-line post-siting DC screening; reads reoptimized sites."""
import sys,json
from pathlib import Path
import pandas as pd,numpy as np
ROOT=Path(__file__).resolve().parent/'MG_Step2_Congestion_Relief'
sys.path.insert(0,str(ROOT))
import step2_model as grid
OUT=Path(__file__).resolve().parent/'results'
def run(year,scope,scen,modes):
 chosen=pd.read_csv(OUT/f'{year}_{scope}_{scen}_chosen_sites.csv')
 port=pd.read_csv(grid.I/f'portfolio_{year}.csv')
 dg=float(port.loc[port.technology=='solar_distributed','new_build_gw'].iloc[0])*1000
 scenlist=[(m,p) for m in modes for p in (.35,.70,.90)]
 def site_inj(yr):
  assert yr==year
  inject=[]
  for mode,pv in scenlist:
   v=np.zeros(grid.nd)
   for r in chosen.itertuples(index=False):
    factor=pv if r.tech=='solar_central' else grid.PFAC[r.tech]
    v[grid.ind[int(r.bus_id)]]+=float(r.new_capacity_mw)*factor
   v[grid.loadidx]+=grid.loadweights*dg*pv
   if mode=='distributed_MG_load':v[grid.loadidx]-=grid.loadweights*v.sum()
   else:v[grid.ref]-=v.sum()
   assert abs(v.sum())<1e-6
   inject.append(v)
  return chosen,port,np.column_stack(inject)
 grid.SCENARIOS=scenlist
 grid.gen_injections=site_inj
 grid.O=OUT/f'{year}_{scope}_{scen}_physical';grid.O.mkdir(exist_ok=True)
 print('START PHYSICAL',year,scope,scen,flush=True)
 r=grid.study(year,max_additions=18)
 print('END PHYSICAL',year,scope,scen,'circuits',r['greedy_added_parallel_circuits'],'cost',round(r['cost_assumed_million_brl'],2),'residual',round(r['remaining_worst_case_sum_of_overloads_mw'],2),flush=True)
 return dict(scope=scope,scenario_set=scen,**r)
if __name__=='__main__':
 results=[]
 for scen,modes in {'local_consumption':['distributed_MG_load'],'local_plus_export_stress':['distributed_MG_load','border_585']}.items():
  for year in [2035,2050]:
   for scope in ['Local_4_North_MG','Regional_12_Areas']:
    results.append(run(year,scope,scen,modes))
 pd.DataFrame(results).to_csv(OUT/'physical_parallel_line_comparison.csv',index=False)
 print('DONE',flush=True)
