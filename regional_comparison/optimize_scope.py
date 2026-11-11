#!/usr/bin/env python3
"""Scope comparison: northern case (4 areas) vs region-wide (all 12 candidate areas).
Tech capacities frozen at latest macro solved stage 1. Compare identical screened PWF model.
Connection MILP uses continuous rating-uplift proxy, not a physical construction plan.
Post-MILP parallel circuit screening adds actual circuit impedances using Step2 engine.
"""
import sys,json,math,time
from pathlib import Path
import numpy as np,pandas as pd
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix
ROOT=Path(__file__).resolve().parent/'MG_Step2_Congestion_Relief'
sys.path.insert(0,str(ROOT/'MG_Nodal_Connection_Optimization'/'code'))
import optimize_siting as ns
OUT=Path(__file__).resolve().parent/'results';OUT.mkdir(parents=True,exist_ok=True)
FOCUS=['Arinos','Janauba','Jaiba','Montes Claros']
SCOPES={'Local_4_North_MG':FOCUS,'Regional_12_Areas':None}
MODES={'local_consumption':['distributed_MG_load'],'local_plus_export_stress':['distributed_MG_load','border_585']}
PV=[.35,.70,.90]
PEN=1.0 # R$ million per MW, only scoring shadow proxy

def run_sites(year,scope,focus,balance_name,modes,time_limit=120):
 p=pd.read_csv(ns.I/f'portfolio_{year}.csv')
 portfolio={r.technology:float(r.new_build_gw) for r in p.itertuples(index=False)}
 cand,cls=ns.candidate_table(year,portfolio)
 k=len(cand);nr=ns.nr;dgpv=portfolio.get('solar_distributed',0)*1000
 src=ns.ptdf_bus(cand.bus_id.to_numpy(int));load_sens=ns.ptdf_bus(ns.B.iloc[ns.loadidx].id.to_numpy())@ns.loadval
 refx=np.zeros(k);refz=np.zeros(k)
 for cl in cls.itertuples(index=False):
  j=np.flatnonzero(cand.cluster_id==cl.cluster_id)[0]
  refx[j]=cl.baseline_mw
  refz[j]=math.ceil(cl.baseline_mw/float(cand.max_circuit_mw.iloc[j])-1e-10)
 scenarios=[]
 for mode in modes:
  for pv in PV:
   factors=np.array([pv if tech=='solar_central' else ns.PFAC[tech] for tech in cand.tech]);sens=src*factors[None,:]
   stat=load_sens*(dgpv*pv)
   if mode=='distributed_MG_load':
    sens-=ns.withdraw[:,None]*factors[None,:]
    stat-=ns.withdraw*(dgpv*pv)
   scenarios.append((mode,pv,ns.F+stat,sens))
 nvar=2*k+nr
 obj=np.r_[cand.variable_cost_mbrl_mw.to_numpy(float),cand.per_circuit_mbrl.to_numpy(float),np.repeat(PEN,nr)]
 low=np.zeros(nvar);up=np.full(nvar,np.inf)
 up[:k]=cand.max_area_mw.to_numpy(float)
 up[k:2*k]=np.ceil(cand.max_area_mw.to_numpy(float)/cand.max_circuit_mw.to_numpy(float))
 integr=np.zeros(nvar,int);integr[k:2*k]=1
 if focus is not None:
  fixed=~cand.area.isin(focus).to_numpy()
  low[:k][fixed]=refx[fixed];up[:k][fixed]=refx[fixed]
  low[k:2*k][fixed]=refz[fixed];up[k:2*k][fixed]=refz[fixed]
 rows=[];cols=[];data=[];lv=[];uv=[];row=0
 def add(jj,vv,l=-np.inf,u=np.inf):
  nonlocal row
  jj=np.asarray(jj,int);vv=np.asarray(vv,float)
  nz=np.abs(vv)>1e-13
  rows.extend([row]*int(nz.sum()));cols.extend(jj[nz].tolist());data.extend(vv[nz].tolist())
  lv.append(l);uv.append(u);row+=1
 for tech,mw in portfolio.items():
  if tech in ns.SUPPORTED and mw>1e-9:
   jj=np.flatnonzero(cand.tech==tech);add(jj,np.ones(len(jj)),mw*1000,mw*1000)
 for cl in cls.itertuples(index=False):
  jj=np.flatnonzero(cand.cluster_id==cl.cluster_id);add(jj,np.ones(len(jj)),u=cl.max_area_mw)
 for j in range(k):add([j,k+j],[1,-float(cand.max_circuit_mw.iloc[j])],u=0)
 for mode,pv,f0,sens in scenarios:
  for ell in range(nr):
   ii=np.r_[np.arange(k),2*k+ell]
   add(ii,np.r_[sens[ell,:],-1],u=float(ns.R[ell]-f0[ell]))
   add(ii,np.r_[-sens[ell,:],-1],u=float(ns.R[ell]+f0[ell]))
 mat=coo_matrix((data,(rows,cols)),shape=(row,nvar)).tocsr()
 st=time.monotonic()
 sol=milp(c=obj,integrality=integr,bounds=Bounds(low,up),constraints=LinearConstraint(mat,lv,uv),options={'time_limit':time_limit,'mip_rel_gap':.0001})
 if sol.x is None:raise RuntimeError(f'NOT SOLVED {year} {scope} {balance_name}: {sol.message}')
 x=sol.x[:k];z=np.rint(sol.x[k:2*k]).astype(int);u=sol.x[2*k:];dgcost=dgpv*ns.C['distribution_upgrade_million_brl_per_mw_assumed']
 cnet=float(x@cand.variable_cost_mbrl_mw.to_numpy(float)+z@cand.per_circuit_mbrl.to_numpy(float)+dgcost)
 refconn=float(refx@cand.variable_cost_mbrl_mw.to_numpy(float)+refz@cand.per_circuit_mbrl.to_numpy(float)+dgcost)
 case_rows=[];lines=[];reference_u=np.zeros(nr)
 for mode,pv,f0,sens in scenarios:
  flow=f0+sens@x;baseflow=f0+sens@refx
  exc=np.maximum(0,np.abs(flow)-ns.R);excref=np.maximum(0,np.abs(baseflow)-ns.R)
  reference_u=np.maximum(reference_u,excref)
  case_rows.append(dict(year=year,scope=scope,scenario_set=balance_name,balance=mode,pv_factor=pv,
                        reference_overload_sum_mw=float(excref.sum()),before_line_construction_overload_sum_mw=float(exc.sum()),
                        over_limit_lines=int((exc>1e-5).sum()),max_loading_pct=float((np.abs(flow)/ns.R).max()*100)))
  for j,ee in enumerate(exc):
   if ee>1e-4:
    ll=ns.L.iloc[ns.monitor[j]];lines.append(dict(balance=mode,pv=pv,from_bus=int(ll.f),to_bus=int(ll.t),circuit=int(ll.c),flow_mw=float(flow[j]),rating_proxy_mw=float(ns.R[j]),overload_mw=float(ee)))
 delta_mat=np.asarray(mat@sol.x)
 maxerr=max(float(np.max(np.array(lv)-delta_mat)),float(np.max(delta_mat-np.array(uv))),0.)
 selected=cand.copy();selected['new_capacity_mw']=x;selected['connection_circuits']=z
 selected['reference_mw']=refx;selected['reference_circuits']=refz
 selected['connection_cost_mbrl']=x*cand.variable_cost_mbrl_mw+z*cand.per_circuit_mbrl
 selected['is_decision_area']=selected.area.isin(focus) if focus is not None else True
 selected=selected[(selected.new_capacity_mw>1e-5)|(selected.connection_circuits>0)].copy()
 case_tag=f'{year}_{scope}_{balance_name}'
 selected.to_csv(OUT/f'{case_tag}_chosen_sites.csv',index=False)
 pd.DataFrame(case_rows).to_csv(OUT/f'{case_tag}_scenario_overloads.csv',index=False)
 pd.DataFrame(lines,columns=['balance','pv','from_bus','to_bus','circuit','flow_mw','rating_proxy_mw','overload_mw']).to_csv(OUT/f'{case_tag}_bottlenecks_prebuild.csv',index=False)
 summ=dict(year=year,scope=scope,scenario_set=balance_name,n_scenarios=len(scenarios),n_areas_considered=int(cand.area.nunique()),
           n_areas_free=int(cand.area.nunique()) if focus is None else int(cand[cand.area.isin(focus)].area.nunique()),
           n_all_bus=ns.n,n_all_branch=ns.m0,n_monitored_lines=ns.nr,n_active_siting_arcs=int((x>1e-4).sum()),
           new_capacity_gw=float(sum(portfolio.values())),optimized_connection_cost_million_brl=cnet,
           reference_connection_cost_million_brl=refconn,connection_saving_million_brl=refconn-cnet,
           worst_case_thermal_uplift_proxy_mw=float(u.sum()),reference_uplift_proxy_mw=float(reference_u.sum()),
           site_plus_uprate_proxy_score_million_brl=float(cnet+PEN*u.sum()),
           screen_penalty_assumed_million_brl_per_mw=PEN,
           solver_status=int(sol.status),solver_message=str(sol.message),solver_mip_gap=float(getattr(sol,'mip_gap',float('nan'))),
           max_constraint_violation=float(maxerr),run_seconds=float(time.monotonic()-st),
           focus_areas=focus if focus else 'All candidate areas',
           fixed_technical_portfolio=True,
           note='Uplift cost is ONLY a linear penalty, not a buildable project cost. 2050 uses 2035 PWF topology.')
 (OUT/f'{case_tag}_summary.json').write_text(json.dumps(summ,indent=2,ensure_ascii=False))
 print('SITE',case_tag,'status',sol.status,'conn',round(cnet,2),'u_proxy',round(u.sum(),2),'score',round(cnet+u.sum(),2),'gap',summ['solver_mip_gap'],'sec',round(summ['run_seconds'],2),flush=True)
 return summ

if __name__=='__main__':
 rows=[]
 for yr in (2035,2050):
  for scen, modes in MODES.items():
   for scope,areas in SCOPES.items():
    rows.append(run_sites(yr,scope,areas,scen,modes))
 pd.DataFrame(rows).to_csv(OUT/'model_comparison_summary.csv',index=False)
 print('ALL SITE RUNS DONE')
