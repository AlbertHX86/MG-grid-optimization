#!/usr/bin/env python3
"""Evaluate source-specific PV/wind curtailment against fixed-site no-line and new-line topologies.
Every curtailed MW must be supplied at the declared balancing sink (unverified headroom).
"""
import json
from pathlib import Path
import pandas as pd,numpy as np
from scipy.optimize import linprog
from scipy.sparse.linalg import splu
import sys
sys.path.insert(0,str(Path(__file__).parent))
import step2_model as M
R=Path(__file__).parent;O=R/'results'

def run(year,with_lines):
    sel,port,inj=M.gen_injections(year)
    upgraded=pd.read_csv(O/f'{year}_parallel_line_candidates.csv') if with_lines else pd.DataFrame(columns=['f','t','circuit_reference'])
    used=[]
    for a in upgraded.itertuples(index=False):
        found=M.L.index[(M.L.f==a.f)&(M.L.t==a.t)&(M.L.c==a.circuit_reference)]
        assert len(found)>0
        used.append(int(found[0]))
    AA=M.A if not used else __import__('scipy.sparse').sparse.vstack([M.A,M.A[used,:]],format='csr')
    ww=np.r_[M.w,M.w[used]]
    y=(AA.T@AA.multiply(ww[:,None])).tocsc()
    fac=splu(y[M.free,:][:,M.free])
    baserhs=np.zeros(M.nd)
    if used:baserhs-=M.A[used,:].T@M.base[used]
    allbaseline=np.r_[M.base,M.base[used]] if used else M.base
    ids=np.r_[M.mon,np.arange(M.nl,M.nl+len(used))] if used else M.mon
    r=np.r_[M.rates,M.L.iloc[used].rate.to_numpy(float)] if used else M.rates
    tbl=[]
    for jj,(mode,pv) in enumerate(M.SCENARIOS):
        dtheta=M.solve_theta(fac,baserhs+inj[:,jj])
        flow=(allbaseline[:,None]+ww[:,None]*(AA@dtheta))[ids,0]
        cc=sel[sel.tech.isin(['solar_central','wind_site'])]
        D=[];ub=[]
        for row in cc.itertuples():
            v=np.zeros(M.nd);v[M.ind[int(row.bus_id)]]=1
            if mode=='border_585':v[M.ref]-=1
            else:v[M.loadidx]-=M.loadweights
            D.append(v);ub.append(float(row.new_capacity_mw)*(pv if row.tech=='solar_central' else M.PFAC[row.tech]))
        ds=M.solve_theta(fac,np.column_stack(D))
        sens=(ww[:,None]*(AA@ds))[ids,:]
        opt=linprog(np.ones(len(ub)),A_ub=np.vstack([sens,-sens]),b_ub=np.r_[flow+r,r-flow],bounds=list(zip(np.zeros(len(ub)),ub)),method='highs')
        ens=np.nan;violation=np.nan
        if opt.success:
            ens=float(opt.x.sum());violation=float(max(0,np.max(np.abs(flow-sens@opt.x)-r)))
        tbl.append({'year':year,'with_12_parallel_circuits':with_lines,'balance':mode,'pv':pv,'feasible_in_screen':bool(opt.success),'minimum_solar_wind_replacement_mw':ens,'max_thermal_violation_after_screen_mw':violation,'data_boundary':'Assumed sink flexible replacement, not resource-backed physical redispatch'})
    return tbl
if __name__=='__main__':
    rows=[]
    for year in [2035,2050]:
        for q in [False,True]:rows.extend(run(year,q))
    p=O/'redispatch_comparison.csv';pd.DataFrame(rows).to_csv(p,index=False)
    print(pd.DataFrame(rows)[['year','with_12_parallel_circuits','balance','pv','feasible_in_screen','minimum_solar_wind_replacement_mw']].to_string(index=False))
