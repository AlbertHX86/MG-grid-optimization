#!/usr/bin/env python3
"""Stage 2: EPE-2035 AC-P flow-calibrated incremental DC/PTDF and parallel-circuit study.

This is a screening model, NOT an AC power-flow expansion certification. It uses
original AC P_from values to anchor DC flow increments. Optional physical
parallel lines are copies of existing circuits, with identical impedance and
rating, and the nodal phase-angle response is recalculated after each addition.
Electricity interchanges are assumed to balance generation in two explicit ways.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix,csc_matrix
from scipy.sparse.linalg import splu
from scipy.optimize import linprog
from pyproj import CRS,Transformer
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'MG_Nodal_Connection_Optimization'
I=SOURCE/'inputs';O=ROOT/'results';O.mkdir(exist_ok=True)
B=pd.read_csv(I/'pwf_2035_buses.csv');L=pd.read_csv(I/'pwf_2035_branches.csv')
C=json.load(open(I/'cost_assumptions.json'))
X=pd.read_csv(I/'epe_ons_station_crosswalk.csv')
nd=len(B);nl=len(L);ids=B.id.to_numpy(int);ind={int(v):i for i,v in enumerate(ids)}
f=L.f.map(ind).to_numpy(int);t=L.t.map(ind).to_numpy(int)
w=100/(L.x.to_numpy(float)*L.tap.to_numpy(float)) # MW per rad
A=coo_matrix((np.r_[np.ones(nl),-np.ones(nl)],(np.r_[np.arange(nl),np.arange(nl)],np.r_[f,t])),shape=(nl,nd)).tocsr()
ref=ind[585];free=np.arange(nd)!=ref
kv=B.kv.to_numpy(float);state=B.state.to_numpy(int)
monmask=(L.kind=='line')&L.rate.between(100,9998)&(L.x>0)&(np.minimum(kv[f],kv[t])>=230)&((state[f]==11)|(state[t]==11))
mon=np.flatnonzero(monmask.to_numpy());nmon=len(mon)
assert nd==1489 and nl==1890 and nmon==283
rates=L.rate.to_numpy(float)[mon]
base=L.p_from_mw.to_numpy(float)
loadidx=np.where((B.state.to_numpy()==11)&(B.pd.to_numpy()>0))[0]
loadweights=B.pd.to_numpy()[loadidx];loadweights/=loadweights.sum()
W=pd.read_csv(I/'allocation_weights.csv')
PFAC={'solar_central':.70,'bio':.50,'small_hydro':.60,'wind_site':.50,'biogas':.70,'bess':0.0}
NAMES=['distributed_MG_load','border_585']
SCENARIOS=[(m,p) for m in NAMES for p in [.35,.70,.90]]
# original nodal incremental DC operator
Y=(A.T@A.multiply(w[:,None])).tocsc();lu=splu(Y[free,:][:,free])

def solve_theta(mat,inj):
    rhs=np.asarray(inj)
    theta=np.zeros((nd,rhs.shape[1] if rhs.ndim==2 else 1))
    sub=rhs[free];theta[free,:]=mat.solve(sub if sub.ndim==2 else sub[:,None]);return theta

# current six-scenario delta MW injection columns; one site plan held fixed

def gen_injections(year):
    stem=f'{year}_ROBUST_distributed_MG_load_border_585_PV35to90_pen1p0'
    chosen=pd.read_csv(SOURCE/'outputs'/f'{stem}_selected.csv')
    port=pd.read_csv(I/f'portfolio_{year}.csv')
    dgpv=float(port.loc[port.technology=='solar_distributed','new_build_gw'].iloc[0])*1000
    inj=[]
    for mode,pv in SCENARIOS:
        v=np.zeros(nd)
        for row in chosen.itertuples():
            factor=pv if row.tech=='solar_central' else PFAC[row.tech]
            v[ind[int(row.bus_id)]]+=float(row.new_capacity_mw)*factor
        # Distributed PV added at load buses; do NOT invent central bus
        v[loadidx]+=loadweights*dgpv*pv
        # replacement power source / sink must keep net P == 0
        if mode=='distributed_MG_load':v[loadidx]-=loadweights*v.sum()
        else:v[ref]-=v.sum()
        assert abs(v.sum())<1e-7
        inj.append(v)
    return chosen,port,np.array(inj).T

station=X[X.pwf_bus_id.notna()&X.lon.notna()&X.lat.notna()].drop_duplicates('pwf_bus_id')
station=station.set_index('pwf_bus_id')
proj=Transformer.from_crs('EPSG:4326',CRS.from_proj4('+proj=aeqd +lat_0=-18.8 +lon_0=-45 +datum=WGS84 +units=m +no_defs'),always_xy=True)

def line_cost(idx):
    row=L.iloc[idx];a=int(row.f);b=int(row.t)
    if a not in station.index or b not in station.index:return np.nan,np.nan
    sta=station.loc[a];stb=station.loc[b]
    aa=np.array(proj.transform(float(sta.lon),float(sta.lat)));bb=np.array(proj.transform(float(stb.lon),float(stb.lat)))
    dist=float(np.linalg.norm(aa-bb)/1000)
    if dist<1.0:return np.nan,np.nan
    v=float(min(kv[f[idx]],kv[t[idx]]))
    voltage=500 if v>=500 else 345 if v>=345 else 230
    cost=dist*C['route_multiplier_base']*C['line_cost_million_brl_per_km'][str(voltage)] + 2*C['bay_cost_million_brl_per_unit'][str(voltage)]
    return dist,cost

costs={}
for idx in mon:
    d,c=line_cost(int(idx))
    costs[int(idx)]=(d,c)

# Candidate physical identical parallel circuits of existing monitored AC lines
# Each new circuit has source P-flow at the initial AC reference state, reflecting
# incremental linearization of same corridor. A new circuit changes Y and all angles.

def study(year,max_additions=18):
    sel,port,injections=gen_injections(year)
    dr=solve_theta(lu,injections)
    base_new=base[:,None]+w[:,None]*(A@dr)
    obs=base_new[mon,:]
    def measures(flow,rt):
        excess=np.maximum(0,abs(flow)-rt[:,None]);return float(np.max(np.sum(excess,axis=0))),float(np.sum(excess)),int(np.count_nonzero(np.max(excess,axis=1)>0)),float(np.max(abs(flow)/rt[:,None]*100))
    initial=measures(obs,rates)
    scenario_rows=[]
    for j,(mode,pv) in enumerate(SCENARIOS):
        exc=np.maximum(0,abs(obs[:,j])-rates)
        scenario_rows.append({'year':year,'method':'existing_grid','balance':mode,'pv_level':pv,'excess_sum_mw':float(exc.sum()),'violated_branches':int((exc>1e-5).sum()),'max_loading_percent':float(np.max(abs(obs[:,j])/rates)*100)})

    # Run transfer reduction LP for each scenario. Results are conditional on
    # assumed replacement balancing at selected slack/sinks and DO NOT prove
    # sufficient firm replacement generation or SOC.
    transfer=[]
    for j,(mode,pv) in enumerate(SCENARIOS):
        # replace portion of the new source injection by less output, proportionally.
        # Single alpha in [0,1]: curtail alpha fraction of incremental transfers.
        f0=base[mon];g=obs[:,j]-f0
        lo=0.;hi=1.
        # find max retain fraction alpha s.t. +/- thermal ratings. If base itself okay.
        for q in range(nmon):
            if abs(g[q])<1e-12:
                if abs(f0[q])>rates[q]:hi=-1
                continue
            r1=(-rates[q]-f0[q])/g[q];r2=(rates[q]-f0[q])/g[q]
            lo=max(lo,min(r1,r2));hi=min(hi,max(r1,r2))
        feasible=bool(lo<=hi and hi>=0)
        alpha=max(0,min(1,hi)) if feasible else np.nan
        # Transfer reduction is a linearized curtailment/redispatch PROXY.
        transfer.append({'year':year,'balance':mode,'pv_level':pv,'retain_fraction':alpha,'curtailed_fraction':1-alpha if feasible else np.nan,'all_283_lines_within_rating':feasible,'qualification':'Assumes available replacement/withdrawal balancing; no hourly supply or storage feasibility guaranteed'})

    # New parallel circuits: select cost-efficient candidates with verified station coordinates.
    # Track each circuit separately (for n-monitor total over original + added).
    additions=[]; Aadd=[];badd=[];baseadd=[];radd=[];totalcost=0.
    rejected=[];improvements=[]
    def current_model(new_indices):
        if not new_indices:return base_new,base_new[mon,:],np.array([]),lu
        af=A[new_indices,:]
        wa=w[new_indices]
        # updated Y with parallel circuits
        Y2=Y+(af.T@af.multiply(wa[:,None])).tocsc()
        fac=splu(Y2[free,:][:,free])
        # f_new baseline initial offset, so baseline original nodal injection preserved:
        # Y_new * deltaTheta = deltaP - A_new^T baseF_new
        init_new=base[new_indices]
        rhs=injections-(af.T@init_new)[:,None]
        th=solve_theta(fac,rhs)
        flown=base[:,None]+w[:,None]*(A@th)
        newfl=init_new[:,None]+wa[:,None]*(af@th)
        return flown,np.vstack([flown[mon,:],newfl]),newfl,fac
    rt_orig=rates
    rows=[]
    flow_cur=base_new
    obj_orig=initial[0]
    for k in range(max_additions):
        rt_cur=np.r_[rt_orig,L.iloc[additions].rate.to_numpy(float)] if additions else rt_orig
        flow_tot=np.vstack([flow_cur[mon,:],added_flows]) if additions else flow_cur[mon,:]
        prev=measures(flow_tot,rt_cur)
        if prev[0]<1e-5:break
        # Evaluate overloaded original lines + their neighboring monitored lines.
        excess=np.maximum(0,np.abs(flow_cur[mon,:])-rates[:,None])
        candidates=set(int(mon[v]) for v in np.flatnonzero(excess.max(axis=1)>1e-5))
        # include a few nearby high impact branches? restrict to qualifying overloaded corridors
        candlist=[j for j in candidates if np.isfinite(costs[j][1]) and additions.count(j)<3]
        if not candlist:break
        best=None
        for j in candlist:
            trial=additions+[j]
            try:
                flown,allflows,newfl,fac=current_model(trial)
            except Exception as exc:
                rejected.append((j,str(exc)));continue
            rtrial=np.r_[rates,L.iloc[trial].rate.to_numpy(float)]
            metric=measures(allflows,rtrial)
            gain=prev[0]-metric[0]
            if gain<=1e-7:continue
            # prioritise min worst-case exceedance, then cost; greedy heuristic, NOT global optimal.
            score=gain/costs[j][1]
            if best is None or score>best[0]:best=(score,j,flown,allflows,newfl,metric)
        if best is None:break
        _,j,flow_cur,_,added_flows,metric=best
        additions.append(j);totalcost+=costs[j][1]
        improvements.append({'year':year,'order':k+1,'f':int(L.f.iloc[j]),'t':int(L.t.iloc[j]),'circuit_reference':int(L.c.iloc[j]),'line_rating_mva_proxy':float(L.rate.iloc[j]),'added_line_X_pu':float(L.x.iloc[j]),'direct_distance_km':costs[j][0], 'assumed_route_km':costs[j][0]*C['route_multiplier_base'],'assumed_new_circuit_cost_mbrl':costs[j][1],'max_overload_before_mw':prev[0],'max_overload_after_mw':metric[0],'line_selected_because':'heuristic maximizing reduction in worst-scenario overload per R$'})
    rt_fin=np.r_[rates,L.iloc[additions].rate.to_numpy(float)] if additions else rates
    allfin=np.vstack([flow_cur[mon,:],added_flows]) if additions else flow_cur[mon,:]
    final=measures(allfin,rt_fin)
    for j,(mode,pv) in enumerate(SCENARIOS):
        exc=np.maximum(0,abs(allfin[:,j])-rt_fin)
        scenario_rows.append({'year':year,'method':'parallel_line_screen','balance':mode,'pv_level':pv,'excess_sum_mw':float(exc.sum()),'violated_branches':int((exc>1e-5).sum()),'max_loading_percent':float(np.max(abs(allfin[:,j])/rt_fin)*100)})
    # Screening of corrective dispatch after physical parallel-circuit changes.
    # Candidate controls: curtail newly built utility-scale PV and wind only, with
    # energy replacement at the declared balancing sink. This is NOT adequacy-verified.
    redispatch=[];residual=[]
    allA=A if not additions else coo_matrix(np.vstack([A.toarray(),A[additions,:].toarray()])).tocsr()
    allw=w if not additions else np.r_[w,w[additions]]
    Ynew=(allA.T@allA.multiply(allw[:,None])).tocsc();funew=splu(Ynew[free,:][:,free])
    # Voltage-angle reference fully recomputed for each network. Only the original
    # calibrated source P flows are taken from EPE; added circuit has same offset.
    allbase=base if not additions else np.r_[base,base[additions]]
    allR=np.r_[rates,L.iloc[additions].rate.to_numpy(float)] if additions else rates
    midx=np.r_[mon,np.arange(nl,nl+len(additions))] if additions else mon
    baserhs=np.zeros(nd)
    if additions:baserhs-=A[additions,:].T@base[additions]
    th0=solve_theta(funew,baserhs)
    baseline_expanded=(allbase[:,None]+allw[:,None]*(allA@th0))[midx,0]
    for jj,(mode,pv) in enumerate(SCENARIOS):
        now=allfin[:,jj]
        # Extract residual bottlenecks after physical additions, including new circuits.
        for rr,flow,lim in zip(midx,now,allR):
            over=max(0,abs(float(flow))-float(lim))
            if over>1e-5:
                if rr<nl:rrow=L.iloc[rr];label='existing'
                else:rrow=L.iloc[additions[rr-nl]];label='new parallel'
                residual.append({'year':year,'balance':mode,'pv_factor':pv,'component':label,'from_bus':int(rrow.f),'to_bus':int(rrow.t),'circuit':int(rrow.c),'rating_mw_proxy':float(lim),'power_mw':float(flow),'excess_mw':over})
        controllable=sel[sel.tech.isin(['solar_central','wind_site'])].copy()
        if len(controllable)==0:
            redispatch.append({'year':year,'balance':mode,'pv_factor':pv,'feasible':float(np.max(np.maximum(0,np.abs(now)-allR)))<1e-5,'minimum_replacement_dispatch_mw':0.0,'replaced_share_of_new_transfer':0.0})
            continue
        cols=[];upper=[]
        for rr in controllable.itertuples():
            cap=float(rr.new_capacity_mw)*(pv if rr.tech=='solar_central' else PFAC[rr.tech])
            u=np.zeros(nd);u[ind[int(rr.bus_id)]]=1
            if mode=='border_585':u[ref]-=1
            else:u[loadidx]-=loadweights
            cols.append(u);upper.append(cap)
        delta=np.column_stack(cols)
        dtheta=solve_theta(funew,delta)
        sensitivity=(allw[:,None]*(allA@dtheta))[midx,:]
        # Flow after corrective curtailment and exact modeled replacement at sink:
        # f_corrected = f_now - sensitivity * curtailment
        cons=np.vstack([sensitivity,-sensitivity])
        rh=np.r_[now+allR,allR-now]
        # -R <= now - S u <= R implies S u <= now+R and -S u <= R-now
        sol=linprog(np.ones(len(upper)),A_ub=cons,b_ub=rh,bounds=list(zip(np.zeros(len(upper)),upper)),method='highs')
        good=bool(sol.success)
        if good:
            resulting=now-sensitivity@sol.x
            error=float(np.max(np.maximum(0,np.abs(resulting)-allR)))
            req=float(sol.x.sum());frac=req/max(sum(upper),1.)
        else:error=np.nan;req=np.nan;frac=np.nan
        redispatch.append({'year':year,'balance':mode,'pv_factor':pv,'feasible':good,'minimum_replacement_dispatch_mw':req,'replaced_share_of_new_PV_wind_output':frac,'max_rating_violation_after_mw':error,'source':'Existing new PV/wind dispatched down, exactly compensated at assumed sink; resource headroom, SOC, and 8760h feasibility not verified'})
    pd.DataFrame(redispatch).to_csv(O/f'{year}_corrective_dispatch_screen.csv',index=False)
    pd.DataFrame(residual,columns=['year','balance','pv_factor','component','from_bus','to_bus','circuit','rating_mw_proxy','power_mw','excess_mw']).to_csv(O/f'{year}_remaining_bottlenecks.csv',index=False)
    pd.DataFrame(scenario_rows).to_csv(O/f'{year}_scenario_comparison.csv',index=False)
    pd.DataFrame(transfer).to_csv(O/f'{year}_transfer_management_proxy.csv',index=False)
    pd.DataFrame(improvements,columns=['year','order','f','t','circuit_reference','line_rating_mva_proxy','added_line_X_pu','direct_distance_km','assumed_route_km','assumed_new_circuit_cost_mbrl','max_overload_before_mw','max_overload_after_mw','line_selected_because']).to_csv(O/f'{year}_parallel_line_candidates.csv',index=False)
    # Check nodal power conservation of the recalculated incremental DC model, not original AC.
    if additions:
        y2=Y+(A[additions,:].T@A[additions,:].multiply(w[additions,None])).tocsc()
        rhs=injections-(A[additions,:].T@base[additions])[:,None]
        th=solve_theta(splu(y2[free,:][:,free]),rhs)
        nodal_res=y2@th-rhs
        # slack row is dependent and generally balances if sum(rhs)==0
        balance_error=float(np.max(np.abs(nodal_res[free,:])))
    else:balance_error=float(np.max(np.abs(Y@dr-injections)[free,:]))
    result={'year':year,'source':'EPE 2035 representative PWF AC P-flow calibration; one fixed nodal siting plan; six concurrent stress cases',
        'n_buses':nd,'original_branch_count':nl,'monitored_existing_lines':nmon,'reference_grid_only':True,
        'initial_worst_case_sum_of_overloads_mw':initial[0],
        'initial_sum_over_all_cases_mw':initial[1],
        'initial_overloaded_unique_monitored_lines':initial[2],
        'initial_max_loading_pct':initial[3],
        'greedy_added_parallel_circuits':len(additions),
        'cost_assumed_million_brl':totalcost,
        'remaining_worst_case_sum_of_overloads_mw':final[0],
        'remaining_sum_over_all_cases_mw':final[1],
        'remaining_unique_overloaded_lines_including_added_circuits':final[2],
        'remaining_max_loading_pct':final[3],
        'all_cases_meet_283_monitored_constraints':bool(final[0]<=1e-5),
        'nodal_equation_max_residual_MW':balance_error,
        'corrective_dispatch_all_cases_feasible':all(q['feasible'] for q in redispatch),
        'corrective_dispatch_worst_case_mw':max((q['minimum_replacement_dispatch_mw'] for q in redispatch if q['feasible']),default=0),
        'n_candidate_branches_without_usable_coords':int(sum(1 for j in mon if not np.isfinite(costs[int(j)][1]))),
        'critical_qualifications':['Line upgrades are illustrative identical parallel circuits with Euclidean route lengths; not licensed or costed projects.',
                                   'Model is incremental linearised DC on 2035 reference AC flows, not post-expansion AC solution.',
                                   'Transformers, Q constraints, all equipment, stability and full 8760h nodal operation are not checked.',
                                   '2050 uses 2035 network and assumes input nodal siting; new-project land/fuel qualifications unverified.',
                                   'The heuristic chooses the next most cost-effective existing monitored line; no optimality guarantee.']}
    (O/f'{year}_summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print('YEAR',year,json.dumps({k:v for k,v in result.items() if k in ['initial_worst_case_sum_of_overloads_mw','greedy_added_parallel_circuits','cost_assumed_million_brl','remaining_worst_case_sum_of_overloads_mw','all_cases_meet_283_monitored_constraints','nodal_equation_max_residual_MW']},indent=2))
    return result

if __name__=='__main__':
    res=[study(2035),study(2050)]
    pd.DataFrame(res).to_csv(O/'summary_all.csv',index=False)
