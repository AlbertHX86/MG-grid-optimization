#!/usr/bin/env python3
"""Location-aware, connection-cost + PTDF thermal-uplift scenario MILP.

All new generator GW totals are fixed from previously solved hourly portfolio LPs.
EPE 2035 *planning* PWF exported bus/branch inputs. The 2050 case is a frozen-
2035 topology sensitivity, NOT a validated 2050 grid or engineering project plan.
Municipality-point sites, capacities, and per-site headroom are hypothetical.
"""
from pathlib import Path
import json, time, math
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csc_matrix, hstack, vstack
from scipy.sparse.linalg import splu
from scipy.optimize import milp, Bounds, LinearConstraint
from pyproj import CRS, Transformer

ROOT=Path(__file__).resolve().parents[1]
I=ROOT/'inputs'; O=ROOT/'outputs'; O.mkdir(parents=True,exist_ok=True)
B=pd.read_csv(I/'pwf_2035_buses.csv');L=pd.read_csv(I/'pwf_2035_branches.csv')
A=pd.read_csv(I/'candidate_areas.csv')
C=json.loads((I/'cost_assumptions.json').read_text())
X=pd.read_csv(I/'epe_ons_station_crosswalk.csv')
W=pd.read_csv(I/'allocation_weights.csv')
assert B.id.is_unique and len(B)==1489 and len(L)==1890
ids=B.id.to_numpy(int);n=len(B);m0=len(L);index={int(v):i for i,v in enumerate(ids)}
f=L.f.map(index).to_numpy(int);t=L.t.map(index).to_numpy(int)
x=L.x.to_numpy(float);tap=L.tap.to_numpy(float);w=1/(x*tap)
assert np.all(np.isfinite(w)) and np.all(x!=0) and np.all(tap>0)
Aa=coo_matrix((np.r_[np.ones(m0),-np.ones(m0)],(np.r_[np.arange(m0),np.arange(m0)],np.r_[f,t])),shape=(m0,n)).tocsr()
Y=(Aa.T@Aa.multiply(w[:,None])).tocsc(); ref=index[585]
free=np.array([j for j in range(n) if j!=ref],int); pos={int(j):k for k,j in enumerate(free)}
lu=splu(Y[free,:][:,free]);pqs=B.kv.to_numpy(float);state=B.state.to_numpy()
mask=(L.kind=='line')&L.rate.between(100,9998)&(L.x>0)&(np.minimum(pqs[f],pqs[t])>=230)&((state[f]==11)|(state[t]==11))
monitor=np.flatnonzero(mask.to_numpy());lm=L.iloc[monitor].reset_index(drop=True)
assert len(monitor)==283
F=lm.p_from_mw.to_numpy(float);R=lm.rate.to_numpy(float);nr=len(F)

# Maps every PWF bus to a vector of the monitored branch flows per 1 MW net injection.
# Positive site injection is balanced by 1 MW withdrawal at the named border bus 585.
def ptdf_bus(bus_ids):
    g=np.zeros((len(free),len(bus_ids)))
    for k,b in enumerate(bus_ids):
        j=index[int(b)]
        if j!=ref:g[pos[j],k]=1
    angle=lu.solve(g) # already per-unit dimensionless GW scaling cancels in ratios
    return ((w[monitor,None])*(Aa[monitor,:][:,free]@angle))

loadidx=B.index[(B.state==11)&(B.pd>0)].to_numpy()
loadval=B.iloc[loadidx].pd.to_numpy(float);loadval=loadval/loadval.sum()
# Nodal load (MG) weighted withdrawal vector for local balancing.
withdraw=(ptdf_bus(B.iloc[loadidx].id.to_numpy())@loadval)

metric=CRS.from_proj4('+proj=aeqd +lat_0=-18.8 +lon_0=-45 +datum=WGS84 +units=m +no_defs')
trans=Transformer.from_crs('EPSG:4326',metric,always_xy=True)
def xyz(lon,lat):return trans.transform(float(lon),float(lat))
S=X[(X.pwf_state=='MG')&X.pwf_bus_id.notna()&X.lat.notna()&X.lon.notna()].copy()
S=S[S.pwf_bus_id.isin(B.id)].drop_duplicates('pwf_bus_id')
S['e_m'],S['n_m']=trans.transform(S.lon.to_numpy(),S.lat.to_numpy())
areas={r.area:r for r in A.itertuples(index=False)}
# Study-only area ceilings: 150% of allocated capacity at that area.
# Not actual resource/land/permits/feeder capacity, so don't claim project feasibility.
alpha=1.5
SUPPORTED={'solar_central','bio','small_hydro','wind_site','biogas','bess'}
PFAC={'solar_central':.70,'bio':.50,'small_hydro':.60,'wind_site':.50,'biogas':.70,'bess':0.0}
PVFACTOR=.70

# Technical constraints: when a cluster is huge, require a higher-voltage station.
def min_voltage(tech, cap):
    if tech=='bess':return 230
    if cap>=2000 and tech in {'solar_central','wind_site'}:return 500
    if cap>=500 and tech in {'solar_central','wind_site','bess'}:return 345
    if cap>=250:return 230
    return 138

def candidate_table(year,portfolio):
    entries=[]; clusters=[]
    Wyr=W[(W.year==year)&(W.technology.isin(SUPPORTED))].copy()
    for item in Wyr.itertuples(index=False):
        cap=float(portfolio.get(item.technology,0)*1000*item.weight)
        if cap<1e-5:continue
        ar=areas[item.area]; ex,ny=xyz(ar.longitude,ar.latitude)
        kvmin=min_voltage(item.technology,cap)
        opts=S[S.pwf_kv>=kvmin].copy()
        opts['distance_km']=np.hypot(opts.e_m-ex,opts.n_m-ny)/1000
        # Prefer three physically distinct eligible substation locations.
        # Several PWF buses share one ONS substation (same coordinates, different kV), so
        # distance alone ties. Break ties deterministically: lowest eligible kV (cheapest
        # bay/line rates), then bus id. A plain sort_values() is not stable, so the kept bus
        # among tied rows was left unspecified. (Lowest kV is a modelling choice.)
        opts=(opts.sort_values(['distance_km','pwf_kv','pwf_bus_id'],kind='mergesort')
                  .drop_duplicates('ons_id_subestacao').head(3))
        if opts.empty:raise RuntimeError(f'No PWF crosswalk candidate for {item.area}')
        cl={'cluster_id':len(clusters),'tech':item.technology,'area':item.area,'baseline_mw':cap,
             'max_area_mw':alpha*cap,'source':ar.resource_basis,'area_lat':ar.latitude,'area_lon':ar.longitude,'mode':item.mode}
        clusters.append(cl)
        for rank,st in enumerate(opts.itertuples(index=False)):
            # Upgrade cost uses line/bay cost per circuit and 0.08 million R$/MW for step-up.
            vv=int(st.pwf_kv)
            line=0 if item.technology=='bess' else float(st.distance_km)*C['route_multiplier_base']
            per_circuit=(line*C['line_cost_million_brl_per_km'][str(vv)]
                          + (1 if item.technology=='bess' else 2)*C['bay_cost_million_brl_per_unit'][str(vv)])
            entries.append({'year':year,'cluster_id':cl['cluster_id'],'tech':item.technology,'area':item.area,'rank':rank,
              'bus_id':int(st.pwf_bus_id),'station':st.ons_name,'station_ons_id':st.ons_id_subestacao,
              'kv':vv,'lon':st.lon,'lat':st.lat,'distance_km':float(st.distance_km),
              'route_length_km':line,'per_circuit_mbrl':per_circuit,
              'max_circuit_mw':C['max_circuit_mw_proxy'][str(vv)],
              'variable_cost_mbrl_mw':C['transformer_cost_million_brl_per_mw_assumed'],
              'max_area_mw':alpha*cap,'baseline_mw':cap,'mode':item.mode,'resource_basis':ar.resource_basis})
    return pd.DataFrame(entries),pd.DataFrame(clusters)


def solve(year, balance, pv_factor, line_penalty, time_limit=90):
    p=pd.read_csv(I/f'portfolio_{year}.csv')
    tech_mw={r.technology:float(r.new_build_gw) for r in p.itertuples(index=False)}
    cand, clusters=candidate_table(year,tech_mw)
    k=len(cand);nc=len(clusters)
    # One x and integer number of connection circuits z per potential cluster-station arc;
    # one MW of extra monitored-line rating u per monitored branch.
    # Fixed statewide technology capacities from hourly LP: no freedom to change technology mix here.
    nvar=2*k+nr
    uidx=2*k
    # DG PV: no high-voltage collector circuit, but distributed injection affects flow.
    dg=float(tech_mw.get('solar_distributed',0))*1000
    static_dg=ptdf_bus(B.iloc[loadidx].id.to_numpy())@loadval*(dg*pv_factor)
    basewithdraw=withdraw if balance=='distributed_MG_load' else np.zeros(nr)
    buses=cand.bus_id.to_numpy(int)
    src_ptdf=ptdf_bus(buses)
    tech_factor=np.array([pv_factor if a=='solar_central' else PFAC[a] for a in cand.tech])
    injection=src_ptdf*tech_factor[None,:]
    # withdrawals include all positive generation, but BESS=0 in snapshot
    # fixed DG is withdrawn locally or at ref according to balancing assumption
    if balance=='distributed_MG_load':
        injection=injection-basewithdraw[:,None]*tech_factor[None,:]
        static_dg=static_dg-basewithdraw*(dg*pv_factor)
    F0=F+static_dg
    Cobj=np.zeros(nvar)
    Cobj[:k]=cand.variable_cost_mbrl_mw.to_numpy(float)
    Cobj[k:2*k]=cand.per_circuit_mbrl.to_numpy(float)
    Cobj[uidx:]=line_penalty
    low=np.zeros(nvar);up=np.full(nvar,np.inf)
    up[:k]=cand.max_area_mw.to_numpy(float)
    up[k:2*k]=np.ceil(cand.max_area_mw.to_numpy()/cand.max_circuit_mw.to_numpy())
    ints=np.zeros(nvar,int);ints[k:2*k]=1
    rows=[];cols=[];vals=[];lhs=[];rhs=[];row=0
    def add_sparse(colidx,v,l=-np.inf,h=np.inf):
        nonlocal row
        colidx=np.asarray(colidx,int); v=np.asarray(v,float)
        nz=np.abs(v)>1e-13
        rows.extend([row]*int(nz.sum()));cols.extend(colidx[nz].tolist());vals.extend(v[nz].tolist());lhs.append(l);rhs.append(h);row+=1
    # Each technology total is fixed exactly (ex-ante hour-by-hour LP solution)
    for tech,cap in tech_mw.items():
        if tech not in SUPPORTED or cap<1e-9:continue
        kk=np.flatnonzero(cand.tech==tech)
        assert len(kk)>0,(year,tech)
        add_sparse(kk,np.ones(len(kk)),l=cap*1000,h=cap*1000)
    # Per-area ceilings (aggregate across station choices)
    for cl in clusters.itertuples(index=False):
        kk=np.flatnonzero(cand.cluster_id==cl.cluster_id)
        add_sparse(kk,np.ones(len(kk)),h=cl.max_area_mw)
    # Required number of connection circuits: x_i <= maxMW_i * integer z_i
    for j in range(k): add_sparse([j,k+j],[1,-float(cand.max_circuit_mw.iloc[j])],h=0)
    # PTDF constrained branch normal thermal envelopes with decision u_l.
    # Absolute baseline AC P and MVA thermal proxy = screening only, not AC operation.
    for j in range(nr):
        inds=np.r_[np.arange(k),uidx+j]
        vv=np.r_[injection[j,:],-1.]
        add_sparse(inds,vv,h=R[j]-F0[j])
        add_sparse(inds,np.r_[-injection[j,:],-1.],h=R[j]+F0[j])
    constraint=coo_matrix((vals,(rows,cols)),shape=(row,nvar)).tocsr()
    start=time.time()
    res=milp(c=Cobj,integrality=ints,bounds=Bounds(low,up),constraints=LinearConstraint(constraint,lhs,rhs),options={'time_limit':time_limit,'mip_rel_gap':0.0001})
    if res.x is None:raise RuntimeError(f'{year} {balance} failed: {res.message}')
    v=res.x
    prim=np.asarray(constraint@v); violation=max(float(np.max(np.asarray(lhs)-prim)),float(np.max(prim-np.asarray(rhs))),0.)
    cap_error=0
    for tech,cap in tech_mw.items():
        if tech not in SUPPORTED or cap<1e-9:continue
        cap_error=max(cap_error,abs(float(v[:k][cand.tech==tech].sum())-cap*1000))
    d=cand.copy();d['new_capacity_mw']=v[:k];d['circuits']=np.rint(v[k:2*k]).astype(int)
    d['line_and_bay_cost_mbrl']=d['circuits']*d['per_circuit_mbrl'];d['step_up_cost_mbrl']=d['new_capacity_mw']*d['variable_cost_mbrl_mw']
    d['connection_cost_mbrl']=d.line_and_bay_cost_mbrl+d.step_up_cost_mbrl
    d['flow_coefficient_at_snapshot']=tech_factor
    selected=d[(d.new_capacity_mw>1e-4)|(d.circuits>0)].copy()
    uplift=v[uidx:]
    f_after=F0+injection@v[:k]
    lines=lm[['f','t','c','kind','rate','p_from_mw']].copy();lines['network_uplift_mw']=uplift;lines['flow_after_mw']=f_after
    lines['over_before_mw']=np.maximum(0,np.abs(F0)-R)
    lines['load_percent_before_uplift']=np.abs(f_after)/R*100
    lines['load_percent_after_uplift']=np.abs(f_after)/(R+uplift)*100
    # Reference allocation: each area gets previous weights, all to the nearest PWF-linked eligible station.
    ref=np.zeros(k);zref=np.zeros(k)
    for cl in clusters.itertuples(index=False):
        js=np.flatnonzero(cand.cluster_id==cl.cluster_id)
        j=js[0]
        ref[j]=cl.baseline_mw
        zref[j]=math.ceil(cl.baseline_mw / float(cand.max_circuit_mw.iloc[j])-1e-10)
    ref_flow=F0+injection@ref
    ref_u=np.maximum(0,np.abs(ref_flow)-R)
    def costfor(x,z,u):
        return float(np.dot(x,cand.variable_cost_mbrl_mw)+np.dot(z,cand.per_circuit_mbrl)),float(u.sum()),float(np.dot(x,cand.variable_cost_mbrl_mw)+np.dot(z,cand.per_circuit_mbrl)+line_penalty*u.sum())
    opt_conn,opt_up,opt_obj=costfor(v[:k],v[k:2*k],uplift)
    ref_conn,ref_up,ref_obj=costfor(ref,zref,ref_u)
    dgcost=dg*C['distribution_upgrade_million_brl_per_mw_assumed']
    name=f'{year}_{balance}_PV{int(round(pv_factor*100))}_pen{str(line_penalty).replace(".","p")}'
    d.to_csv(O/f'{name}_candidates.csv',index=False)
    selected.to_csv(O/f'{name}_selected.csv',index=False)
    lines.to_csv(O/f'{name}_lines.csv',index=False)
    # Since half the electricity could be withdrawn at the MG area, reports conditional on balancing.
    report={'name':name,'year':year,'balancing':balance,'pv_factor':pv_factor,'thermal_uplift_penalty_mbrl_mw':line_penalty,
        'solver_status':int(res.status),'solver_message':str(res.message),'solver_mip_gap':getattr(res,'mip_gap',None),
        'optimal_certified':bool(res.status==0),'n_candidates':k,'n_clusters':nc,'n_physical_branches':nr,
        'fixed_new_portfolio_mw':sum(float(t)*1000 for t in tech_mw.values()),'monitored_network_frozen_2035':True,
        'candidate_cap_multiplier_assumed':alpha,'max_constraint_violation':violation,'max_technology_allocation_error_mw':cap_error,
        'reference_connection_nominal_mbrl':ref_conn+dgcost,'optimized_connection_nominal_mbrl':opt_conn+dgcost,
        'reference_thermal_uplift_mw':ref_up,'optimized_thermal_uplift_mw':opt_up,
        'reference_combined_screen_cost_mbrl':ref_obj+dgcost,'optimized_combined_screen_cost_mbrl':opt_obj+dgcost,
        'nominal_delta_mbrl':opt_obj-ref_obj,'time_seconds':time.time()-start,
        'n_selected_sites':len(selected),'n_uprated_monitored_branches':int((uplift>1e-4).sum()),
        'snapshot_energy_factors':dict(PFAC,**{'solar_central':pv_factor,'solar_distributed':pv_factor}),
        'assumption_note':'NOT a joint portfolio-choice optimization; fixed tech total. Local proxy station options and site caps are hypothetical. Thermal uplift is not a line project or price.'}
    (O/f'{name}_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(name,report['solver_message'],'connection',round(opt_conn+dgcost,2),'uplift',round(opt_up,2),'baseline',round(ref_conn+dgcost,2),round(ref_up,2), 'delta',round(opt_obj-ref_obj,2), 'seconds',round(report['time_seconds'],1),flush=True)
    return report

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--years',nargs='+',type=int,default=[2035,2050]);ap.add_argument('--balancing',nargs='+',default=['distributed_MG_load','border_585']);ap.add_argument('--pv',nargs='+',type=float,default=[.7]);ap.add_argument('--penalty',type=float,default=1.0);ap.add_argument('--time-limit',type=float,default=90.)
    args=ap.parse_args()
    summaries=[]
    for yr in args.years:
        for b in args.balancing:
            for pv in args.pv:
                summaries.append(solve(yr,b,pv,args.penalty,args.time_limit))
    pd.DataFrame(summaries).to_csv(O/'summary_all.csv',index=False)

# Conservative common-siting solution: one x,z and one set of line uplifts must pass all
# the declared cases. This avoids selecting a new generator site for each scenario.
def solve_robust(year, balance_modes=('distributed_MG_load','border_585'), pv_levels=(.35,.70,.90), line_penalty=1., time_limit=90):
    p=pd.read_csv(I/f'portfolio_{year}.csv')
    portfolio={r.technology:float(r.new_build_gw) for r in p.itertuples(index=False)}
    cand,clusters=candidate_table(year,portfolio)
    k=len(cand);nvar=2*k+nr
    dg=float(portfolio.get('solar_distributed',0)*1000)
    src=ptdf_bus(cand.bus_id.to_numpy(int))
    load_ptdf=ptdf_bus(B.iloc[loadidx].id.to_numpy())@loadval
    models=[]
    for mode in balance_modes:
        for pv in pv_levels:
            factor=np.array([pv if tech=='solar_central' else PFAC[tech] for tech in cand.tech])
            sens=src*factor[None,:]
            static=load_ptdf*(dg*pv)
            if mode=='distributed_MG_load':
                sens=sens-withdraw[:,None]*factor[None,:]
                static=static-withdraw*dg*pv
            models.append((mode,pv,F+static,sens))
    obj=np.zeros(nvar);obj[:k]=cand.variable_cost_mbrl_mw.to_numpy(float)
    obj[k:2*k]=cand.per_circuit_mbrl.to_numpy(float);obj[2*k:]=line_penalty
    low=np.zeros(nvar);upper=np.full(nvar,np.inf)
    upper[:k]=cand.max_area_mw.to_numpy(float)
    upper[k:2*k]=np.ceil(cand.max_area_mw.to_numpy()/cand.max_circuit_mw.to_numpy())
    ints=np.zeros(nvar,int);ints[k:2*k]=1
    rows=[];cols=[];data=[];l=[];u=[];row=0
    def add(jj,vv,lo=-np.inf,hi=np.inf):
        nonlocal row
        jj=np.asarray(jj,int);vv=np.asarray(vv,float);nz=abs(vv)>1e-13
        rows.extend([row]*int(sum(nz)));cols.extend(jj[nz].tolist());data.extend(vv[nz].tolist());l.append(lo);u.append(hi);row+=1
    for tech,mw in portfolio.items():
        if tech in SUPPORTED and mw>1e-9:
            jj=np.flatnonzero(cand.tech==tech);add(jj,np.ones(len(jj)),mw*1000,mw*1000)
    for cl in clusters.itertuples(index=False):
        jj=np.flatnonzero(cand.cluster_id==cl.cluster_id);add(jj,np.ones(len(jj)),hi=cl.max_area_mw)
    for j in range(k): add([j,k+j],[1,-float(cand.max_circuit_mw.iloc[j])],hi=0)
    for mode,pv,F0,sens in models:
        for ell in range(nr):
            jj=np.r_[np.arange(k),2*k+ell]
            add(jj,np.r_[sens[ell,:],-1.],hi=R[ell]-F0[ell])
            add(jj,np.r_[-sens[ell,:],-1.],hi=R[ell]+F0[ell])
    M=coo_matrix((data,(rows,cols)),shape=(row,nvar)).tocsr()
    started=time.time()
    rr=milp(c=obj,integrality=ints,bounds=Bounds(low,upper),constraints=LinearConstraint(M,l,u),options={'time_limit':time_limit,'mip_rel_gap':.0001})
    if rr.x is None:raise RuntimeError('ROBUST_FAIL '+str(rr.message))
    sol=rr.x;xx=sol[:k];zz=np.rint(sol[k:2*k]).astype(int);uplift=sol[2*k:]
    csel=cand.copy();csel['new_capacity_mw']=xx;csel['circuits']=zz
    csel['connection_cost_mbrl']=xx*cand.variable_cost_mbrl_mw+zz*cand.per_circuit_mbrl
    dgcost=dg*C['distribution_upgrade_million_brl_per_mw_assumed']
    csel=csel[(csel.new_capacity_mw>1e-5)|(csel.circuits>0)].copy()
    ref=np.zeros(k);refz=np.zeros(k)
    for cl in clusters.itertuples(index=False):
        j=np.flatnonzero(cand.cluster_id==cl.cluster_id)[0]
        ref[j]=cl.baseline_mw;refz[j]=math.ceil(cl.baseline_mw / float(cand.max_circuit_mw.iloc[j])-1e-10)
    ref_u=np.zeros(nr);case_rows=[];branch_rows=[]
    for mode,pv,F0,sens in models:
        fres=F0+sens@xx;fb=F0+sens@ref
        ref_u=np.maximum(ref_u,np.maximum(0,np.abs(fb)-R))
        screenexcess=np.maximum(0,np.abs(fres)-R)
        case_rows.append({'year':year,'mode':mode,'pv_factor':pv,
             'reference_excess_mw':float(np.maximum(0,np.abs(fb)-R).sum()),
             'optimized_excess_mw':float(screenexcess.sum()),
             'max_loading_pct_without_uprate':float((np.abs(fres)/R*100).max()),
             'max_loading_pct_after_uprate':float((np.abs(fres)/(R+uplift)*100).max()),
             'n_violated_without_uprate':int(sum(screenexcess>1e-5))})
        for i,line in enumerate(lm.itertuples(index=False)):
            if uplift[i]>1e-3:
                branch_rows.append({'year':year,'mode':mode,'pv_factor':pv,'from_bus':int(line.f),'to_bus':int(line.t),'circuit':line.c,
                     'base_rating_mva_proxy':float(R[i]),'base_flow_mw':float(F0[i]),
                     'new_flow_mw':float(fres[i]),'rating_uplift_mw':float(uplift[i])})
    reference_conn=float(np.dot(ref,cand.variable_cost_mbrl_mw)+np.dot(refz,cand.per_circuit_mbrl)+dgcost)
    optimized_conn=float(csel.connection_cost_mbrl.sum()+dgcost)
    residual=np.asarray(M@sol); maxviol=max(float(np.max(np.asarray(l)-residual)),float(np.max(residual-np.asarray(u))),0.)
    per_tech={t:float(csel.loc[csel.tech==t,'new_capacity_mw'].sum()) for t in SUPPORTED}
    error=max([abs(per_tech.get(t,0)-v*1000) for t,v in portfolio.items() if t in SUPPORTED and v>1e-7]+[0])
    suffix=f'{year}_ROBUST_{"_".join(balance_modes)}_PV{int(min(pv_levels)*100)}to{int(max(pv_levels)*100)}_pen{str(line_penalty).replace(".","p")}'
    csel.to_csv(O/f'{suffix}_selected.csv',index=False)
    pd.DataFrame(case_rows).to_csv(O/f'{suffix}_cases.csv',index=False)
    pd.DataFrame(branch_rows).to_csv(O/f'{suffix}_line_scenarios.csv',index=False)
    summary={'name':suffix,'year':year,'solver_status':int(rr.status),'success':rr.status==0,'solver_message':str(rr.message),
       'mip_gap':getattr(rr,'mip_gap',None),'max_constraint_violation':maxviol,'capacity_error_mw':error,
       'n_candidate_arcs':k,'n_resource_clusters':len(clusters),'n_monitored_lines':nr,'n_scenarios':len(models),
       'yearly_portfolio_fixed':True,'dg_mw':dg,'dg_cost_mbrl':dgcost,
       'optimized_connection_nominal_mbrl':optimized_conn,'reference_connection_nominal_mbrl':reference_conn,
       'optimized_worstcase_uplift_mw':float(uplift.sum()),'reference_worstcase_uplift_mw':float(ref_u.sum()),
       'optimized_screen_objective_mbrl':optimized_conn+line_penalty*uplift.sum(),
       'reference_screen_objective_mbrl':reference_conn+line_penalty*ref_u.sum(),
       'n_uprated_branches':int(sum(uplift>1e-5)),'n_active_arcs':len(csel),
       'cost_assumption':'R$m/MW thermal-uplift is a valuation penalty NOT actual line project price',
       'time_seconds':time.time()-started}
    (O/f'{suffix}_summary.json').write_text(json.dumps(summary,indent=2))
    print('ROBUST',suffix,'optimized',round(summary['optimized_screen_objective_mbrl'],2),'reference',round(summary['reference_screen_objective_mbrl'],2),'connection',round(optimized_conn,2),'uplift',round(float(uplift.sum()),2),'status',rr.message,'seconds',round(time.time()-started,2),flush=True)
    return summary
