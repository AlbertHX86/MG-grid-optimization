"""Transparent multistage hourly MG capacity-expansion screening LP.

Units inside the LP: GW, GWh, million constant-2024 BRL.
No representative days, fabricated load curves, free imports, or free batteries.
The physical scope is a single ONS electrical area, not a full SIN SCOPF.
"""
from pathlib import Path
import argparse, json, time, platform, gzip, hashlib, io, os
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
TECH=['solar_central','solar_distributed','bio','small_hydro','wind','wind_site','biogas','bess']
FIELDS=['hydro','vre_used','gas','imports','charge','discharge','soc','shed','biogas_dispatch']

class SparseRows:
    def __init__(self):self.r=[];self.c=[];self.v=[];self.b=[];self.nr=0
    def add(self,terms,rhs):
        rhs=np.atleast_1d(np.asarray(rhs,dtype=float)); n=len(rhs)
        rows=np.arange(self.nr,self.nr+n,dtype=np.int32)
        for cols,vals in terms:
            cols=np.broadcast_to(np.asarray(cols,dtype=np.int32),(n,))
            vals=np.broadcast_to(np.asarray(vals,dtype=float),(n,))
            nz=vals!=0
            self.r.append(rows[nz]);self.c.append(cols[nz]);self.v.append(vals[nz])
        self.b.append(rhs);self.nr+=n
    def matrix(self,nvars):
        return coo_matrix((np.concatenate(self.v),(np.concatenate(self.r),np.concatenate(self.c))),shape=(self.nr,nvars)).tocsc(),np.concatenate(self.b)

def solve_case(name,policy='terminal',demand='low',hydro_stress=False,import_scale=1.,phaseout=False,
               fixed_case=None,only_year=None,allow_shed=False,price_scale=1.,cost_scale=1.,solver='highs-ds',resource_envelope=True,
               netzero_mode='none',fossil_emergency_reserve=False, scenario_config=None):
    started=time.time()
    p=json.loads((ROOT/'macro/processed/macro_parameters.json').read_text())
    cfg = json.loads(Path(scenario_config).read_text()) if scenario_config else {}
    p['technology']['wind_site']=dict(p['technology']['wind'])
    p['technology']['wind_site']['source']='Assumed high-resource candidate based on 2019 MG temporal shape; no verified candidate station'
    p['technology']['biogas']={
        'life_years':20.,'capex_brl_kw':float(cfg.get('biogas_capex_brl_kw',7200.)),
        'fixed_cost_brl_kw_year':float(cfg.get('biogas_fixed_brl_kw_year',300.)),
        'construction_interest_fraction':0.064,
        'source':'Unverified research proxy; requires MG fuel supply, lifecycle and project costs'
    }
    p['base_capacity_gw']['wind_site']=0.;p['base_capacity_gw']['biogas']=0.
    p['bess_duration_hours']=float(cfg.get('bess_duration_hours',p['bess_duration_hours']))
    assert p['bess_duration_hours']>=4, 'This study supports >=4-hour duration only'
    config_name=Path(scenario_config).name if scenario_config else 'baseline_unchanged'
    code_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    f=pd.read_csv(ROOT/'macro/processed/macro_hourly_inputs.csv',parse_dates=['hour_start_brt'])
    h=len(f);assert h==8760
    windpath=ROOT/'macro/processed/mg_wind_2019_brt_aligned.csv'
    if windpath.exists():
        wind=pd.read_csv(windpath)['capacity_factor'].to_numpy(float)
        assert len(wind)==h
    else:
        # Not permitted to silently substitute a guessed wind profile.
        wind=np.zeros(h)
    # Candidate wind is explicitly a resource scenario, not a measured site profile.
    wind_target=float(cfg.get('wind_site_mean_cf',0.30))
    if not (0 < wind_target < 1): raise ValueError('Wind site CF should lie between 0 and 1')
    if np.mean(wind)>0:
        lo,hi=0.,100.
        for _ in range(50):
            m=(lo+hi)/2
            if np.minimum(1,wind*m).mean()<wind_target:lo=m
            else:hi=m
        candidate_wind=np.minimum(1.,wind*(lo+hi)/2)
    else:candidate_wind=np.zeros(h)
    years=[int(only_year)] if only_year else p['years'];n=len(years);k=len(TECH)
    starts=[y-4 for y in years]
    nc=n*k;nv=nc+n*len(FIELDS)*h
    cap=lambda s,t:s*k+TECH.index(t)
    ix=lambda s,var:nc+(s*len(FIELDS)+FIELDS.index(var))*h+np.arange(h,dtype=np.int32)
    bounds=np.zeros((nv,2));bounds[:,1]=np.inf
    objective=np.zeros(nv)
    ae=SparseRows();au=SparseRows();month=f.hour_start_brt.dt.month.to_numpy()
    rate=p['discount_rate'];eta=np.sqrt(p['bess_roundtrip_efficiency'])
    annual_weight={y:sum((1+rate)**(-(z-2025)) for z in range(y-4,y+1)) for y in years}
    cf={'solar_central':f.new_central_pv_cf.to_numpy(),'solar_distributed':f.new_dg_pv_cf.to_numpy(),
        'bio':f.new_bio_cf.to_numpy(),'small_hydro':f.new_small_hydro_cf.to_numpy(),'wind':wind,'wind_site':candidate_wind}
    if hydro_stress:
        # The same disclosed stress analogue also reduces new small-hydro supply.
        cf['small_hydro']=cf['small_hydro']*np.array([p['hydro_stress_month_ratio'][str(int(m))] for m in month])
    carbon_price=float(cfg.get('carbon_price_brl_tco2',0.))
    gas_factor=float(cfg.get('gas_tco2_per_mwh',0.40))
    carbon_cap=cfg.get('gas_emissions_cap_tonnes',None)
    fiscal_rate=float(cfg.get('eligible_purchase_tax_relief',0.0))
    fiscal_share=float(cfg.get('eligible_purchase_share_of_capex',0.0))
    assert 0 <= fiscal_rate <= 0.10 and 0 <= fiscal_share <= 1
    fiscal_technologies=set(cfg.get('fiscal_eligible_technologies',[]))
    wind_site_cap=float(cfg.get('wind_site_build_cap_gw',0.))
    biogas_cap=float(cfg.get('biogas_build_cap_gw',0.))
    build_limits=cfg.get('block_new_capacity_caps_gw',{})
    biogas_cf=float(cfg.get('biogas_availability_factor',0.70))
    biogas_fuel_cost=float(cfg.get('biogas_fuel_cost_brl_mwh',250.))
    bess_energy_cost=float(cfg.get('additional_bess_energy_cost_brl_kw_h',450.))
    fiscal_benefit_terms=[]
    allmeta=[];fixed=None
    if fixed_case:
        fixed=pd.read_csv(ROOT/'output'/fixed_case/'portfolio.csv').set_index(['year','technology'])
        if len(years)!=1:raise ValueError('Fixed portfolio stress requires --year')
        allow_shed=True
    def active(s,t):
        if fixed is not None:return [0]
        return [v for v in range(s+1) if starts[s]-starts[v]<p['technology'][t]['life_years']]
    for s,y in enumerate(years):
        if demand=='low':load_factor=(1+p['low_demand_growth'])**(y-2025)
        else:
            nat={int(k):v for k,v in p['national_reference_average_load_gw'].items()}
            load_factor=nat[min(y,2035)]/nat[2025]*(1+p['high_demand_growth_after_2035'])**max(0,y-2035)
        load=f.load_mw.to_numpy()/1000*load_factor
        retirement=max(0,(2050-y)/25) if (phaseout or netzero_mode!='none') else 1.
        existing={'central_pv':f.existing_solar_central_mw.to_numpy()/1000,
                  'mmgd':f.existing_mmgd_mw.to_numpy()/1000,
                  'biomass_observed':f.existing_biomass_observed_mw.to_numpy()/1000,
                  'mixed_thermal':f.existing_mixed_thermal_mw.to_numpy()/1000*retirement}
        vrebase=sum(existing.values())
        for t in TECH:
            j=cap(s,t);tp=p['technology'][t]
            if fixed is not None:
                value=float(fixed.loc[(y,t),'new_surviving_capacity_gw'])
                bounds[j]=[value,value]
            else:
                life=tp['life_years'];used=min(life,2051-starts[s]);remaining=max(0,life-used)/life
                base_capex=tp['capex_brl_kw']+(p['bess_duration_hours']-4)*bess_energy_cost if t=='bess' else tp['capex_brl_kw']
                # Scenario proxy: PV of technology-level mean connection costs, from the solved nodal MILP.
                # This captures marginal connection expense but not discrete circuit costs or DC congestion feedback.
                base_capex += float(cfg.get('interconnection_addon_brl_kw_by_tech',{}).get(t,0.))
                capital_before_tax=base_capex*(1+tp['construction_interest_fraction'])*cost_scale
                eligible_discount=fiscal_rate*fiscal_share if t in fiscal_technologies else 0.
                capital=capital_before_tax*(1-eligible_discount)
                # 1 GW * R$/kW = 1 million R$: no hidden 1000 multiplier.
                pv_capital=capital/(1+rate)**(starts[s]-2026)
                om=sum(tp['fixed_cost_brl_kw_year']/(1+rate)**(z-2025)
                       for z in range(starts[s],min(2051,int(starts[s]+life))))
                salvage=capital*remaining/(1+rate)**25
                objective[j]=pv_capital+om-salvage
                discount_pv=capital_before_tax*eligible_discount* (
                    1/(1+rate)**(starts[s]-2026)-remaining/(1+rate)**25)
                fiscal_benefit_terms.append((j,discount_pv))
            if t=='wind' and not windpath.exists():bounds[j]=[0,0]
            if t=='wind_site':bounds[j,1]=max(0.,wind_site_cap)
            if t=='biogas':bounds[j,1]=max(0.,biogas_cap)
            if t in build_limits:bounds[j,1]=min(bounds[j,1],float(build_limits[t]))
        for t in ['solar_central','solar_distributed','bio','small_hydro']:
            target=p['policy_2030_gw'][t]+(p['policy_2050_gw'][t]-p['policy_2030_gw'][t])*(y-2030)/20
            if policy=='addition':target+=p['policy_addition_baseline_2022_gw'][t]
            if fixed is None:
                au.add([(cap(v,t),-1.) for v in active(s,t)],p['base_capacity_gw'][t]-target)
        if resource_envelope and fixed is None:
            # A conservative candidate-list ceiling based on the policy horizon.
            # This is a RESEARCH restriction, not an estimate of technical potential.
            # It prevents unlimited unverified biomass fuel and new hydro sites.
            for t in ['bio','small_hydro']:
                # The 2035 biomass target is a policy floor; no independent fuel-supply ceiling is available.
                if y == 2035 and t == 'bio':
                    continue
                ceiling=max(p['policy_2030_gw'][t],p['policy_2050_gw'][t])
                if policy=='addition':ceiling+=p['policy_addition_baseline_2022_gw'][t]
                au.add([(cap(v,t),1.) for v in active(s,t)],ceiling-p['base_capacity_gw'][t])
        # Supply/load identity uses gross load and includes existing MMGD exactly once.
        terms=[(ix(s,t),1.) for t in ['hydro','vre_used','gas','imports','discharge','shed','biogas_dispatch']]
        terms.append((ix(s,'charge'),-1.));ae.add(terms,load)
        # Sequential 8760-hour cyclic state, no daily SOC reset.
        soc=ix(s,'soc');prev=np.roll(soc,1)
        ae.add([(soc,1.),(prev,-1.),(ix(s,'charge'),-eta),(ix(s,'discharge'),1/eta)],np.zeros(h))
        # Non-reservoir availability, including observed baseline delivery and additions.
        vterms=[(ix(s,'vre_used'),1.)]
        for t in cf:
            for v in active(s,t):vterms.append((cap(v,t),-cf[t]))
        au.add(vterms,vrebase)
        # Biogas is dispatchable and fuel-priced, unlike the zero-VOM nondispatchable pool.
        au.add([(ix(s,'biogas_dispatch'),1.)]+[(cap(v,'biogas'),-biogas_cf) for v in active(s,'biogas')],np.zeros(h))
        # A single bidirectional power converter and a four-hour usable energy bound.
        au.add([(ix(s,'charge'),1.),(ix(s,'discharge'),1.)]+[(cap(v,'bess'),-1.) for v in active(s,'bess')],np.zeros(h))
        au.add([(soc,1.)]+[(cap(v,'bess'),-p['bess_duration_hours']) for v in active(s,'bess')],np.zeros(h))
        ratio=np.array([p['hydro_stress_month_ratio'][str(int(m))] for m in month]) if hydro_stress else np.ones(h)
        bounds[ix(s,'hydro'),1]=f.hydro_hourly_upper_mw.to_numpy()/1000
        # Dispatch analogue reduces energy, not turbine nameplate. No invented inflows.
        for m in range(1,13):
            inds=np.flatnonzero(month==m);energy=float((f.observed_hydro_mw.to_numpy()[inds]/1000*ratio[inds]).sum())
            # A scalar monthly row assembled directly from individual hour columns.
            row=au.nr;au.r.append(np.full(len(inds),row,dtype=np.int32));au.c.append(ix(s,'hydro')[inds]);au.v.append(np.ones(len(inds)));au.b.append(np.array([energy]));au.nr+=1
        # Empirical aggregate maximum ramp, incl. cyclic year boundary, explicit envelope.
        hy=ix(s,'hydro');hp=np.roll(hy,1);ramp=p['hydro_hourly_ramp_gw']
        au.add([(hy,1.),(hp,-1.)],np.full(h,ramp));au.add([(hy,-1.),(hp,1.)],np.full(h,ramp))
        if carbon_cap is not None:
            # Each hourly GW corresponds to 1000 MWh; direct combustion only.
            inds=ix(s,'gas');row=au.nr
            au.r.append(np.full(h,row,dtype=np.int32));au.c.append(inds)
            au.v.append(np.full(h,gas_factor*1000.));au.b.append(np.array([float(carbon_cap)]));au.nr+=1
        bounds[ix(s,'gas'),1]=(p['gas_operating_capacity_gw'] if (fixed is not None and fossil_emergency_reserve) else 0.) if netzero_mode!='none' else p['gas_operating_capacity_gw']*retirement
        bounds[ix(s,'imports'),1]=(0. if (netzero_mode=='local_zero_2050' and y==2050) else p['import_bound_gw']*import_scale)
        bounds[ix(s,'shed'),1]=load if allow_shed else 0.
        if fixed is None:
            w=annual_weight[y]
            objective[ix(s,'imports')]=w*p['import_price_brl_mwh']*price_scale/1000
            objective[ix(s,'gas')]=w*(p['gas_dispatch_cost_brl_mwh']+carbon_price*gas_factor)/1000
            objective[ix(s,'biogas_dispatch')]=w*biogas_fuel_cost/1000
            # Declared numerical throughput tie-break, not a degradation-cost estimate.
            objective[ix(s,'charge')]=w*.001
            objective[ix(s,'discharge')]=w*.001
            objective[ix(s,'shed')]=w*1000. # only used in an explicitly soft-feasibility run
        else:
            # Exact conditional ENS minimization, with no invented event probability.
            objective[ix(s,'shed')]=1.
            # Minimize ENS first; positive gas proxy avoids needless fossil in ties.
            objective[ix(s,'gas')]=0.00001 if fossil_emergency_reserve else 0.0
        allmeta.append({'year':y,'load':load,'load_factor':load_factor,'existing':existing,'vrebase':vrebase,'hydro_ratio':ratio,'retirement':retirement})
    Aeq,beq=ae.matrix(nv);Aub,bub=au.matrix(nv)
    out=ROOT/'output'/name;out.mkdir(parents=True,exist_ok=True)
    (out/'parameters_used.json').write_text(json.dumps(p,indent=2,ensure_ascii=False))
    print(f'{name}: {nv:,} variables, {Aeq.shape[0]:,} equalities, {Aub.shape[0]:,} inequalities, {(Aeq.nnz+Aub.nnz):,} nonzeros',flush=True)
    native_info={}
    if solver=='hipo':
        import highspy
        hs=highspy.Highs()
        hs.setOptionValue('threads',2)
        status=hs.setOptionValue('solver','hipo')
        if status!=highspy.HighsStatus.kOk:raise RuntimeError('HiPO unavailable; install highspy[extras]==1.15.1 or use --solver highs-ds')
        hs.setOptionValue('run_crossover','off')
        hs.setOptionValue('time_limit',1200.)
        hs.setOptionValue('primal_feasibility_tolerance',1e-7)
        hs.setOptionValue('dual_feasibility_tolerance',1e-7)
        hs.setOptionValue('ipm_optimality_tolerance',1e-8)
        amat=vstack([Aeq,Aub],format='csc')
        lp=highspy.HighsLp();lp.num_col_=nv;lp.num_row_=amat.shape[0]
        lp.col_cost_=objective;lp.col_lower_=bounds[:,0];lp.col_upper_=bounds[:,1]
        lp.row_lower_=np.r_[beq,np.full(len(bub),-np.inf)];lp.row_upper_=np.r_[beq,bub]
        lp.a_matrix_.format_=highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_=amat.indptr.astype(np.int32);lp.a_matrix_.index_=amat.indices.astype(np.int32);lp.a_matrix_.value_=amat.data
        hs.passModel(lp);hs.run();info=hs.getInfo();status=hs.getModelStatus()
        solution=hs.getSolution();ok=status==highspy.HighsModelStatus.kOptimal
        result=SimpleNamespace(success=ok,status=0 if ok else int(status),message=hs.modelStatusToString(status),
           fun=info.objective_function_value,x=np.array(solution.col_value),nit=info.ipm_iteration_count)
        for key in ['max_primal_infeasibility','max_dual_infeasibility','primal_dual_objective_error','num_primal_infeasibilities','num_dual_infeasibilities']:
            if hasattr(info,key):native_info[key]=getattr(info,key)
        native_info['highs_version']=hs.version();native_info['crossover']='off; validated interior solution'
    else:
        result=linprog(objective,A_ub=Aub,b_ub=bub,A_eq=Aeq,b_eq=beq,bounds=bounds,method=solver,
                       options={'presolve':True,'disp':True,'time_limit':1200.,'dual_feasibility_tolerance':1e-7,'primal_feasibility_tolerance':1e-7,'ipm_optimality_tolerance':1e-8})
    record={'case':name,'success':bool(result.success),'status':int(result.status),'message':str(result.message),
       'objective':None if result.fun is None else float(result.fun),'objective_unit':'GWh of conditional unserved load' if fixed is not None else 'million constant-2024 BRL, end-2025 PV',
       'policy':policy,'demand':demand,'hydro_stress':hydro_stress,'import_scale':import_scale,'phaseout':phaseout,
       'fixed_case':fixed_case,'years':years,'price_scale':price_scale,'capex_scale':cost_scale,
       'netzero_mode':netzero_mode,'fossil_emergency_reserve':fossil_emergency_reserve,
       'scenario_config':config_name,'scenario_assumptions':cfg,
       'direct_gas_carbon_price_brl_t':carbon_price,'direct_gas_emission_factor_assumed_t_mwh':gas_factor,
       'fiscal_reidi_rate_assumed':fiscal_rate,'fiscal_eligible_purchase_share_assumed':fiscal_share,
       'resource_envelope':resource_envelope,'resource_envelope_status':'research ceiling for total bio/small-hydro equal to interpreted2050policy; not proven physical resource potential'
           + ('; excludes biomass in 2035' if resource_envelope and 2035 in years else ''),
       'code_sha256_at_start':code_sha,
       'hourly_input_sha256':hashlib.sha256((ROOT/'macro/processed/macro_hourly_inputs.csv').read_bytes()).hexdigest(),
       'variables':nv,'equality_rows':Aeq.shape[0],'inequality_rows':Aub.shape[0],
       'solver':solver,'scipy_version':scipy.__version__,'python_version':platform.python_version(),
       'iterations':int(getattr(result,'nit',0)),'elapsed_seconds':time.time()-started,
       'native_solver_certificate':native_info,
       'wind_profile_included':windpath.exists(),
       'new_small_hydro_stress_applied':bool(hydro_stress),
       'economic_boundary':'incremental assets, existing gas dispatch proxy, border reference purchases; excludes existing fleet renewal, network projects, retail transfers and unpriced externalities',
       'milestone_representation':'end-of-block demand and policy floor apply to every year in its preceding five-year block; new investment at block start',
       'no_engineering_security_certificate':True}
    if not result.success:
        (out/'solver.json').write_text(json.dumps(record,indent=2,ensure_ascii=False));print(json.dumps(record,ensure_ascii=False));return record
    x=result.x
    record.update({'max_balance_residual_gw':float(np.max(np.abs(Aeq@x-beq))),
      'max_inequality_violation':float(max(0,np.max(Aub@x-bub))),
      'max_bound_violation':float(max(0,np.max(bounds[:,0]-x),np.max(x-bounds[:,1]))),
      'max_simultaneous_charge_discharge_gw':float(max(np.minimum(x[ix(s,'charge')],x[ix(s,'discharge')]).max() for s in range(n)))})
    portfolios=[];annual=[];snaps=[];costcomponents=[]
    for s,meta in enumerate(allmeta):
        y=meta['year'];df=pd.DataFrame({'hour_start_brt':f.hour_start_brt,'load_gw':meta['load']})
        for var in FIELDS:df[var+'_gw' if var!='soc' else 'soc_gwh']=x[ix(s,var)]
        avail=meta['vrebase'].copy();parts={k:v.copy() for k,v in meta['existing'].items()}
        for t in TECH:
            addition=0. if fixed is not None else float(x[cap(s,t)])
            survive=sum(float(x[cap(v,t)]) for v in active(s,t))
            total=survive+p['base_capacity_gw'][t]
            tp=p['technology'][t]
            portfolios.append({'year':y,'commission_block_start':starts[s],'technology':t,'new_build_gw':addition,
              'new_surviving_capacity_gw':survive,'existing_policy_capacity_gw':p['base_capacity_gw'][t],
              'total_policy_capacity_gw':total,'bess_energy_gwh':total*p['bess_duration_hours'] if t=='bess' else 0.,
              'raw_capex_million_brl':addition*tp['capex_brl_kw']*cost_scale,'investment_objective_million_brl':float(x[cap(s,t)]*objective[cap(s,t)]),
              'fixed_portfolio_recourse':fixed is not None})
            if t in cf:parts['new_'+t]=survive*cf[t];avail+=survive*cf[t]
        df['available_nondispatchable_gw']=avail
        df['biogas_mwh']=df['biogas_dispatch_gw']*1000.
        df['priced_gas_direct_tco2']=df['gas_gw']*1000.*gas_factor
        df['curtailed_nondispatchable_gw']=np.maximum(0,avail-df.vre_used_gw)
        frac=np.divide(df.vre_used_gw.to_numpy(),avail,out=np.zeros(h),where=avail>0)
        # All zero-marginal-cost availability can be split proportionally after LP:
        # this is a feasible reporting allocation, not observed plant dispatch.
        for t,v in parts.items():df['used_'+t+'_gw']=v*frac
        # Build each compressed artifact completely before a single atomic write.
        # Read-back is an integrity gate; a success marker cannot precede it.
        csvbytes=df.to_csv(index=False,float_format='%.9f').encode('utf-8')
        gzbytes=gzip.compress(csvbytes,mtime=0)
        dest=out/f'dispatch_{y}.csv.gz';temp=out/f'.dispatch_{y}.csv.gz.tmp'
        with open(temp,'wb') as stream:stream.write(gzbytes);stream.flush();os.fsync(stream.fileno())
        os.replace(temp,dest)
        assert gzip.decompress(dest.read_bytes())==csvbytes, 'Dispatch artifact failed byte integrity check'
        rec={'year':y,'load_twh':float(df.load_gw.sum()/1000),'peak_load_gw':float(df.load_gw.max()),
          'hydro_twh':float(df.hydro_gw.sum()/1000),'gas_twh':float(df.gas_gw.sum()/1000),
          'biogas_twh':float(df.biogas_dispatch_gw.sum()/1000),
          'priced_gas_direct_tco2':float(df.priced_gas_direct_tco2.sum()),
          'imports_twh':float(df.imports_gw.sum()/1000),'max_import_gw':float(df.imports_gw.max()),
          'nonreservoir_used_twh':float(df.vre_used_gw.sum()/1000),'curtailment_twh':float(df.curtailed_nondispatchable_gw.sum()/1000),
          'bess_charge_twh':float(df.charge_gw.sum()/1000),'bess_discharge_twh':float(df.discharge_gw.sum()/1000),
          'conditional_ens_mwh':float(df.shed_gw.sum()*1000),'hours_with_shed':int((df.shed_gw>1e-7).sum()),
          'max_shed_mw':float(df.shed_gw.max()*1000),'load_factor_vs_2025':meta['load_factor'],
          'import_cost_million_brl_year':float(df.imports_gw.sum()*p['import_price_brl_mwh']*price_scale/1000),
          'gas_cost_million_brl_year':float(df.gas_gw.sum()*p['gas_dispatch_cost_brl_mwh']/1000)}
        for t in parts:rec['used_'+t+'_twh']=float(df['used_'+t+'_gw'].sum()/1000)
        annual.append(rec)
        criteria={'peak_load':df.load_gw.idxmax(),'peak_nondispatchable':df.available_nondispatchable_gw.idxmax(),
          'max_import':df.imports_gw.idxmax(),'min_residual_load':(df.load_gw-df.available_nondispatchable_gw).idxmin(),
          'max_bess_discharge':df.discharge_gw.idxmax()}
        for label,i in criteria.items():
            d=df.iloc[i].to_dict();d['year']=y;d['criterion']=label;snaps.append(d)
    pd.DataFrame(portfolios).to_csv(out/'portfolio.csv',index=False)
    pd.DataFrame(annual).to_csv(out/'annual_energy.csv',index=False)
    pd.DataFrame(snaps).to_csv(out/'critical_snapshots.csv',index=False)
    record['estimated_fiscal_tax_transfer_pv_million_brl']=float(sum(x[j]*v for j,v in fiscal_benefit_terms))
    record['priced_gas_carbon_payment_pv_million_brl']=float(sum(
       annual_weight[meta['year']]*carbon_price*gas_factor/1000 * x[ix(s,'gas')].sum() for s,meta in enumerate(allmeta)))
    record['system_resource_cost_proxy_pv_million_brl']=float(result.fun+record['estimated_fiscal_tax_transfer_pv_million_brl']-record['priced_gas_carbon_payment_pv_million_brl'])
    record['carbon_scope_note']='Only explicit gas dispatch priced; historical mixed thermal in pooled nondispatchable supply has unverified fuel and CO2. Imports and biomass lifecycle emissions are unpriced.'
    record['wind_site_proxy_note']='Scenario annual mean CF '+str(round(candidate_wind.mean(),4))+' from scaled observed regional wind shape; candidate sites not identified.'
    record['capital_and_fixed_cost_pv_million_brl']=float(objective[:nc]@x[:nc]) if fixed is None else None
    record['operating_cost_pv_million_brl']=float(objective[nc:]@x[nc:]) if fixed is None else None
    (out/'solver.json').write_text(json.dumps(record,indent=2,ensure_ascii=False))
    binary=io.BytesIO();np.savez_compressed(binary,investments=x[:nc],full_solution=x,objective=np.array([result.fun]))
    (out/'solution_compact.npz').write_bytes(binary.getvalue())
    print(json.dumps(record,ensure_ascii=False,indent=2),flush=True)
    print(pd.DataFrame(annual)[['year','load_twh','imports_twh','curtailment_twh','conditional_ens_mwh']].to_string(index=False),flush=True)
    return record

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True)
    ap.add_argument('--policy',choices=['terminal','addition'],default='terminal');ap.add_argument('--demand',choices=['low','high'],default='low')
    ap.add_argument('--hydro-stress',action='store_true');ap.add_argument('--import-scale',type=float,default=1.)
    ap.add_argument('--phaseout',action='store_true');ap.add_argument('--fixed-case');ap.add_argument('--year',type=int)
    ap.add_argument('--allow-shed',action='store_true');ap.add_argument('--price-scale',type=float,default=1.);ap.add_argument('--cost-scale',type=float,default=1.)
    ap.add_argument('--solver',default='highs-ds',choices=['hipo','highs','highs-ipm','highs-ds'])
    ap.add_argument('--no-resource-envelope',action='store_true')
    ap.add_argument('--netzero-mode',default='none',choices=['none','local_zero_2050','clean_import'])
    ap.add_argument('--fossil-emergency-reserve',action='store_true')
    ap.add_argument('--scenario-config',default=None,help='JSON policy / technology scenario; all experimental values explicit')
    a=ap.parse_args();solve_case(a.name,a.policy,a.demand,a.hydro_stress,a.import_scale,a.phaseout,a.fixed_case,a.year,a.allow_shed,a.price_scale,a.cost_scale,a.solver,not a.no_resource_envelope,a.netzero_mode,a.fossil_emergency_reserve,a.scenario_config)
