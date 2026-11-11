#!/usr/bin/env python3
"""Independent output accounting and figure production for MG location-aware screen.
Does not assert engineering or carbon-certification validity.
"""
from pathlib import Path
import json,glob,hashlib,math
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
from pyproj import CRS,Transformer

ROOT=Path(__file__).resolve().parents[1];I=ROOT/'inputs';O=ROOT/'outputs';F=ROOT/'figures';F.mkdir(exist_ok=True)
N=ROOT/'macro_reopt/output'
P=pd.read_csv(I/'portfolio_2050.csv')
COMB='ROBUST_distributed_MG_load_border_585_PV35to90_pen1p0'
R={}; checks=[]
for yr in [2035,2050]:
 prefix=f'{yr}_{COMB}'
 s=json.loads((O/(prefix+'_summary.json')).read_text());R[yr]=s
 sel=pd.read_csv(O/(prefix+'_selected.csv'))
 cases=pd.read_csv(O/(prefix+'_cases.csv'))
 port=pd.read_csv(I/f'portfolio_{yr}.csv')
 for rr in port.itertuples(index=False):
  tech=str(rr.technology)
  if tech=='solar_distributed' or float(rr.new_build_gw)<1e-8:continue
  actual=sel.loc[sel.tech==tech,'new_capacity_mw'].sum()
  checks.append({'item':f'{yr} {tech} allocation','error':abs(actual-float(rr.new_build_gw)*1000),'tol':1e-4})
 assert set(sel.bus_id.astype(int)).issubset(set(pd.read_csv(I/'pwf_2035_buses.csv').id.astype(int)))
 expected_cost=sel.new_capacity_mw*sel.variable_cost_mbrl_mw+sel.circuits*sel.per_circuit_mbrl
 checks += [
  {'item':f'{yr} cost by arc','error':float(abs(sel.connection_cost_mbrl-expected_cost).max()),'tol':1e-5},
  {'item':f'{yr} per-arc circuits','error':float(max(0,(sel.new_capacity_mw-sel.circuits*sel.max_circuit_mw).max())),'tol':1e-5},
  {'item':f'{yr} robust post-uplift peak','error':float(max(0,cases.max_loading_pct_after_uprate.max()-100)),'tol':1e-5},
  {'item':f'{yr} MILP gap','error':float(s['mip_gap']),'tol':.0001},
  {'item':f'{yr} solver constraints','error':float(s['max_constraint_violation']),'tol':1e-5},
  {'item':f'{yr} technology capacity residual','error':float(s['capacity_error_mw']),'tol':1e-4},
 ]
 conn=float(sel.connection_cost_mbrl.sum()+s['dg_cost_mbrl'])
 checks.append({'item':f'{yr} connection total','error':abs(conn-s['optimized_connection_nominal_mbrl']),'tol':1e-4})
 disp=pd.read_csv(N/f'REFRESH_{yr}'/f'dispatch_{yr}.csv.gz')
 def get(col):return disp[col].to_numpy(float)
 # Validate the actual balance equation; unused load shedding must be zero.
 residual=(get('hydro_gw')+get('vre_used_gw')+get('gas_gw')+get('imports_gw')+
     get('discharge_gw')+get('shed_gw')+get('biogas_dispatch_gw')-get('load_gw')-get('charge_gw'))
 checks.append({'item':f'{yr} hourly supply balance','error':float(np.max(np.abs(residual))),'tol':1e-6})
 checks.append({'item':f'{yr} normal shed GWh','error':float(disp.shed_gw.abs().sum()),'tol':1e-5})
 if yr==2050:
  checks += [{'item':'2050 zero gas/foreign imports','error':float((disp.gas_gw.abs()+disp.imports_gw.abs()).sum()),'tol':1e-6}]
 res=json.loads((N/f'REFRESH_{yr}'/'solver.json').read_text())
 assert res['success'] and res['status']==0
 checks.append({'item':f'{yr} solver balance','error':float(res['max_balance_residual_gw']),'tol':1e-6})

A=pd.DataFrame(checks);A['passed']=A.error<=A.tol
A.to_csv(O/'INDEPENDENT_AUDIT.csv',index=False)
if not A.passed.all():
 print(A[~A.passed].to_string(index=False));raise RuntimeError('AUDIT FAILED')
(O/'audit_summary.json').write_text(json.dumps({'status':'PASS','checked_conditions':len(A),'max_relative_tolerance_fraction':float((A.error/(A.tol+1e-50)).max()),'note':'Checks numerical feasibility and arithmetic only. AC, project permits and actual transmission construction are NOT certified.'},indent=2))

plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none','axes.labelcolor':'#26364A','text.color':'#26364A'})
colors=['#3C5488','#00A087'];xpos=np.arange(2); width=.34
fig,ax=plt.subplots(figsize=(10.8,5.3),dpi=160)
for k,(tag,color) in enumerate([('Baseline nearest PWF-linked', '#A5AFB9'),('Optimized multi-station','#3C5488')]):
 values=[R[y]['reference_connection_nominal_mbrl'] if k==0 else R[y]['optimized_connection_nominal_mbrl'] for y in [2035,2050]]
 bars=ax.bar(xpos+(k-.5)*width,np.array(values)/1000,width,label=tag,color=color,edgecolor='white')
 ax.bar_label(bars,fmt='%.3f',padding=4,fontsize=11,fontweight='bold')
ax.set_xticks(xpos,['2035 policy pathway','2050 diversified fossil-free'])
ax.set_ylabel('Connection investment proxy (billion BRL, undiscounted)')
ax.set_ylim(0,6.5);ax.set_axisbelow(True);ax.grid(axis='y',alpha=.15)
ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False,fontsize=10)
ax.set_title('Location-aware siting reduces indicative connection cost',loc='left',fontsize=15,fontweight='bold',pad=16)
fig.text(.08,.035,'Fixed technology quantities per horizon; EPE/ONS PWF-linked stations; study-only location and engineering-cost assumptions.',fontsize=9,color='#657587')
fig.subplots_adjust(left=.08,right=.97,bottom=.15,top=.86)
for extension in ['png','pdf','svg']:fig.savefig(F/f'01_connection_cost.{extension}',dpi=300,facecolor='white',bbox_inches='tight')
plt.close(fig)

fig,ax=plt.subplots(figsize=(10.8,5.5),dpi=160)
sc=pd.read_csv(O/(f'2050_{COMB}_cases.csv'))
labels=[]; vals=[]; groups=[]
for row in sc.itertuples():
 labels.append(f"{int(row.pv_factor*100)}% PV\n"+('Local sink' if row.mode=='distributed_MG_load' else 'Border sink'))
 vals.append(row.optimized_excess_mw/1000);groups.append('#00A087' if row.mode=='distributed_MG_load' else '#E64B35')
bars=ax.bar(np.arange(len(vals)),vals,color=groups,width=.66)
ax.set_xticks(np.arange(len(labels)),labels,fontsize=9)
ax.set_ylabel('Post-siting thermal exceedance without uprating (GW, summed)')
ax.set_title('2050 stress overlays still exceed monitored line ratings',loc='left',fontsize=14,weight='bold',pad=15)
ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
for bar,v in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,v+.12,f'{v:.2f}',ha='center',fontsize=9)
ax.set_ylim(0,max(vals)*1.18)
fig.text(.08,.025,'Thermal uplift required is a nonphysical rating proxy, NOT a priced transmission-line construction project.',fontsize=9,color='#657587')
fig.subplots_adjust(left=.08,right=.98,bottom=.2,top=.86)
for ext in ['png','pdf','svg']:fig.savefig(F/f'02_2050_stress.{ext}',dpi=300,bbox_inches='tight',facecolor='white')
plt.close(fig)

# Geographical illustration. Crosswalk points: verified names, NOT verified injection headroom.
try:
 import geopandas as gpd
 from matplotlib.lines import Line2D
 geo=gpd.read_file(I/'mg_boundary.geojson')
 cross=pd.read_csv(I/'epe_ons_station_crosswalk.csv')
 techcolors={'solar_central':'#3C5488','bio':'#00A087','small_hydro':'#849E4D','wind_site':'#4DBBD5','biogas':'#D9823B','bess':'#7B62A4'}
 coords={r.area:(float(r.longitude),float(r.latitude)) for r in pd.read_csv(I/'candidate_areas.csv').itertuples(index=False)}
 for yr in [2035,2050]:
  sel=pd.read_csv(O/(f'{yr}_{COMB}_selected.csv'))
  fig,ax=plt.subplots(figsize=(9.5,7),dpi=160); geo.boundary.plot(ax=ax,color='#8594A2',linewidth=.9)
  cr=cross[(cross.pwf_state=='MG')&cross.lat.notna()&cross.lon.notna()]
  ax.scatter(cr.lon,cr.lat,s=4,c='#A8B2BB',alpha=.55,zorder=1)
  for r in sel.itertuples(index=False):
   color=techcolors.get(r.tech,'#333')
   if r.tech=='bess':
    # Deliberate co-location at station, no assumed long new dedicated line.
    ax.scatter(r.lon,r.lat,marker='s',s=20+float(r.new_capacity_mw)/9,c=color,alpha=.8,edgecolors='white',linewidth=.6,zorder=4)
   else:
    lon,lat=coords[r.area]
    ax.plot([lon,r.lon],[lat,r.lat],color=color,lw=.85,alpha=.55,zorder=2)
    ax.scatter([lon],[lat],s=20+max(0,r.new_capacity_mw)/13,c=color,edgecolors='white',linewidth=.5,zorder=3)
  ax.set_xlim(-51.8,-39.5);ax.set_ylim(-24,-13.7);ax.set_aspect('equal',adjustable='box')
  ax.set_title(f'{yr} optimized candidate clusters & grid connections',fontsize=14,weight='bold')
  ax.set_xlabel('Longitude (° W)');ax.set_ylabel('Latitude (° S)')
  handles=[Line2D([0],[0],marker='o',color='w',markerfacecolor=c,label=k.replace('_',' ').title(),markersize=8) for k,c in techcolors.items() if k in set(sel.tech)]
  ax.legend(handles=handles,loc='lower left',fontsize=8,frameon=False,ncol=2)
  fig.text(.07,.03,'Areas are municipality proxies; station coordinates are ONS snapshots; straight segments are NOT surveyed routes.',fontsize=8,color='#657587')
  fig.subplots_adjust(top=.92,bottom=.14)
  for ext in ['png','pdf','svg']:fig.savefig(F/f'03_{yr}_site_map.{ext}',dpi=300,facecolor='white',bbox_inches='tight')
  plt.close(fig)
except Exception as e:
 print('Map warning:',e)
print('PASS',len(A),'checks, figures in',F)
