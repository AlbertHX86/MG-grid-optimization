from pathlib import Path
import pandas as pd,numpy as np,matplotlib.pyplot as plt
from matplotlib.lines import Line2D
R=Path(__file__).parent;O=R/'results';D=R/'figures';D.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'pdf.fonttype':42,'svg.fonttype':'none'})
for yr in (2035,2050):
 x=pd.read_csv(O/f'{yr}_scenario_comparison.csv')
 x['scenario']=x.balance.map({'distributed_MG_load':'MG local','border_585':'Border'})+' / '+(x.pv_level*100).round().astype(int).astype(str)+'% PV'
 # grouped before/after six scenario bars
 x0=x[x.method=='existing_grid'].copy();x1=x[x.method=='parallel_line_screen'].copy()
 assert list(x0.scenario)==list(x1.scenario)
 fig,ax=plt.subplots(figsize=(11.8,5.5));ind=np.arange(6)
 ax.bar(ind-.2,x0.excess_sum_mw,width=.38,label='Existing 2035 reference network',color='#415B70')
 ax.bar(ind+.2,x1.excess_sum_mw,width=.38,label='With tested parallel circuits',color='#75A5A1')
 ax.set_xticks(ind,x0.scenario,rotation=20,ha='right');ax.set_ylabel('Sum of line-rating exceedance (MW)')
 ax.set_title(f'{yr} | Transmission congestion under six screening cases',loc='left',weight='bold')
 ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
 ax.legend(frameon=False,loc='upper left')
 fig.text(.07,.035,'283 selected high-voltage lines; MW thermal-rating proxy; no AC/N-1 certification.',fontsize=9,color='#5f6872')
 fig.tight_layout(rect=[0,.1,1,1])
 for ext in ['png','pdf','svg']:fig.savefig(D/f'01_{yr}_congestion_comparison.{ext}',dpi=220,bbox_inches='tight',facecolor='white')
 plt.close(fig)

# compact costs and residual breakdown
lines=pd.read_csv(O/'2050_parallel_line_candidates.csv');fig,ax=plt.subplots(figsize=(11,5.6))
labels=[f"{int(a)}–{int(b)}" for a,b in zip(lines.f,lines.t)]
ax.bar(np.arange(len(lines)),lines.assumed_new_circuit_cost_mbrl,color='#5B7486')
ax.set_xticks(range(len(lines)),labels,rotation=48,ha='right');ax.set_ylabel('Assumed nominal cost (million R$)')
ax.set_title('2050 | Indicative parallel-line screening cost',weight='bold',loc='left')
ax.text(.02,.94,'12 candidate added circuits · one greedy sequence · no line permits or route survey',transform=ax.transAxes,color='#627082',va='top',fontsize=10)
ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
fig.tight_layout()
for ext in ['png','pdf','svg']:fig.savefig(D/f'02_2050_candidate_circuit_costs.{ext}',dpi=220,bbox_inches='tight',facecolor='white')
plt.close(fig)

b=pd.read_csv(O/'2050_remaining_bottlenecks.csv')
p=pd.read_csv(O/'2050_corrective_dispatch_screen.csv')
fig,(a1,a2)=plt.subplots(1,2,figsize=(12,5))
x=np.array([35,70,90]);a1.bar(x-.08,b[b.balance=='distributed_MG_load'].groupby('pv_factor').excess_mw.sum().reindex([.35,.7,.9]).to_numpy(),width=.18,color='#9D6674')
a1.set_xticks(x,['35%','70%','90%']);a1.set_title('Residual line exceedance (MG-local balance)');a1.set_ylabel('Exceedance (MW)');a1.grid(axis='y',alpha=.15)
a2.plot(x,p[p.balance=='distributed_MG_load'].minimum_replacement_dispatch_mw.to_numpy()/1000,'o-',color='#3C6B85',lw=2)
a2.set_xticks(x,['35%','70%','90%']);a2.set_title('Conditional replacement dispatch required');a2.set_ylabel('Dispatch replaced (GW)');a2.grid(axis='y',alpha=.15)
for ax in (a1,a2):ax.spines[['top','right']].set_visible(False);ax.set_axisbelow(True)
fig.suptitle('2050 | Physical-circuit screening leaves local 230 kV constraints',fontsize=16,weight='bold',x=.02,y=1.04,ha='left')
fig.tight_layout()
for ext in ['png','pdf','svg']:fig.savefig(D/f'03_2050_residual_and_dispatch.{ext}',dpi=220,bbox_inches='tight',facecolor='white')
plt.close(fig)

# Simplified geospatial corridor plot
import sys
sys.path.insert(0,str(R));import step2_model as M
s=M.station; A=M.B; L=M.L
path=pd.read_csv(O/'2050_parallel_line_candidates.csv');
fig,ax=plt.subplots(figsize=(9,7));
# all verifiable 2035 existing lines with both station coords
seen=set()
for row in L.itertuples(index=False):
 f=int(row.f);t=int(row.t)
 if f not in s.index or t not in s.index:continue
 k=tuple(sorted([f,t]));
 if k in seen:continue
 seen.add(k)
 u,v=s.loc[f],s.loc[t];
 if abs(u.lon-v.lon)+abs(u.lat-v.lat)>7:continue
 ax.plot([u.lon,v.lon],[u.lat,v.lat],color='#C8D0D5',alpha=.48,lw=.5,zorder=1)
for row in path.itertuples(index=False):
 u,v=s.loc[row.f],s.loc[row.t]
 ax.plot([u.lon,v.lon],[u.lat,v.lat],color='#C35140' if row.assumed_route_km>100 else '#364C65',lw=2.3,alpha=.77,zorder=5)
ax.set_xlim(-51,-41);ax.set_ylim(-24,-12);ax.set_xlabel('Longitude');ax.set_ylabel('Latitude')
ax.set_title('2050 candidate parallel corridors on EPE 2035 network',loc='left',weight='bold')
ax.legend(handles=[Line2D([0],[0],color='#C8D0D5',lw=1,label='Georeferenced existing links'),Line2D([0],[0],color='#C35140',lw=2.3,label='Assumed new long-distance parallel circuit'),Line2D([0],[0],color='#364C65',lw=2.3,label='Assumed shorter parallel circuit')],loc='lower left',frameon=False,fontsize=9)
ax.spines[['top','right']].set_visible(False)
fig.text(.07,.02,'Straight electrical connections and candidate duplicates, not surveyed routes. Multiple circuits overlap geographically.',fontsize=9,color='dimgray')
fig.tight_layout(rect=[0,.04,1,1]);
for ext in ['png','pdf','svg']:fig.savefig(D/f'04_2050_candidate_corridor_map.{ext}',dpi=220,bbox_inches='tight',facecolor='white')
plt.close(fig)
print('Saved figures',sorted(str(q.name) for q in D.glob('*.png')))
# Comparison of policy-relevant local net load vs cross-border transfer sensitivity
loc=pd.read_csv(O/'local_only'/'2050_scenario_comparison.csv')
full=pd.read_csv(O/'2050_scenario_comparison.csv')
fig,(a,b)=plt.subplots(1,2,figsize=(12.2,5.1),gridspec_kw={'width_ratios':[1.45,1]})
xs=np.array([0,1,2]);local0=loc[loc.method=='existing_grid'].excess_sum_mw.to_numpy();local1=loc[loc.method=='parallel_line_screen'].excess_sum_mw.to_numpy()
border0=full[(full.method=='existing_grid')&(full.balance=='border_585')].excess_sum_mw.to_numpy();border1=full[(full.method=='parallel_line_screen')&(full.balance=='border_585')].excess_sum_mw.to_numpy()
a.bar(xs-.15,local0,.30,color='#415B70',label='Existing grid')
a.bar(xs+.15,local1,.30,color='#75A5A1',label='After 7 circuits')
# Border transfer assumptions are excluded from local-balancing exceedance bars.
a.set_xticks(xs,['35% PV','70% PV','90% PV']);a.set_ylabel('Local-balancing exceedance (MW)');a.set_title('MG-local congestion relief',weight='bold',loc='left')
a.legend(frameon=False,fontsize=9);a.grid(axis='y',alpha=.15);a.set_axisbelow(True)
b.bar(['Local balancing\n7 new circuits','Border + local\n12 new circuits'],[236.92578405032975,4424.920162697376],color=['#75A5A1','#B87664'])
b.set_ylabel('Illustrative corridor cost (million R$)');b.set_title('Indicative construction cost',weight='bold',loc='left')
for j,v in enumerate([236.92578405032975,4424.920162697376]):b.text(j,v+70,f'{v:,.0f} M',ha='center',fontsize=10)
b.set_ylim(0,5200);b.grid(axis='y',alpha=.15);b.set_axisbelow(True)
for q in (a,b):q.spines[['top','right']].set_visible(False)
fig.text(.04,.025,'Additional circuits are engineering proxies; the local-only case still has 172 MW residual exceedance.',fontsize=9,color='#6b747e')
fig.tight_layout(rect=[0,.06,1,1])
for ext in ['png','pdf','svg']:fig.savefig(D/f'05_local_vs_border_planning_assumptions.{ext}',dpi=240,bbox_inches='tight',facecolor='white')
plt.close(fig)
