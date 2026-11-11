#!/usr/bin/env python3
"""Independent file-level numerical/accounting audit of Step 2 screening outputs."""
import json,sys,hashlib
from pathlib import Path
import numpy as np,pandas as pd
R=Path(__file__).resolve().parent;O=R/'results';S=R/'MG_Nodal_Connection_Optimization';tests=[]
def check(label,actual,expected,tol=1e-6):
    success=bool(abs(float(actual)-float(expected))<=tol)
    tests.append({'name':label,'pass':success,'observed':float(actual),'target':float(expected),'tolerance':tol})
    return success
for y in [2035,2050]:
    d=json.load(open(O/f'{y}_summary.json'));cases=pd.read_csv(O/f'{y}_scenario_comparison.csv');names=cases.balance.drop_duplicates().tolist()
    check(f'{y} 6 cases for each method',len(cases),12)
    check(f'{y} network bus count',d['n_buses'],1489)
    check(f'{y} network branch count',d['original_branch_count'],1890)
    check(f'{y} selected thermal line count',d['monitored_existing_lines'],283)
    check(f'{y} incremental power-flow residual',d['nodal_equation_max_residual_MW'],0.,1e-6)
    check(f'{y} largest post-line overload from cases',cases[cases.method=='parallel_line_screen'].excess_sum_mw.max(),d['remaining_worst_case_sum_of_overloads_mw'],1e-6)
    # independent starting-flow reference check against previous solved nodal robust archive
    orig=pd.read_csv(S/'outputs'/f'{y}_ROBUST_distributed_MG_load_border_585_PV35to90_pen1p0_cases.csv')
    pre=cases[cases.method=='existing_grid'].sort_values(['balance','pv_level'])
    orig=orig.sort_values(['mode','pv_factor'])
    check(f'{y} reproduces old six-case PTDF flow screening',np.max(abs(pre.excess_sum_mw.to_numpy()-orig.optimized_excess_mw.to_numpy())),0.,1e-5)
    conn=pd.read_csv(O/f'{y}_parallel_line_candidates.csv')
    check(f'{y} actual number of candidate circuits',len(conn),d['greedy_added_parallel_circuits'])
    check(f'{y} total nominal circuit cost recomputed',conn.assumed_new_circuit_cost_mbrl.sum(),d['cost_assumed_million_brl'],1e-6)
    if len(conn):
        check(f'{y} candidate phases finite',(np.isfinite(conn.added_line_X_pu)).all(),1,0)
        check(f'{y} all proposed circuits positive cost',int((conn.assumed_new_circuit_cost_mbrl>0).all()),1,0)
    rd=pd.read_csv(O/f'{y}_corrective_dispatch_screen.csv')
    check(f'{y} dispatch test count',len(rd),6)
    if rd.feasible.all():check(f'{y} conditional redispatch test residual',rd.max_rating_violation_after_mw.max(),0.,1e-4)
check('2035 no selected-line overload',json.load(open(O/'2035_summary.json'))['initial_worst_case_sum_of_overloads_mw'],0.,1e-5)
local=json.load(open(O/'local_only'/'2050_summary.json'));allc=json.load(open(O/'2050_summary.json'))
check('2050 local-only 3 scenarios',len(pd.read_csv(O/'local_only'/'2050_scenario_comparison.csv')),6)
check('2050 local-only computed line count',len(pd.read_csv(O/'local_only'/'2050_parallel_line_candidates.csv')),local['greedy_added_parallel_circuits'])
check('2050 local-only line costs',sum(pd.read_csv(O/'local_only'/'2050_parallel_line_candidates.csv').assumed_new_circuit_cost_mbrl),local['cost_assumed_million_brl'])
check('2050 all-case line costs positive',int(allc['cost_assumed_million_brl']>local['cost_assumed_million_brl']),1,0)
# input SHA for future reproducibility
hashes={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (S/'inputs').glob('*') if p.is_file()}
(R/'INPUT_SHA256.json').write_text(json.dumps(hashes,indent=2))
obj={'passed':all(v['pass'] for v in tests),'n_checks':len(tests),'checks':tests,'source_data_checksums':'INPUT_SHA256.json','known_gaps':['AC power flow after new circuits not run','2050 official network not available','unpriced two remaining 230 kV interfaces','conditional dispatch replacement headroom not validated']}
(O/'audit_summary.json').write_text(json.dumps(obj,indent=2))
print('PASS' if obj['passed'] else 'FAIL',len(tests),'checks')
for t in tests:
    if not t['pass']:print('FAIL',t)
if not obj['passed']:sys.exit(1)
