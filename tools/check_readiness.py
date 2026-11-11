#!/usr/bin/env python3
"""Read-only validation of local inputs, portfolio totals, basic import and syntax readiness."""
from pathlib import Path
import csv,ast,sys
root=Path(__file__).resolve().parents[1]
i=root/'MG_Step2_Congestion_Relief/MG_Nodal_Connection_Optimization/inputs'
required={'pwf_2035_buses.csv':1489,'pwf_2035_branches.csv':1890,'candidate_areas.csv':12}
errors=[]
for file,count in required.items():
    path=i/file
    if not path.exists():errors.append(f'Missing {path.relative_to(root)}');continue
    with path.open(encoding='utf-8-sig',newline='') as fh: rows=list(csv.DictReader(fh))
    if len(rows)!=count:errors.append(f'{file}: expected {count} rows, found {len(rows)}')
    else: print(f'PASS {file}: {count} rows')
for year,expected in [(2035,3.217),(2050,30.10063393185335)]:
    path=i/f'portfolio_{year}.csv'
    if not path.exists():errors.append(f'Missing {path}');continue
    with path.open(encoding='utf-8-sig',newline='') as fh:
        rows=list(csv.DictReader(fh))
    value=sum(float(r['new_build_gw']) for r in rows)
    print(f'PORTFOLIO {year}: {value:.6f} GW incremental capacity including BESS')
    if abs(value-expected)>1e-3:errors.append(f'{year} portfolio sum mismatch: {value}')
for p in root.rglob('*.py'):
    try:ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
    except Exception as e:errors.append(f'Python parsing: {p.relative_to(root)} {e}')
print(f'Python syntax scan: {len(list(root.rglob("*.py")))} files')
if errors:
    for error in errors:print('ERROR',error,file=sys.stderr)
    sys.exit(1)
print('PASS: basic data/syntax readiness. This is NOT a full optimization reproduction.')
