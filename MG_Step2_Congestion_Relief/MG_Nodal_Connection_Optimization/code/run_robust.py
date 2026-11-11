#!/usr/bin/env python3
from optimize_siting import solve_robust
import argparse
p=argparse.ArgumentParser()
p.add_argument('--years',nargs='+',type=int,default=[2035,2050])
p.add_argument('--balance-modes',nargs='+',default=['distributed_MG_load','border_585'])
p.add_argument('--pv',nargs='+',type=float,default=[.35,.70,.90])
p.add_argument('--penalty',type=float,default=1.0)
p.add_argument('--time-limit',type=float,default=90)
a=p.parse_args()
for year in a.years:
 solve_robust(year,tuple(a.balance_modes),tuple(a.pv),a.penalty,a.time_limit)
