#!/usr/bin/env python3
"""Restore access-controlled/source-derived model data locally, without staging it for Git.

Usage: python tools/restore_local_inputs.py /path/to/PRIVATE_INPUTS_FOR_LOCAL_REPRODUCTION.zip
Only use if you have the relevant data usage rights. Never add restored files to a public commit.
"""
import argparse
from pathlib import Path
import zipfile
import sys

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "MG_Step2_Congestion_Relief/MG_Nodal_Connection_Optimization/inputs/pwf_2035_buses.csv",
    "MG_Step2_Congestion_Relief/MG_Nodal_Connection_Optimization/inputs/pwf_2035_branches.csv",
    "MG_Step2_Congestion_Relief/MG_Nodal_Connection_Optimization/inputs/epe_ons_station_crosswalk.csv",
    "MG_Step2_Congestion_Relief/MG_Nodal_Connection_Optimization/inputs/mg_boundary.geojson",
    "MG_Step2_Congestion_Relief/MG_Nodal_Connection_Optimization/macro_reopt/macro/processed/macro_hourly_inputs.csv",
    "MG_Step2_Congestion_Relief/MG_Nodal_Connection_Optimization/macro_reopt/macro/processed/mg_wind_2019_brt_aligned.csv",
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("private_zip", type=Path)
    args=p.parse_args()
    if not args.private_zip.exists():p.error("Private input ZIP not found")
    with zipfile.ZipFile(args.private_zip) as z:
        names=set(z.namelist())
        if not FILES.issubset(names):p.error("Required input files missing: " + str(sorted(FILES-names)))
        for name in sorted(FILES):
            dest=(ROOT/name).resolve()
            if ROOT.resolve() not in dest.parents: raise RuntimeError("Unsafe archive path")
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(z.read(name))
            print("Restored (local only):",name)
    print("REMINDER: Never force-add these inputs to a public Git repository.")

if __name__=="__main__":main()
