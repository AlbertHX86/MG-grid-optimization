#!/usr/bin/env python3
"""Repeat local-Minas balancing cases; physically identical parallel lines (not certified)."""
from pathlib import Path
import step2_model as M
M.O=Path(__file__).resolve().parent/'results'/'local_only';M.O.mkdir(parents=True,exist_ok=True)
M.SCENARIOS=[(b,p) for b,p in M.SCENARIOS if b=='distributed_MG_load']
M.study(2050,max_additions=18)
