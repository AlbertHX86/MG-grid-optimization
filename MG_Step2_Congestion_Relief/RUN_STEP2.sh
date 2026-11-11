#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python step2_model.py
python local_only_run.py
python redispatch_compare.py
python audit_step2.py
python create_outputs.py
printf '\nSTEP2 COMPLETE: see results/ and figures/\n'
