#!/usr/bin/env bash
# Pre-deploy smoke + test gate. Run BEFORE `systemctl restart kol-scraper`.
# Exits non-zero if anything fails so /deploy can abort the push.
#
# Checks (fast, <30s):
#   1. Python imports for all core modules load without error
#   2. pytest -x (stop on first failure)
#   3. No syntax errors on critical files (py_compile)
#
# Usage (locally):
#   bash scripts/pre_deploy_check.sh
# Or from /deploy before ssh vps restart.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT/scraper"

echo "=== 1/3 py_compile critical modules ==="
python -m py_compile \
    safe_scraper.py \
    paper_trader.py \
    live_trader.py \
    pipeline.py \
    strategies.py \
    enrich.py \
    enrich_jupiter.py \
    alerter.py
echo "  OK"

echo ""
echo "=== 2/3 import smoke ==="
python -c "
import sys
sys.path.insert(0, '.')
modules = [
    'safe_scraper',
    'paper_trader',
    'live_trader',
    'pipeline',
    'strategies',
    'enrich',
    'enrich_jupiter',
    'alerter',
    'chain_detect',
    'sim',
]
for m in modules:
    __import__(m)
    print(f'  OK {m}')
print(f'  {len(modules)} modules imported clean')
"

echo ""
echo "=== 3/3 pytest -x ==="
python -m pytest tests/ -x -q --tb=line 2>&1 | tail -20

echo ""
echo "=== Pre-deploy check PASSED ==="
