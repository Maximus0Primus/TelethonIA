"""Analyze shadow trade performance over last 4 days (since March 15, 2026)."""
import requests
from collections import defaultdict

SUPABASE_URL = 'https://xbcasrywqqmnotknzbpg.supabase.co'
SUPABASE_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InhiY2Fzcnl3cXFtbm90a256YnBnIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3MDQxNjg5NCwiZXhwIjoyMDg1OTkyODk0fQ.W8Ziw6nfB9da9oXFRh_Z_SwGRPH5sRj-YKUwpmaHhRQ'

headers = {'apikey': SUPABASE_KEY, 'Authorization': f'Bearer {SUPABASE_KEY}'}

# Fetch ALL completed trades since March 15
all_trades = []
offset = 0
PAGE = 1000
while True:
    r = requests.get(
        f'{SUPABASE_URL}/rest/v1/paper_trades?status=neq.open&created_at=gte.2026-03-15T00:00:00Z'
        f'&select=strategy,pnl_pct,is_shadow,kol_group,status'
        f'&order=created_at.asc&limit={PAGE}&offset={offset}',
        headers=headers
    )
    if r.status_code != 200:
        print('Error:', r.status_code, r.text)
        break
    batch = r.json()
    all_trades.extend(batch)
    if len(batch) < PAGE:
        break
    offset += PAGE

print(f'Total completed trades since Mar 15: {len(all_trades)}')

from collections import Counter
status_counts = Counter(t['status'] for t in all_trades)
print(f'Status breakdown: {dict(status_counts)}')

main_trades = [t for t in all_trades if not t.get('is_shadow')]
shadow_trades = [t for t in all_trades if t.get('is_shadow')]
print(f'Main trades: {len(main_trades)}, Shadow trades: {len(shadow_trades)}')
print()

TRADE_SIZE = 15.0
# pnl_pct is fractional: 0.0224 = +2.24%, -0.50 = -50%


def calc_stats(trades_list):
    stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'sum_pnl_pct': 0.0, 'trades': 0, 'tp': 0, 'sl': 0, 'to': 0})
    for t in trades_list:
        s = t['strategy']
        pnl = t.get('pnl_pct') or 0
        stats[s]['trades'] += 1
        stats[s]['sum_pnl_pct'] += pnl
        if pnl > 0:
            stats[s]['wins'] += 1
        else:
            stats[s]['losses'] += 1
        if t['status'] == 'tp_hit':
            stats[s]['tp'] += 1
        elif t['status'] == 'sl_hit':
            stats[s]['sl'] += 1
        elif t['status'] == 'timeout':
            stats[s]['to'] += 1
    return stats


strat_stats = calc_stats(all_trades)

def sort_key(item):
    return item[1]['sum_pnl_pct'] * TRADE_SIZE

sorted_strats = sorted(strat_stats.items(), key=sort_key, reverse=True)


def print_table(title, items):
    print('=' * 115)
    print(title)
    print('=' * 115)
    print(f'| {"Strategy":<30} | {"Trades":>6} | {"Win%":>6} | {"Avg PnL%":>9} | {"Tot PnL$":>10} | {"TP":>4} | {"SL":>4} | {"TO":>4} |')
    print(f'|{"-"*32}|{"-"*8}|{"-"*8}|{"-"*11}|{"-"*12}|{"-"*6}|{"-"*6}|{"-"*6}|')
    for name, st in items:
        n = st['trades']
        wr = st['wins'] / n * 100 if n else 0
        avg_pnl = st['sum_pnl_pct'] / n * 100 if n else 0
        total_usd = st['sum_pnl_pct'] * TRADE_SIZE
        print(f'| {name:<30} | {n:>6} | {wr:>5.1f}% | {avg_pnl:>+8.2f}% | {total_usd:>+9.2f}$ | {st["tp"]:>4} | {st["sl"]:>4} | {st["to"]:>4} |')


print_table('TOP 10 BEST STRATEGIES (by total PnL$, $15/trade, last 4 days)', sorted_strats[:10])
print()
print_table('TOP 10 WORST STRATEGIES (by total PnL$)', list(reversed(sorted_strats[-10:])))

# =========================================
# MAIN vs SHADOW
# =========================================
print()
print('=' * 115)
print('MAIN vs SHADOW TRADES COMPARISON')
print('=' * 115)

for label, trades in [('MAIN', main_trades), ('SHADOW', shadow_trades), ('ALL', all_trades)]:
    if not trades:
        print(f'  {label}: No trades')
        continue
    wins = sum(1 for t in trades if (t.get('pnl_pct') or 0) > 0)
    total_pnl_frac = sum((t.get('pnl_pct') or 0) for t in trades)
    total_usd = total_pnl_frac * TRADE_SIZE
    avg_pnl = total_pnl_frac / len(trades) * 100
    wr = wins / len(trades) * 100
    tp_cnt = sum(1 for t in trades if t['status'] == 'tp_hit')
    sl_cnt = sum(1 for t in trades if t['status'] == 'sl_hit')
    to_cnt = sum(1 for t in trades if t['status'] == 'timeout')
    print(f'  {label:>8}: {len(trades):>6} trades | WR: {wr:>5.1f}% | Avg PnL: {avg_pnl:>+7.2f}% | Total PnL$: {total_usd:>+10.2f}$ | TP:{tp_cnt:>5} SL:{sl_cnt:>5} TO:{to_cnt:>5}')

# Main-only per-strategy
if main_trades:
    print()
    main_strat = calc_stats(main_trades)
    sorted_main = sorted(main_strat.items(), key=lambda x: x[1]['sum_pnl_pct'] * TRADE_SIZE, reverse=True)
    print('  MAIN trades per strategy:')
    print(f'  | {"Strategy":<25} | {"Trades":>6} | {"Win%":>6} | {"Avg PnL%":>9} | {"Tot PnL$":>10} |')
    print(f'  |{"-"*27}|{"-"*8}|{"-"*8}|{"-"*11}|{"-"*12}|')
    for name, st in sorted_main:
        n = st['trades']
        wr = st['wins'] / n * 100 if n else 0
        avg_pnl = st['sum_pnl_pct'] / n * 100 if n else 0
        total_usd = st['sum_pnl_pct'] * TRADE_SIZE
        print(f'  | {name:<25} | {n:>6} | {wr:>5.1f}% | {avg_pnl:>+8.2f}% | {total_usd:>+9.2f}$ |')

# =========================================
# PER-KOL for top 5 strategies
# =========================================
top5_strats = set(name for name, _ in sorted_strats[:5])

print()
print('=' * 115)
print(f'PER-KOL PERFORMANCE (top 5 strategies: {", ".join(sorted(top5_strats))})')
print('=' * 115)

kol_stats = defaultdict(lambda: {'wins': 0, 'trades': 0, 'sum_pnl_pct': 0.0})

for t in all_trades:
    if t['strategy'] not in top5_strats:
        continue
    kol = t.get('kol_group') or 'unknown'
    pnl = t.get('pnl_pct') or 0
    kol_stats[kol]['trades'] += 1
    kol_stats[kol]['sum_pnl_pct'] += pnl
    if pnl > 0:
        kol_stats[kol]['wins'] += 1

sorted_kols = sorted(kol_stats.items(), key=lambda x: x[1]['sum_pnl_pct'] * TRADE_SIZE, reverse=True)

print(f'| {"KOL Group":<35} | {"Trades":>6} | {"Win%":>6} | {"Avg PnL%":>9} | {"Tot PnL$":>10} |')
print(f'|{"-"*37}|{"-"*8}|{"-"*8}|{"-"*11}|{"-"*12}|')
for kol, st in sorted_kols:
    n = st['trades']
    wr = st['wins'] / n * 100 if n else 0
    avg_pnl = st['sum_pnl_pct'] / n * 100 if n else 0
    total_usd = st['sum_pnl_pct'] * TRADE_SIZE
    print(f'| {kol:<35} | {n:>6} | {wr:>5.1f}% | {avg_pnl:>+8.2f}% | {total_usd:>+9.2f}$ |')

# =========================================
# PER-KOL across ALL strategies (overall KOL ranking)
# =========================================
print()
print('=' * 115)
print('PER-KOL OVERALL PERFORMANCE (all strategies combined, top 20 + bottom 10)')
print('=' * 115)

kol_all = defaultdict(lambda: {'wins': 0, 'trades': 0, 'sum_pnl_pct': 0.0})

for t in all_trades:
    kol = t.get('kol_group') or 'unknown'
    pnl = t.get('pnl_pct') or 0
    kol_all[kol]['trades'] += 1
    kol_all[kol]['sum_pnl_pct'] += pnl
    if pnl > 0:
        kol_all[kol]['wins'] += 1

sorted_kol_all = sorted(kol_all.items(), key=lambda x: x[1]['sum_pnl_pct'] * TRADE_SIZE, reverse=True)

print(f'| {"KOL Group":<35} | {"Trades":>6} | {"Win%":>6} | {"Avg PnL%":>9} | {"Tot PnL$":>10} |')
print(f'|{"-"*37}|{"-"*8}|{"-"*8}|{"-"*11}|{"-"*12}|')
for kol, st in sorted_kol_all[:20]:
    n = st['trades']
    wr = st['wins'] / n * 100 if n else 0
    avg_pnl = st['sum_pnl_pct'] / n * 100 if n else 0
    total_usd = st['sum_pnl_pct'] * TRADE_SIZE
    print(f'| {kol:<35} | {n:>6} | {wr:>5.1f}% | {avg_pnl:>+8.2f}% | {total_usd:>+9.2f}$ |')

if len(sorted_kol_all) > 20:
    print(f'| {"...":<35} | {"":>6} | {"":>6} | {"":>9} | {"":>10} |')
    for kol, st in sorted_kol_all[-10:]:
        n = st['trades']
        wr = st['wins'] / n * 100 if n else 0
        avg_pnl = st['sum_pnl_pct'] / n * 100 if n else 0
        total_usd = st['sum_pnl_pct'] * TRADE_SIZE
        print(f'| {kol:<35} | {n:>6} | {wr:>5.1f}% | {avg_pnl:>+8.2f}% | {total_usd:>+9.2f}$ |')

print()
print(f'Total strategies: {len(strat_stats)}')
print(f'Total KOL groups: {len(kol_all)}')
grand_pnl = sum(st['sum_pnl_pct'] for st in strat_stats.values()) * TRADE_SIZE
print(f'Grand total PnL$ (all {len(all_trades)} trades @ $15): {grand_pnl:+.2f}$')
