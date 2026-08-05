"""Live simulation — FAST_TP50_SL30_MCAP_S40 + sentiment band 0.5-0.6.

Pool: 69 deduped (rolling 24h), blacklist-current (olympeqg banned), corrupt rows
excluded (pnl_pct<=20), 2026-04-20 -> 2026-08-05 (~107 days).
Paper pnl_pct already carries buy slip 225bps + dynamic sell slip + fees.
This adds what paper does NOT model: MEV, failed-txn retries, priority-fee
competition, partial fills.
"""
import random, statistics as st

PNL = [0.1830,-0.1250,0.1599,-0.0514,0.1695,0.3044,0.0134,-0.0957,0.1080,-0.1402,
-0.1524,-0.0872,-0.5044,0.1630,1.0517,0.5476,0.9257,1.0417,0.6923,-0.5228,1.0914,
0.6134,-0.0161,-0.3294,-0.2114,0.5099,-0.1514,0.5161,-0.0344,-0.0082,0.0311,-0.3501,
-0.2171,-0.0498,-0.4010,-0.1353,0.5798,-0.1055,0.3984,1.1081,-0.0095,-0.0398,-0.0003,
-0.3657,-0.2783,0.1372,-0.0110,-0.2961,0.5938,-0.5202,-0.3459,-0.0067,0.0461,0.3505,
0.0390,0.8159,0.1958,-0.3360,0.0359,-0.1480,-0.2480,0.0904,-0.0550,0.4908,-0.0440,
1.0606,1.3010,-0.1425,-0.2881]

DAYS = 107.0
LIQ_MED = 27580.0
EXTRA = 0.004          # MEV + retry + priority fee + partial fill (skill Phase 2)


def adj(p, pos):
    """Live-adjusted return. Size impact only bites above $500 (skill formula),
    which neither $50 nor $100 reaches."""
    return p - EXTRA - max(0.0, (pos - 500) / 5000) * 0.01


def metrics(pool, pos):
    r = [adj(p, pos) for p in pool]
    wins = [x for x in r if x > 0]
    loss = [x for x in r if x <= 0]
    gp, gl = sum(wins), -sum(loss)
    dstd = st.pstdev([min(x, 0) for x in r]) or 1e-9
    return {
        "n": len(r), "moy": st.mean(r), "wr": len(wins)/len(r),
        "pf": gp/gl if gl else float("inf"),
        "sharpe": st.mean(r)/(st.pstdev(r) or 1e-9),
        "sortino": st.mean(r)/dstd,
        "gain_med": st.median(wins) if wins else 0,
        "perte_med": st.median(loss) if loss else 0,
    }


def walk(pool, start, pos_fn):
    bank, peak, mdd = start, start, 0.0
    for p in pool:
        pos = pos_fn(bank)
        if pos <= 0:
            break
        bank += pos * adj(p, pos)
        peak = max(peak, bank)
        mdd = max(mdd, (peak - bank) / peak)
    return bank, mdd


def mc(pool, start, pos_fn, n_trades, runs=10000, seed=7):
    rnd = random.Random(seed)
    fins, dds, ruin = [], [], 0
    for _ in range(runs):
        bank, peak, mdd = start, start, 0.0
        for _ in range(n_trades):
            pos = pos_fn(bank)
            if pos <= 0:
                break
            bank += pos * adj(rnd.choice(pool), pos)
            peak = max(peak, bank)
            mdd = max(mdd, (peak - bank) / peak)
            if bank < start * 0.10:
                ruin += 1
                break
        fins.append(bank); dds.append(mdd)
    fins.sort(); dds.sort()
    q = lambda a, p: a[int(p * (len(a) - 1))]
    return {"p10": q(fins,.10), "p25": q(fins,.25), "med": q(fins,.50),
            "p75": q(fins,.75), "p90": q(fins,.90),
            "dd_med": q(dds,.50), "dd_p90": q(dds,.90), "ruin": ruin/runs}


print("=" * 74)
print("SIMULATION LIVE — FAST_TP50_SL30_MCAP_S40 + sentiment 0.5-0.6")
print("=" * 74)
print(f"Pool: n={len(PNL)} dedup 24h | {DAYS:.0f} jours | {len(PNL)/DAYS*7:.1f} trades/semaine")
print(f"Liquidite mediane du pool: ${LIQ_MED:,.0f}")
print(f"Cout live additionnel applique: -{EXTRA*100:.1f}pp/trade (MEV, retry, priority fee, fill partiel)\n")

print("--- EV par trade, avant vs apres couts live ---")
brut = st.mean(PNL)
for pos in (50, 100):
    m = metrics(PNL, pos)
    print(f"  ${pos:<4} brut {brut*100:+6.2f}%  ->  net {m['moy']*100:+6.2f}%   "
          f"WR {m['wr']*100:.0f}%  PF {m['pf']:.2f}  Sharpe {m['sharpe']:.2f}  Sortino {m['sortino']:.2f}")
m50 = metrics(PNL, 50)
print(f"  gain median {m50['gain_med']*100:+.1f}%  |  perte mediane {m50['perte_med']*100:+.1f}%  "
      f"|  position vs liq mediane: $100 = {100/LIQ_MED*100:.2f}% du pool\n")

print("--- Walk-forward chronologique (les 69 trades dans l'ordre reel) ---")
print(f"{'sizing':<26}{'$300':>10}{'$1000':>10}{'$3000':>10}{'maxDD':>9}")
sizings = [
    ("fixe $50",            lambda b: 50 if b > 50 else 0),
    ("fixe $100",           lambda b: 100 if b > 100 else 0),
    ("5% du capital",       lambda b: b * 0.05),
    ("10% du capital",      lambda b: b * 0.10),
    ("20% du capital",      lambda b: b * 0.20),
]
for name, fn in sizings:
    row, dd_max = [], 0
    for start in (300, 1000, 3000):
        b, dd = walk(PNL, start, fn)
        row.append(b); dd_max = max(dd_max, dd)
    print(f"{name:<26}{row[0]:>10,.0f}{row[1]:>10,.0f}{row[2]:>10,.0f}{dd_max*100:>8.0f}%")

print("\n--- Monte Carlo 10 000 tirages, 100 trades (~7 mois au rythme actuel) ---")
print(f"{'sizing':<26}{'P10':>9}{'P25':>9}{'median':>9}{'P75':>9}{'P90':>9}{'DDmed':>8}{'ruine':>8}")
for name, fn in sizings:
    r = mc(PNL, 1000, fn, 100)
    print(f"{name:<26}{r['p10']:>9,.0f}{r['p25']:>9,.0f}{r['med']:>9,.0f}"
          f"{r['p75']:>9,.0f}{r['p90']:>9,.0f}{r['dd_med']*100:>7.0f}%{r['ruin']*100:>7.1f}%")
print("  (capital de depart $1 000)")

print("\n--- Stress tests (fixe $100, capital $1 000, 100 trades) ---")
srt = sorted(PNL)
p20, p40 = srt[int(.20*len(srt))], srt[int(.40*len(srt))]
froid = [p for p in PNL if p20 <= p <= p40] or srt[:len(srt)//3]
rnd = random.Random(11)
scenarios = [
    ("Base",                      PNL,  0.004),
    ("Slippage double",           PNL,  0.008),
    ("Attaque MEV (-2% sur 5%)",  [p - (0.02 if rnd.random() < .05 else 0) for p in PNL], 0.004),
    ("Crise de liquidite (-5% sur 10%)", [p - (0.05 if rnd.random() < .10 else 0) for p in PNL], 0.004),
    ("Cold streak (P20-P40)",     froid, 0.004),
]
print(f"{'scenario':<36}{'median':>10}{'DDmed':>9}{'ruine':>8}  verdict")
for name, pool, cost in scenarios:
    old = EXTRA
    globals()["EXTRA"] = cost
    r = mc(pool, 1000, lambda b: 100 if b > 100 else 0, 100, seed=3)
    globals()["EXTRA"] = old
    ok = r["med"] > 1000 and r["ruin"] < 0.05
    print(f"{name:<36}{r['med']:>10,.0f}{r['dd_med']*100:>8.0f}%{r['ruin']*100:>7.1f}%  {'PASS' if ok else 'FAIL'}")

print("\n--- Marge de securite: quel cout live tue l'edge ? ---")
for c in (0.004, 0.01, 0.02, 0.04, 0.06, 0.08):
    ev = st.mean([p - c for p in PNL])
    print(f"  cout {c*100:>4.1f}pp/trade -> EV {ev*100:+6.2f}%/trade" + ("   <-- seuil de rentabilite" if abs(ev) < 0.01 else ""))
