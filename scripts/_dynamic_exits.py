"""E20b — sorties dynamiques rejouées tick par tick.

Dernier axe non exploré du registre. Toutes les stratégies de la grille ont un
TP/SL FIXE décidé à l'entrée. Ici on rejoue le flux de ticks et on laisse la
sortie dépendre de ce que le prix a fait depuis.

Discipline héritée de la session
--------------------------------
- **Mono-source obligatoire.** `price_ticks` entrelace jupiter/fast/full toutes
  les 11-20 s avec un désaccord p1 −85.8 % / p99 +640 %. Le rejouer mélangé avait
  fabriqué un edge de +12.6 %/trade qui n'existait pas.
- **Slippage de production** via `sim_engines._exit` (`_dynamic_sell_slip_factor`),
  pas une constante.
- **Univers filtré** = bande de sentiment 0.30-0.70, la config validée. Optimiser
  la sortie sur un univers qu'on ne trade pas n'aurait aucun sens.
- **Jugement en composition** à f=0.10 (E22), pas en somme brute : E27 a montré
  que somme brute et rendement composé peuvent pointer en sens opposés.
- ~10 règles testées ⇒ risque de sélection modéré. On lit l'ÉCART entre règles et
  on exige que le gagnant batte la référence **mois par mois**, pas seulement au
  total.

Note: les familles TRAIL/DTRAIL de la grille sont déjà connues comme mauvaises EN
LIVE (DTRAIL10_ACT15_SL70 = −$45 réel). On les re-teste ici quand même mais avec
le slippage de production, pour savoir si c'était le modèle de slip ou la règle.

    python scripts/_dynamic_exits.py [--source jupiter|ds]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scraper"))

import sim_engines  # noqa: E402
from strategies import BUY_SLIPPAGE_BPS  # noqa: E402

CACHE = Path(__file__).parent / "_dynexit_cache.json"
WINDOW_DAYS = 30            # price_ticks ne retient que ~30 j
MAX_HOLD_MIN = 240
BUY_SLIP = BUY_SLIPPAGE_BPS / 10_000
F = 0.10                    # sizing retenu en E22
LIVE_COST = 0.004


def _ts(iso):
    return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp())


def fetch():
    from supabase import create_client
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
    since = (datetime.now(timezone.utc) - timedelta(days=WINDOW_DAYS)).isoformat()
    cfg = sb.table("scoring_config").select("paper_trade_config").eq("id", 1).execute().data[0]
    bl = set((cfg["paper_trade_config"].get("kol_chain_blacklist") or {}).get("solana") or [])

    def page(table, select, eq, key):
        out, cur = [], None
        while True:
            q = sb.table(table).select(select)
            for k, v in eq:
                q = q.eq(k, v)
            rows = q.gte(key, cur or since).order(key).limit(1000).execute().data
            if not rows:
                break
            out.extend(rows)
            if len(rows) < 1000 or rows[-1][key] == cur:
                break
            cur = rows[-1][key]
        return out

    calls = page("paper_trades", "token_address,kol_group,created_at,rt_liquidity_usd",
                 [("chain", "solana"), ("source", "rt"),
                  ("strategy", "FAST_TP50_SL30_MCAP_S40")], "created_at")
    men = page("kol_mentions", "kol_group,resolved_ca,sentiment,message_date",
               [("chain", "solana")], "message_date")
    ticks = page("price_ticks", "token_address,price_usd,fetched_at,source",
                 [("chain", "solana")], "fetched_at")

    sent = {}
    for r in sorted(men, key=lambda r: r["message_date"]):
        k = (r.get("kol_group"), r.get("resolved_ca"))
        if k not in sent and r.get("sentiment") is not None:
            sent[k] = float(r["sentiment"])

    seen, uni = set(), {}
    for c in sorted(calls, key=lambda r: r["created_at"]):
        a = c["token_address"]
        if a in seen or c.get("kol_group") in bl:
            continue
        s = sent.get((c.get("kol_group"), a))
        if s is None or not (0.30 <= s < 0.70):     # bande validée
            continue
        seen.add(a)
        uni[a] = {"t0": c["created_at"], "liq": float(c.get("rt_liquidity_usd") or 0)}

    series = defaultdict(list)
    for r in ticks:
        p = r.get("price_usd")
        if p and float(p) > 0 and r["token_address"] in uni:
            series[r["token_address"]].append([_ts(r["fetched_at"]), float(p), r.get("source")])
    for a in series:
        series[a].sort()
    return {"uni": uni, "series": {a: s for a, s in series.items()}}


# --- règles de sortie -------------------------------------------------------
# Chacune reçoit la série (ts, prix) depuis l'entrée et rend (pnl_fraction, raison).

def _finish(px, entry, reason, liq):
    """Applique le slippage de vente de PRODUCTION, pas une constante."""
    sim_engines._sim_liquidity_usd = liq or 50_000
    sim_engines._sim_chain = "solana"
    sim_engines._sim_position_usd = 100
    return sim_engines._exit(reason, px, entry, 0.0)["pnl_pct"], reason


def fixed(path, entry, liq, tp=1.50, sl=0.70, horizon=30):
    t0 = path[0][0]
    for ts, p in path:
        if p <= entry * sl:
            return _finish(entry * sl, entry, "sl_hit", liq)
        if p >= entry * tp:
            return _finish(entry * tp, entry, "tp_hit", liq)
        if (ts - t0) / 60 >= horizon:
            return _finish(p, entry, "timeout", liq)
    return _finish(path[-1][1], entry, "timeout", liq)


def trailing(path, entry, liq, arm=1.20, give=0.25, sl=0.70, horizon=60):
    """Trailing classique: s'arme à +20 %, puis suit le sommet à -25 %."""
    t0, peak, armed = path[0][0], entry, False
    for ts, p in path:
        peak = max(peak, p)
        if not armed and p >= entry * arm:
            armed = True
        if armed and p <= peak * (1 - give):
            return _finish(peak * (1 - give), entry, "trail_stop", liq)
        if not armed and p <= entry * sl:
            return _finish(entry * sl, entry, "sl_hit", liq)
        if (ts - t0) / 60 >= horizon:
            return _finish(p, entry, "timeout", liq)
    return _finish(path[-1][1], entry, "timeout", liq)


def decay_tp(path, entry, liq, tp0=2.00, tp1=1.20, sl=0.70, horizon=60):
    """TP qui décroît avec le temps: on demande beaucoup tôt, on se contente
    de peu si ça traîne. L'inverse d'un trailing."""
    t0 = path[0][0]
    for ts, p in path:
        frac = min((ts - t0) / 60 / horizon, 1.0)
        tp = tp0 + (tp1 - tp0) * frac
        if p <= entry * sl:
            return _finish(entry * sl, entry, "sl_hit", liq)
        if p >= entry * tp:
            return _finish(entry * tp, entry, "tp_hit", liq)
        if frac >= 1.0:
            return _finish(p, entry, "timeout", liq)
    return _finish(path[-1][1], entry, "timeout", liq)


def momentum_exit(path, entry, liq, n_down=3, sl=0.70, horizon=60):
    """Sort après n ticks baissiers consécutifs — pari sur le retournement."""
    t0, down, prev = path[0][0], 0, None
    for ts, p in path:
        if prev is not None:
            down = down + 1 if p < prev else 0
        prev = p
        if p <= entry * sl:
            return _finish(entry * sl, entry, "sl_hit", liq)
        if down >= n_down and p > entry:
            return _finish(p, entry, "trail_stop", liq)
        if (ts - t0) / 60 >= horizon:
            return _finish(p, entry, "timeout", liq)
    return _finish(path[-1][1], entry, "timeout", liq)


RULES = {
    "REFERENCE TP50/SL30 30m": lambda p, e, l: fixed(p, e, l),
    "TP50/SL30 60m":           lambda p, e, l: fixed(p, e, l, horizon=60),
    "TP70/SL30 60m":           lambda p, e, l: fixed(p, e, l, tp=1.70, horizon=60),
    "trailing arm+20 give25":  lambda p, e, l: trailing(p, e, l),
    "trailing arm+50 give30":  lambda p, e, l: trailing(p, e, l, arm=1.50, give=0.30),
    "trailing arm+20 give15":  lambda p, e, l: trailing(p, e, l, give=0.15),
    "TP decroissant 200->120": lambda p, e, l: decay_tp(p, e, l),
    "TP decroissant 150->110": lambda p, e, l: decay_tp(p, e, l, tp0=1.50, tp1=1.10),
    "sortie 3 ticks baissiers": lambda p, e, l: momentum_exit(p, e, l),
    "sortie 5 ticks baissiers": lambda p, e, l: momentum_exit(p, e, l, n_down=5),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="jupiter", choices=["jupiter", "ds"])
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()

    if CACHE.exists() and not a.refresh:
        data = json.loads(CACHE.read_text())
    else:
        data = fetch()
        CACHE.write_text(json.dumps(data))

    keep = {"fast", "full"} if a.source == "ds" else {a.source}
    results = defaultdict(list)
    n_tok = 0
    for addr, meta in data["uni"].items():
        raw = data["series"].get(addr) or []
        path = [[ts, p] for ts, p, src in raw if src in keep]
        t0 = _ts(meta["t0"])
        path = [x for x in path if t0 <= x[0] <= t0 + MAX_HOLD_MIN * 60]
        if len(path) < 5:
            continue
        n_tok += 1
        entry = path[0][1] * (1 + BUY_SLIP)
        for name, fn in RULES.items():
            try:
                pnl, _ = fn(path, entry, meta["liq"])
            except Exception:
                continue
            results[name].append(pnl - LIVE_COST)

    print(f"SOURCE={a.source} — {n_tok} tokens de la bande avec assez de ticks "
          f"(fenetre {WINDOW_DAYS}j)\n")
    print(f"{'regle de sortie':<28}{'n':>5}{'EV':>9}{'WR':>6}"
          f"{'capital f=0.10':>16}{'vs reference':>14}")
    print("-" * 78)
    ref = None
    rows = []
    for name, v in results.items():
        if len(v) < 20:
            continue
        ev = sum(v) / len(v)
        cap = math.exp(sum(math.log(max(1 + F * x, 1e-4)) for x in v)) - 1
        rows.append((name, len(v), ev, sum(1 for x in v if x > 0) / len(v), cap))
        if name.startswith("REFERENCE"):
            ref = cap
    for name, n, ev, wr, cap in rows:
        delta = "" if ref is None or name.startswith("REFERENCE") else f"{100*(cap-ref):>+13.0f}pp"
        print(f"{name:<28}{n:>5}{100*ev:>8.1f}%{100*wr:>5.0f}%{100*cap:>15.0f}%{delta:>14}")


if __name__ == "__main__":
    main()
