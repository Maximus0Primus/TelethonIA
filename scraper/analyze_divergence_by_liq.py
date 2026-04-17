"""
Analyse la divergence paper→live (paper_exit_price vs exit_price) par bucket de liquidité
et par flag bonding curve. Read-only — aucun effet sur la prod.

Hypothèse à tester : liq=$0 / bonding curve produit des divergences >20%, alors que
liq >$25K reste sous 10%. N=3 à date (Apr 17) insuffisant pour conclure.

ATTENTION : deux effets confondus contribuent à la divergence :
  (A) **Polling lag** — délai paper_decision → live_sell. LIVE MONITOR tombe toutes
      les 3-11min en observé. Sur bonding qui crash, chaque seconde coûte cher.
  (B) **Slippage exécution pur** — sell Jupiter dans curve illiquide déplace le prix.

Ces deux sont CORRÉLÉS à liq=$0 (bonding = illiquide ET volatile). Le script les sépare
via `delay_sec` (gap paper→live) et `drift_during_delay` (via price_ticks si dispo).

Seuil de décision : N ≥ 15 par bucket + décomposition (A)/(B) claire avant patch.

NB v141 : `paper_exit_price` persisté en DB mesure désormais le vrai prix paper-simulé
(dynamic slip + fee + Ultra SELL quote) — pas le niveau SL brut comme en v122-v140.
Les données collectées AVANT v141 sont biaisées (mélange 3 effets), à filtrer par
`created_at >= v141_deploy_ts` pour toute analyse propre.

Usage :
    python scraper/analyze_divergence_by_liq.py
    python scraper/analyze_divergence_by_liq.py --days 14
"""

import argparse
import os
import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from supabase import create_client


LIQ_BUCKETS = [
    ("bonding_0",     0,       1),
    ("micro_1_5K",    1,       5_000),
    ("small_5_25K",   5_000,   25_000),
    ("mid_25_100K",   25_000,  100_000),
    ("large_100K+",   100_000, float("inf")),
]


def _bucket(liq: float | None, is_bonding: bool) -> str:
    if is_bonding or not liq:
        return "bonding_0"
    for name, lo, hi in LIQ_BUCKETS:
        if lo <= liq < hi:
            return name
    return "large_100K+"


def _pct(values, p):
    if not values:
        return 0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(len(s) * p / 100)))
    return s[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7, help="lookback window (default 7d)")
    args = ap.parse_args()

    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    client = create_client(url, key)

    since = (datetime.now(timezone.utc) - timedelta(days=args.days)).isoformat()

    # Live trades only (trade_type='rt' + execution signature present) that are closed
    # and have a divergence computed. paper_exit_price populated since v122.
    # eval_history (v138) gives us the poll cadence — last poll before sell = paper decision ts.
    resp = (
        client.table("paper_trades")
        .select(
            "id,symbol,token_address,strategy,status,pnl_pct,"
            "entry_price,exit_price,paper_exit_price,price_divergence_pct,"
            "liquidity_usd,is_bonding_curve,dex,trade_type,created_at,exit_at,"
            "eval_history"
        )
        .eq("trade_type", "rt")
        .not_.is_("tx_signature_exit", "null")
        .not_.is_("paper_exit_price", "null")
        .not_.is_("price_divergence_pct", "null")
        .gte("created_at", since)
        .execute()
    )
    rows = resp.data or []

    # v141-dbg: enrich each row with delay_sec (paper_decision → live_sell gap)
    # Extracted from eval_history: last poll timestamp = moment paper saw paper_exit_price.
    for r in rows:
        r["_delay_sec"] = None
        hist = r.get("eval_history")
        exit_at = r.get("exit_at")
        if hist and exit_at:
            try:
                last_ts = hist[-1].get("ts") if isinstance(hist, list) and hist else None
                if last_ts:
                    from datetime import datetime as _dt
                    t_paper = _dt.fromisoformat(str(last_ts).replace("Z", "+00:00"))
                    t_live = _dt.fromisoformat(str(exit_at).replace("Z", "+00:00"))
                    r["_delay_sec"] = max(0, (t_live - t_paper).total_seconds())
            except Exception:
                pass
    if not rows:
        print(f"No closed live trades with divergence data since {since}.")
        return

    buckets = defaultdict(list)
    for r in rows:
        liq = r.get("liquidity_usd")
        is_bond = bool(r.get("is_bonding_curve")) or (r.get("dex") or "").lower().startswith("pumpfun")
        b = _bucket(liq, is_bond)
        buckets[b].append(r)

    order = [b[0] for b in LIQ_BUCKETS]
    print(f"\n=== Divergence paper→live par bucket liq (window={args.days}d, N_total={len(rows)}) ===\n")
    header = f"{'bucket':<16} {'N':>4} {'div_mean%':>10} {'div_p50%':>9} {'div_p95%':>9} {'delay_s':>8} {'ratio':>7}"
    print(header)
    print("-" * len(header))

    for b in order:
        rs = buckets.get(b, [])
        n = len(rs)
        if n == 0:
            print(f"{b:<16} {n:>4}    —         —         —         —         —")
            continue
        divs = [float(r["price_divergence_pct"]) * 100 for r in rs if r.get("price_divergence_pct") is not None]
        real_pnl = [float(r.get("pnl_pct") or 0) * 100 for r in rs]
        paper_pnl = []
        for r in rs:
            e = r.get("entry_price")
            p = r.get("paper_exit_price")
            if e and p and e > 0:
                paper_pnl.append((float(p) / float(e) - 1) * 100)
        delays = [r["_delay_sec"] for r in rs if r.get("_delay_sec") is not None]
        delay_med = _pct(delays, 50) if delays else 0
        div_mean = statistics.mean(divs) if divs else 0
        real_mean = statistics.mean(real_pnl) if real_pnl else 0
        paper_mean = statistics.mean(paper_pnl) if paper_pnl else 0
        ratio = (real_mean / paper_mean) if paper_mean else 0
        flag = "  ⚠ N<15" if n < 15 else ""
        print(
            f"{b:<16} {n:>4} {div_mean:>+10.2f} {_pct(divs,50):>+9.2f} {_pct(divs,95):>+9.2f} {delay_med:>8.0f} "
            f"{ratio:>7.2f}{flag}"
        )

    # === (A)/(B) decomposition : delay vs pure slippage ===
    # Correlate |divergence| with delay_sec — split by bucket.
    # If |div| increases sharply with delay_sec → (A) polling dominates → fix = polling.
    # If |div| stays high at delay=0 → (B) pure exec slippage → fix = Jupiter Ultra / liq filter.
    print("\n=== Décomposition (A) polling-lag vs (B) slippage exec ===")
    print("      Si |div| corrèle fort avec delay_sec → polling est cause principale")
    print("      Si |div| reste haut à delay≈0 → slippage pur, liq filter requis\n")
    header2 = f"{'bucket':<16} {'N_delay':>8} {'corr(delay,|div|)':>20} {'|div|@delay<30s':>16} {'|div|@delay>120s':>17}"
    print(header2)
    print("-" * len(header2))
    for b in order:
        rs = [r for r in buckets.get(b, []) if r.get("_delay_sec") is not None]
        if not rs:
            print(f"{b:<16} {'0':>8}          —              —                 —")
            continue
        xs = [r["_delay_sec"] for r in rs]
        ys = [abs(float(r["price_divergence_pct"]) * 100) for r in rs]
        # Pearson corr (manual, avoid numpy dep)
        n2 = len(xs)
        if n2 < 3:
            corr = float("nan")
        else:
            mx, my = sum(xs) / n2, sum(ys) / n2
            num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n2))
            dx2 = sum((xs[i] - mx) ** 2 for i in range(n2))
            dy2 = sum((ys[i] - my) ** 2 for i in range(n2))
            corr = num / ((dx2 * dy2) ** 0.5) if dx2 > 0 and dy2 > 0 else 0
        fast = [ys[i] for i in range(n2) if xs[i] < 30]
        slow = [ys[i] for i in range(n2) if xs[i] > 120]
        fast_s = f"{statistics.mean(fast):.1f}%  (N={len(fast)})" if fast else "—"
        slow_s = f"{statistics.mean(slow):.1f}%  (N={len(slow)})" if slow else "—"
        print(f"{b:<16} {n2:>8} {corr:>+20.2f} {fast_s:>16} {slow_s:>17}")

    # Highlight outliers
    print("\n=== Outliers |divergence| > 20% ===\n")
    outliers = [r for r in rows if abs(float(r.get("price_divergence_pct") or 0)) > 0.20]
    if not outliers:
        print("Aucun outlier.")
    else:
        for r in sorted(outliers, key=lambda x: abs(float(x.get("price_divergence_pct") or 0)), reverse=True)[:20]:
            d = float(r["price_divergence_pct"]) * 100
            liq = r.get("liquidity_usd") or 0
            print(f"  {r['symbol']:>14} {r['strategy']:<24} liq=${liq:>8.0f} bond={int(bool(r.get('is_bonding_curve')))} div={d:+.1f}% pnl_real={float(r.get('pnl_pct') or 0)*100:+.1f}%")

    # Decision hint
    print("\n=== Décision ===")
    ready = all(len(buckets.get(b, [])) >= 15 for b in ("bonding_0", "mid_25_100K"))
    if ready:
        print("✅ N ≥ 15 dans bonding_0 ET mid_25_100K → quantification possible. Compare divergence mean et ratio.")
    else:
        need = {b: max(0, 15 - len(buckets.get(b, []))) for b in order}
        print(f"⏳ Pas assez de data. Besoin encore : {need}")
        print("   Au rythme ~2-3 bondings/jour, attendre ~5-7 jours supplémentaires.")


if __name__ == "__main__":
    main()
