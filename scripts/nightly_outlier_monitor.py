"""Nightly monitor — alert on paper↔live outliers where sync=True.

Why the sync=True filter: v142E (entry) + v143.5 (exit) should close the gap
for every new trade. An outlier with sync=False is historic or live-swap-fail
and not actionable. An outlier with sync=True is a NEW logic bug — that's what
we want to catch early.

Exits with code 2 when any sync=True outlier found (so the GH Actions job fails
and alerts Telegram). Also prints summary stats from diverge_report logic.

v14e.96 — LE COTE LIVE BORNE LE TRAVAIL. La version precedente tirait toutes
les lignes `paper_trades` de la fenetre 48 h pour n'apparier qu'une poignee de
trades live. La grille shadow est passee de ~10 k a ~33 k lignes/jour (bras
v14e.93-95) => 64 k lignes, 64 pages OFFSET sans ORDER BY => `57014 statement
timeout`, et le monitor mourait en postant une fausse alerte trading aux champs
vides. On lit maintenant le cote live D'ABORD (filtre SQL), et le cote paper
uniquement sur ses tokens. Le cout ne depend plus du nombre de bras shadow.
"""
import os, sys, time, statistics as st, json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

# Les prints contiennent Δ et — : sur une console Windows en cp1252 le script
# meurt en UnicodeEncodeError APRES avoir fait son travail (run local 13/08).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Last 48h rolling window so we always see enough N without old sync=False noise
SINCE = os.environ.get("OUTLIER_SINCE",
    (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat())
THRESHOLD_PP = float(os.environ.get("OUTLIER_THRESHOLD_PP", "10"))

COLS = ("id,symbol,strategy,status,source,pnl_pct,entry_price,exit_price,entry_source,"
        "created_at,exit_at,rt_is_pump_fun,token_address,paper_sim_pnl_pct")
CLOSED_STATUSES = ("sl_hit", "trail_stop", "tp_hit", "timeout", "be_stop")

# Le statement timeout Supabase est chronique sur ce projet (~13/h) : une seule
# tentative transforme un hoquet d'infra en fausse alerte trading.
STATEMENT_TIMEOUT = "57014"


def _execute(q, tries=3):
    for attempt in range(tries):
        try:
            return q.execute()
        except Exception as e:
            if STATEMENT_TIMEOUT not in str(e) or attempt == tries - 1:
                raise
            wait = 2 ** attempt
            print(f"  [retry {attempt+1}/{tries-1}] statement timeout — retrying in {wait}s")
            time.sleep(wait)


def fetch_all(tbl, sel, order=("id",), **f):
    """`order` DOIT correspondre a l'index qui sert le filtre.

    Mesure sur la fenetre 01/06 (EXPLAIN ANALYZE, 100 tokens) :
      order by id             -> top-N heapsort sur 21 384 lignes, 1631 ms/page
                                 (et le tri recommence a chaque OFFSET => timeout)
      order by token_address  -> Index Scan idx_paper_trades_token,     40 ms
      order by token_address, id -> Incremental Sort presorted, 14 ms a OFFSET 3000
    Trier sur la pkey pendant qu'on filtre sur token_address force le planner a
    materialiser tout le resultat : c'est ce qui tuait le monitor.
    """
    out, step, off = [], 1000, 0
    while True:
        q = sb.table(tbl).select(sel)
        for k, v in f.items():
            if k.startswith("gte_"): q = q.gte(k[4:], v)
            elif k.startswith("eq_"): q = q.eq(k[3:], v)
            elif k.startswith("in_"): q = q.in_(k[3:], v)
        # .range() sans ordre stable = pages non deterministes (doublons/trous)
        for col in order:
            q = q.order(col)
        r = _execute(q.range(off, off+step-1))
        if not r.data: break
        out.extend(r.data)
        if len(r.data) < step: break
        off += step
    return out


def _closed(rows):
    return [r for r in rows
            if r.get("status") in CLOSED_STATUSES
            and not str(r.get("strategy", "")).startswith("DTRAIL")]


def _emit(path_payload, gh_lines):
    """Ecrit l'artefact JSON + les outputs GH. Toujours appele, meme sans live."""
    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "nightly_outliers.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(path_payload, f, indent=2, default=str)
    print(f"  saved -> {out_path}")

    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as gh:
            for line in gh_lines:
                gh.write(line + "\n")
            # Marqueur de fin de parcours : distingue "0 outlier" de "crash".
            gh.write("monitor_status=completed\n")


def main():
    print(f"Nightly outlier monitor — window since {SINCE}")

    # 1) Cote live d'abord : il borne tout le reste (filtre SQL, pas en Python).
    live_rows = fetch_all("paper_trades", COLS, gte_created_at=SINCE, eq_source="rt_live")
    live = _closed(live_rows)
    print(f"  closed live trades: {len(live)}")

    # Live coupe depuis le 05/06 => aucune paire possible. Tirer le cote paper
    # scannerait toute la grille shadow (~64 k lignes) pour construire 0 paire.
    if not live:
        print("\nNo live trades in window — nothing to compare. (live trading off)")
        _emit({
            "window_since": SINCE, "threshold_pp": THRESHOLD_PP,
            "n_live": 0, "n_pairs": 0, "sim_live_median_pp": None,
            "outliers_synced": [], "outliers_unsynced_count": 0,
            "outliers_mev_pump_count": 0, "note": "no live trades in window",
        }, ["sync_true_count=0", "sync_false_count=0", "pairs=0"])
        return

    # 2) Cote paper restreint aux tokens vus en live (chunk : longueur d'URL).
    tokens = sorted({r["token_address"] for r in live if r.get("token_address")})
    paper_rows = []
    for i in range(0, len(tokens), 100):
        paper_rows += fetch_all("paper_trades", COLS, order=("token_address", "id"),
                                gte_created_at=SINCE,
                                in_token_address=tokens[i:i+100])
    paper = [r for r in _closed(paper_rows) if r.get("source") != "rt_live"]
    print(f"  closed paper trades on those {len(tokens)} tokens: {len(paper)}")

    pairs = defaultdict(dict)
    for r in live:
        pairs[(r["token_address"], r["strategy"])]["live"] = r
    for r in paper:
        pairs[(r["token_address"], r["strategy"])]["paper"] = r

    matched = [(v["live"], v["paper"]) for v in pairs.values() if "live" in v and "paper" in v]
    print(f"  matched pairs: {len(matched)}")

    sim_diffs = []
    for lv, _ in matched:
        if lv.get("paper_sim_pnl_pct") is not None:
            d = (float(lv.get("pnl_pct") or 0) - float(lv["paper_sim_pnl_pct"])) * 100
            sim_diffs.append(d)
    if sim_diffs:
        print(f"  sim-live diff median: {st.median(sim_diffs):+.2f}pp  "
              f"mean: {st.mean(sim_diffs):+.2f}pp  max|{max(abs(d) for d in sim_diffs):.2f}|pp")

    outliers_synced = []
    outliers_unsynced = 0
    outliers_mev_pump = 0  # v144.19: expected Jupiter Ultra positive-slip edge
    for lv, pp in matched:
        pnl_lv = float(lv.get("pnl_pct") or 0)
        pnl_pp = float(pp.get("pnl_pct") or 0)
        dpp = (pnl_lv - pnl_pp) * 100
        if abs(dpp) <= THRESHOLD_PP:
            continue
        # v144.19: skip tp_hit/tp_hit where live caught a MEV pump above paper's
        # clean TP exit. This is expected execution edge (positive Jupiter Ultra
        # slippage on pumps, documented v144-11-live-paper-divergence.md), NOT a
        # logic bug. Still alerts on: opposite statuses, paper > live (sim
        # over-estimate), or any non-TP tp/sl asymmetry.
        if (lv.get("status") == "tp_hit" and pp.get("status") == "tp_hit"
                and pnl_lv > pnl_pp > 0):
            outliers_mev_pump += 1
            continue
        synced = (lv.get("entry_source") == "live_sync") or (pp.get("entry_source") == "live_sync")
        if synced:
            outliers_synced.append({
                "symbol": lv["symbol"], "strategy": lv["strategy"],
                "pnl_live": round(pnl_lv * 100, 2),
                "pnl_paper": round(pnl_pp * 100, 2),
                "delta_pp": round(dpp, 2),
                "status_live": lv["status"], "status_paper": pp["status"],
                "exit_at": lv.get("exit_at"),
                "is_pump": lv.get("rt_is_pump_fun"),
            })
        else:
            outliers_unsynced += 1

    print(f"\n  outliers |L-P|>{THRESHOLD_PP}pp : "
          f"{len(outliers_synced)+outliers_unsynced+outliers_mev_pump} total")
    print(f"    sync=False (expected, historic):  {outliers_unsynced}")
    print(f"    MEV-pump tp/tp (expected edge):   {outliers_mev_pump}")
    print(f"    sync=True  (BUG SIGNAL):          {len(outliers_synced)}")

    gh_lines = [
        f"sync_true_count={len(outliers_synced)}",
        f"sync_false_count={outliers_unsynced}",
        f"pairs={len(matched)}",
    ]
    if outliers_synced:
        first = outliers_synced[0]
        gh_lines += [
            f"first_symbol={first['symbol']}",
            f"first_strategy={first['strategy']}",
            f"first_delta={first['delta_pp']}",
        ]

    _emit({
        "window_since": SINCE,
        "threshold_pp": THRESHOLD_PP,
        "n_live": len(live),
        "n_pairs": len(matched),
        "sim_live_median_pp": st.median(sim_diffs) if sim_diffs else None,
        "outliers_synced": outliers_synced,
        "outliers_unsynced_count": outliers_unsynced,
        "outliers_mev_pump_count": outliers_mev_pump,
    }, gh_lines)

    for o in outliers_synced:
        print(f"    [SYNC=TRUE] {o['symbol']} {o['strategy']} L={o['pnl_live']:+.1f}% "
              f"P={o['pnl_paper']:+.1f}% Δ={o['delta_pp']:+.1f}pp "
              f"statL={o['status_live']} statP={o['status_paper']}")

    if outliers_synced:
        print(f"\n::error::Found {len(outliers_synced)} sync=True outlier(s) — post-v143.5 logic bug signal")
        sys.exit(2)

    print("\nNo sync=True outliers. Alignment holding.")


if __name__ == "__main__":
    main()
