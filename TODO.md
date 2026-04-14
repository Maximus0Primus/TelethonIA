# TODO

## Latence live trade (Plan A — msg→buy delay)

Contexte: buys live entrent avec 20-60s de retard sur le call KOL → front-running par bots concurrents → "slippage" apparent 20-50% (ex: $BBC call @ 10k → entré @ 14k). Voir analyse session 2026-04-14.

### Fait
- [x] **B** — fix `_rt_price_at_message` (bug clé de dict, toujours NULL)
- [x] **A.1** — instrumentation latence : logs `RT timing:` + `LIVE LATENCY:` (msg→ds, ds→pre_buy, buy_exec)
- [x] **A.3** — cache SOL price 5s + wallet balance 10s dans live_trader.py
- [x] **A.3** — vérifié qu'il n'y a pas de double fetch DexScreener (faux blocker)

### À faire
- [ ] **A.2** — analyser les logs `LIVE LATENCY:` après 24h (≥5-10 buys live) pour localiser le vrai bottleneck parmi :
  - `msg→ds` : temps entre message Telegram et fin fetch DexScreener (API externe)
  - `ds→pre_buy` : enrichment + scoring + sizing + `_rt_open_trades` → `open_live_trade` → pre-`execute_buy`
  - `buy_exec` : temps d'exécution Jupiter Ultra (signature + envoi)
- [ ] **A.2** — cibler 2-3 gros bloqueurs universels (pas de shortcut S-tier). Hypothèses :
  - `_fetch_dexscreener_by_address` (safe_scraper.py:1655) sync dans executor — voir si async HTTP client aide
  - enrichment synchrone avant buy dans `_rt_open_trades` — paralléliser ou déplacer post-buy
  - checks séquentiels dans `open_live_trade` (max_open, dedup_cooldown, min_sol_reserve) — combinable

### Données à récolter avant de coder A.2
Query type:
```sql
-- après 24h, filtrer les logs VPS: `journalctl -u kol-scraper | grep LIVE LATENCY`
-- puis moyenne/p50/p95 sur msg→ds, ds→pre_buy, buy_exec
```

## Repo hygiene
- [ ] Nettoyer commit `007db6b` qui a pushé 2400+ fichiers cache (`scraper/ohlcv_cache/`, `scraper/jupiter_candles_cache/`, `grid_search_*.csv`). Options : revert + force-push, ou ajouter au `.gitignore` et laisser.

## Data quality (depuis /check-data 2026-04-14)
- [ ] Enrichment Jupiter LDS sous-fill (35% vs seuil 70%) — vérifier quotas API / erreurs silent
- [ ] Enrichment holders sous-fill (27%) — idem
- [ ] CA resolution 71.9% (sous seuil 75%) — revoir le resolver pour les messages récents
- [ ] Backlog labels 24h = 774, 7d = 2346 — voir outcome_tracker cadence
