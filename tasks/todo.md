# Operational Backlog

> 🎯 **Strategy decisions live in `tasks/strategy_candidates.md`.** This file tracks tech infrastructure, deploys, scripts to create, and operational state — pas de strategy ranking ici.

## 🛑 Live status

`live_trading.enabled = False` depuis 2026-05-02 22:12 UTC. Backup config : `data/rt_trade_config_pre_pause_20260501T221242Z.json`. Conditions de resume documentées dans `strategy_candidates.md` (§9 critères + recette stat).

## ✅ Recently shipped (May 7-8, 2026)

- [x] Companion shadow post-promote — `paper_trader.py:1644` (commit `985a11d`)
- [x] Sim-align-gate skip clean quand N=0 (commit `d670fba`)
- [x] Cleanup 14 backup tables Supabase + 6 search_path + 10 RPC lockdown (3 migrations)
- [x] 5 v14e.57 picks ajoutés `strategies.py` (commit `4ee1cb0`)
- [x] Combo proposer enrichi axes filter (commits `80b7f0c` + `f4bf886`)
- [x] Live blacklist Option B per-chain infra (commit `25ff5de`)
- [x] Audit blacklist + DB updates (migrations `v14e57_blacklist_audit_may7` + `v14e57_split_venom_*`)

## 🔧 Operational backlog

### Tech features

- [ ] **Fix #2 verb-proximity ticker filter** — patterns `$X (whales|holders|dev) (buying|aping) $Y`. Demande N≥30 messages historiques pour calibrer.
- [ ] **Score V3 walk-forward audit** (Mai 9-16) — script à créer : `scripts/_score_v3_walk_forward.py`. Sur N≥30 trades fermés post-v14e.55. Si AUC V3 ≥ V1 + 0.015 → swap `min_rt_score` filter.
- [ ] **LOCK polling alignment** — appliquer `polling_sec=60 + median_5` aux LOCK SOL/ETH si BE25 confirme post-bump.

### Perf / monitoring

- [ ] **R2 Profiler `process_and_push`** si lag >30s revient.
- [ ] **T3 ETH round-trip smoke** — `_eth_round_trip_smoke.py --execute` mensuel ou si base_fee >5 gwei.

### Scripts à créer (référencés depuis `strategy_candidates.md` Q1)

- [ ] `scripts/_kol_blacklist_audit.py` — paired-test KOL-allowed vs banned trades.
- [ ] `scripts/_kol_per_strat_breakdown.py` — matrice KOL × strat ($/d, WR) — répond à "blacklist optimale per-famille ?"
- [ ] `scripts/_blacklist_counterfactual.py` — recompute Tier S avec/sans blacklist active. Output : delta $/d, sensitivity score.

### Cleanup différé

- [ ] **DROP `_backup_blacklist_audit_20260507`** dans 7j (vers Mai 14) si pas de rollback nécessaire — règle `sql_backup_tables_lifecycle.md`.

## 📌 Quick reference

### Slippage (single source = `strategies.py`)

- **SOL** : `BUY_SLIPPAGE_BPS = 225`
- **ETH paper** : `BUY 350 / SELL 650`. JSONB miroir : `eth_buy_slippage_bps=350`, `eth_sell_slippage_bps=650`
- **ETH live tx tolerance** : `eth_buy_slippage_bps=500, eth_sell_slippage_bps=600`

### Cron schedules (GitHub Actions)

| Workflow | Schedule | Notes |
|---|---|---|
| `mega-sweep-48h` (SOL) | 02:00 UTC tous les 2j | matrix 6 shards (~1.5-2h chacun) |
| `mega-sweep-eth-48h` | 22:00 UTC tous les 2j | single job (3-shard tient) |
| `train-models` | daily | auto-deploy ML model files |
| `sim-align-gate` | 04:00 UTC daily | skip clean si N=0 (post-fix) |
| `outcomes` (Fill Labels) | every ~2-4h | label backfill |
| `nightly-shadow-audit` | daily 07:30 UTC | |
| `nightly-outlier-monitor` | daily 07:20 UTC | |
| `daily-summary` | daily | Telegram report |
| `kol-weekly-audit` | weekly | KOL stats refresh |

### Pre-deploy gate

```bash
bash scripts/pre_deploy_check.sh
# 1/3 py_compile critical modules
# 2/3 import smoke (10 modules)
# 3/3 pytest tests/ -x (~124 tests + ~367 subtests)
```

### Deploy

```bash
git push origin master
ssh vps "cd /opt/TelethonIA && git pull origin master && systemctl restart kol-scraper"
```

### VPS service

- Service : `kol-scraper.service`
- Working dir : `/opt/TelethonIA`
- Logs : `journalctl -u kol-scraper -f` (or via `/logs` skill)
- Wallet : `9t3yNhWUV7f3EfyMAiFHrL6qDU8oT4rA9Agt8tSmBeSM`
