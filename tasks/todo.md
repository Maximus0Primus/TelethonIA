# Pipeline Status — Updated Apr 23, 2026 (v14e.11 — audit paper + refresh)

## v14e.11 — Apr 23 PM — audit paper mains + retrait 3 perdantes

Les 11 strats Solana mains auditées sur 7d + 14d N≥30 :

| Statut | Strats |
|---|---|
| ✅ Gardées (top earners) | FAST_TP50_SL30 (+$361), FAST_TP80_SL25 (+$329), TP50_SL15 (+$277), FAST_TP40_SL30 (+$256), BE25_TP80_SL30 (+$202) |
| 🟡 Gardées watch | FAST_TP100_SL20 (+$170, avg +4.2%), BE25_TP80_SL30_S30_HYST (+$83), FAST_TP50_SL30_HYST (+$48) |
| 🔴 **Retirées** | BE25_TP80_SL30_NZS30_HYST (−$32), BE15_TP70_SL50_NZ (−$46), HIGHSCORE_TP200_SL40 (−$70) |

Total évité : **−$148/7j = +$21/jour** libérés.

**17 allocations restantes** : 8 SOL mains + 3 ETH + 3 BSC + 3 BASE
```
BE25_TP80_SL30, BE25_TP80_SL30_S30_HYST, FAST_TP100_SL20,
FAST_TP40_SL30, FAST_TP50_SL30, FAST_TP50_SL30_HYST,
FAST_TP80_SL25, TP50_SL15
+ 3 ETH_* + 3 BSC_* + 3 BASE_*
```

**Pas de promotion de shadow** — les top shadows (DTRAIL10_ACT10_SL70 +10.6%, DTRAIL10_ACT30_SL50 +9.6%, DIP30_B10_T10_A20_SL60_120m +9.5%) sont famille trail/dtrail/dip = **artefact sim** (cf. `docs/known_issues.md §2`, slip 47× live vs paper). Les seuls safe candidats single-exit sont des variants kernel (MED3/LAZY/JUPITER) de mains déjà actives → attendre paired-test.

**Watch list** (rétrograde si pattern persiste 14j) :
- FAST_TP50_SL30_HYST +$48 / +1.4% → borderline, garder N≥60 avant décision
- BE25_TP80_SL30_S30_HYST +$83 / +4.0% → single filter HYST, garder mais watch
- FAST_TP100_SL20 avg 14d +3.7% (vs 7d +4.2%) — stable mais sous la médiane des top 5

---

# Pipeline Status — Updated Apr 23, 2026 (v14e.6 — 6 tâches batch)

## v14e.5 + v14e.6 — Apr 23 PM — batch P0-P5

- **P0 ✅** : `NOZEROLIQ_TP200_SL40` retiré de `hybrid_strategy.allocations`
  (N=33 sur 7j, avg −5.1%, WR 24%, PnL −$85 → seuil N≥30 + pattern perdant
  atteint). 20 strats actives désormais (12 SOL - 1 + 3 ETH + 3 BSC + 3 BASE).
- **P1 ✅** : 4 bugs sim-align ($BUZZED/$XBT/$ZACHXBT/$ACHI) déjà résolus par
  v144.19 (decontamination paper_sim_pnl_pct). Tous les 4 ont maintenant
  live vs sim <5pp. Replay drift résiduel sur be_stop historiques (ex $WIF2
  +17.61pp) = divergence code actuel vs paper_sim stocké — résoluble par
  `scripts/backfill_paper_sim_pnl_pct.py` si besoin.
- **P2 ✅** : `sim-align-gate.yml` FAIL depuis 5 jours (avg −11.33pp à cause
  de 2 MEV-pumps). Fix v14e.5: `verify_sim_live_alignment.py` tag `[MEV]`
  et exclut les rows `tp_hit/tp_hit + live > paper > 0 + |diff| > 50pp` du
  gate metric (mirror nightly_outlier_monitor v144.19). Bonus: fixé bug
  parse bash qui double-comptait N (32/66 → 32/33). Manual run post-fix:
  avg=+1.31pp, within=100% (après exclusion MEV).
- **P3 ✅** : `docs/known_issues.md` créé (10 règles : HYST/DTRAIL/paired-test/
  family-slip-mult/MEV-pump/be_stop replay drift/chain gates/LAZY
  throttling/`_pt_ultra_override` ban/hygiène générale).
- **P4 ✅** : Audit 4 tables (paper_trades, price_ticks, token_snapshots,
  tokens) + 0 leak shape vs chain. **Leak résiduelle trouvée** : 11 orphelins
  bankroll dans `strategy_bankrolls_per_chain['ethereum']` qui étaient des
  strats Solana (résidu rollback post-cleanup). Nettoyage SQL :
  `jsonb_object_agg WHERE key LIKE 'ETH_%'` — chaque bucket EVM ne contient
  maintenant QUE ses strats. Post-fix: SOL 21 / ETH 3 / BSC 3 / BASE 3, purs.
- **P5 ✅** : Slip model refinement v14e.6. `_dynamic_sell_slip_factor` passe
  de 3 buckets (5k/20k/50k → 2.0/1.3/1.0) à courbe log-continue
  `1.0 + 0.5 × log10(50_000 / max(liq, 500))`. Motivation: éliminer les
  discontinuités 54% à 5k / 23% à 20k qui biaisaient les paired-tests sur
  les bords. Valeurs: 500→2.00, 5k→1.50 (vs 2.00 ancien, plus doux), 10k→1.35,
  20k→1.20, 50k+→1.00. Clamped [1.0, 2.5]. 4 tests dédiés (anchors, monotone,
  continuity, clamped) + compatibilité EVM branches préservée.

**Reste ouvert pour plus tard** :
- Volume-volatility component dans slip model (P5 v2) : quand N≥30 par
  (liq_band × exit_type × vol_band).
- Legacy `strategy_bankrolls` flat dict (mirror) → removal au prochain reset
  bankroll. Aujourd'hui kept for backward compat.

---

# Pipeline Status — Updated Apr 23, 2026 (v14e.4 — full multi-chain paper)

## État actuel — snapshot Apr 23 18:00 UTC

**Versions déployées depuis ce matin** :
- v14e → chain gates hard (live_trader + enrich_jupiter + safe_scraper), alertes séparées par strat, bankroll per-chain schema
- v14e.2 → BSC + Base paper strats (3 chacune), fee models per-chain, DS routing via paper_trades.chain
- v14e.2b → fix price_ticks 400 (jupiter tick rows missing `chain` column)
- v14e.3 → bot commands Telegram chain-aware (tous les /cmd acceptent `sol|eth|bsc|base|all`)
- v14e.4 → drop `min_liquidity_usd` des 9 strats EVM (fee model encode déjà le slip des pools shallow)

**Registry strats actuel (post-v14e.4)** :
- 🟣 Solana : 302 strats (12 mains + 290 shadows)
- 🔷 Ethereum : 3 strats — ETH_TP100_SL50, ETH_TP80_SL40_T2H, ETH_BE50_TP150_SL50
- 🟡 BSC : 3 strats — BSC_TP100_SL50, BSC_TP80_SL40_T2H, BSC_BE50_TP150_SL50
- 🔵 Base : 3 strats — BASE_TP100_SL50, BASE_TP80_SL40_T2H, BASE_BE50_TP150_SL50
- Filter unique : `{"chain": "<chain>"}`. Pas de min_liquidity, pas de min_score — le fee model encode le coût réel des pools.

**`hybrid_strategy.allocations` en DB** : 21 strats total (12 SOL + 3 ETH + 3 BSC + 3 BASE, chacune alloc=1).

**Bankroll per-chain seedé** : SOL 21 strats (preserves historique), ETH/BSC/Base 3 strats × $1000 chacune.

**Cleanup DB effectué (18:00 UTC)** :
- 2367 `paper_trades` supprimées : strats Solana (SCALE_OUT, MOONBAG, BE25…) ouvertes sur un token $PAXI (0xa9fd...) à 16:48 UTC, AVANT le deploy v14e à 16:57. Cause : token_entry n'avait pas `chain` → filter default solana → strats SOL passaient sur 0x token. Le bankroll Solana a été remboursé du −$111.31 artificiel par strategy (voir memory/cleanup_pollution_v14e2.md si créé).
- Ticks (382 ETH, 52520 SOL) conservés — vraie market data.
- Post-cleanup: 0 trades ETH/BSC/Base encore (attend un KOL call qui résolve vers ces chains).

**Live trading (Solana-only)** :
- 4 strats actives : BE25_TP80_SL30 + FAST_TP50_SL30 + FAST_TP80_SL25 + BOND_FAST_TP50_SL20_T20
- BE25 : 7 jours verts consécutifs (14-22 Apr) + 1er rouge aujourd'hui (23 Apr, N=5 −$0.85, WR 20%)
- Market-wide rouge aujourd'hui (paper main Solana −$327 N=52 WR 25%) → pas une régression du setup, mauvais jour système

**Live ETH/BSC/Base** : NotImplementedError stubs. Phase 2 ETH conditionnée à WR≥65% + EV≥+10%/trade à N≥50 (ETA Mai 07). BSC/Base attendent décision Phase 2 ETH.

**Décisions tranchées** :
- ~~allocations split per-chain en DB ?~~ → NON, le naming prefix ETH_/BSC_/BASE_ + registry CHAIN_STRATEGIES suffit
- ~~min_liquidity_usd sur strats EVM ?~~ → NON, retiré v14e.4, fee model suffit
- ~~revert chain='ethereum' backfill des 2367 rows polluées ?~~ → delete définitif, bankroll remboursé

**À surveiller** :
- Prochain KOL call EVM → vérifier que les 3 strats (et **seulement** les 3) de la bonne chain s'ouvrent
- Drift live vs paper sur BE25 si 2-3 jours rouges consécutifs → signal pour ajuster
- `SELECT chain, COUNT(*), SUM(pnl_usd) FROM paper_trades WHERE source='rt' AND is_shadow=false GROUP BY chain` — validation isolation à J+3

---

## v14e.2 — Apr 23 PM — BSC + Base paper strats live

User wants symmetric paper trading for BSC + Base (same 3 strategies as ETH).
Alerts must clearly tag the chain on every KOL trade / close with correct
DEX + block-explorer links. Solana must stay untouched.

**Done** :
- `scraper/strategies.py` — 3 BSC strats + 3 Base strats added (TP100/SL50, TP80/SL40, BE50/TP150). Fee constants `BSC_*` ($0.30 gas, 250 bps) and `BASE_*` ($0.10 gas, 150 bps) parallel to `ETH_*`. CHAIN_STRATEGIES registry now shows 302 SOL / 3 ETH / 3 BSC / 3 BASE — zero leakage asserted.
- `scraper/paper_trader.py` — consolidated fee branches: `_EVM_FEE_PARAMS` + `_evm_slip_bps_with_gas(pos, chain, side)` + `_evm_min_position_usd(chain)` replace the ETH-only override. Entry slip, shadow slip, exit slip, min-position, Jupiter skip — all cover ETH/BSC/Base uniformly. Solana path unchanged (Jupiter Ultra RFQ + `_dynamic_sell_slip_factor` legacy).
- `scraper/paper_trader.py::_fetch_prices_batch` — accepts `chain_by_addr` map. Without it, 0x addresses fall to ethereum (the "dexscreener token pair not found" cause when CA was BSC/Base). `check_paper_trades`, `check_paper_trades_fast`, `correct_closed_prices`, `live_trader.check_live_trades`, `bot_commands.cmd_positions_live` all pass the map built from `paper_trades.chain`.
- `_log_price_ticks` accepts `chain_by_addr` too so price_ticks.chain rows are correct on BSC/Base tokens.
- `scraper/enrich.py::_fetch_dexscreener_by_address` — chain support for BSC (PancakeSwap V3 > V2 > Biswap) and Base (Uniswap V3 > Aerodrome). Address-shape sanity check widened to all EVM chains.
- `scraper/safe_scraper.py::_rt_open_trades` handler — 0x CA now disambiguated via `resolve_evm_chain(ca)` (DexScreener chainId lookup), cached in `_rt_evm_chain_cache`. Eliminates silent mislabel where a BSC token was queried against ETH endpoints.
- `scraper/alerter.py::alert_kol_trade` — BSC adds PancakeSwap + BscScan links; Base adds Uniswap + BaseScan. Every trade open alert wears `chain_tag` (🟣SOL / 🔷ETH / 🟡BSC / 🔵BASE).
- `scraper/alerter.py::alert_trade_closed` — per-chain explorer link appended next to DexScreener.
- `scraper/safe_scraper.py::alert_kol_trade call site` — per-strategy positions filtered by `_passes_strategy_filter` on the token's chain. Strats of other chains no longer appear in the alert (fix of user complaint: "alertes mélangées avec les différentes stratégies paper").
- Supabase migration v14e_bankroll_per_chain applied. 9 new bankroll entries seeded at $1000 each (3 ETH + 3 BSC + 3 BASE). `rt_trade_config.hybrid_strategy.allocations` updated — 21 strategies total (12 SOL + 3 ETH + 3 BSC + 3 BASE).
- Tests: 97/100 pass (3 skipped — pre-existing pipeline skips). `test_paper_trader.TestFetchPricesBatch` made deterministic (cache reset). New smoke checks confirm Solana alert template unchanged and BSC/Base alerts carry the right links.

**Verified non-regression on Solana** :
- 302 SOL strats kept in registry — same count as before v14e
- Solana tokens still pass Solana strats, still reject ETH/BSC/Base strats
- `alert_kol_trade(..., chain='solana')` still outputs `dexscreener.com/solana/{ca}` with bonding pump.fun fallback
- `_fetch_prices_batch` default shape-inference preserves pre-v14e behaviour for callers that don't pass `chain_by_addr`
- Jupiter Ultra path (paper entry + live) gated by `chain == "solana"` — ETH/BSC/Base skip it deterministically

**Next** (when data flows) :
- Monitor N≥30 per chain in `paper_trades` at 7-day horizon
- Validate no cross-chain bankroll drift: `SELECT chain, SUM(pnl_usd) FROM paper_trades WHERE source='rt' AND is_shadow=false GROUP BY chain`
- BSC/Base KOL discovery: today relies on existing Solana KOLs happening to post 0x addresses. If zero calls in 3 days, add chain-specific KOL groups.

---

# Pipeline Status — Updated Apr 23, 2026 (v14e — chain isolation hardening)

## v14e — Apr 23 PM — hard chain isolation

Regression fix + architectural hardening. Three user-reported symptoms:
1. Jupiter 400 Bad Request storms — ETH `0x...` mints reaching `/ultra/v1/order` because v14b promoted ETH strats to main paper without a chain gate on live_trader.
2. Telegram alerts mixing all strategies' bankrolls into every single trade close.
3. Bankroll / strategies not isolated per chain — BSC/Base rollout blocked.

**Done** :
- `scraper/live_trader.py` — `_is_solana_mint` gate at `execute_buy` / `execute_sell` / `open_live_trade`. The 400 storm stops here.
- `scraper/enrich_jupiter.py` — defence-in-depth 0x reject in `fetch_ultra_quote_price` + `fetch_ultra_sell_quote_price`.
- `scraper/safe_scraper.py` — `_rt_open_trades` resolves `chain` once, propagates on `token_entry`, skips the live branch entirely for non-Solana (paper-only until Phase 2 ETH greenlit).
- `scraper/alerter.py` — one `chain_tag()` helper (🟣SOL / 🔷ETH / 🟡BSC / 🔵BASE), used by every trade alert. Bankroll block in `alert_trade_closed` / `alert_live_buy` / `alert_live_sell` now scopes to THIS strategy only — no more cross-strategy dump. 24h drift block scoped to this strategy too.
- `supabase/migrations/v14e_bankroll_per_chain.sql` — widen `chain` CHECK to allow bsc/base, add `strategy_bankrolls_per_chain` JSONB (nested by chain), backfill from flat dict by ETH_/BSC_/BASE_ naming heuristic, add `risk_limits_per_chain` for per-chain daily_loss_limit + max_open_positions.
- `scraper/safe_scraper.py` — `_rt_strategy_bankrolls_for_chain(row, chain)` reader with legacy fallback, `_rt_update_bankroll(..., chain=)` writes to both new nested and legacy flat dict.
- `paper_trader.py` + `live_trader.py` — 4 call sites rewired to pass chain + scope alert bankroll to chain bucket.
- `scraper/strategies.py` — `CHAIN_STRATEGIES` registry built at import + `strategies_for_chain(chain)`. Partition post-v14e.2: 302 solana / 3 ethereum / 3 bsc / 3 base.
- `scraper/chain_detect.py` — `resolve_evm_chain(addr)` disambiguates 0x via DexScreener chainId (ETH/BSC/Base share the same 0x+40hex shape).
- `scraper/live_trader_eth.py` + `live_trader_bsc.py` + `live_trader_base.py` — explicit `NotImplementedError` stubs so a misrouted call fails loud, not silently.
- `scraper/tests/test_live_trader.py` — regression tests: `test_rejects_eth_mint` for buy + sell + `TestOpenLiveTradeChainGate` (14/14 pass). All existing chain_detect + pipeline_eth tests still green (38/38).

**Applied Apr 23** ✅ :
- Migration `v14e_bankroll_per_chain.sql` exécutée sur Supabase (CHECK widened, `strategy_bankrolls_per_chain` column, `risk_limits_per_chain` column, backfill from flat dict)
- Code pushed + VPS restart (commits c3173d6 → 82bb143)

**Decision tranchée (v14e.2)** : allocations restent un flat dict. BSC_/BASE_ naming prefix + `_passes_strategy_filter` chain gate + CHAIN_STRATEGIES registry suffisent pour l'isolation. Pas de refactor DB nécessaire.

---

# Pipeline Status — Updated Apr 23, 2026 (v14b — ETH paper mains live)

## Sprint #ETH-1 — ETH L1 paper mains (Phase 1 LIVE, zero capital)

**État : ✅ Phase 1 déployée Apr 23** — 3 strats ETH paper mains avec alertes Telegram identiques à Solana. Collecting data.

**Stack déployée** :
- Migration `v14_chain_column.sql` appliquée Supabase : colonne `chain TEXT NOT NULL DEFAULT 'solana'` sur 5 tables + indexes compound (chain, token_address/symbol)
- `scraper/chain_detect.py` + 25 tests : détection 0x vs base58, rejet tx hashes, normalisation lowercase ETH
- `pipeline.extract_tokens` scanne `ETH_CA_REGEX` **en plus** du Solana base58 — tag chain dans le ca_cache
- DexScreener chain-parameterized : `/tokens/v1/{chain}/{address}`, ranking DEX spécifique (Uniswap V3 > V2 > Sushi sur ETH)
- Enrichers Solana-only (RugCheck, Helius, Jupiter, Bubblemaps, outcome OHLCV) skip 0x silencieusement
- Paper trader : fee model ETH ($7.50 gas/side + 200bps MEV), `position_usd=$200` forcé (cohérence fee accounting), branche chain dans `_dynamic_sell_slip_factor` + `_override_exit_with_ultra_quote`
- `_passes_strategy_filter` : chain gate strict — strat sans `filt["chain"]` = solana-only implicite (ETH doit déclarer)
- Alertes Telegram `alert_kol_trade` + `alert_trade_closed` chain-aware : tag 🔷ETH, URL `dexscreener.com/{chain}/`, links Uniswap + Etherscan pour ETH
- `scoring_config.rt_trade_config.hybrid_strategy.allocations` : 21 strats (12 Solana + 3 ETH + 3 BSC + 3 BASE à alloc=1) — post-v14e.2
- 13 tests ETH pipeline + fee model + filter chain gate : 38/38 pass

**3 strats ETH paper mains actives (depuis commit 9635e65)** :
- `ETH_TP100_SL50` : TP 100% / SL 50% / timeout 4h — let-it-run
- `ETH_TP80_SL40_T2H` : TP 80% / SL 40% / timeout 2h — conservateur
- `ETH_BE50_TP150_SL50` : BE +50%, TP 150%, SL 50% — pour KOLs big moves
- **v14e.4** : `min_liquidity_usd=25_000` retiré de ces 3 filters. Le fee model ($7.50 gas + 200 bps MEV, amorti sur position virtuelle $200) encode le coût réel des pools shallow. Chain gate seul.

**Hypothèses à valider (N≥50 calls sur 2-3 semaines)** :
- WR ≥ 65% (vs ~50% Solana)
- EV net après frais $15/trade positif à $200/pos
- **Abandon si WR < 55% ou EV net < +5%/trade**

**Phase 2 — décision à N≥50 / 14 jours (ETA Mai 07)** :
- Si WR ≥ 65% AND EV net ≥ +10%/trade @ $200 → Phase 3 (dev live Uniswap V3 + Flashbots Protect)
- Sinon → archive, reste 100% Solana

**Phase 3 — live ETH (PAS lancée, conditionnée Phase 2)** :
- `live_trader_eth.py` séparé : web3.py + Uniswap V3 SwapRouter02
- MEV Protect RPC obligatoire (`rpc.flashbots.net` ou `rpc.mevblocker.io`)
- Wallet EVM séparé du Solana, bankroll distincte $500-1000
- Position min $200/trade

**Risques monitoring Phase 1** :
- Si aucun call ETH détecté en 3j → vérifier que les KOLs postent bien des 0x (sinon pivot vers détection CA par URL Etherscan/Uniswap)
- Si WR très bas dès N=10 → claims KOL étaient trompeurs (exit Phase 2 early)
- MEV 2026 prend 2-5% sans protection — si ça passe >6% le modèle $15 gas est sous-estimé

---

## v144.19 Apr 23 — alert noise reduction + sim-align fix

**Done (committed + deployed)** :
1. **API health Telegram alerts désactivées** (`scraper/safe_scraper.py:524-525`) — miroir de v144.17 pour `api_errors`. Fill rates toujours loggés, juste plus d'alerte.
2. **`paper_sim_pnl_pct` contamination fix** (`live_trader.py:1213-1221`) — retiré `_pt_ultra_override` qui capturait le fill Jupiter live au lieu d'une vraie ref sim pure. Résout les faux drift +148pp sur pumps ($MHGA, $8) détectés par sim-align-gate.
3. **Nightly outlier monitor MEV-pump filter** (`scripts/nightly_outlier_monitor.py`) — skip les paires tp_hit/tp_hit où live > paper > 0 (edge positive-slip attendue), comptées dans `outliers_mev_pump_count`. Les vraies alertes (statuts opposés, paper > live) continuent.

---

## v144.19b Apr 23 — shadow audit nightly CI + KOL tick quality

**Done** :
- `.github/workflows/nightly-shadow-audit.yml` (05:00 UTC) : `verify_shadow_main_parity.py` en gate dur (alerte Telegram si régression v144.3, tolérance 5 rows ou 0.1% de N) + `paired_all_v144_shadows.py` en artefact info.
- `scripts/kol_tick_quality.py` : leaderboard KOL par qualité intrinsèque du call (win-rate path-dependent +10% avant -20% sur price_ticks, indépendant de TP/SL/timeout). Top sur 30j : `gubbinscalls` 92.9% WR N=14.
- Backfill `paper_sim_pnl_pct` sur 49,746 lignes historiques completed (exit 0).

---

# Pipeline Status — Apr 22, 2026 (v144.15 — 4 live strats A/B)

## Current state

**Live (4 strats)** — Allocations dans `rt_trade_config.live_trading.allocations` :
- `BE25_TP80_SL30` : alloc 0.5 (median_5/240s, base size ~$1.70/trade) — champion courant, 6/6 jours verts live
- `FAST_TP50_SL30` : alloc 0.5 (median_3/30s + LAZY, ~$1.70/trade)
- `FAST_TP80_SL25` : alloc 0.5 (ds/30s, ~$1.70/trade) — **NEW v144.15** : +10.14% paper 7d N=94, single-exit crédible (R:R 3.2:1), Live>Paper attendu +5pp → cible ~+15%/trade live
- `BOND_FAST_TP50_SL20_T20` : alloc 0.5 (hyst/60s, ~$1.70/trade) — **NEW v144.15c** : niche bonding (`max_liquidity_usd=3000`, filtre vérifié 26/26 liq=0), +23.86% paper 7d N=26 WR 50%, orthogonal aux autres (pas d'overlap). Full size — filtre auto-throttle (1-2 trades/j max), $1.70 sur pool $5-15k = 0.01-0.03% impact = négligeable

Position base `max_position_sol=0.02` (~$3.40 plein). **max_open_positions: 12** (v144.15b — bumped from 6 pour garder ratio 3 slots/strat avec 4 strats). Daily loss limit 0.5 SOL (~$85).

**NOT live** (shadow-only) : `DTRAIL10_ACT15_SL70` (paper −$91/j/15j), `BE15_TP100_SL50` (retirée v144.12 — avg +0.30% R:R mauvais), `DTRAIL3_ACT10_SL70`, et toutes les variantes v144.x.

## v144.15 deployed Apr 22 — live A/B expansion (BE25 + FAST_TP50 + FAST_TP80 + BOND_FAST)

### Rationale
- **BE25 seule = concentration risque** : 6/6 verts (+$13.90 live) mais N=59 seulement sur 6 jours. Seule strat crédible doit pas être seule.
- **FAST_TP80_SL25** : meilleur R:R du paper (TP 80% / SL 25% = 3.2:1), N=94 sur 7j, WR 39%, +10.14% avg. Aucune structure sim-risky (pas de trail, pas de HYST, pas de BE). Si Live>Paper +5pp se tient → ~+15%/trade en live = potentiellement meilleur que BE25.
- **BOND_FAST_TP50_SL20_T20** : +23.86% paper N=26 WR 50% sur pump.fun bondings (liq=0). Filtre `max_liquidity_usd=3000` vérifié → **aucun overlap** avec les 3 autres strats (qui prennent tokens migrés/indexés). Size réduite 60% car slippage pump.fun bonding incertain.

### ❌ Rejetés pour le live A/B (artefacts sim)
- `FAST_TP50_SL30_LAZYMED` (+16.05% paper) — LAZY kernel = sim bias (cf. `hyst_artifacts_apr20.md`)
- `FAST_TP100_SL20_COMBO` (+14.44%) — COMBO multi-price-source = artefact, +0.8pp vs base = bruit
- `BE25_TP80_SL30_DS` (+16.47% paper vs +13.66% live BE25) — N=22 trop faible, +2.8pp non-significatif. Reste en shadow, paired-test vs BE25 à N≥50.
- `DTRAIL10_ACT15_SL70` (paper +17.28% / live −3.87%) — gap 21pp confirmé artefact sim
- `TP50_SL15` (+9.62% paper) — SL ultra-tight 15%, sim exagère hit rate

### Decision rules (semaine 1-2 monitoring)
- Si `FAST_TP80_SL25` live >= +12% avg après N≥20 → scale-up full size, candidat substitute pour FAST_TP50
- Si `BOND_FAST` live >= +15% après N≥15 → scale à alloc 0.5 (full size)
- Si `FAST_TP80` ou `BOND_FAST` live <= +3% ou < 0 → retirer, retour à 2 strats
- Paired-test `BE25_DS` vs `BE25` shadow : attendre N≥50 avant décision config swap

### Monitoring
- `scripts/recap_daily.py` : PnL $/j par strat (toutes les 24h)
- `scripts/verify_sim_live_alignment.py` : drift live vs paper_sim_pnl_pct (gate: mean<-3pp ou |med|>5pp avec N≥5 = exit 2)
- Alerts Telegram existantes enrichies per-strategy (v144.11)

**Paper hybrid — 12 mains + 294 shadows** (300 distinct strats tradées last 14d). Alignment audit (`verify_shadow_main_parity.py`): **0 violations sur 805 shadows post-v144.3**.

**Jupiter Trigger V2 — DÉSACTIVÉ (Apr 21, v144.14)**. `trigger_orders_enabled=false` en DB. Raison : risque de perdre le positive slippage Jupiter Ultra (+5pp/trade observé sur FAST live vs paper_sim). Re-activable ponctuellement pour TP200 cluster (TP/SL 100% static) après validation à $10+ sur polling. Détails : `v144-14-trigger-disabled.md`.

---

## v144.6-9 deployed Apr 21 (sim alignment overhaul)

### v144.6 — Fix LAZY throttling for live_sync shadows
Nightly_outlier_monitor a flaggé 4 outliers sync=True post-v144.3 (ASMORA +21pp, SAEP +25pp, TRUST x2). Cause : v144.3 a retiré le shortcut `if pos_usd==0: return True` dans `_should_evaluate_exit`, donc les paper rows `entry_source="live_sync"` (v142E shadow-sync) se sont retrouvées LAZY-throttled (180-600s) alors qu'elles doivent mirror la cadence live (30s). Fix : bypass LAZY quand `entry_source="live_sync"`. Shadows A/B purs gardent LAZY.

### v144.7 — Sim-align gate via eval_history (not price_ticks)
`sim-align-gate.yml` fail chronique 3 jours (Apr 19-20-21). Root cause : `verify_sim_live_alignment.py` reconstruisait l'input prix depuis `price_ticks` qui sample Jupiter à 3-min batch vs live 30s polling. Tokens hors rotation active → 0% coverage Jupiter → sim fallback `timeout_eod` bidons. Fix : switched to `paper_trades.eval_history` JSONB (v138+, chaque poll persisté), replay via `sim._replay_from_eval_history`. **avg=-3.78pp → -1.16pp** (3.3× mieux).

### v144.8 — Gate compares replay vs paper_sim_pnl_pct (apples-to-apples)
Encore des "divergences" trompeuses parce que le gate comparait sim_replay vs live.pnl_pct, et live.pnl_pct inclut le fill Jupiter Ultra réel (slippage positif sur spikes, ex: $CHUCHU TP=+50% fill=+120%). Fix : compare vs `paper_sim_pnl_pct` (colonne v143.6 persistée par live_trader.py:1174 — "ce que paper aurait book avec le même input"). Colonne "Jup slip" ajoutée en info. **avg=-1.16pp → -0.61pp**, max Jup slip ±0.5pp typique confirme Ultra RFQ near-zero. Aussi migré `scripts/diverge_report.py` pour préférer eval_history.

### v144.10 — 10 new shadows from EH A/B (hidden gems)
Le Spearman ρ=0.058 entre PT et EH sweeps confirme le biais structurel de price_ticks. 10 shadows ajoutées depuis les rankings EH propres :
- **7 nouvelles strats** dans STRATEGIES (TP200/TP150 cluster, rank EH 46-113) : `BE25_TP200_SL40_4H`, `TP200_SL30_2H`, `BE50_TP200_SL30_4H`, `TP200_SL30_4H`, `TP200_SL40_2H`, `TP200_SL50_4H`, `TP150_SL40_2H`
- **3 existantes** promues en shadow (MOONBAG, WIDE_RUNNER, SCALE_OUT — let-it-run profile, WR 60.9% med +8.58% sur SCORE30 subset)
- Skipped : HYST variants (v142 redundant), DIP30/DTRAIL (artifacts live), dupes TP300/500_SL50 (weak median)

ETA verdict paper paired : **Apr 28-Maj 02** (N≥30 paired vs base attendu)

### v144.9 — mega_sweep A/B price_ticks vs eval_history
Le mega_sweep (discovery de strats, dernier output = BE25_S35 + FAST_TP100_S35 v144.4/5) lisait `price_ticks` → même biais structurel 3-min Jupiter. Deux patches :
- **A (minimaliste)** : warning coverage dans `_mega_sweep_run`. Affiche `median jup ticks/token`, `% zero_jup`, `% <10_jup`. Alerte si >15% zero_jup ⇒ résultats biaisés DS fallback.
- **B (propre)** : nouveau flag `--mega-sweep-eval-history`. Universe = tokens tradés avec `eval_history`. Source forcée à jupiter (eval_history n'a pas de DS stream). Output `_mega_sweep_eh.csv`.

Usage A/B :
```
python scraper/sim.py --mega-sweep                  # legacy price_ticks
python scraper/sim.py --mega-sweep-eval-history     # ground truth
# Compare rankings; strats avec delta rank ≥ 5 = suspectes.
```

---

## v144.x deployed Apr 20

### v144.1 — 4 retraits HYST/DS losers from hybrid
Pair-test 7d (N=38-69) :
- FAST_TP80_SL25_HYST (−$62 vs base +$427)
- FAST_TP100_SL20_HYST (−$54 vs base +$137)
- BE25_TP80_SL30_HYST (+$6 vs base +$191)
- BE25_TP80_SL30_DS (−$0 vs base +$191)

### v144.2 — Bug routing paper FAST_TP50/BE25
Root cause: `paper_trader.py` open/cooldown queries n'excluaient pas `source='rt_live'` → live row bloquait paper sibling. Fix : 3 queries patchées avec `.neq("source", "rt_live")`. Avant fix, FAST_TP50 paper stoppé 32h, BE25 paper stoppé 52h.

### v144.3 — Shadow ↔ main parity
3 changements pour aligner shadows sur mains (zéro biais A/B) :
1. `_should_evaluate_exit` : LAZY throttling appliqué aux shadows aussi
2. `_override_exit_with_ultra_quote` : Ultra SELL quote sur shadows (legacy pos=0 bypass auto)
3. Shadow row creation : `position_usd = alloc_usd × tranche_pct × bot_ml_mult` (= main), entry_source tagué, ML gate appliqué

Cosmétique préservé : telegram alerts + bankroll updates restent skippés via `is_shadow=True`.

### v144.4 — `FAST_TP100_SL20_S35` shadow (top robust)
Top robust cluster sim (`analyze_mega_sweep.py` Bonferroni × 508K) : N=35, WR 62.86%, avg +28.06%, fdr_q≈0. Orch : LAZY + median_3 + jupiter.

### v144.5 — `BE25_TP80_SL30_S35` + LAZY_STRATEGIES cleanup
Sweet-spot SCORE35 sur BE25 (extrapolation FAST_TP100_S35). LAZY_STRATEGIES nettoyé : retiré 4 entrées qui référençaient des mains supprimées par v144.1.

---

## 12 Mains actives (post v144.1) — état 7d

| Strat | $/jour | Note |
|---|---|---|
| FAST_TP80_SL25 ⭐ | +$45 | top earner paper |
| FAST_TP50_SL30 (live) | +$53 | top + en live |
| BE25_TP80_SL30_S30_HYST 🚀 | +$44 | WR 56% |
| TP50_SL15 | +$40 | simple, robuste |
| HIGHSCORE_TP200_SL40 | +$35 | asymétrique |
| FAST_TP40_SL30 | +$34 | |
| BE25_TP80_SL30 (live) | +$30 | |
| FAST_TP100_SL20 | +$11 | |
| BE25_TP80_SL30_NZS30_HYST | +$8 | N=17 |
| FAST_TP50_SL30_HYST | +$8 | watch |
| BE15_TP70_SL50_NZ | +$6 | N=22 |
| NOZEROLIQ_TP200_SL40 | −$8 | 🔴 perdant N=18, retirer si pattern persiste |

**Paper 14d actualisé (Apr 21, v144.12) — les 3 strats historiquement "live":**
| Strat | N 14d | Avg% | WR% | $/jour | statut |
|---|---|---|---|---|---|
| FAST_TP50_SL30 | 218 | +1.94% | 41.3% | +$19.19 | live ✅ |
| BE25_TP80_SL30 | 83 | +8.20% | 36.1% | +$48.62 | live ✅ |
| BE15_TP100_SL50 | 226 | +0.30% | 21.2% | +$11.04 | retirée live (avg trop faible, WR 21% mauvais R:R) |

**TOTAL paper 7d : ~+$2027 = +$290/jour** (positions $50/trade).

---

## Live 7d actual (avant swap v144.1)

- BE25_TP80_SL30 : N=38, WR 42%, +$4.90 → +$0.70/jour
- FAST_TP50_SL30 : N=66, WR 41%, +$1.16 → +$0.17/jour
- (legacy DTRAIL/BE15 résiduels) : −$0.30/jour
- **Total live : +$0.58/jour**, projection post-swap **+$1.4/jour**

---

## 🧪 Shadows v144.x — verdicts en attente data

| Dim | Shadows | ETA verdict |
|---|---|---|
| **NOLAZY paired** (4) | FAST_TP40/50/80, TP50_SL15 | Apr 23-25 N≥30 paired |
| **Source BOTH/JUPITER** (8) | FAST_TP40/50/80/100, BE25 | Apr 25-27 |
| **Smoothing DS/MED3** (8) | FAST_TP40/50/80/100, TP50_SL15 | Apr 25-27 |
| **SCORE filter S35/S40/S30** (10) | BE25, FAST_TP50/80/100, TP50_SL15 | Apr 25-30 |
| **MCAP_S40 / COMBO** (5) | sur top earners | Apr 25-30 |
| **LAZY cadence FAST/MED/SLOW/XSLOW** (4) | FAST_TP50_SL30 only | Apr 25-27 |
| **LAZYSLOW** (3) | FAST_TP50/80, BE25 | Apr 25-27 |
| **HIGHSCORE_*_BOTH/DS/MED3/NOLAZY** (4) | nouveaux v144.2 | Apr 27-30 |
| **v144.10 TP200/TP150 cluster** (7) | BE25_TP200_SL40_4H, TP200_SL30_2H/4H, BE50_TP200_SL30_4H, TP200_SL40_2H, TP200_SL50_4H, TP150_SL40_2H | Apr 25-27 (launch 2026-04-21 09:25, couverture paired **100%** vs REF depuis, rate ~7 trades/j) |
| **v144.10 let-it-run** (3) | MOONBAG, WIDE_RUNNER, SCALE_OUT | Apr 28-Maj 02 |

**Règle** : N≥30 paired (pas raw) avant promotion. Re-run `paired_all_v144_shadows.py` quotidien.

---

## 📋 Reste à faire

### ⏳ Data wait (laisser tourner)
- **ETH Phase 1 N≥50 / 14j** (ETA Mai 07) — verdict go/no-go live ETH. Monitor via : `SELECT strategy, COUNT(*), AVG(pnl_pct)*100 FROM paper_trades WHERE chain='ethereum' AND status != 'open' GROUP BY 1;`
- Verdicts paired shadows v144.x (Apr 23-30)
- Slip per-cell N≥15 sur pump×tp_hit + non-pump×* (Apr 25)
- Validation FAST_TP100_SL20_S35 paper paired vs base (sim dit +28%/trade)
- Validation BE25_TP80_SL30_S35 paper paired vs base
- LIVE post-swap projection vs réel (Apr 27)

### 🟢 Maintenance rapide (faisable maintenant)
- ~~Documenter règles HYST/DTRAIL/paired-test dans `docs/known_issues.md`~~ ✅ v14e.6 P3
- `analyze_mega_sweep.py` en nightly CI (faible priorité — post-processeur on-demand, pas un gate quotidien)

### 🔵 Sim-align follow-up
- ~~**4 bugs logiques**~~ ✅ Résolus par v144.19 (decontamination `paper_sim_pnl_pct`). Les 4 cas ont maintenant diff <5pp. Documenté `known_issues.md §7`.
- ~~**MEV-pump filter dans le gate**~~ ✅ v14e.5 (`verify_sim_live_alignment.py` tag `[MEV]` + parse bash fix double-count N).
- Vérif gate vert au cron 04:00 UTC demain — si rouge, réinvestiguer.
- **A/B mega sweep rappel (Apr 21)** : Spearman ρ=**0.225** (weak), 99.9% configs suspectes.
  - ~~`HIGHSCORE_TP200_SL40` hidden gem~~ → retirée v14e.11 (avg 7d −3.5%, le PT 12665 était un faux positif).
  - `FAST_TP80_SL25` ⭐ rank 1 des DEUX sweeps — confirmée, live candidat.
  - `FAST_TP100_SL20_S35` : shadow-only, attend paired-test.
  - Famille let-it-run TP100 sous-estimée par PT — à revisiter si shadows confirment sur N≥30.

### 🟠 Actions après verdicts
- ~~**NOZEROLIQ_TP200_SL40**~~ ✅ Retirée v14e.6 P0 (N=33 perdante).
- ~~**HYST + filtre S30/NZS30**~~ ✅ Arbitré v14e.11 : NZS30_HYST retirée (perdante), S30_HYST gardée en watch.
- **Top winners shadow paired** : promouvoir 1-2 en main paper si Δpp ≥ +5pp (data-wait Apr 23-30).
- **FAST_TP100_SL20_S35** (sim top robust) : si paper paired confirme → main paper + envisager live (data-wait).

### 🟡 Scale-up live (après verdict paper)
- BE25 → remplacer par 2e FAST avec TP différent (FAST_TP80 ou FAST_TP100) après FAST_TP50 stable + N≥30
- max_open_positions 6 → 8-10 si bankroll grandit
- Position size live $3.40 → $10-20/trade (gain x3-x6 attendu)
- **Trigger V2 policy au scale-up** : laisser DÉSACTIVÉ par défaut. Valider d'abord 48-72h à $10/trade sur polling pur pour mesurer si le positive slippage Jupiter Ultra (+5pp/trade) tient à cette taille. Si oui → garder trigger off. Si le positive slippage disparaît (le spread Ultra peut se compresser à position plus grosse) → envisager trigger uniquement sur TP200 cluster (TP/SL 100% static, pas de PATCH nécessaire). Ne JAMAIS activer trigger sur BE25/BE15 (activation BE impose 1 PATCH non testé en prod) ni sur DTRAIL/TRAIL/DIP (patch-à-chaque-poll = gas × 10).

### 🔒 Bloqué / dormant
- **Jupiter Trigger V2** — 0 fills historiques, **désactivé v144.14 (Apr 21)**. Config DB `trigger_orders_enabled=false`. Autres paramètres gardés (min_usd=10, expiry=14400, sl_slip_bps=2000). Re-activation discutée au scale-up.

---

## 🛠 Chantiers planifiés (sprint format)

### Sprint #1 — Refinement slip model ✅ DONE v14e.6 P5
- Log-continu remplace les 3 buckets (1.0 + 0.5 × log10(50k/max(liq,500)))
- Clamped [1.0, 2.5], 4 tests dédiés
- **Reste (v2)** : composante volume-volatility quand N≥30 par (liq_band × exit_type × vol_band)

### Sprint #2 — Coherence sim trail/dtrail/dip family (post Apr 25)
**Problème** : sim mega_sweep top picks famille trail/dtrail/dip alors que paper/live confirment artefact (DTRAIL10 sim top vs live 65% reconciled, slip 47×)
**Options** :
- (a) Modéliser `position_reconciler` dans sim (~150 lignes)
- (b) **✅ DONE v144.13 (Apr 21)** — `_mega_family_slip_mult` applique ×10 DTRAIL, ×8 TRAIL, ×6 DIP, ×5 SCALP, ×4 SPLIT. Le reste (FAST/BE/TP*/HIGHSCORE/MOONBAG) inchangé. Hybrides = worst-family wins. Les prochains `--mega-sweep` et `--mega-sweep-eval-history` utiliseront la nouvelle calibration automatiquement.
- (c) Post-process flag `family_realism` dans `analyze_mega_sweep.py` — **fait Apr 20**, à itérer
**Reco** : (b) data-driven simple, puis (a) si rigueur nécessaire
**Next** : re-run mega_sweep extended overnight (~3h) et comparer rankings vs `_mega_sweep_extended.csv` pre-v144.13. Shadow DTRAIL/TRAIL/DIP devraient dégringoler de 30-70%, FAST/BE inchangés.

---

## 🧠 Gotcha
- Supabase PostgREST cap 1000 rows même avec `.limit(10000)`. Toujours paginer via `.range(off, off+999)`.
- **Sim mega_sweep over-estimates trail/dtrail/dip/HYST** (historique 45-57×). **Partiellement corrigé v144.13** via `_mega_family_slip_mult` (×10 DTRAIL, ×8 TRAIL, ×6 DIP, ×5 SCALP, ×4 SPLIT). Calibration conservative — re-calibrer quand N≥30 par famille live.
- `slippage_actual_bps` column : signe opposé à `_dynamic_sell_slip_factor`. Utiliser per-pair PnL delta pour calibration.
- **Dedup paper/live asymétrique** (intentionnel) : paper exclut rt_live, live n'inclut que rt_live. Edge case sur KOL recall <24h après SL — bias ~5-10% optimiste paper. Pas un bug, design OK.
- **Per-trade Spearman ρ ≠ Per-strategy Spearman ρ** — par-trade ~0.9, par-strat ~0.7. Toujours préciser le niveau.

---

## Sim ↔ Live/Paper coherence (post v144.3)

### Status actuel
- **shadow ↔ paper main** : 100% parité (post v144.3) ✅
- **paper ↔ live médiane** : ≤2pp (target tenu)
- **paper ↔ live mean** : +5pp paper > live (queue lourde)
- **sim per-trade ↔ paper** : ρ ≈ +0.9
- **sim per-strategy ↔ paper** : ρ ≈ +0.7 (excluant shadow v144 polluants : ρ +0.71)

### Slip calibration v144
`_dynamic_sell_slip_factor` : offset global −100 bps. Splits per-cell pas faits faute de N. Revisit Apr 25-28.

### CI Monitoring
- `sim-align-gate.yml` (04:00 UTC) — alert si drift > 5pp
- `nightly-outlier-monitor.yml` (04:30 UTC) — alert si outlier sync=True

### Méthodologie 3 canaux
1. `paper_trades.paper_sim_pnl_pct` (v143.6) — PnL sim joint per-trade
2. `scripts/verify_sim_live_alignment.py` — CI nightly
3. `sim.py --mega-sweep` + `ranking_compare.py` — Spearman rank

---

## Architecture summary

**Scoring** : rt_score v141 (40.5/13.5/40.5/5.4 + 3 bonuses).
**Trading** : Paper slip `_dynamic_sell_slip_factor` v144 (offset −100bps), live Jupiter Ultra RFQ. Loss limit 0.5 SOL/jour.
**Orch v144** : `source` + `smoothing` split via `strategy_overrides` JSONB. `source=both` supporté.
**Alerting** : ML disabled (anti-predictive). Sim-align + outlier nightly alerts.
**Shadow ↔ main** : 100% parité comportementale post v144.3 (sauf alerts/bankroll).

## Workflow sim

| Mode | Flag | Source | Biais | Use case |
|---|---|---|---|---|
| Focused grid | `--from-ticks` | price_ticks | ⚠️ 3-min jup batch | Ranking rapide legacy |
| Ground truth | `--from-trades` | paper_trades.pnl_pct | ✅ exact | Vérité historique (strats déjà tradées) |
| 0% bias | `--from-eval-history` | eval_history JSONB | ✅ 30s exact | Perfect replay per-trade |
| Standard sweep | `--mega-sweep` | price_ticks | ⚠️ biaisé | Discovery legacy (warning coverage depuis v144.9) |
| Extended sweep | `--mega-sweep-extended` | price_ticks | ⚠️ biaisé | 874K configs (~3h) |
| **Ground truth sweep** | `--mega-sweep-eval-history` | eval_history | ✅ 30s | **v144.9 — A/B vs legacy, discover sans biais** |
| Annotation | `analyze_mega_sweep.py` | — | — | Multi-test correction (FDR/Bonferroni) + family_realism flag |

## Scripts (`scripts/`)

| Script | Usage |
|---|---|
| `recap_daily.py` | $/jour paper & live |
| `refresh_main_stats.py` | top earners ranking |
| `compare_lazy_vs_nolazy.py` | paired LAZY verdict |
| `paired_all_v144_shadows.py` | **paired audit + gap detection v144** |
| `verify_shadow_main_parity.py` | **invariants v144.3 shadows** |
| `diverge_report.py` | tableau sim/paper/live unifié |
| `slip_per_exit_type.py` | per pump×exit_type calibration |
| `spearman_drift_check.py` | Spearman 4×4 matrix |
| `analyze_mega_sweep.py` | **multi-test correction + family_realism** |
| `backfill_paper_sim_pnl_pct.py` | backfill `paper_sim_pnl_pct` historique |
| `audit_strategies.py` | audit alignement mains+live+shadows |
| `verify_sim_live_alignment.py` | CI sim vs live audit |

---

## Historique récent

- **v14b** (Apr 23 PM) ETH strats promues de shadow à main paper + alertes Telegram chain-aware (🔷ETH tag, `dexscreener.com/{chain}/`, Uniswap + Etherscan links). `position_usd=$200` forcé sur main path ETH aussi. 3 strats ajoutées aux `hybrid_strategy.allocations` en DB (15 total).
- **v14** (Apr 23 PM) **Sprint #ETH-1 Phase 1 deployed** : migration `chain` column sur 5 tables, `chain_detect.py` module + 25 tests, `ETH_CA_REGEX` scan dans `extract_tokens`, DexScreener chain-parameterized, guards 0x sur RugCheck/Helius/Jupiter/Bubblemaps/outcome, fee model ETH ($15 gas + 200bps MEV), `_passes_strategy_filter` chain gate strict. 3 strats ETH initiales. 38 tests pass. Sprint #ETH-1 Phase 1 live.
- **v144.19b** (Apr 23 AM) Nightly shadow audit CI (`verify_shadow_main_parity.py` + `paired_all_v144_shadows.py`). Crash fix + tolerance sur parity script. Backfill `paper_sim_pnl_pct` historique (49,746 rows) completed.
- **v144.19** (Apr 23 AM) API health Telegram alerts désactivées. `paper_sim_pnl_pct` decontamination : retiré `_pt_ultra_override` dans `live_trader._paper_sim_ev` qui faisait que la ref sim stockée suivait le fill Jupiter au lieu de rester sim pure (faux drift +148pp sur pumps $MHGA/$8). Nightly outlier monitor skip les paires tp/tp MEV-pump (live>paper>0) — seuls les vrais bugs logiques alertent.
- **v144.17-18** (Apr 22 eve) API error alerts désactivées (noisy). +2 KOLs A-tier (leoclub69, markdegens).
- **v144.16** (Apr 22 PM) STRATEGY_FILTERS appliqué au live (paper-only avant → BOND_FAST live achetait non-bonding comme $OOO). Live = miroir strict du shadow.
- **v144.15** (Apr 22) Live 4-strat A/B : BE25 + FAST_TP50 + FAST_TP80_SL25 (new) + BOND_FAST_TP50_SL20_T20 (new).
- **v144.14** (Apr 21 eve) Jupiter Trigger V2 désactivé en DB (`trigger_orders_enabled=false`). Risque de détruire le +5pp positive slippage Ultra observé sur FAST live. Re-évalué au scale-up $10+.
- **v144.13** (Apr 21 eve) Per-family slip multiplier dans mega_sweep : ×10 DTRAIL, ×8 TRAIL, ×6 DIP, ×5 SCALP, ×4 SPLIT. Hybrides = worst-family wins. Corrige le biais 44% du sweep universe (Sprint #2b). Static TP/SL inchangés.
- **v144.12b** (Apr 21 eve) Scope fix gate SIM-vs-PAPER : itère `paper_by_strat.keys()` pour capturer FAST/DTRAIL sans `paper_sim_pnl_pct`. Révèle +55.9% sim-drift sur FAST_TP50_SL30, +40.2% BE25.
- **v144.12** (Apr 21 eve) Gate économique bidirectionnel (|mean|>3pp, |median|>5pp) + nouveau gate SIM-vs-PAPER ($/day paper vs sim médiane, flag |diff|>30%). Paired test cross-source aware (flag ⚠️CROSS-SRC quand price_source diffère, leaderboard SAME-SOURCE isolé).
- **v144.11** (Apr 21 eve) Alertes live enrichies : bankroll + per-strategy breakdown sur buy/sell, bloc 🔀 Paper vs Live per-trade (paper_sim_pnl_pct + fill Δ), bloc 📊 Drift 24h par strat via `_live_paper_strategy_drift_24h` (cache 5min).
- **v144.9** (Apr 21) mega_sweep A/B : warning coverage jup (A) + `--mega-sweep-eval-history` mode (B)
- **v144.8** (Apr 21) Sim-align gate apples-to-apples (vs `paper_sim_pnl_pct`, Jup slip info) + diverge_report migration
- **v144.7** (Apr 21) Sim-align gate switched from price_ticks to eval_history replay (−3.78pp → −1.16pp)
- **v144.6** (Apr 21) Fix LAZY throttling bypass pour live_sync shadows (4 outliers Apr 21)
- **v144.5** (Apr 20 PM) BE25_TP80_SL30_S35 + LAZY_STRATEGIES cleanup (4 dead entries)
- **v144.4** (Apr 20 PM) FAST_TP100_SL20_S35 — top robust sweep cluster
- **v144.3** (Apr 20 PM) Shadow ↔ main behavioral parity (LAZY + Ultra exit + position)
- **v144.2** (Apr 20 PM) Bug routing paper FAST_TP50/BE25 (rt_live blocking sibling) + 19 new shadows pour gaps couverture
- **v144.1** (Apr 20) 4 retraits HYST/DS losers from hybrid_strategy.allocations
- **v144** (Apr 19) Slip offset −100bps + extended mega sweep + price_source split + 34 A/B shadows + audit_strategies tool
- **v143.6** (Apr 19) DS cache TTL + `paper_sim_pnl_pct` column + CI gate
- **v143.5** (Apr 19) Live exit shadow-sync
- **v143.1-4** (Apr 18-19) Sim alignment fixes + 7 smoothing modes ports
- **v142E** (Apr 18) Entry shadow-sync
- **v141** (Apr 17) rt_score +3 bonuses data-driven
- **v140** (Apr 17) 8 new strats, bankroll reset $18K
- **v138.5** (Apr 17) Slip recalibration per exit-type
