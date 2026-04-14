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

## Stratégies post-v132 (Apr 13+, régime actuel)

### Découvertes clés du sweep 2026-04-14
- **+11pp de WR depuis Apr 13** — attribué majoritairement aux modifs code (v130-v132: source cohérence, polling per-strat, hybrid) — pas à un "régime marché" pur. Tendance graduelle avant v132 ~+2pp/sem.
- Kelly + MC extrapolés à N<50 = overfit garanti. Traiter toute proj avec CI±5pp sur WR.
- **Le sweep sur N=133 mélangeant shadows pré/post-v132 a produit des conclusions fausses** (notamment "désactiver DTRAIL10 et DIP30"). Post-v132 uniquement, ces strats redeviennent profitables. Ne pas désactiver avant plus de données.

### Classement top strats post-Apr 13 (N=30/strat, shadows inclus)

**Profil "consistent" (WR haut, médiane positive, drawdown contenu):**
| Rank | Strategy | avg | WR | med |
|---|---|---|---|---|
| 5 | **FAST_TP70_SL50** | +10.14% | **56.7%** | +8.97% |
| 6 | TP70_SL70 | +10.06% | 50.0% | +10.84% |
| 12 | FAST_TP100_SL50 | +8.55% | 53.3% | +8.02% |
| **16** | **FAST_TP50_SL30 (LIVE actuel)** | **+8.42%** | 53.3% | +6.86% |

**Profil "home-run" (avg élevée mais médiane négative = > 50% losers):**
| Rank | Strategy | avg | WR | med |
|---|---|---|---|---|
| 1 | TP70_SL30 | +12.25% | 43.3% | **−20.30%** |
| 2 | TP80_SL30 | +11.92% | 40.0% | −27.89% |
| 3 | TP90_SL30 | +10.92% | 36.7% | −30.00% |

→ Pas recommandés pour live : drawdown long, Kelly réel faible malgré avg.

### À faire — exploration strat
- [ ] **A/B paper FAST_TP50_SL30 (live actuel) vs FAST_TP70_SL50** — 2 semaines, allocation 50/50. FAST_TP70_SL50 a WR 56.7% vs 53.3% et +1.72pp d'edge sur N=30.
- [ ] Re-run sweep smoothing sur **top-10 strats consistent** (pas juste les 4 actuelles) pour voir si le gain smoothing est stable inter-strats.
- [ ] **Ne pas désactiver DTRAIL10/DIP30** — post-v132 elles redeviennent profitables (DTRAIL10 both/60s/ema_slow Kelly 5.4%, DIP30 both/60s/winsor_p95 Kelly 11.4%). Laisser tourner en paper, décision après N=100 par strat.
- [ ] Attendre **N≥100/strat en régime actuel** avant tout tuning définitif (actuel N=30-41/strat).

### Stratégies synthétiques testées (v134 sweep, N=30 post-Apr 13)

**Top candidates consistent (WR ≥53%, médiane positive):**
| Strat | Config | avg | WR | med | Note |
|---|---|---|---|---|---|
| **FAST_TP80_SL25** | Jup/120s/dual_confirm | +13.23% | 53.3% | +7.79% | **+3pp sur FAST_TP70_SL50** |
| BE25_TP80_SL30 | Jup/120s/ema_fast | +13.40% | 53.3% | +7.79% | BE ratchet safety |
| FAST_TP60_SL20 | Jup/60s/ema_slow | +11.69% | 53.3% | +3.10% | Ultra tight |
| BE15_TP70_SL30 | Jup/120s/ema_slow | +11.52% | 53.3% | +7.79% | Conservative |

**Top candidates home-run (avg haute, médiane négative):**
| Strat | Config | avg | WR | med | Kelly |
|---|---|---|---|---|---|
| FAST_TP100_SL20 | DS/120s/raw | +16.36% | 50.0% | −2.01% | **31.6%** (!) |
| BE30_TP100_SL30 | DS/120s/raw | +13.55% | 50.0% | +2.19% | — |

### Plan pragmatique
- [ ] **Ajouter FAST_TP80_SL25 à paper_trader.STRATEGIES** — candidate #1 consistent. Laisser tourner 1 semaine paper pour confirmer le +3pp vs FAST_TP70_SL50.
- [ ] **Ajouter BE25_TP80_SL30** idem — test du BE ratchet.
- [ ] Si validé en paper (N≥50), A/B live vs FAST_TP50_SL30 actuel.
- [ ] FAST_TP100_SL20 → **prudence**. Le Kelly 31.6% est un piège à N=30, médiane négative = drawdown long. Ne tester que si on accepte volatilité.
- [ ] Re-run synthetic-sweep dans 1 semaine avec N double pour trancher.

### Smoothing — conclusions
- À N=133 mélangé : `raw` ≥ smoothed sur FAST (+2.03% vs +1.53%).
- À N=30 post-v132 : `dual_confirm` > `raw` pour FAST (Kelly 17.2% vs moins), mais N trop petit pour trancher.
- **Verdict honnête : pas de gain prouvé du smoothing. Laisser FAST en `hybrid/60s/raw` (prod actuelle) jusqu'à plus de données.**
- [ ] Si smoothing validé plus tard : étendre `paper_trader._decision_price` avec `median_3/5, winsor_p95, dual_confirm, ema_fast/slow, hysteresis, volume_gated`.

## Repo hygiene
- [ ] Nettoyer commit `007db6b` qui a pushé 2400+ fichiers cache (`scraper/ohlcv_cache/`, `scraper/jupiter_candles_cache/`, `grid_search_*.csv`). Options : revert + force-push, ou ajouter au `.gitignore` et laisser.

## Data quality (depuis /check-data 2026-04-14)
- [ ] Enrichment Jupiter LDS sous-fill (35% vs seuil 70%) — vérifier quotas API / erreurs silent
- [ ] Enrichment holders sous-fill (27%) — idem
- [ ] CA resolution 71.9% (sous seuil 75%) — revoir le resolver pour les messages récents
- [ ] Backlog labels 24h = 774, 7d = 2346 — voir outcome_tracker cadence
