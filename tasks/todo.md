# Pipeline Status — Updated Apr 25, 2026 (v14e.22 — LOWSCORE shadow + routing audit)

## v14e.22 — Apr 25 PM — LOWSCORE_TP50_SL15 shadow + 4 routing ideas tested

### Audit routing (sur 22,475 trades closed depuis reset Apr 17)

Script `_analyze_routing_ideas.py` + apples-to-apples `_quantify_routing_dollar_day.py`. Verdicts honnêtes après correction du biais d'échantillonnage :

| Idée | Verdict | Δ apples-to-apples N=52 | Live $/d ($1.74/trade) | Annualisé |
|---|---|---:|---:|---:|
| BASELINE BE25_TP80_SL30 | (référence) | +9.41% | +$16.24/d | +$5,926/yr |
| Per-KOL family routing | ❌ NEGATIVE | −0.88pp | −$1.51/d | −$551/yr |
| **Score-gate (TP50 si score<30)** | ✅ POSITIVE | **+1.93pp** | **+$3.34/d** | **+$1,217/yr** |
| Combined score+per-KOL | ≈ score-gate seul | +1.73pp | +$2.99/d | +$1,091/yr |
| Cooccurrence multi-KOL | ❌ N/A | dataset trop petit | — | — |
| Per-KOL custom timeout | ❌ NEGATIVE | +0.3pp | <$1/d | — |

### Implémenté en v14e.22

- **`LOWSCORE_TP50_SL15`** : nouvelle strat ajoutée à `strategies.py` (TP +50% / SL -15% / 120min / chain=solana / max_score=29). Mode **shadow uniquement** — pas dans `hybrid_strategy.allocations` ni `paper_trade_config.active_strategies` → s'ouvre en `is_shadow=True` automatiquement sur tout call SOL avec score<30. Aucune alerte Telegram, aucun impact bankroll. Track N≥150 events avant promotion main.
- **Mega-sweep extended workflow GH** étendu (`mega-sweep-48h.yml`) : runner intégré pour `_analyze_routing_ideas.py` + `_quantify_routing_dollar_day.py` après chaque sweep 48h. Sortie dans artefacts. Trace l'évolution du signal à mesure que N grossit.
- **Documentation `docs/known_issues.md §11`** mise à jour avec verdicts apples-to-apples (la version naïve a été corrigée — le per-KOL +13.7pp était un biais de sample, pas un signal réel).

### Pourquoi le per-KOL routing n'a pas marché

Intuition utilisateur : "chaque KOL a sa meilleure stratégie définie, donc le routing devrait gagner". **Réalité** : c'est de l'**overfitting / data mining bias**.
- 30 KOLs × 8 familles = 240 cellules (KOL, famille). Avec N moyen 30-100/cellule, la **variance** sur l'avg pnl_pct est large (±10-20pp 95% CI).
- Quand je sélectionne "le best fit" pour chaque KOL, je picke la famille qui s'est trouvée chanceuse sur ce sample précis.
- Sur les MÊMES (KOL, token) events out-of-sample, le routing perd −0.88pp car la "préférence" était bruit, pas structure.
- Avec N>150 events apples-to-apples (~3 semaines de plus), si signal réel >0pp persiste alors implémenter ; sinon idée définitivement abandonnée.

### Reste à faire

- **Surveiller LOWSCORE_TP50_SL15** : N≥30 attendu sous 5-7j (les calls score<30 représentent ~30-40% du flux). Décision de promotion main paper à ce moment.
- **Re-runner `_quantify_routing_dollar_day.py` automatiquement** via mega-sweep workflow toutes les 48h → tracking N et Δ.

---

# Pipeline Status — Updated Apr 25, 2026 (v14e.17 — bat_gamble ETH-only purge)

## v14e.17 — Apr 25 — bat_gamble Solana wipe + per-KOL chain whitelist

Verdict bat_gamble v2 atteint après ~24h de paper post-v14e.14 (et l'historique pre-v108 toujours visible) :

| Chain | N main closed | sum_pnl | strats | Verdict |
|---|---:|---:|---:|---|
| 🟣 Solana | 727 | **−$11,386.30** | 9 | Tous saignent — 0/290 shadows profitables @ N≥15 (les 2 borderline winners sont HYST/score-filter artefacts). Késako : late/post-pump sur low-liq, chaque exit type puni. |
| 🔷 Ethereum | 23 | **+$1,627.84** | 6 | Tous gagnent (let-it-run profile, BE+TP100/150). N=3-5 par strat, verdict préliminaire mais cohérent. |

**Action exécutée (commit cc61c5c → c6b60b2 push, déployé VPS 12:52 UTC)** :
- 32,359 paper_trades rows SOL deleted (727 main + 31,632 shadows). 18 ETH rows kept.
- Round 1 refund (+$11,386) sur 9 SOL strats — TROP GROS (incluait l'historique pre-v108 déjà wiped par le reset Apr 17 v138.3).
- Round 2 (-$557.43) : leak post-deploy avant restart VPS, retiré.
- **Round 3 (canonical) — full rebuild** : `_batgamble_rebuild_bankroll.py`. Cutoff = v138.3 reset (2026-04-17 14:36 UTC). Pour chaque strat SOL : `target_balance = starting_balance + sum(pnl_usd closed main rt SOL since cutoff WHERE kol_group != 'bat_gamble')`. Strats sans trades préservées telles quelles. starting_balance healed (cur_bal-cur_pnl) quand le bookkeeping le mettait à 0.
- **Détection annexe** : top-level `strategy_bankrolls` inflaté **+$10,000 par strat** (phantom historique), maintenant mirrored depuis per_chain.solana (single source of truth).
- État final : `current_balance` $24,937.68 → **$19,347.22** ; `total_pnl` $3,933.42 → **+$1,347.22** ; `total_trades` 1,162 → 1,006. SOL active strats dans la fourchette **$861-$1,247** (correspond au "$1000-$1300 avant bat_gamble" attendu par l'user). ETH/BSC/BASE inchangés.
- Backups : `data/rt_bankroll_pre_batgamble_purge.json` (R1), `_round2.json` (R2), `_pre_rebuild_*.json` (R3).

**Filtre per-KOL chain whitelist** (réutilisable) :
- `GROUPS_DATA[kol]["chains"]` optionnel (default = toutes chains).
- Gate dans `safe_scraper._rt_open_trades` après résolution `token_chain` — return 0 avant live ET paper paths.
- Log : `RT SKIP (kol chain filter v14e.17): %s on %s — KOL allowed only on %s`.
- bat_gamble = `["ethereum"]`. Pattern réutilisable pour future restriction par-KOL.

**Reste blacklist live (`live_trading.kol_blacklist`)** : inchangée — bat_gamble toujours blacklist live (paper ETH only).

**Slippage drift audit (en parallèle)** : suspicion utilisateur d'outliers en hausse post-v14e.6 → **infirmée**. Mean |Δ| paper_sim vs live 9.57pp (PRE) → **3.16pp (POST, −67%)**, max 215pp → 17.7pp, 0 outliers >50pp post (vs 8 pre). Le modèle continu écrase la dispersion sur 5-20k. À revérifier dans 5-7j (POST N=28 trop petit).

---

# Pipeline Status — Updated Apr 24, 2026 (v14e.16 — age-window A/B shadows)

## v14e.16 — Apr 24 — A/B test fenêtre d'âge (paper-only, zero-regression)

### Contexte (audit logs 48h)

- **79.2% des skips sont "token too old"** (175/220 detects).
- Sim naïf sur `token_snapshots` : relax 12h → 72h donnerait **+$3.16 / 48h** sur 27 tokens observables, mais **biais optimiste** (seulement 17-28% coverage, les dumps sans trace sont invisibles).
- Verdict : **ne pas relaxer aveuglément**. A/B-test propre avec shadows paper-only.

### Design A/B

Global RT gate `max_token_age_hours_rt` relaxé 12h → **72h** (safe_scraper:1916).
Existant intact : chaque strat sans `max_age_hours` dans son filter ignore le check → comportement pre-v14e.16 préservé.

**3 nouvelles shadows FAST_TP50_SL30** en fenêtres **disjointes** :

| Strat | Fenêtre | Mesure |
|---|---|---|
| `AGE24_FAST_TP50_SL30` | `[12, 24]h` | Incrément de relaxer 12→24h |
| `AGE48_FAST_TP50_SL30` | `[24, 48]h` | Incrément 24→48h |
| `AGE72_FAST_TP50_SL30` | `[48, 72]h` | Incrément 48→72h |

Somme des 3 = PnL total du full relax 12→72h.

### Safeguards

- Paper-only : **aucune** de ces 3 n'est dans `live_trading.allocations`. Zéro risque capital.
- `_passes_strategy_filter` applique age filter uniquement si `max_age_hours`/`min_age_hours` explicitement déclarés (opt-in). Toutes les strats existantes inchangées.
- 4 tests de régression ajoutés (`TestStrategyAgeFilter`).
- Bankroll seeded $1000 chacune, disjoint SOL.

### Règles de décision (N≥30 par bucket, ETA ~10-14j)

| Bucket | PnL net < $0 | PnL net > 0, avg < $0.05/trade | PnL net > 0, avg ≥ $0.05/trade |
|---|---|---|---|
| AGE24 | retire | garder shadow, watch | promote 12h → 24h global |
| AGE48 | retire | retire | promote si AGE24 aussi promote |
| AGE72 | retire | retire | rare — garder shadow N=60 avant trancher |

### Monitoring

- Query `paper_trades` WHERE strategy LIKE 'AGE%' GROUP BY strategy (N, PnL, WR).
- Si aucun trade après 48h → vérifier que le flag `token_age_hours` est bien populated dans token_info RT (DexScreener sortie `pair_created_at`).

---

# Pipeline Status — Updated Apr 24, 2026 (v14e.14d — ryoshikdegen + skip-audit reminder)

## 🔔 REMINDER — À faire le dimanche 26 Apr 2026 ~14:30 UTC (48h post-fix)

**Audit manuel des raisons de skip RT pipeline** (Option C choisie par l'user).

Le remote agent a été rejeté car il ne peut pas SSH le VPS ni lire
`scraper/.env` local. Option C = tu le lances manuellement.

Commande à lancer (copie-colle) :

```bash
# 1) Pull + aggregate logs VPS
ssh vps "journalctl -u kol-scraper --since '48h ago' --no-pager" > /tmp/kol_48h.log

# 2) Count skip reasons
grep -oP 'RT SKIP: .+?— \S+' /tmp/kol_48h.log | sort | uniq -c | sort -rn | head -20
grep -cE 'RT detect:' /tmp/kol_48h.log         # denominator
grep -cE 'RT: unmatched chat_id' /tmp/kol_48h.log
grep -cE 'RT LIVE SKIP \(kol blacklist' /tmp/kol_48h.log  # v14e.14 fire count
grep -cE '\$WETH' /tmp/kol_48h.log             # should be ~0 post-v14e.15
```

Ou lance Claude Code à ce moment avec : "fais l'audit des raisons de skip RT
sur les 48 dernières heures de logs VPS + cross-ref avec Supabase (calls →
detects → skips → trades funnel)".

Objectif : identifier si le taux 45% de CAs non-enrich baisse post-v14e.15
(WETH exclu). Cibler la plus grosse catégorie de skip restante pour next fix.

---

## v14e.14d — Apr 24 — ryoshikdegen (correct channel)

`ryoshikushama` retiré précédemment (user, pas channel). Après probe
Telegram : **`ryoshikdegen` = Channel "Ryoshi Degen" (broadcast=1)**, ajouté
en remplacement. `ryoshidegen` (sans le "k") n'existe pas.

**Blacklist live finale (9 KOLs)** :
`bagcalls, bat_gamble, batman_gem, mad_apes_gambles, maestrodegen,
reapergamble, ryoshigamble, ryoshikdegen, venom_gambles`.

---

# Pipeline Status — Updated Apr 24, 2026 (v14e.14b + v14e.15 — trim blacklist + fix WETH noise)

## v14e.15 — Apr 24 PM — WETH/wrapped tokens excluded (enrichment noise fix)

**Signal découvert** : **45.1% des tokens calls ne sont jamais enrich** sur 7j
(200 CAs manqués sur 443, cf. diagnostic AIB). Top cas : **$WETH, 87 callers
sur 25 groupes** — c'est `0xc02aaa39b223fe8d0a0e5c4f27ea` = le contrat
Wrapped ETH natif, extrait à tort quand les KOLs postent des liens Uniswap.

**Fix immédiat** : ajouté `WETH, WBTC, WSOL, WBNB, WMATIC, WAVAX, WFTM` à
`EXCLUDED_TOKENS` dans `pipeline.py`. L'extraction via `ETH_CA_REGEX` va
toujours matcher le CA mais `_resolve_ca_to_symbol` retourne "WETH" qui est
maintenant filtré avant d'être ajouté à `tokens`. Économie estimée : ~87
faux callers/semaine = 12/jour de bruit retiré.

**Reste à investiguer** (NON FAIT — nécessite trace logs VPS) :
- Les 113 autres CAs missing avec ≥3 callers. Causes possibles :
  - Pump.fun bonding curve pre-migration (DS pas indexé) — comme $AIB.
  - DS API failure ponctuel (rate limit, 500).
  - Token rug avant enrichment (liquidité < $500).
- Cf. top manquants : $ELIEN (70), $ASTEROID (35, EVM), $FLORK (21),
  $AIB (18), $AYYLIEN (16), $CRYPTOCURRENCY (14), $BRITAIN (14).
- **Action suggérée** : activer DEBUG log dans `_rt_open_trades` pour tracer
  où exactement chaque call skip (no data / low_liq / low_score / dedup). Sur
  24h on saura où vont les 12-30 calls/jour perdus.

---

## v14e.14b — Apr 24 PM — trim blacklist (retirer KOLs déjà présents)

Per-user : `TheReaperGems` était déjà dans `GROUPS_DATA` avant v14e.14 donc
était déjà live-eligible implicitement. Retiré de `live_trading.kol_blacklist`.
Variants `thereapergems` retiré aussi (aucun risque de mismatch casse).

**Blacklist live finale (9 KOLs vraiment nouveaux)** :
`bagcalls, bat_gamble, batman_gem, mad_apes_gambles, maestrodegen,
reapergamble, ryoshigamble, ryoshikushama, venom_gambles`

## v14e.14 join status

- **99 groupes rejoints** (vs 91 avant) → +8 succès.
- **1 fail : `ryoshikushama`** — c'est un **user individuel Telegram**, pas un
  channel. Erreur : `Cannot cast InputPeerUser to any kind of InputChannel`.
  Telethon ne peut pas join un user comme un channel. Action user requise :
  soit donner un lien vers le channel où cet user poste ses calls, soit
  retirer de la liste. Pour l'instant gardé dans `GROUPS_DATA` avec commentaire
  — le symbole peut potentiellement apparaître via d'autres groupes qui
  forward ses messages.

---

# Pipeline Status — Updated Apr 24, 2026 (v14e.14 — 10 KOLs paper-only test)

## v14e.14 — Apr 24 — 10 nouveaux KOLs paper+shadow, exclus du live

**Ajoutés dans `GROUPS_DATA`** (9 nouveaux + `TheReaperGems` déjà présent capitalisé) :
batman_gem, venom_gambles, ryoshigamble, ryoshikushama, mad_apes_gambles,
maestrodegen, bat_gamble (re-ajout post-v108), reapergamble, bagcalls.
+ thereapergems (variant lowercase, juste au cas où).

**Nouveau gate** `live_trading.kol_blacklist` dans `safe_scraper._rt_open_trades` :
KOLs listés exécutent **paper + shadow normalement** mais bypass la branche
`open_live_trade`. Capital live zéro pour ces 10 pendant la phase test.

### Objectifs de l'expérimentation

1. **Build N≥30 paper trades par KOL** (attendu 1-2 semaines sur les spammers,
   plus long sur les autres).
2. **Mesurer rentabilité individuelle** par KOL : `avg pnl_pct`, WR, $/jour
   sur l'ensemble des 8 strats mains actives. Seuil promotion live : pnl >$0
   sur N≥30 sur 14j (aligné kol_filter v92).
3. **Détecter combos profitables** : certains KOLs sont profitables **uniquement
   quand ils overlapent** avec un autre caller (confirmation multi-sources).
   Script `scripts/kol_combo_analysis.py` à créer — analyser les trades où
   ≥2 KOLs de la liste + ≥1 existant = confirmation.
4. ~~**Verdict bat_gamble v2**~~ ✅ **Tranché Apr 25 (v14e.17)** : SOL = perdant
   définitif (−$11,386 / 727 main / 0/290 shadows profitables) → wipe SOL +
   chain whitelist ETH-only. ETH = positif préliminaire (+$1,628 / 23 main /
   6 strats N=3-5) → keep observation, paper-only.
5. **Anti-spam detection** : certains (batman_gem, bat_gamble, reapergames)
   sont connus pour spammer. Mesurer si leurs 20e-30e calls/jour dilue le
   signal (WR tombe avec le volume) — indicateur anti-KOL vs bon signal.

### Règles de décision (ETA 14 jours = Mai 08)

| Résultat | Action |
|---|---|
| `pnl > 0` et `WR ≥ 50%` sur N≥30 | Retirer de kol_blacklist → live activé |
| `pnl > 0` mais `WR < 50%` (asymétrique) | Laisser paper, surveillance N=60 |
| `pnl < 0` avec `N ≥ 30` | Garder blacklist, considerer blacklist scraping aussi |
| `pnl < -$50` avec `N ≥ 30` (chain-specific) | Per-KOL chain whitelist (pattern v14e.17 bat_gamble : `chains=["ethereum"]`) ou retirer entièrement de `GROUPS_DATA` si saigne sur toutes les chains |
| Spam signal (>30 calls/jour) avec `WR < 40%` | Downweight conviction 7→3 ou exclure totalement |

### Monitoring

- `scripts/kol_stats.py` si existant, sinon query directe paper_trades group by kol_group.
- Alerte Telegram nightly via `alerter._kol_leaderboard_24h` (déjà en place).
- Vérifier logs VPS : chercher `RT LIVE SKIP (kol blacklist v14e.14)` pour
  confirmer que le gate fire bien.

### À noter — $AIB / bounty_journal investigation

L'user a demandé pourquoi `$AIB` (call bounty_journal Apr 23 22:53) n'a pas
été tradé. Root cause : **token jamais enrich** (0 rows dans `tokens`, 0
rows dans `token_snapshots`). 20+ KOLs l'ont call mais DexScreener n'a
probablement pas indexé ce bonding curve pump.fun (CA ending `bonk`) au
moment des calls → `_rt_open_trades` skip car pas de données enrichies.
Pas un bug spécifique à bounty_journal. Pattern classique sur très-jeunes
pump.fun bondings. Rien à fixer côté code.

---

# Pipeline Status — Updated Apr 24, 2026 (v144.20 + ETH first blood)

## ETH Phase 1 — premiers trades (Apr 23-24)

**12 paper trades ETH** sur 4 tokens, 28h (first blood depuis v14b Apr 23) :

| Token | Time | BE50_TP150_SL50 | TP80_T2H | TP100_SL50 |
|---|---|---|---|---|
| $ASTEROID | 23/19:59 | sl_hit −$106 | **tp_hit +$139** | **tp_hit +$177** |
| $GENZ | 23/22:12 | sl_hit −$106 | sl_hit −$87 | sl_hit −$106 |
| $CHINESEASTEROID | 23/23:57 | timeout −$12 | timeout −$14 | timeout −$12 |
| $EIB | 24/00:21 | sl_hit −$106 | sl_hit −$87 | sl_hit −$106 |

- **Total −$525** (paper, zéro capital réel).
- **WR 3/12 = 25%** (vs thèse Phase 2 ≥65%).
- **Par strat** : TP100 −$47, TP80 −$48, BE50 −$430.
- **SL 50% full-slip = −53% net** sur position $200 (fee model $15 gas + 200 bps MEV + slip). Chaque trade perdant mange 53%.

**Signaux précoces (N=12, trop petit pour trancher)** :
- BE50_TP150_SL50 n'a jamais activé le BE (peak < +50% sur les 4 tokens) → joue juste SL 50%. À watcher si N≥30 continue ce pattern, retirer.
- 3/4 tokens full dump → sélection KOL EVM à questionner. Les KOLs Solana postent parfois des 0x mais pas pour autant leurs meilleurs calls.
- $ASTEROID = seule win → pump réel ~2x. R:R d'un TP100 fire = +88% net après frais = compense ~1.7 SL.

**Règle Phase 2** (v14b) : décision à N≥50 / 14j, seuils WR ≥65% + EV net ≥+10%/trade. **ETA Mai 07**. On est à N=12 / J+1. Trop tôt pour kill — mais si le pattern 25% WR persiste à N=30, Phase 2 sera no-go mécanique.

**Pas d'action immédiate** : laisser tourner, monitorer. Alert si N=30 avec WR < 40% → review précoce + éventuelle pause allocations ETH.

### v14e.13 — 3 nouveaux ETH BE variants paper (Apr 24)

Grid sim (`scripts/_sim_eth_grid.py`) sur les 4 tokens révèle que **BE20/BE30** transforme les 3 dumps en BE stop à ~−$19 (gas + slip seulement) au lieu de −$106 full SL. Le peak des 4 tokens a atteint >+20% avant de dumper, mais JAMAIS >+50% → BE50_TP150_SL50 structurellement dead.

Grid total sur 4 tokens (paper, $200/trade, fee model ETH) :
- `BE20_TP100_SL50_H240` : **+$129** (vs −$72 ETH_TP100_SL50, −$269 BE50_TP150_SL50)
- 3 perdants limités à −$19 chacun (BE stop), ASTEROID +$177 préservé.

**Ajoutés paper (not live)** :
- `ETH_BE20_TP100_SL50` : mirror TP100_SL50 + BE active à +20%
- `ETH_BE30_TP100_SL40` : SL plus serré (−40%) + BE +30%
- `ETH_BE20_TP80_SL40_T2H` : mirror TP80 + BE +20%

DB patched : `hybrid_strategy.allocations` 17→20, `strategy_bankrolls_per_chain.ethereum` 3→6 (seed $1000 each). Aucun tick supplémentaire requis — les 3 shadows partagent les mêmes `_fetch_prices_batch` que les 3 mains existantes sur chaque token ETH.

**N=4 est statistiquement rien** — mais si le pattern "peak>+20% avant dump" tient à N=30, BE50 sera retirée avant Mai 07 et BE20/BE30 pourraient candidater pour Phase 2.

---

## v144.20 — Apr 24 — alignement LAZY throttle live ↔ shadow

**Alerte reçue** : `PAPER-LIVE OUTLIER (sync=True)` — 7 outliers |L−P|>10pp sur 33 pairs, pire : $AINI / FAST_TP80_SL25 Δ=+59.89pp.

**Root cause** (non post-v143.5 bug) : asymétrie LAZY throttle.
- `paper_trader._should_evaluate_exit` throttle LAZY strats à 1 eval/180s (v118).
- v144.6 a fait bypass LAZY pour paper shadows `entry_source=live_sync`.
- MAIS live trades (`source=rt_live`) passent par le même code → **étaient throttled** contrairement au comment v144.6.
- Résultat : shadow réactif à chaque wick Jupiter, live skippait 180s → divergence structurelle.
- Exemple $AINI : Jupiter wick à −25.04% à 22:28:56 sur pump.fun liq $75k. Shadow SL (bypass LAZY), live skippé 3s avant la prochaine eval (177s/180s), puis rebond à +32%.

**A/B 14j / 75 LAZY live trades** (`scripts/_ab_lazy_bypass_live.py`) :
- Impact total bypass = **−$0.03** (noise pur).
- FAST_TP50_SL30 : +$1.67 (+116%) avec bypass.
- FAST_TP80_SL25 : −$1.69 (−105%) avec bypass. **MAIS N=16 sur 48h de marché bear (Apr 23 = −14.71% avg / 21% WR) — verdict prématuré**.
- Les deux strats se compensent exactement au total.

**Fix appliqué** : `paper_trader.py:1672-1673` — live `rt_live` bypass LAZY comme les shadows. Paper mains gardent le throttle (A/B baseline v144.3 inchangée).

**Cleanup** :
- `nightly_outlier_monitor.py` : retour code propre (pas de filtre hack LAZY — le fix règle l'asymétrie à la racine).
- `tests/test_paper_trader.py::TestShouldEvaluateExitLazyBypass` : 3 tests régression.
- Les 7 outliers historiques Apr 22-23 vont rester visibles ~48h le temps que la fenêtre 48h roule. Nouveaux trades post-deploy : zéro asymétrie LAZY.

**TODO dans 7-10j** : re-run `_ab_lazy_bypass_live.py` avec N≥30 FAST_TP80 sur fenêtre marché mixte. Trancher garder/killer. Même règle que la memory pour shadows (N≥30 paired-test).

**Allocs live inchangées** : décision explicite — pas de kill FAST_TP80 sur N=16 + 48h bear. Re-score post-fix sur data solide.

---

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
- **bat_gamble ETH-only** (v14e.17, Apr 25) — re-vérifier dans 5-7j que le filtre tient (logs `RT SKIP (kol chain filter v14e.17)`) et que les 6 ETH strats restent positifs sur N≥10/strat
- **Slip drift POST v14e.6** — recheck Apr 30 avec N≥80 trades post-Apr-23. PRE 9.57pp → POST (N=28) 3.16pp ; si converge stable <3pp on garde, si signed bias persiste >−1pp sur 5-20k tighten +5-10%
- Verdicts paired shadows v144.x (Apr 23-30)
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

### Slip calibration v144 + v14e.6
`_dynamic_sell_slip_factor` : offset global −100 bps + courbe log-continue v14e.6 sur liq mult (1.0 + 0.5·log10(50k/liq), clamped [1.0, 2.5]). Audit Apr 25 N=28 POST : mean |Δ| 3.16pp (vs 9.57pp PRE), max 17.7pp (vs 215pp), 0 outliers >50pp. Recheck Apr 30 avec N≥80 ; tighten +5-10% sur 5-20k si signed bias persiste <−1pp.

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

- **v14e.17** (Apr 25) bat_gamble Solana wipe (32,359 rows deleted, 727 main = −$11,386 + 31K shadows) + ETH kept (23 main = +$1,628 / 6 strats). SOL bankrolls refunded +$11,386 (current 13.56k → 24.95k, total_pnl flipped +$3,943). Per-KOL chain whitelist `GROUPS_DATA[kol]["chains"]` enforced in safe_scraper RT gate (réutilisable). Slippage drift audit en parallèle : modèle continu v14e.6 a divisé outliers par 3 (mean |Δ| 9.57pp → 3.16pp), pas de régression — perception utilisateur infirmée.
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
