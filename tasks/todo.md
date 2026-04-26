# Pipeline Status — Updated Apr 26, 2026 (v14e.27 deployed)

État courant : 34 nouvelles shadow strats post-mega-sweep + 12 AGE clones + age dimension dans le mega-sweep grid + SELL slip health-check récurrent. ETH bleeders demoted en shadow, 3 mains ETH restants. Live SOL = BE25_TP80_SL30 + BOND_FAST_TP50_SL20_T20.

L'historique des décisions se lit dans le git log (commits e8239b4, a6772df, 0deef5b, 4f6ee3c…) — ce TODO ne garde que ce qui est encore à faire.

---

## 🎯 EN COURS — surveillance des 34 nouvelles shadows + AGE clones

Pas de code à écrire, juste observer la data grossir.

### À N≥30 par strat (estimé Mai 02-06)

- [ ] **W1.** Paired-test `SCALP_TP15_SL20_S35` vs base `SCALP_TP15_SL20` (sans filter). Si score gate ajoute >+2pp avg/trade → promouvoir 1-2 SCALP en paper main + Telegram.
- [ ] **W2.** Ranking ETH-only sur les 17 ETH clones — sortir top 3 et virer les autres.
- [ ] **W3.** Paired-test des AGE clones vs leur parent : `AGE24_BE25_TP80_SL30` vs `BE25_TP80_SL30`, etc. Mesurer le delta avg/WR par bande d'âge.

### Mega sweep auto (cron 48h)

- [ ] **W4.** Au prochain run (~Apr 28 21:00 UTC), vérifier le ranking avec age dimension activée. Le top robust devrait maintenant inclure des `(strat, age_band)` couples — voir si AGE24/AGE48 surperforme `ALL` sur les top strats.

---

## 🔧 BACKLOG TECH

### SELL slip drift monitoring (recurring health-check)

- [x] Calibrator twin-pair `scripts/_calibrate_sell_slip.py` créé. Mesure `(pnl_live − paper_sim_pnl_pct)` sur chaque rt_live trade (méthodo v144). Verdict Apr 26 : N=120, weighted median Δ = −0.16 pp → **v144 calibration toujours valide**, pas de code change.
- [ ] **T1.** Câbler le script dans `mega-sweep-48h.yml` workflow comme step post-sweep — log dans artifacts. Détecte automatiquement si la dérive franchit le seuil 1pp.
- [ ] **T2.** Re-run quand N≥200 twin pairs (~Mai 03-05) pour confirmer la stabilité.

### Dead-day filter en RT (priorité basse — exploratoire)

- [ ] **T3.** Brancher le `_compute_day_regime` (sim.py:4252) dans le pipeline RT : flagger `regime=dead` côté safe_scraper et couper les ouvertures de strats non-robust quand le pump_rate live tombe <15%. Hypothèse : BE25/FAST classics arrêtent de saigner les jours dead.
- [ ] **T4.** Tester en shadow d'abord : ajouter un nouveau set de `DEADGATE_*` shadows qui n'ouvrent que les jours non-dead.

---

## 🚧 PHASE A ETH live — SUSPENDU

Mis en pause après le crash ETH paper du Apr 25 PM (148 trades / −$5,628 en 17h, 6 strats demoted).

Reprend quand :
- Les 3 ETH paper mains restantes (`ETH_TP80_SL40_T2H`, `ETH_FAST_TP100_SL20`, `ETH_FAST_TP40_SL30`) atteignent N≥30 chacune avec cum_pnl positif sur 7j glissants
- ET le ranking ETH-only des 17 ETH clones produit ≥1 strat positive cross-régime à N≥20

À ce moment-là, reprendre Phase A (calibration empirique gas + dry-run + swap réel test) — détails en commit history `880f0ea` ou redemander.

---

## 📌 RAPPELS PERSISTANTS

- **KOL whitelist** : actuellement DISABLED. Les 6 KOL destroyers (jadendegens, CarnagecallsGambles, ChairmanDN1, bounty_journal, DegenSeals, explorer_gems) saignent en paper+live. Décision pendante : blacklist dans `live_trading.kol_blacklist` ou activer whitelist mode ?
- **Bankroll** : current $19,766 / starting $23,000 (DD −$3,234, peak $25,831). SOL live = $20 (post nuit Apr 25-26).
- **Triple gate mega-sweep** : FDR<alpha désormais opt-in via `--require-fdr` (default OFF). Top robust publie 30 SCALP_TP15_SL20 variants depuis le run Apr 26.
- **SELL slip model** : v144 calibration (`_dynamic_sell_slip_factor` avec type_bps + GLOBAL_OFFSET=−100) confirmée valide par twin-pair drift checker. Ne PAS toucher sans re-runner `scripts/_calibrate_sell_slip.py` d'abord.
