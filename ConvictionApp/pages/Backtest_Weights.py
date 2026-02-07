# pages/Backtest_Weights.py
# v8 – + Saisie manuelle des prix (éditeur), pré-remplissage calendrier, ffill optionnel

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import streamlit as st

from backtest_weights_core import (
    FeatureWindow, SentimentKnobs,
    try_load_app_dataset, compute_features_from_app,
    PairCache, OhlcvCache,
    build_prices_auto_from_tokens_debug,
    build_prices_from_contracts,
    align_trade_prices,
    fetch_history_birdeye, fetch_history_coingecko,
    _load_user_hints, _save_user_hints
)

st.set_page_config(page_title="Backtest & Weights – Contrats + Optuna", layout="wide")
st.title("🧪 Backtest & Weights – Contrats + Optuna")
st.caption("`pairAddress` manuelle = priorité totale. Si /day est vide, on tente /minute puis on recompose en daily.")

# -------- helpers chemin --------
def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

def resolve_data_dir() -> Path:
    return resolve_repo_root() / "data" / "telegram"

def resolve_pair_cache_path() -> Path:
    return resolve_repo_root() / "cache" / "pair_cache.json"

def resolve_ohlcv_cache_dir() -> Path:
    return resolve_repo_root() / "cache" / "ohlcv"

def resolve_token_hints_path() -> Path:
    return resolve_repo_root() / "cache" / "token_hints.json"

# -------- lecture dataset --------
def get_dataset_from_session_state() -> Optional[pd.DataFrame]:
    for _, v in st.session_state.items():
        if isinstance(v, pd.DataFrame):
            cols = set([c.lower() for c in v.columns])
            if {"id", "date"}.issubset(cols):
                tmp = v.copy()
                if "tokens" not in tmp.columns:
                    tmp["tokens"] = [[] for _ in range(len(tmp))]
                return tmp
    return None

def _tokens_from_signals(df: pd.DataFrame) -> List[str]:
    toks = sorted(set([str(t).upper().replace("$","").strip() for t in df["token"].unique()]))
    return toks

# ---------------- Source des données ----------------
st.subheader("📦 Source des signaux / features")

src = st.radio(
    "D'où viennent les signaux/features ?",
    ["Construire depuis l'app (automatique)", "Uploader CSVs"],
    index=0, horizontal=True
)

signals_df: Optional[pd.DataFrame] = None
prices_df: Optional[pd.DataFrame] = None

if src == "Construire depuis l'app (automatique)":
    with st.expander("Réglages build auto (alignés Dashboard)", expanded=True):
        colA, colB, colC = st.columns(3)
        with colA:
            step_h = st.slider("Pas (heures) pour le lookback", 1, 24, 2, 1)
            bins_n = st.slider("Nombre de pas", 4, 48, 12, 1, help="Ex: 12×2h = 24h d'historique.")
            tau_hours = st.slider("Demi-vie (co-mentions)", 1.0, 48.0, 12.0, 1.0)
        with colB:
            use_hf = st.checkbox("Utiliser modèle local (HF)", value=False)
            w_hf = st.slider("Poids modèle local (HF)", 0.0, 1.0, 0.50, 0.05)
            w_vader = st.slider("Poids VADER", 0.0, 1.0, 0.35, 0.05)
            w_crypto = st.slider("Poids Lexique crypto", 0.0, 1.0, 0.15, 0.05)
        with colC:
            rule_weight = st.slider("Poids règles/boosts lexicaux", 0.0, 2.0, 1.0, 0.05)
            group_alpha = st.slider("Poids conviction groupe", 0.0, 2.0, 1.0, 0.05)
            alias_no_dollar = st.checkbox("Détecter alias sans $", value=True)

        if st.button("🧱 Construire les features", type="secondary"):
            raw = get_dataset_from_session_state()
            if raw is None or raw.empty:
                data_dir = resolve_data_dir()
                if not data_dir.exists():
                    st.error(f"Dossier introuvable: {data_dir}")
                    st.stop()
                raw = try_load_app_dataset(str(data_dir))

            if raw is None or raw.empty:
                st.error("Dataset vide. Dépose des JSON dans data/telegram/ ou charge via l’app (autres pages).")
                st.stop()

            win = FeatureWindow(step_h=step_h, bins_n=bins_n)
            knobs = SentimentKnobs(
                use_hf=use_hf, w_hf=w_hf, w_vader=w_vader, w_crypto=w_crypto,
                rule_weight=rule_weight, group_alpha=group_alpha, alias_no_dollar=alias_no_dollar
            )
            with st.spinner("Construction des features…"):
                feats = compute_features_from_app(raw, win, knobs, tau_hours=tau_hours)

            if feats is None or feats.empty:
                st.warning("Aucun token dans la fenêtre sélectionnée.")
            else:
                st.success(f"Features construites pour {feats.shape[0]} tokens.")
                st.session_state["_features_auto"] = feats

    _fe = st.session_state.get("_features_auto")
    if _fe is not None and not _fe.empty:
        st.dataframe(_fe.head(50), use_container_width=True)
        signals_df = _fe.copy()
else:
    st.info("Uploader **signals_features.csv** (token, ts_signal_utc, horizon, <features…>) et **prices.csv** (token, ts_price_utc, price).")
    up1, up2 = st.columns(2)
    with up1:
        f_sig = st.file_uploader("signals_features.csv", type=["csv"])
    with up2:
        f_prc = st.file_uploader("prices.csv", type=["csv"])
    if f_sig:
        signals_df = pd.read_csv(f_sig, parse_dates=["ts_signal_utc"])
    if f_prc:
        prices_df = pd.read_csv(f_prc, parse_dates=["ts_price_utc"])

if signals_df is None or signals_df.empty:
    st.stop()

# ---------------- 📇 Associer tokens → contrats/pools ----------------
st.subheader("📇 Associer tokens → contrats / pools")

tokens = _tokens_from_signals(signals_df)
hints_path = resolve_token_hints_path()
existing_hints = _load_user_hints(str(hints_path))

rows = []
for t in tokens:
    rec = existing_hints.get(t, {})
    rows.append({
        "token": t,
        "chainId": rec.get("chainId", ""),
        "address": rec.get("address", ""),
        "pairAddress": rec.get("pairAddress", ""),
    })
contracts_df_default = pd.DataFrame(rows)

contracts_df_prev = st.session_state.get("_manual_contracts_df")
if isinstance(contracts_df_prev, pd.DataFrame) and not contracts_df_prev.empty:
    contracts_df = (pd.concat([contracts_df_default, contracts_df_prev], ignore_index=True)
                    .sort_values("token")
                    .drop_duplicates(subset=["token"], keep="last")
                    .reset_index(drop=True))
else:
    contracts_df = contracts_df_default.copy()

chains = ["solana","ethereum","base","bsc","arbitrum","polygon","avax","optimism","fantom",
          "sei","blast","zksync","linea","scroll","opbnb"]

contracts_df = st.data_editor(
    contracts_df,
    num_rows="dynamic",
    use_container_width=True,
    key="contracts_editor",
    column_config={
        "token": st.column_config.TextColumn("Token", disabled=True),
        "chainId": st.column_config.SelectboxColumn("Chain", options=chains, required=False),
        "address": st.column_config.TextColumn("Contract address (CA)"),
        "pairAddress": st.column_config.TextColumn("Pair/pool (optionnel, prioritaire si présent)"),
    }
)

cA, cB, cC = st.columns([1,1,1])
with cA:
    btn_save = st.button("💾 Enregistrer dans hints (facultatif)", type="primary")
with cB:
    btn_clear = st.button("🧹 Supprimer lignes vides", type="secondary")
with cC:
    strict_manual = st.checkbox("🔒 Strict manual (pas de fallback auto si ligne présente)", value=True)

if btn_clear:
    contracts_df = contracts_df[(contracts_df["chainId"].astype(str) != "") & (contracts_df["address"].astype(str) != "")]
    st.session_state["_manual_contracts_df"] = contracts_df.reset_index(drop=True)
    st.experimental_rerun()

if btn_save:
    valid = contracts_df.copy()
    valid["token"] = valid["token"].astype(str).str.upper().str.replace("$","", regex=False).str.strip()
    valid = valid[(valid["token"]!="") & (valid["chainId"].astype(str)!="") & (valid["address"].astype(str)!="")]
    hints = {**existing_hints}
    for _, r in valid.iterrows():
        hints[r["token"]] = {
            "chainId": str(r["chainId"]).strip().lower(),
            "address": str(r["address"]).strip(),
            "pairAddress": str(r.get("pairAddress","")).strip()
        }
    _save_user_hints(hints, str(hints_path))
    st.session_state["_manual_contracts_df"] = valid.reset_index(drop=True)
    st.success(f"{len(valid)} lignes enregistrées dans {hints_path.name}.")

# ---------------- Prix ----------------
st.subheader("💵 Prix")

price_mode = st.radio(
    "Source des prix",
    [
        "Saisie manuelle (éditeur)",                      # <-- NOUVEAU
        "Contrats/pools saisis (prioritaire)",
        "Automatique (Dexscreener -> GeckoTerminal + caches)",
        "Uploader prices.csv",
        "API Birdeye / CoinGecko Pro"
    ],
    index=0, horizontal=True
)

token_map: Optional[pd.DataFrame] = None
birdeye_key = None
cg_key = None

with st.expander("⚙️ Caches & Hints", expanded=False):
    cache_path = st.text_input("Chemin du cache paires", value=str(resolve_pair_cache_path()))
    ttl_pairs = st.number_input("TTL paires (jours)", min_value=1, max_value=90, value=7, step=1)
    ohlcv_dir = st.text_input("Dossier cache OHLCV", value=str(resolve_ohlcv_cache_dir()))
    ttl_ohlcv = st.number_input("TTL OHLCV (jours)", min_value=1, max_value=30, value=3, step=1)
    max_pages = st.number_input("Pagination max GeckoTerminal", min_value=1, max_value=20, value=8, step=1)
    pcache = PairCache(path=cache_path, ttl_days=int(ttl_pairs))
    ocache = OhlcvCache(dirpath=ohlcv_dir, ttl_days=int(ttl_ohlcv))
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Voir le cache paires"):
            st.json({"items": pcache.data, "count": len(pcache.data)})
    with c2:
        if st.button("Vider cache paires"):
            pcache.clear()
            st.success("Cache des paires vidé.")

# ==== NOUVEAU : ÉDITEUR DE PRIX MANUELS ====
manual_prices_df: Optional[pd.DataFrame] = None
manual_resample_ffill = st.session_state.get("_manual_resample_ffill", True)

def _prep_manual_template(tokens: List[str], ts_min: pd.Timestamp, ts_max: pd.Timestamp) -> pd.DataFrame:
    # calendrier 1D, une ligne par (token, date)
    dr = pd.date_range(ts_min.floor("D"), ts_max.ceil("D"), freq="1D")
    idx = pd.MultiIndex.from_product([tokens, dr], names=["token", "ts_price_utc"])
    df = pd.DataFrame(index=idx).reset_index()
    df["token"] = df["token"].astype(str).str.upper().str.replace("$","", regex=False).str.strip()
    df["price"] = np.nan
    return df

if price_mode == "Saisie manuelle (éditeur)":
    tmin_sig = pd.to_datetime(signals_df["ts_signal_utc"]).min()
    tmax_sig = pd.to_datetime(signals_df["ts_signal_utc"]).max()
    horizon_max = max( int(signals_df.get("horizon","mid").map({"mid":30,"long":90}).fillna(30).max()) , 30)
    tmax_need = tmax_sig + pd.Timedelta(days=horizon_max)

    with st.expander("📝 Saisie manuelle des prix (collez depuis Excel possible)", expanded=True):
        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            prefill = st.checkbox("Pré-remplir calendrier (1D)", value=not bool(st.session_state.get("_manual_prices_editor")))
        with c2:
            manual_resample_ffill = st.checkbox("Remplir jours manquants (ffill)", value=bool(manual_resample_ffill),
                                                help="Après saisie, on re-échantillonne en quotidien et propage la dernière valeur.")
            st.session_state["_manual_resample_ffill"] = manual_resample_ffill
        with c3:
            st.download_button(
                "⬇️ Template CSV",
                data="token,ts_price_utc,price\nPEPE,2024-01-01,0.000001\n",
                file_name="prices_template.csv", mime="text/csv"
            )

        if prefill and st.button("🧩 Générer les lignes (tokens × jours)"):
            st.session_state["_manual_prices_editor"] = _prep_manual_template(tokens, tmin_sig, tmax_need)

        # éditeur
        manual_prices_df = st.data_editor(
            st.session_state.get("_manual_prices_editor", pd.DataFrame(columns=["token","ts_price_utc","price"])),
            num_rows="dynamic",
            use_container_width=True,
            key="manual_prices_editor",
            column_config={
                "token": st.column_config.TextColumn("Token"),
                "ts_price_utc": st.column_config.DatetimeColumn("Date (UTC)"),
                "price": st.column_config.NumberColumn("Prix USD", step=0.00000001, format="%.10f"),
            }
        )
        # sauvegarde en session
        if isinstance(manual_prices_df, pd.DataFrame):
            st.session_state["_manual_prices_editor"] = manual_prices_df.copy()

# ==== existants ====
if price_mode == "API Birdeye / CoinGecko Pro":
    st.markdown("Fournis un mapping **token_map.csv** (`token,chain,address,cg_id`) + clés API si dispo.")
    f_map = st.file_uploader("token_map.csv", type=["csv"])
    if f_map:
        token_map = pd.read_csv(f_map)
        st.dataframe(token_map.head(20), use_container_width=True)
    birdeye_key = st.text_input("Birdeye API KEY (optionnel)", type="password")
    cg_key = st.text_input("CoinGecko PRO API KEY (optionnel)", type="password")

prices_csv_df: Optional[pd.DataFrame] = None
if price_mode == "Uploader prices.csv":
    if prices_df is None:
        st.warning("Upload prices.csv ou choisis un autre mode.")
    else:
        prices_csv_df = prices_df.copy()

# ---------------- Choix des features ----------------
num_cols = [c for c in signals_df.columns if c not in ["token", "ts_signal_utc", "horizon"] and np.issubdtype(signals_df[c].dtype, np.number)]
st.subheader("🧮 Choix des features (pondérées)")
default_feats = [c for c in ["score_conviction_graph","score_conviction","score_quick_win",
                             "pagerank","breadth","polarisation","wilson_low",
                             "momentum","accel","mentions","sentiment"] if c in num_cols]
features = st.multiselect(
    "Sélectionne les features",
    options=num_cols,
    default=default_feats if default_feats else (num_cols[:6] if len(num_cols) >= 6 else num_cols)
)
if not features:
    st.warning("Sélectionne au moins une feature.")
    st.stop()

# ---------------- Horizons & règle d’entrée ----------------
st.subheader("⏱️ Horizons & règle d’entrée")
c1, c2, c3 = st.columns(3)
with c1:
    horizon_mid = st.select_slider("Horizon MID (jours)", options=[21, 30, 45], value=30)
    horizon_long = st.select_slider("Horizon LONG (jours)", options=[60, 90, 120], value=90)
with c2:
    select_mode = st.selectbox("Mode de sélection", ["threshold", "topn"])
    threshold = None
    topn = None
    if select_mode == "threshold":
        threshold = st.slider("Seuil du score (z-score)", -3.0, 3.0, 0.0, 0.1)
    else:
        topn = st.select_slider("Top-N / jour", options=[1, 2, 3, 5, 10], value=3)
with c3:
    metric = st.selectbox("Objectif d’optimisation", ["winrate", "median_return"], index=0)
    folds = st.number_input("Folds (walk-forward)", 2, 12, 4, 1)
    test_days = st.number_input("Taille de chaque fold (jours)", 30, 365, 90, 15)
    opt_trials = st.number_input("Essais Optuna", 50, 2000, 300, 50)

min_trades_per_fold = st.number_input("Min trades / fold", 5, 200, 20)
min_total_trades = st.number_input("Min total trades (pénalité winrate)", 20, 5000, 100)

# ---------------- Helpers prix ----------------
def _normalize_prices_table(df: pd.DataFrame, do_resample_ffill: bool) -> Dict[str, pd.DataFrame]:
    """
    Attend colonnes: token, ts_price_utc, price
    - nettoie, uppercase token
    - convertit ts
    - option: resample quotidien + ffill
    Retour: dict token -> dataframe [ts_price_utc, price] trié
    """
    out: Dict[str, pd.DataFrame] = {}
    if df is None or df.empty:
        return out
    tmp = df.copy()
    tmp["token"] = tmp["token"].astype(str).str.upper().str.replace("$","", regex=False).str.strip()
    tmp["ts_price_utc"] = pd.to_datetime(tmp["ts_price_utc"], utc=True, errors="coerce").dt.tz_convert(None)
    tmp["price"] = pd.to_numeric(tmp["price"], errors="coerce")
    tmp = tmp.dropna(subset=["token","ts_price_utc","price"])
    for tok, g in tmp.groupby("token"):
        gd = g[["ts_price_utc","price"]].dropna().sort_values("ts_price_utc").reset_index(drop=True)
        if gd.empty:
            continue
        if do_resample_ffill:
            dfr = (gd.set_index("ts_price_utc")
                     .resample("1D").last()
                     .ffill()
                     .dropna()
                     .reset_index())
            gd = dfr
        out[tok] = gd
    return out

# ---------------- Prix par token ----------------
@st.cache_data(show_spinner=False, ttl=600)
def build_prices_by_token(signals: pd.DataFrame,
                          price_mode: str,
                          prices_csv_df: Optional[pd.DataFrame],
                          token_map: Optional[pd.DataFrame],
                          birdeye_key: Optional[str],
                          cg_key: Optional[str],
                          manual_contracts_df: Optional[pd.DataFrame],
                          pcache_path: str,
                          pcache_ttl: int,
                          ohlcv_dir: str,
                          ohlcv_ttl: int,
                          gt_max_pages: int,
                          strict_manual: bool,
                          manual_prices_df: Optional[pd.DataFrame],
                          manual_resample_ffill: bool) -> Tuple[Dict[str, pd.DataFrame], List[dict]]:
    out: Dict[str, pd.DataFrame] = {}
    dbg: List[dict] = []

    tmin = pd.to_datetime(signals["ts_signal_utc"]).min()
    tmax = pd.to_datetime(signals["ts_signal_utc"]).max() + pd.Timedelta(days=int(max(horizon_mid, horizon_long)))

    # -1) Saisie manuelle (prioritaire si choisie)
    if price_mode == "Saisie manuelle (éditeur)":
        mp = manual_prices_df if (manual_prices_df is not None and not manual_prices_df.empty) else st.session_state.get("_manual_prices_editor")
        if mp is None or mp.empty:
            return {}, [{"status":"no_manual_rows"}]
        out = _normalize_prices_table(mp, do_resample_ffill=bool(manual_resample_ffill))
        for tok, dfp in out.items():
            dbg.append({"token": tok, "status": "ok_manual", "n_candles": int(len(dfp))})
        return out, dbg

    # 0) Contrats/pools saisis (prioritaire)
    if price_mode == "Contrats/pools saisis (prioritaire)":
        pcache = PairCache(path=pcache_path, ttl_days=int(pcache_ttl))
        ocache = OhlcvCache(dirpath=ohlcv_dir, ttl_days=int(ohlcv_ttl))

        manual = pd.DataFrame(columns=["token","chainId","address","pairAddress"])
        editor_df = st.session_state.get("contracts_editor")
        if isinstance(editor_df, pd.DataFrame) and not editor_df.empty:
            manual = editor_df.copy()
        elif isinstance(manual_contracts_df, pd.DataFrame) and not manual_contracts_df.empty:
            manual = manual_contracts_df.copy()

        if not manual.empty:
            manual["token"] = manual["token"].astype(str).str.upper().str.replace("$","", regex=False).str.strip()
            manual = manual[(manual["token"]!="") & (manual["chainId"].astype(str)!="") & (manual["address"].astype(str)!="")]
            manual = manual[manual["token"].isin([str(t).upper().replace("$","").strip() for t in signals["token"].unique()])]

        manual_tokens: set = set()
        if not manual.empty:
            px, dbg1, manual_tokens = build_prices_from_contracts(
                manual, tmin, tmax, pair_cache=pcache, ohlcv_cache=ocache,
                hints_path=str(resolve_token_hints_path()), max_pages=int(gt_max_pages),
                strict_manual=bool(strict_manual)
            )
            out.update(px)
            dbg.extend(dbg1)

        # Fallback auto UNIQUEMENT pour tokens SANS ligne manuelle (ou strict_manual=False)
        remaining = sorted(set([str(t).upper().replace("$","").strip() for t in signals["token"].unique()]) - set(out.keys()))
        if remaining:
            if strict_manual:
                remaining = [t for t in remaining if t not in manual_tokens]
            if remaining:
                px2, dbg2 = build_prices_auto_from_tokens_debug(
                    remaining, tmin, tmax, pair_cache=pcache, ohlcv_cache=ocache, max_pages=int(gt_max_pages)
                )
                out.update(px2)
                dbg.extend(dbg2)
        return out, dbg

    # 1) CSV
    if price_mode == "Uploader prices.csv" and prices_csv_df is not None and not prices_csv_df.empty:
        out = _normalize_prices_table(prices_csv_df, do_resample_ffill=True)
        for tok, dfp in out.items():
            dbg.append({"token": tok, "status": "ok_csv", "n_candles": int(len(dfp))})
        return out, dbg

    # 2) Automatique
    if price_mode.startswith("Automatique"):
        toks = [str(t).upper().replace("$", "").strip() for t in signals["token"].unique().tolist()]
        pcache = PairCache(path=pcache_path, ttl_days=int(pcache_ttl))
        ocache = OhlcvCache(dirpath=ohlcv_dir, ttl_days=int(ohlcv_ttl))
        out, dbg = build_prices_auto_from_tokens_debug(
            toks, tmin, tmax, pair_cache=pcache, ohlcv_cache=ocache, max_pages=int(gt_max_pages)
        )
        return out, dbg

    # 3) Birdeye / CG Pro
    if price_mode == "API Birdeye / CoinGecko Pro":
        if token_map is None or token_map.empty:
            return {}, []
        for _, row in token_map.iterrows():
            tok = str(row.get("token", "")).upper().replace("$", "").strip()
            chain = row.get("chain", None)
            address = row.get("address", None)
            cg_id = row.get("cg_id", None)

            df_b = fetch_history_birdeye(chain, address, tmin, tmax, birdeye_key) if chain and address else pd.DataFrame()
            df_c = fetch_history_coingecko(cg_id, tmin, tmax, cg_key) if cg_id else pd.DataFrame()
            df_use = df_b if not df_b.empty else df_c
            if not df_use.empty:
                out[tok] = df_use
                dbg.append({"token": tok, "status": "ok_api", "n_candles": int(len(df_use))})
            else:
                dbg.append({"token": tok, "status": "no_prices_api"})
        return out, dbg

    return out, dbg

# ---------------- Standardisation robuste ----------------
def robust_standardize(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    z = df.copy()
    for c in cols:
        x = z[c].astype(float).values
        med = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - med))
        scale = 1.4826 * mad if mad > 0 else (np.nanstd(x) or 1.0)
        z[c] = (x - med) / (scale if scale != 0 else 1.0)
    return z

# ---------------- Évaluation OOS ----------------
def evaluate_oos(signals_z: pd.DataFrame,
                 prices_by_token: Dict[str, pd.DataFrame],
                 weights: List[float],
                 select_mode: str,
                 threshold: Optional[float],
                 topn: Optional[int],
                 horizon_days: Dict[str, int],
                 metric: str,
                 folds: int,
                 test_days: int,
                 min_trades_per_fold: int,
                 min_total_trades: int):
    w = np.array(weights, dtype=float); s = w.sum() or 1.0; w = w / s
    sc = (signals_z[features].astype(float).values * w).sum(axis=1)
    sdf = signals_z.copy(); sdf["score"] = sc

    tmin = sdf["ts_signal_utc"].min(); tmax = sdf["ts_signal_utc"].max()
    total_days = int((tmax - tmin).days) if pd.notna(tmax) and pd.notna(tmin) else 0
    folds = max(1, min(int(folds), total_days // max(1, int(test_days)))) if total_days > 0 else 1
    step = max(1, (total_days - int(test_days)) // max(1, folds)) if total_days > int(test_days) else 1
    starts = [tmin + pd.Timedelta(days=i * step) for i in range(folds)] if total_days > 0 else [tmin]

    fold_scores, total_trades, diag = [], 0, {"folds": []}
    for stt in starts:
        enn = stt + pd.Timedelta(days=int(test_days))
        test = sdf[(sdf["ts_signal_utc"] >= stt) & (sdf["ts_signal_utc"] < enn)].copy()
        if select_mode == "threshold":
            th = threshold if threshold is not None else 0.0
            sel = test[test["score"] >= th]
        else:
            test["date"] = test["ts_signal_utc"].dt.floor("D")
            sel = (test.sort_values(["date", "score"], ascending=[True, False])
                   .groupby("date", as_index=False, group_keys=False).head(int(topn or 3)))
            sel = sel.drop(columns=["date"])
        if sel.empty:
            diag["folds"].append({"start": stt, "end": enn, "n_trades": 0, "score": None})
            continue

        trades = align_trade_prices(sel[["token", "ts_signal_utc", "horizon"]],
                                    prices_by_token,
                                    {"mid": horizon_mid, "long": horizon_long})
        n = len(trades); total_trades += n
        if n < int(min_trades_per_fold) or n == 0:
            score_val = np.nan
        else:
            score_val = trades["ret"].median() if metric == "median_return" else trades["win"].mean()
        diag["folds"].append({"start": stt, "end": enn, "n_trades": n,
                              "score": None if np.isnan(score_val) else float(score_val)})
        if not np.isnan(score_val):
            fold_scores.append(score_val)

    if len(fold_scores) == 0:
        return 0.0, total_trades, diag
    avg = float(np.mean(fold_scores))
    penalty = 0.10 if (metric == "winrate" and total_trades < int(min_total_trades)) else 0.0
    return max(0.0, avg - penalty), total_trades, diag

def build_objective(signals: pd.DataFrame, prices_by_token: Dict[str, pd.DataFrame]):
    z = robust_standardize(signals, features)
    def objective(trial: optuna.trial.Trial) -> float:
        w = [trial.suggest_float(f"w_{c}", 0.0, 1.0) for c in features]
        if select_mode == "threshold":
            thr = trial.suggest_float("threshold", (threshold or 0.0) - 0.5, (threshold or 0.0) + 0.5)
            tn = None
        else:
            thr = None
            tn = trial.suggest_categorical("topn", [topn, 1, 2, 3, 5, 10])
        score, total_trades, _ = evaluate_oos(
            z, prices_by_token, w, select_mode, thr, tn,
            {"mid": horizon_mid, "long": horizon_long},
            metric, folds, test_days, min_trades_per_fold, min_total_trades
        )
        trial.set_user_attr("total_trades", int(total_trades))
        return float(score)
    return objective

# ---------------- RUN ----------------
col_run1, col_run2 = st.columns([1, 1])
with col_run1:
    run = st.button("🚀 Optimiser (Optuna)", type="primary", use_container_width=True)
with col_run2:
    show_diag = st.checkbox("Afficher diagnostics par fold", value=True)

if run:
    manual_contracts_df = None
    ed = st.session_state.get("contracts_editor")
    if isinstance(ed, pd.DataFrame) and not ed.empty:
        manual_contracts_df = ed.copy()
    else:
        manual_contracts_df = st.session_state.get("_manual_contracts_df")

    with st.spinner("Préparation des prix…"):
        prices_by_token, dbg = build_prices_by_token(
            signals_df, price_mode, prices_csv_df, token_map, birdeye_key, cg_key,
            manual_contracts_df=manual_contracts_df,
            pcache_path=str(resolve_pair_cache_path()), pcache_ttl=7,
            ohlcv_dir=str(resolve_ohlcv_cache_dir()), ohlcv_ttl=3,
            gt_max_pages=8, strict_manual=bool(strict_manual),
            manual_prices_df=manual_prices_df,
            manual_resample_ffill=bool(manual_resample_ffill)
        )

    with st.expander("🔎 Diagnostics résolution des prix", expanded=True):
        if dbg:
            st.dataframe(pd.DataFrame(dbg), use_container_width=True)
        missing = sorted(set([t for t in _tokens_from_signals(signals_df) if t not in prices_by_token.keys()]))
        if missing:
            st.warning(f"Tokens sans prix: {missing[:50]}{' …' if len(missing)>50 else ''}")
            st.markdown(
                "➡️ Si tu as renseigné une `pairAddress` et que tu vois `gt_404_pool`, c’est que **cette valeur n’est pas une pool** "
                "(ex: un mint SOL ou un contrat ERC20) ou que la pool n’est pas encore indexée chez GeckoTerminal. "
                "Vérifie l’adresse de **POOL** exacte."
            )

    if len(prices_by_token) == 0:
        st.error("Pas de prix disponibles. Utilise le mode 'Saisie manuelle (éditeur)' / 'Uploader CSV' ou vérifie tes pools.")
        st.stop()

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    with st.spinner("Recherche des meilleurs poids…"):
        study.optimize(build_objective(signals_df, prices_by_token), n_trials=int(opt_trials), show_progress_bar=True)
    st.success("Optimisation terminée.")

    best = study.best_params
    best_weights = {k.replace("w_", ""): v for k, v in best.items() if k.startswith("w_")}
    opt_threshold = best.get("threshold", threshold if select_mode == "threshold" else None)
    opt_topn = best.get("topn", topn if select_mode == "topn" else None)

    st.subheader("🏆 Meilleurs paramètres trouvés")
    st.json({
        "features": features,
        "weights": best_weights,
        "threshold": opt_threshold,
        "topn_per_day": opt_topn,
        "horizon_days": {"mid": int(horizon_mid), "long": int(horizon_long)},
        "metric": metric,
        "folds": int(folds),
        "test_days": int(test_days)
    })

    # Recalc diag + trades
    z = robust_standardize(signals_df, features)
    w_list = [best_weights[c] for c in features]
    sc = (z[features].astype(float).values * (np.array(w_list) / (sum(w_list) or 1.0))).sum(axis=1)
    z2 = z.copy(); z2["score"] = sc

    if select_mode == "threshold":
        best_sel = z2[z2["score"] >= float(opt_threshold)]
    else:
        z2["date"] = z2["ts_signal_utc"].dt.floor("D")
        best_sel = (z2.sort_values(["date", "score"], ascending=[True, False])
                    .groupby("date", as_index=False, group_keys=False).head(int(opt_topn or 3)))
        best_sel = best_sel.drop(columns=["date"])

    horizon_days = {"mid": int(horizon_mid), "long": int(horizon_long)}
    trades = align_trade_prices(best_sel[["token", "ts_signal_utc", "horizon"]],
                                prices_by_token, horizon_days)

    st.metric("Trades totaux (sélection finale)", f"{len(trades)}")
    if show_diag:
        sc_best, total_tr, diag = evaluate_oos(z, prices_by_token, w_list,
                                               select_mode, opt_threshold, opt_topn,
                                               horizon_days, metric, folds, test_days,
                                               min_trades_per_fold, min_total_trades)
        st.metric("Score OOS (moyenne folds)", f"{sc_best:.4f}")
        st.write(pd.DataFrame(diag["folds"]))

    st.subheader("📈 Trades sélectionnés")
    st.dataframe(trades.head(200), use_container_width=True)

    # Exports
    csv_trades = trades.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Télécharger trades.csv", data=csv_trades, file_name="trades.csv", mime="text/csv")

    best_config = {
        "features": features,
        "weights": best_weights,
        "select_mode": select_mode,
        "threshold": float(opt_threshold) if opt_threshold is not None else None,
        "topn_per_day": int(opt_topn) if opt_topn is not None else None,
        "horizon_days": horizon_days,
        "metric": metric,
        "folds": int(folds),
        "test_days": int(test_days),
        "min_trades_per_fold": int(min_trades_per_fold),
        "min_total_trades": int(min_total_trades),
        "price_mode": price_mode
    }
    st.download_button("⬇️ Télécharger best_config.json",
                       data=json.dumps(best_config, indent=2),
                       file_name="best_config.json",
                       mime="application/json")
else:
    st.info("Renseigne (si besoin) `chainId`, `address` et la **pairAddress (pool)**. "
            "Astuce Solana : la **pool Raydium/Orca** n’est pas le **mint** du token.")
