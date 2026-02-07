# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import json
import networkx as nx
import plotly.express as px

from utils import (
    load_many_jsons,
    parse_messages_json,
    add_sentiment,
    explode_tokens,
    summarizer_available,  # indicateur
    # --- UI/dataset unifiés ---
    ensure_session_dataset,
    ui_dataset_loader,
    ui_status_banner,
    remix_sentiment_weights,
    dataset_signature,
)

st.set_page_config(layout="wide")
st.title("💹 Investissement — Consensus & Classement des tokens")

# Dataset persistant entre pages
ensure_session_dataset()

# ---------------- Session defaults ----------------
def ensure_defaults():
    ss = st.session_state
    # période
    ss.setdefault("period", "24h")
    ss.setdefault("use_custom_period", False)
    ss.setdefault("custom_start_date", None)
    ss.setdefault("custom_start_time", None)
    ss.setdefault("custom_end_date", None)
    ss.setdefault("custom_end_time", None)

    # sentiment (recalcule léger si besoin)
    ss.setdefault("use_hf", True)
    ss.setdefault("w_hf", 0.50)
    ss.setdefault("w_vader", 0.35)
    ss.setdefault("w_crypto", 0.15)
    ss.setdefault("rule_weight", 1.0)
    ss.setdefault("group_alpha", 1.0)

    # consensus & score (investissables)
    ss.setdefault("cons_top_k", 5)            # Top-K intra-groupe
    ss.setdefault("cons_min_groups", 2)       # min groupes retenus
    ss.setdefault("min_avg_sent", 0.0)        # filtre avg_sent
    ss.setdefault("topN_display", 25)         # nb lignes affichées

    # poids du score final (0..10)
    ss.setdefault("w_score", 0.35)            # avg_score (0..10 -> /10)
    ss.setdefault("w_sent", 0.25)             # avg_sent [-1,1] -> [0,1]
    ss.setdefault("w_pr", 0.20)               # PageRank token
    ss.setdefault("w_groups", 0.20)           # groups_count (normalisé)

    # Demie-vie / graphe
    ss.setdefault("tau_hours", 12.0)          # pondération temporelle co-mentions

    # Bump chart "investissables maintenant"
    ss.setdefault("inv_bump_show", False)
    ss.setdefault("inv_bump_step_h", 2)       # pas (h)
    ss.setdefault("inv_bump_bins", 12)        # nb de pas (ex 12x2h=24h)

    # Mode expert (Super Classement)
    ss.setdefault("exp_show", False)
    ss.setdefault("exp_step_h", 2)            # pas temporel pour features
    ss.setdefault("exp_bins", 12)             # nb de pas (lookback)

    # Poids super score (tous positifs sauf polar_penalty)
    ss.setdefault("exp_w_quality_sent", 0.22)
    ss.setdefault("exp_w_quality_wilson", 0.12)
    ss.setdefault("exp_w_cons_breadth", 0.14)
    ss.setdefault("exp_w_cons_groups", 0.10)
    ss.setdefault("exp_w_network_pr", 0.15)
    ss.setdefault("exp_w_dyn_mom", 0.17)
    ss.setdefault("exp_w_dyn_acc", 0.05)
    ss.setdefault("exp_w_stability", 0.10)
    ss.setdefault("exp_w_polar_penalty", 0.15)  # pénalité
    ss.setdefault("exp_topN", 15)
    ss.setdefault("exp_show_bump", True)

    # BONUS PERSISTANCE (musclé)
    ss.setdefault("exp_use_persist", False)        # ON/OFF
    ss.setdefault("exp_persist_hours", 24)         # fenêtre (6..168h)
    ss.setdefault("exp_persist_rank", 5)           # Top-R
    ss.setdefault("exp_w_persist", 0.25)           # poids de base
    ss.setdefault("exp_persist_gain", 0.75)        # amplification (0..2)
    ss.setdefault("exp_persist_multiplicative", True)  # mode fort

    # 🔹 Réglage d’alias $
    ss.setdefault("alias_no_dollar", False)  # False = uniquement via $ ; True = inclure alias sans $

    # Data partagé
    ss.setdefault("RAW_ALL", pd.DataFrame())
    ss.setdefault("RAW_DF", pd.DataFrame())

ensure_defaults()

# ---------------- Sidebar ----------------
st.sidebar.title("⚙️ Configuration")

# --- UI unifié Sources & chargement (multi JSON + reset + compteur) ---
raw_all = ui_dataset_loader(page_key="invest")

def _custom_range_datetimes():
    try:
        sdate = st.session_state.get("custom_start_date")
        stime = st.session_state.get("custom_start_time")
        edate = st.session_state.get("custom_end_date")
        etime = st.session_state.get("custom_end_time")
        if not (sdate and stime and edate and etime):
            return None, None
        start = pd.Timestamp.combine(sdate, stime).tz_localize("Europe/Paris").tz_convert(None)
        end = pd.Timestamp.combine(edate, etime).tz_localize("Europe/Paris").tz_convert(None)
        return start, end
    except Exception:
        return None, None

with st.sidebar.expander("Période d'analyse", expanded=True):
    st.session_state["period"] = st.radio(
        "Fenêtre",
        ["2h","6h","12h","24h","48h","Tout"],
        index=["2h","6h","12h","24h","48h","Tout"].index(st.session_state["period"]),
        horizontal=True,
        help="Fenêtre d’analyse appliquée aux messages.",
    )
    st.session_state["use_custom_period"] = st.checkbox(
        "Utiliser une plage personnalisée", value=st.session_state["use_custom_period"]
    )
    if st.session_state["use_custom_period"]:
        c1, c2 = st.columns(2)
        st.session_state["custom_start_date"] = c1.date_input(
            "Date début (Europe/Paris)", value=st.session_state.get("custom_start_date", None)
        )
        st.session_state["custom_start_time"] = c1.time_input(
            "Heure début", value=st.session_state.get("custom_start_time", None)
        )
        st.session_state["custom_end_date"] = c2.date_input(
            "Date fin (Europe/Paris)", value=st.session_state.get("custom_end_date", None)
        )
        st.session_state["custom_end_time"] = c2.time_input(
            "Heure fin", value=st.session_state.get("custom_end_time", None)
        )
        st.caption("Quand activée, la plage personnalisée remplace la fenêtre rapide.")

with st.sidebar.expander("Sentiment – canaux & poids (local)", expanded=False):
    st.session_state["use_hf"] = st.checkbox("Utiliser modèle local (HF)", value=st.session_state["use_hf"], help="Active le blend local (HF + VADER + lexique).")
    st.session_state["w_hf"] = st.slider("Poids modèle local (HF)", 0.0, 1.0, st.session_state["w_hf"], 0.05)
    st.session_state["w_vader"] = st.slider("Poids VADER", 0.0, 1.0, st.session_state["w_vader"], 0.05)
    st.session_state["w_crypto"] = st.slider("Poids Lexique crypto", 0.0, 1.0, st.session_state["w_crypto"], 0.05)
    st.session_state["rule_weight"] = st.slider("Poids règles/boosts lexicaux", 0.0, 1.0, st.session_state["rule_weight"], 0.05, help="Influence des mots-clés (ATH, listing, rug...).")
    st.session_state["group_alpha"] = st.slider("Poids conviction de groupe", 0.0, 2.0, st.session_state["group_alpha"], 0.1, help="Renforce le sentiment des groupes très confiants.")

with st.sidebar.expander("Consensus & Score", expanded=True):
    st.session_state["cons_top_k"] = st.slider("Top-K intra-groupe (consensus)", 3, 10, st.session_state["cons_top_k"], 1, help="Un token compte dans un groupe s’il est dans son Top-K local.")
    st.session_state["cons_min_groups"] = st.slider("Min groupes (consensus)", 1, 10, st.session_state["cons_min_groups"], 1, help="Nombre minimum de groupes qui poussent le token.")
    st.session_state["min_avg_sent"] = st.slider("Filtre: avg_sent min", -1.0, 1.0, float(st.session_state["min_avg_sent"]), 0.05, help="Retire les tokens dont la moyenne de sentiment inter-groupes est trop faible.")
    st.session_state["topN_display"] = st.slider("Top N à afficher", 10, 100, st.session_state["topN_display"], 5)

    st.markdown("**Poids du score final (0→10)**")
    st.session_state["w_score"]   = st.slider("Poids avg_score (0→10)  ?", 0.0, 1.0, st.session_state["w_score"], 0.05, help="Score intra-groupe (mentions+sentiment) moyen sur les groupes.")
    st.session_state["w_sent"]    = st.slider("Poids avg_sent (−1→1)  ?", 0.0, 1.0, st.session_state["w_sent"], 0.05, help="Biais moyen de sentiment inter-groupes.")
    st.session_state["w_pr"]      = st.slider("Poids PR_token       ?", 0.0, 1.0, st.session_state["w_pr"], 0.05, help="Importance structurelle (PageRank) dans le graphe de co-mentions.")
    st.session_state["w_groups"]  = st.slider("Poids groups_count   ?", 0.0, 1.0, st.session_state["w_groups"], 0.05, help="Convergence: combien de groupes poussent le token.")
    st.caption("Conseil: garde la somme ≈ 1.0 (mais ce n’est pas obligatoire).")

with st.sidebar.expander("Paramètres graphe", expanded=False):
    st.session_state["tau_hours"] = st.slider("Demi-vie temporelle (τ) pour co-mentions (h)", 1.0, 72.0, float(st.session_state["tau_hours"]), 1.0, help="Plus τ est court, plus les co-mentions récentes pèsent.")

# 🔹 Réglage d’alias $
with st.sidebar.expander("Détection des tokens", expanded=False):
    st.session_state["alias_no_dollar"] = st.checkbox(
        "Inclure les alias sans $ (tolérant)",
        value=bool(st.session_state["alias_no_dollar"]),
        help="Coché = accepte aussi des alias sans signe $. Décoché = uniquement des mentions explicites via $."
    )

# Bannière de statut dataset
ui_status_banner(compact=True)

# Indicateur résumeur (info visuelle)
ready = summarizer_available()
if ready:
    st.caption("🟢 Résumeur DistilBART disponible (utilisé ailleurs pour les résumés)")
else:
    st.caption("🟠 Résumeur indisponible / fallback extractif (pas critique pour cette page)")

# ---------------- Helpers ----------------
def _apply_period_filter(df: pd.DataFrame, period_choice: str) -> pd.DataFrame:
    if df.empty or "date" not in df.columns:
        return df
    if st.session_state.get("use_custom_period"):
        start, end = _custom_range_datetimes()
        if start and end:
            return df[(df["date"] >= start) & (df["date"] <= end)]
    now = pd.Timestamp.now(tz="Europe/Paris").tz_localize(None)
    if period_choice == "Tout":
        return df
    hours = int(period_choice.replace("h", ""))
    cutoff = now - pd.Timedelta(hours=hours)
    return df[df["date"] >= cutoff]

# 🔹 Wrapper centralisé pour gérer le mode alias $
def _explode_tokens(df_like: pd.DataFrame) -> pd.DataFrame:
    allow = bool(st.session_state.get("alias_no_dollar", False))
    try:
        # Si utils.explode_tokens supporte ce paramètre, on l’utilise.
        return explode_tokens(df_like, allow_alias_no_dollar=allow)
    except TypeError:
        # Fallback rétro-compatible : on appelle sans argument.
        dt = explode_tokens(df_like)
        # Si l’on voulait forcer "uniquement via $", on ne peut filtrer
        # que si une colonne brute est fournie par utils (ex: token_raw).
        if not allow and "token_raw" in dt.columns:
            return dt[dt["token_raw"].astype(str).str.startswith("$")]
        return dt

def _compute_group_token_scores(df: pd.DataFrame, w_m: float = 0.6) -> pd.DataFrame:
    """(group, token) -> mentions, sentiment (w_sentiment), score_conviction 0..10, rang local."""
    dt = _explode_tokens(df)
    if dt.empty:
        return pd.DataFrame(columns=["group","token","mentions","sentiment","score_conviction","rank_in_group"])
    agg = dt.groupby(["group","token"]).agg(
        mentions=("id","count"),
        sentiment=("w_sentiment","mean")
    ).reset_index()
    agg["m_norm"] = agg.groupby("group")["mentions"].transform(lambda s: s / max(1, s.max()))
    agg["s_norm"] = (agg["sentiment"] + 1.0) / 2.0
    agg["score_conviction"] = (w_m*agg["m_norm"] + (1-w_m)*agg["s_norm"]) * 10.0
    agg["score_conviction"] = agg["score_conviction"].round(2)
    agg["rank_in_group"] = agg.groupby("group")["score_conviction"].rank(ascending=False, method="first").astype(int)
    return agg

def _consensus_table(scores: pd.DataFrame, top_k: int, min_groups: int) -> pd.DataFrame:
    """Garde, pour chaque groupe, les Top-K; agrège par token."""
    if scores.empty:
        return pd.DataFrame(columns=["token","groups_count","groups_list","avg_score","avg_sent","mentions_total"])
    topk = scores[scores["rank_in_group"] <= top_k].copy()
    by_tok = (topk.groupby("token")
                    .agg(
                        groups_count=("group","nunique"),
                        groups_list=("group", lambda s: ", ".join(sorted(set(s)))),
                        avg_sent=("sentiment","mean"),
                        avg_score=("score_conviction","mean"),
                        mentions_total=("mentions","sum")
                    )
                    .reset_index()
             )
    by_tok["avg_score"] = by_tok["avg_score"].astype(float).round(2)
    by_tok["avg_sent"] = by_tok["avg_sent"].astype(float).round(3)
    by_tok = by_tok[by_tok["groups_count"] >= min_groups]
    return by_tok

def _token_pagerank(df: pd.DataFrame, tau_h: float = 12.0) -> pd.DataFrame:
    """PageRank (token↔token) sur la période, avec pondération temporelle simple (demi-vie τ)."""
    cols = ["id","date","tokens"]
    if "w_sentiment" in df.columns:
        cols.append("w_sentiment")
    dt = _explode_tokens(df[cols])
    if dt.empty:
        return pd.DataFrame(columns=["token","pagerank","pr_norm"])
    now = pd.Timestamp.now(tz="Europe/Paris").tz_localize(None)
    edges = []
    for mid, g in dt.groupby("id"):
        toks = sorted(set(g["token"].tolist()))
        if len(toks) < 2:
            continue
        # poids temporel
        try:
            ts = df.loc[df["id"]==mid, "date"].iloc[0]
            age_h = max(0.0, (now - ts).total_seconds()/3600.0)
        except Exception:
            age_h = 0.0
        decay = float(np.exp(-age_h/max(1e-6, tau_h)))
        for i in range(len(toks)):
            for j in range(i+1, len(toks)):
                edges.append((toks[i], toks[j], decay))
    if not edges:
        return pd.DataFrame(columns=["token","pagerank","pr_norm"])
    G = nx.Graph()
    for a,b,w in edges:
        if G.has_edge(a,b):
            G[a][b]["weight"] += w
        else:
            G.add_edge(a,b, weight=w)
    try:
        pr = nx.pagerank(G, weight="weight")
    except Exception:
        pr = {n: 0.0 for n in G.nodes()}
    pr_df = pd.DataFrame([{"token":k, "pagerank":float(v)} for k,v in pr.items()])
    if not pr_df.empty and pr_df["pagerank"].max() > 0:
        pr_df["pr_norm"] = pr_df["pagerank"] / pr_df["pagerank"].max()
    else:
        pr_df["pr_norm"] = 0.0
    return pr_df

# ---------------- Load / parse ----------------
# raw_all déjà fourni par ui_dataset_loader(page_key="invest")
if raw_all.empty:
    st.info("Charge des JSON d’abord.")
    st.stop()

# Bannière (rappel après check)
ui_status_banner(compact=True)

# filtre période
raw = _apply_period_filter(raw_all, st.session_state["period"])

# récupère df avec sentiments (si dispo) ou recalcule rapidement
cached_df = st.session_state.get("RAW_DF", pd.DataFrame())
cached_alias = st.session_state.get("_RAW_DF_ALIAS_NO_DOLLAR", None)
alias_flag = bool(st.session_state.get("alias_no_dollar", False))

if not cached_df.empty and cached_alias == alias_flag:
    # ✅ on a le DF complet (toutes périodes) déjà enrichi
    df_all = cached_df.copy()
else:
    # ❗ recalcul sentiments sur TOUT le dataset brut (PAS la période)
    with st.spinner("Calcul du sentiment..."):
        df_all = add_sentiment(
            raw_all,  # ⬅️ important : tout le dataset
            use_hf=st.session_state["use_hf"],
            w_vader=st.session_state["w_vader"],
            w_crypto=st.session_state["w_crypto"],
            w_hf=st.session_state["w_hf"],
            rule_weight=st.session_state["rule_weight"],
            group_weight_alpha=st.session_state["group_alpha"],
            alias_no_dollar=alias_flag,
        )
    # cache le DF complet + le flag
    st.session_state["RAW_DF"] = df_all.copy()
    st.session_state["_RAW_DF_ALIAS_NO_DOLLAR"] = alias_flag

#
# ⚙️ Nouveau : canaux 1× par dataset (pas de ré-inférence quand les poids changent)
#
sig = dataset_signature(raw_all)
channels_key = f"CHAN__{sig}__alias{int(alias_flag)}"

if channels_key not in st.session_state:
    with st.spinner("Calcul des canaux de sentiment (1× par dataset)…"):
        # On calcule CryptoBERT/VADER/Lexique sur TOUT le dataset brut (pas filtré par période)
        base = add_sentiment(
            raw_all,
            use_hf=st.session_state["use_hf"],
            # poids temporaires (n’ont pas d’importance, on remélange ensuite)
            w_vader=0.33, w_crypto=0.34, w_hf=0.33,
            rule_weight=0.0,            # pas de règles ici
            group_weight_alpha=1.0,
            alias_no_dollar=alias_flag,
            gain=1.0,
        )
        keep_cols = [
            "id","date","group","tokens","text","remark","conviction",
            "sentiment_vader","sentiment_crypto","sentiment_hf","text_for_sent"
        ]
        st.session_state[channels_key] = base[keep_cols].copy()

# Remix instantané suivant les sliders (zéro ré-inférence)
channels_df = st.session_state[channels_key]
df_all = remix_sentiment_weights(
    channels_df,
    w_vader=float(st.session_state["w_vader"]),
    w_crypto=float(st.session_state["w_crypto"]),
    w_hf=float(st.session_state["w_hf"]),
    rule_weight=float(st.session_state["rule_weight"]),
    gain=1.20,
    group_weight_alpha=float(st.session_state["group_alpha"]),
)
# Appliquer la période APRÈS sur le DF complet remélangé
df = _apply_period_filter(df_all, st.session_state["period"])

# garde-fous colonnes
for c in ["sentiment","w_sentiment"]:
    if c not in df.columns:
        df[c] = pd.Series(dtype=float)

if df.empty:
    st.info("Aucun message dans cette période/filtre. Élargis la fenêtre.")
    st.stop()

# ---------------- Consensus + Classement (investissables) ----------------
with st.spinner("Construction du consensus & des scores…"):
    scores = _compute_group_token_scores(df, w_m=0.6)
    cons = _consensus_table(scores, top_k=int(st.session_state["cons_top_k"]), min_groups=int(st.session_state["cons_min_groups"]))
    cons = cons[cons["avg_sent"] >= float(st.session_state["min_avg_sent"])]
    pr_df = _token_pagerank(df, tau_h=float(st.session_state["tau_hours"]))

    # score final 0..10 (table "investissables")
    if cons.empty:
        final = pd.DataFrame()
    else:
        final = cons.merge(pr_df[["token","pr_norm","pagerank"]], on="token", how="left")
        final["pr_norm"] = final["pr_norm"].fillna(0.0)
        final["pagerank"] = final["pagerank"].fillna(0.0)
        # normalisations
        final["score_norm"] = (final["avg_score"]/10.0).clip(0,1)
        final["sent_norm"]  = ((final["avg_sent"]+1.0)/2.0).clip(0,1)
        gc_max = max(1, int(final["groups_count"].max()))
        final["groups_norm"] = (final["groups_count"]/gc_max).clip(0,1)
        w_score  = float(st.session_state["w_score"])
        w_sent   = float(st.session_state["w_sent"])
        w_pr     = float(st.session_state["w_pr"])
        w_groups = float(st.session_state["w_groups"])
        w_sum = max(1e-9, w_score + w_sent + w_pr + w_groups)
        final["score_invest"] = 10.0 * (
            (w_score*final["score_norm"] +
             w_sent*final["sent_norm"] +
             w_pr*final["pr_norm"] +
             w_groups*final["groups_norm"]) / w_sum
        )
        final["score_invest"] = final["score_invest"].round(2)

if final.empty:
    st.warning("Aucun token ne remplit les critères de consensus. Essaie de baisser 'Min groupes' ou 'avg_sent min'.")
    st.stop()

# tri + export
final = final.sort_values(["score_invest","groups_count","avg_score","mentions_total"], ascending=[False, False, False, False])
csv = final.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Exporter CSV (classement actuel)", data=csv, file_name="investissement_classement.csv", mime="text/csv")

# tableau principal lisible
st.subheader("🏆 Classement — Tokens les plus investissables maintenant")
cols_show = ["token","score_invest","groups_count","avg_score","avg_sent","pagerank","mentions_total"]
col_cfg = {
    "token": st.column_config.TextColumn(
        "Token",
        help=("Symbole du memecoin — mode actuel : "
              + ("alias avec ou sans $" if st.session_state.get("alias_no_dollar") else "uniquement via $"))
    ),
    "score_invest": st.column_config.ProgressColumn(
        "Score (0→10)",
        help="Score final combinant avg_score (0→10), avg_sent, PageRank et convergence (groups_count).",
        min_value=0.0, max_value=10.0, format="%.2f",
    ),
    "groups_count": st.column_config.NumberColumn("Groupes", help="Nombre de groupes où le token est Top-K.", format="%d"),
    "avg_score": st.column_config.NumberColumn("Avg score (0→10)", help="Moyenne des scores intra-groupe (mentions+sentiment) sur les groupes Top-K.", format="%.2f"),
    "avg_sent": st.column_config.NumberColumn("Avg sent (−1→1)", help="Moyenne du sentiment (pondéré par conviction de groupe) sur ces groupes.", format="%.2f"),
    "pagerank": st.column_config.NumberColumn("PR_token", help="PageRank (co-mentions).", format="%.3g"),
    "mentions_total": st.column_config.NumberColumn("Mentions (Top-K)", help="Somme des mentions sur les groupes où il est Top-K.", format="%d"),
}
topN = int(st.session_state["topN_display"])
st.dataframe(final.head(topN)[cols_show], use_container_width=True, column_config=col_cfg, hide_index=True)

with st.expander("ℹ️ Comment lire ce tableau ?"):
    st.markdown("""
- **Score (0→10)** = combinaison **normalisée** de :
  - `avg_score` (score intra-groupe 0→10, mix **mentions+sentiment**),
  - `avg_sent` (qualité pure),
  - `PR_token` (centralité réseau),
  - `Groupes` (convergence inter-communautés).
- **Tips** :
  - **Groupes** ≥ 3 + **Score** ≥ 6.5 + **avg_sent** ≥ 0.25 → picks robustes.
  - **PR_token** haut = tokens “pivots” de la narrative.
""")

# --------- Bump chart : évolution du classement "investissables maintenant" ---------
with st.expander("📈 Évolution du classement (bump) — Investissables"):
    st.toggle("Afficher la bump chart des investissables", key="inv_bump_show")
    if st.session_state["inv_bump_show"]:
        st.info("Note: calcule un mini-classement par pas temporel (peut prendre quelques secondes).")
        c1, c2 = st.columns(2)
        st.session_state["inv_bump_step_h"] = c1.slider("Pas (heures)", 1, 12, int(st.session_state["inv_bump_step_h"]), 1)
        st.session_state["inv_bump_bins"]   = c2.slider("Nombre de pas", 4, 24, int(st.session_state["inv_bump_bins"]), 1)

        step_h_b = int(st.session_state["inv_bump_step_h"])
        bins_n_b = int(st.session_state["inv_bump_bins"])

        end_time_b = df["date"].max()
        if pd.isna(end_time_b):
            end_time_b = pd.Timestamp.now()
        start_time_b = end_time_b - pd.Timedelta(hours=bins_n_b*step_h_b)
        bins_all_b = pd.date_range(start=start_time_b.floor(f"{step_h_b}h"),
                                   end=end_time_b.ceil(f"{step_h_b}h"),
                                   freq=f"{step_h_b}h")
        ranks_rows_b = []

        def _calc_final_invest(df_sub: pd.DataFrame) -> pd.DataFrame:
            if df_sub.empty:
                return pd.DataFrame(columns=["token","score_invest"])
            sc = _compute_group_token_scores(df_sub, w_m=0.6)
            ctable = _consensus_table(sc, top_k=int(st.session_state["cons_top_k"]), min_groups=int(st.session_state["cons_min_groups"]))
            if ctable.empty:
                return pd.DataFrame(columns=["token","score_invest"])
            ctable = ctable[ctable["avg_sent"] >= float(st.session_state["min_avg_sent"])]
            pr_local = _token_pagerank(df_sub, tau_h=float(st.session_state["tau_hours"]))
            out = ctable.merge(pr_local[["token","pr_norm"]], on="token", how="left")
            out["pr_norm"] = out["pr_norm"].fillna(0.0)
            out["score_norm"] = (out["avg_score"]/10.0).clip(0,1)
            out["sent_norm"]  = ((out["avg_sent"]+1.0)/2.0).clip(0,1)
            gc_max = max(1, int(out["groups_count"].max()))
            out["groups_norm"] = (out["groups_count"]/gc_max).clip(0,1)
            w_score  = float(st.session_state["w_score"])
            w_sent   = float(st.session_state["w_sent"])
            w_pr     = float(st.session_state["w_pr"])
            w_groups = float(st.session_state["w_groups"])
            w_sum = max(1e-9, w_score + w_sent + w_pr + w_groups)
            out["score_invest"] = 10.0 * (
                (w_score*out["score_norm"] +
                 w_sent*out["sent_norm"] +
                 w_pr*out["pr_norm"] +
                 w_groups*out["groups_norm"]) / w_sum
            )
            return out[["token","score_invest"]]

        with st.spinner("Calcul des rangs temporels…"):
            for b in bins_all_b:
                # Fenêtre [b - step, b]
                b_start = b - pd.Timedelta(hours=step_h_b)
                df_b = df[(df["date"] > b_start) & (df["date"] <= b)].copy()
                res_b = _calc_final_invest(df_b)
                if res_b.empty:
                    continue
                res_b["rank"] = res_b["score_invest"].rank(ascending=False, method="min")
                for _, r in res_b.iterrows():
                    ranks_rows_b.append({"bin": b, "token": r["token"], "rank": int(r["rank"])})

        if ranks_rows_b:
            ranks_b = pd.DataFrame(ranks_rows_b)
            keep = final.head(min(topN, 15))["token"].tolist()  # montrer top actuel (<=15) pour lisibilité
            ranks_b = ranks_b[ranks_b["token"].isin(keep)].sort_values(["bin","rank"])
            figb = px.line(ranks_b, x="bin", y="rank", color="token", markers=True,
                           title="Évolution des rangs — Investissables (plus bas = mieux)")
            figb.update_yaxes(autorange="reversed")
            figb.update_layout(height=420, margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(figb, use_container_width=True)
        else:
            st.info("Pas assez de données pour tracer la bump chart des investissables.")

# --------------------- MODE EXPERT : SUPER CLASSEMENT ---------------------
st.markdown("## 🧠 Super Classement (expert)")
st.toggle("Activer le Super Classement", key="exp_show", help="Score composite avancé : qualité, réseau, dynamique, stabilité, consensus.")
if st.session_state["exp_show"]:

    # ---- UI expert (poids + fenêtre temporelle)
    c1, c2, c3, c4 = st.columns([1.3,1.3,1,1.2])
    with c1:
        st.markdown("**Fenêtre temporelle**")
        st.session_state["exp_step_h"] = st.slider("Pas (heures)", 1, 24, int(st.session_state["exp_step_h"]), 1)
        st.session_state["exp_bins"]   = st.slider("Nombre de pas (lookback)", 4, 48, int(st.session_state["exp_bins"]), 1,
                                                   help="Ex: 12 pas × 2h = 24h d'historique.")
    with c2:
        st.markdown("**Poids positifs**")
        st.session_state["exp_w_quality_sent"] = st.slider("Qualité: Sentiment moyen", 0.0, 1.0, float(st.session_state["exp_w_quality_sent"]), 0.01)
        st.session_state["exp_w_quality_wilson"] = st.slider("Qualité: Wilson (positifs)", 0.0, 1.0, float(st.session_state["exp_w_quality_wilson"]), 0.01)
        st.session_state["exp_w_cons_breadth"] = st.slider("Consensus: Breadth", 0.0, 1.0, float(st.session_state["exp_w_cons_breadth"]), 0.01,
                                                           help="Couverture inter-groupes du token (0→1).")
        st.session_state["exp_w_cons_groups"] = st.slider("Consensus: #Groupes", 0.0, 1.0, float(st.session_state["exp_w_cons_groups"]), 0.01)
    with c3:
        st.markdown("**Réseau & dynamique**")
        st.session_state["exp_w_network_pr"] = st.slider("Réseau: PageRank", 0.0, 1.0, float(st.session_state["exp_w_network_pr"]), 0.01)
        st.session_state["exp_w_dyn_mom"] = st.slider("Dynamique: Momentum", 0.0, 1.0, float(st.session_state["exp_w_dyn_mom"]), 0.01)
        st.session_state["exp_w_dyn_acc"] = st.slider("Dynamique: Accélération", 0.0, 1.0, float(st.session_state["exp_w_dyn_acc"]), 0.01)
        st.session_state["exp_w_stability"] = st.slider("Stabilité: (1−CI95)", 0.0, 1.0, float(st.session_state["exp_w_stability"]), 0.01)
        st.session_state["exp_w_polar_penalty"] = st.slider("Pénalité: Polarisation", 0.0, 1.0, float(st.session_state["exp_w_polar_penalty"]), 0.01,
                                                            help="Plus la polarisation est forte (50/50 pos/neg), plus on pénalise.")
    with c4:
        st.markdown("**🎯 Bonus de persistance (fort)**")
        st.session_state["exp_use_persist"] = st.checkbox("Activer le bonus de persistance", value=bool(st.session_state["exp_use_persist"]),
                                                          help="Bonus aux tokens restés longtemps dans le haut du classement.")
        st.session_state["exp_persist_hours"] = st.slider("Fenêtre (heures)", 6, 168, int(st.session_state["exp_persist_hours"]), 6,
                                                          help="Ex: 24h = 1j ; 168h = 1 semaine.")
        st.session_state["exp_persist_rank"] = st.slider("Seuil de rang (Top-R)", 3, 10, int(st.session_state["exp_persist_rank"]), 1,
                                                         help="Considéré 'haut' si rang ≤ R.")
        st.session_state["exp_w_persist"] = st.slider("Poids (base) du bonus", 0.0, 1.0, float(st.session_state["exp_w_persist"]), 0.01)
        st.session_state["exp_persist_gain"] = st.slider("Amplification du bonus", 0.0, 2.0, float(st.session_state["exp_persist_gain"]), 0.05,
                                                         help=">0 pour un effet nettement plus visible.")
        st.session_state["exp_persist_multiplicative"] = st.checkbox("Mode multiplicatif (très fort)", value=bool(st.session_state["exp_persist_multiplicative"]),
                                                                     help="Multiplie le score par (1 + poids×persistance×(1+ampli)).")

    # ---- Features par token (qualité/consensus/stabilité)
    dt = _explode_tokens(df)
    if dt.empty:
        st.info("Pas de tokens dans cette période.")
        st.stop()

    # binning pour dynamique
    step_h = int(st.session_state["exp_step_h"])
    bins_n = int(st.session_state["exp_bins"])
    end_time = dt["date"].max()
    if pd.isna(end_time):
        end_time = pd.Timestamp.now()
    start_time = end_time - pd.Timedelta(hours=bins_n*step_h)
    dt_win = dt[(dt["date"] >= start_time) & (dt["date"] <= end_time)].copy()
    if dt_win.empty:
        st.info("Fenêtre experte vide. Augmente le lookback.")
        st.stop()

    dt_win["bin"] = dt_win["date"].dt.floor(f"{step_h}h")

    # agrégats globaux
    tok_grp = dt_win.groupby("token")
    agg_base = tok_grp.agg(
        n=("id","count"),
        sent=("w_sentiment","mean"),
        std=("w_sentiment","std"),
        groups_used=("group","nunique")
    ).reset_index()
    agg_base["std"] = agg_base["std"].fillna(0.0)
    agg_base["ci95"] = 1.96*agg_base["std"]/np.sqrt(np.maximum(1, agg_base["n"]))
    agg_base["ci95"] = agg_base["ci95"].fillna(0.0).clip(0,1)

    # breadth
    total_groups = max(1, int(df["group"].nunique()))
    agg_base["breadth"] = (agg_base["groups_used"] / total_groups).clip(0,1)

    # polarisation
    grp_sent = dt_win.groupby(["token","group"])["w_sentiment"].mean().reset_index()
    grp_pos = grp_sent.assign(pos=(grp_sent["w_sentiment"]>0).astype(int)).groupby("token")["pos"].mean().reset_index(name="p_pos_groups")
    agg_base = agg_base.merge(grp_pos, on="token", how="left")
    agg_base["p_pos_groups"] = agg_base["p_pos_groups"].fillna(0.0).clip(0,1)
    agg_base["polarisation"] = 4.0*agg_base["p_pos_groups"]*(1.0-agg_base["p_pos_groups"])

    # Wilson
    msg_pos = dt_win.assign(pos=(dt_win["w_sentiment"]>0).astype(int)).groupby("token")["pos"].sum().reset_index(name="pos_count")
    agg_base = agg_base.merge(msg_pos, on="token", how="left")
    agg_base["pos_count"] = agg_base["pos_count"].fillna(0).astype(int)

    def wilson_lower_bound(pos, n, z=1.96):
        if n <= 0: return 0.0
        p = pos/n
        denom = 1 + z*z/n
        centre = p + z*z/(2*n)
        margin = z*np.sqrt((p*(1-p)/n) + (z*z/(4*n*n)))
        return float((centre - margin)/denom)

    agg_base["wilson"] = [wilson_lower_bound(int(p), int(n)) for p,n in zip(agg_base["pos_count"], agg_base["n"])]

    # PageRank (période courante)
    pr_df2 = _token_pagerank(df, tau_h=float(st.session_state["tau_hours"]))
    agg_base = agg_base.merge(pr_df2[["token","pr_norm","pagerank"]], on="token", how="left")
    agg_base["pr_norm"] = agg_base["pr_norm"].fillna(0.0)

    # Dynamique: momentum & accélération (mentions)
    bins_all = pd.date_range(start=start_time.floor(f"{step_h}h"), end=end_time.ceil(f"{step_h}h"), freq=f"{step_h}h")
    m = (dt_win.groupby(["token","bin"])["id"].count()
               .reindex(pd.MultiIndex.from_product([agg_base["token"], bins_all], names=["token","bin"]))
               .fillna(0.0).reset_index())
    x = np.arange(len(bins_all))
    def _slope_acc(vals):
        if len(vals) < 2: return 0.0, 0.0
        y = np.asarray(vals, dtype=float)
        try: b1 = np.polyfit(x, y, 1)[0]
        except Exception: b1 = 0.0
        try: a2 = np.polyfit(x, y, 2)[0]
        except Exception: a2 = 0.0
        return float(b1), float(2.0*a2)
    mom = m.groupby("token")["id"].apply(lambda s: _slope_acc(s.values)[0]).reset_index(name="momentum")
    acc = m.groupby("token")["id"].apply(lambda s: _slope_acc(s.values)[1]).reset_index(name="accel")
    agg_base = agg_base.merge(mom, on="token", how="left").merge(acc, on="token", how="left")
    agg_base[["momentum","accel"]] = agg_base[["momentum","accel"]].fillna(0.0)

    # Normalisations 0..1
    def norm01(s):
        s = s.astype(float); lo, hi = float(s.min()), float(s.max())
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12: return pd.Series(0.0, index=s.index)
        return (s - lo)/(hi - lo)

    agg_base["nz_sent"]   = norm01(agg_base["sent"])
    agg_base["nz_wilson"] = norm01(agg_base["wilson"])
    agg_base["nz_breadth"]= agg_base["breadth"].clip(0,1)
    agg_base["nz_groups"] = norm01(agg_base["groups_used"])
    agg_base["nz_pr"]     = agg_base["pr_norm"].clip(0,1)
    agg_base["nz_mom"]    = norm01(agg_base["momentum"])
    agg_base["nz_acc"]    = norm01(agg_base["accel"])
    agg_base["nz_stab"]   = (1.0 - agg_base["ci95"].clip(0,1))
    agg_base["nz_polar"]  = agg_base["polarisation"].clip(0,1)

    # Super score de base
    wq_s   = float(st.session_state["exp_w_quality_sent"])
    wq_w   = float(st.session_state["exp_w_quality_wilson"])
    wc_b   = float(st.session_state["exp_w_cons_breadth"])
    wc_g   = float(st.session_state["exp_w_cons_groups"])
    wn_pr  = float(st.session_state["exp_w_network_pr"])
    wd_m   = float(st.session_state["exp_w_dyn_mom"])
    wd_a   = float(st.session_state["exp_w_dyn_acc"])
    ws_st  = float(st.session_state["exp_w_stability"])
    w_pen  = float(st.session_state["exp_w_polar_penalty"])

    pos_sum = wq_s + wq_w + wc_b + wc_g + wn_pr + wd_m + wd_a + ws_st
    pos_sum = max(1e-9, pos_sum)

    agg_base["super_score"] = 10.0 * (
        (wq_s*agg_base["nz_sent"] +
         wq_w*agg_base["nz_wilson"] +
         wc_b*agg_base["nz_breadth"] +
         wc_g*agg_base["nz_groups"] +
         wn_pr*agg_base["nz_pr"] +
         wd_m*agg_base["nz_mom"] +
         wd_a*agg_base["nz_acc"] +
         ws_st*agg_base["nz_stab"]) / pos_sum
        - (w_pen*agg_base["nz_polar"]) / max(1e-9, w_pen + pos_sum)
    )

    # --------- BONUS PERSISTANCE : plus fort ----------
    ranks_rows = []
    need_ranks = bool(st.session_state["exp_use_persist"]) or bool(st.session_state["exp_show_bump"])
    if need_ranks:
        for b in bins_all:
            sub_b = dt_win[dt_win["bin"] == b]
            if sub_b.empty: continue
            tb = sub_b.groupby("token").agg(
                n=("id","count"),
                sent=("w_sentiment","mean"),
                std=("w_sentiment","std"),
                groups_used=("group","nunique")
            ).reset_index()
            tb["std"] = tb["std"].fillna(0.0)
            tb["ci95"] = 1.96*tb["std"]/np.sqrt(np.maximum(1, tb["n"]))
            tb["ci95"] = tb["ci95"].fillna(0.0).clip(0,1)
            tb["breadth"] = (tb["groups_used"] / max(1,total_groups)).clip(0,1)
            gb = sub_b.groupby(["token","group"])["w_sentiment"].mean().reset_index()
            gpos = gb.assign(pos=(gb["w_sentiment"]>0).astype(int)).groupby("token")["pos"].mean().reset_index(name="p_pos_groups")
            tb = tb.merge(gpos, on="token", how="left").fillna({"p_pos_groups":0.0})
            tb["polarisation"] = 4.0*tb["p_pos_groups"]*(1.0-tb["p_pos_groups"])
            mpos = sub_b.assign(pos=(sub_b["w_sentiment"]>0).astype(int)).groupby("token")["pos"].sum().reset_index(name="pos_count")
            tb = tb.merge(mpos, on="token", how="left").fillna({"pos_count":0})
            def wilson_lower_bound(pos, n, z=1.96):
                if n <= 0: return 0.0
                p = pos/n
                denom = 1 + z*z/n
                centre = p + z*z/(2*n)
                margin = z*np.sqrt((p*(1-p)/n) + (z*z/(4*n*n)))
                return float((centre - margin)/denom)
            tb["wilson"] = [wilson_lower_bound(int(p), int(n)) for p,n in zip(tb["pos_count"], tb["n"])]

            def nz(s):
                lo, hi = float(s.min()), float(s.max())
                return (s - lo)/(hi-lo) if (np.isfinite(lo) and np.isfinite(hi) and hi>lo) else s*0.0
            tb["nz_sent"]   = nz(tb["sent"])
            tb["nz_wilson"] = nz(tb["wilson"])
            tb["nz_breadth"]= tb["breadth"].clip(0,1)
            tb["nz_groups"] = nz(tb["groups_used"])
            tb["nz_pr"]     = 0.0
            tb["nz_mom"]    = 0.5
            tb["nz_acc"]    = 0.5
            tb["nz_stab"]   = (1.0 - tb["ci95"].clip(0,1))
            tb["nz_polar"]  = tb["polarisation"].clip(0,1)

            tb["super_bin"] = 10.0 * (
                (wq_s*tb["nz_sent"] + wq_w*tb["nz_wilson"] + wc_b*tb["nz_breadth"] + wc_g*tb["nz_groups"] +
                 wn_pr*tb["nz_pr"] + wd_m*tb["nz_mom"] + wd_a*tb["nz_acc"] + ws_st*tb["nz_stab"]) / pos_sum
                - (w_pen*tb["nz_polar"]) / max(1e-9, w_pen + pos_sum)
            )
            tb["rank"] = tb["super_bin"].rank(ascending=False, method="min")
            for _, r in tb.iterrows():
                ranks_rows.append({"bin": b, "token": r["token"], "rank": int(r["rank"])})

    ranks = pd.DataFrame(ranks_rows) if ranks_rows else pd.DataFrame(columns=["bin","token","rank"])

    # Appliquer bonus persistance "fort"
    if st.session_state["exp_use_persist"] and not ranks.empty:
        cutoff = end_time - pd.Timedelta(hours=int(st.session_state["exp_persist_hours"]))
        R = int(st.session_state["exp_persist_rank"])
        w_pers = float(st.session_state["exp_w_persist"])
        gain   = float(st.session_state["exp_persist_gain"])
        mult   = bool(st.session_state["exp_persist_multiplicative"])

        r_recent = ranks[ranks["bin"] >= cutoff]
        if not r_recent.empty:
            persist = (r_recent.assign(top=lambda d: (d["rank"] <= R).astype(int))
                                .groupby("token")["top"].mean()
                                .reset_index(name="persist_frac"))
            agg_base = agg_base.merge(persist, on="token", how="left")
            agg_base["persist_frac"] = agg_base["persist_frac"].fillna(0.0).clip(0,1)

            if mult:
                agg_base["super_score"] = agg_base["super_score"] * (1.0 + w_pers*agg_base["persist_frac"]*(1.0 + gain))
            else:
                agg_base["super_score"] = agg_base["super_score"] + 10.0 * w_pers * (1.0 + gain) * agg_base["persist_frac"]

    agg_base["super_score"] = agg_base["super_score"].clip(0,10).round(2)

    # ----- Affichage TOP (visuel)
    st.subheader("⭐ Super Classement (composite)")
    exp_topN = int(st.session_state["exp_topN"])
    sort_keys = ["super_score","nz_pr","nz_mom","nz_breadth"]
    top_super = agg_base.sort_values(sort_keys, ascending=[False, False, False, False]).head(exp_topN)

    try:
        fig = px.bar(
            top_super.sort_values("super_score"),
            x="super_score", y="token", orientation="h",
            color="sent", color_continuous_scale="RdYlGn",
            labels={"super_score":"Super score (0→10)", "sent":"Sentiment moyen", "token":"Token"},
            title="Top — Super score (couleur = sentiment moyen)"
        )
        fig.update_layout(height=480, margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass

    # Tableau compact
    show_cols = ["token","super_score","groups_used","breadth","ci95","polarisation","pagerank"]
    if "persist_frac" in top_super.columns:
        show_cols += ["persist_frac"]
    col_cfg2 = {
        "token": st.column_config.TextColumn("Token"),
        "super_score": st.column_config.ProgressColumn("Super score (0→10)", min_value=0.0, max_value=10.0, format="%.2f"),
        "groups_used": st.column_config.NumberColumn("Groupes", help="Nb. de groupes ayant mentionné le token."),
        "breadth": st.column_config.NumberColumn("Breadth", help="Couverture inter-groupes (0→1).", format="%.2f"),
        "ci95": st.column_config.NumberColumn("CI95", help="Intervalle de confiance du sentiment (plus petit = plus stable).", format="%.2f"),
        "polarisation": st.column_config.NumberColumn("Polarisation", help="0 (consensus) → 1 (conflit 50/50).", format="%.2f"),
        "pagerank": st.column_config.NumberColumn("PR_token", help="PageRank (importance réseau).", format="%.3g"),
    }
    if "persist_frac" in show_cols:
        col_cfg2["persist_frac"] = st.column_config.NumberColumn("Persistance", help="Fraction des pas récents où le token est Top-R.", format="%.2f")
    st.dataframe(top_super[show_cols], use_container_width=True, column_config=col_cfg2, hide_index=True)

    # ----- Évolution des rangs (bump chart expert)
    if st.checkbox("Voir l’évolution du classement (bump chart) — Super score", value=bool(st.session_state["exp_show_bump"])):
        if not ranks.empty:
            keep = top_super["token"].tolist()
            ranks2 = ranks[ranks["token"].isin(keep)].sort_values(["bin","rank"])
            fig2 = px.line(ranks2, x="bin", y="rank", color="token", markers=True,
                           title="Évolution des rangs — Super score (plus bas = mieux)")
            fig2.update_yaxes(autorange="reversed")
            fig2.update_layout(height=420, margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Pas assez de pas temporels pour construire la bump chart.")

    with st.expander("ℹ️ Détails du Super score"):
        st.markdown("""
**Le Super score** combine 4 axes (+ bonus optionnel), normalisés 0→1 puis agrégés (puis *10*) :

1. **Qualité** — *Sentiment moyen* (nz_sent), *Wilson* (nz_wilson)  
2. **Consensus** — *Breadth* (nz_breadth), *#Groupes* (nz_groups)  
3. **Réseau & Dynamique** — *PageRank* (nz_pr), *Momentum* (nz_mom), *Accélération* (nz_acc)  
4. **Stabilité & Conflit** — *(1 − CI95)* (nz_stab), **Pénalité** *Polarisation* (nz_polar)  
5. **🎯 Bonus persistance** — fraction des pas récents où le token est **Top-R** :
   - **Additif amplifié** : `+ 10 × w × (1+gain) × persistance`
   - **Multiplicatif (fort)** : `× (1 + w × persistance × (1+gain))`

Ajuste **fenêtre**, **Top-R**, **poids**, **amplification** pour favoriser les **convictions durables**.
        """)

# ---------------- Fin ----------------
st.caption("Tip: joue avec la bump chart des investissables et le bonus de persistance pour isoler les leaders vraiment stables.")
