# -*- coding: utf-8 -*- 
import streamlit as st
import pandas as pd
import json
import plotly.express as px
import networkx as nx

from utils import (
    available_tokens,
    load_many_jsons,
    per_token_view,
    add_sentiment,
    parse_messages_json,
    summarizer_available,
    explode_tokens,
    # --- AJOUTS: loader/bannière/dataset partagé ---
    ensure_session_dataset,
    ui_dataset_loader,
    ui_status_banner,
)

st.set_page_config(layout="wide")
st.title("🧩 Vue par groupe")


# ---------- Defaults (session) ----------
def ensure_defaults():
    ss = st.session_state
    # Période
    ss.setdefault("period", "24h")
    ss.setdefault("use_custom_period", False)
    ss.setdefault("custom_start_date", None)
    ss.setdefault("custom_start_time", None)
    ss.setdefault("custom_end_date", None)
    ss.setdefault("custom_end_time", None)

    # Sentiment local + poids
    ss.setdefault("use_hf", True)
    ss.setdefault("w_hf", 0.50)
    ss.setdefault("w_vader", 0.35)
    ss.setdefault("w_crypto", 0.15)
    ss.setdefault("rule_weight", 1.0)
    ss.setdefault("group_alpha", 1.0)
    ss.setdefault("tau_hours", 12.0)
    ss.setdefault("enable_summarizer", False)  # indicateur résumeur (hérité entre pages)

    # Conviction
    ss.setdefault("conv_w_mentions", 0.60)  # poids mentions vs sentiment pour score_conviction

    # Heatmap & convergence
    ss.setdefault("hm_top_tokens", 20)
    ss.setdefault("hm_cluster_sort", True)
    ss.setdefault("consensus_top_k", 5)
    ss.setdefault("consensus_min_groups", 2)

    # Data holders
    ss.setdefault("RAW_ALL", pd.DataFrame())
    ss.setdefault("RAW_DF", pd.DataFrame())

ensure_defaults()
# Dataset persistant (entre pages)
ensure_session_dataset()


# ---------- Sidebar ----------
st.sidebar.title("⚙️ Configuration")

# --- Sources & chargement : UI unifiée (multi JSON + compteur + reset) ---
raw_all = ui_dataset_loader(page_key="group")

def _custom_range_datetimes():
    """Construit les datetimes (timezone Europe/Paris -> naive) depuis la plage personnalisée."""
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
    # Fenêtres rapides
    st.session_state["period"] = st.radio(
        "Fenêtre",
        ["2h", "6h", "12h", "24h", "48h", "Tout"],
        index=["2h", "6h", "12h", "24h", "48h", "Tout"].index(st.session_state["period"]),
        horizontal=True,
        key="period_group",
        help="Fenêtre d’analyse appliquée aux messages.",
    )
    # Plage personnalisée
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

with st.sidebar.expander("Sentiment – canaux & poids (local)", expanded=True):
    st.session_state["use_hf"] = st.checkbox(
        "Utiliser modèle local (HF)",
        value=st.session_state["use_hf"],
        key="use_hf_group",
        help="Active le blend local (HF + VADER + lexique).",
    )
    st.session_state["w_hf"] = st.slider(
        "Poids modèle local (HF)", 0.0, 1.0, st.session_state["w_hf"], 0.05, key="w_hf_group"
    )
    st.session_state["w_vader"] = st.slider(
        "Poids VADER", 0.0, 1.0, st.session_state["w_vader"], 0.05, key="w_vader_group"
    )
    st.session_state["w_crypto"] = st.slider(
        "Poids Lexique crypto", 0.0, 1.0, st.session_state["w_crypto"], 0.05, key="w_crypto_group"
    )

with st.sidebar.expander("Ajustements & pondération", expanded=True):
    st.session_state["rule_weight"] = st.slider(
        "Poids règles/boosts lexicaux", 0.0, 1.0, st.session_state["rule_weight"], 0.05, key="rule_weight_group"
    )
    st.session_state["group_alpha"] = st.slider(
        "Poids conviction de groupe", 0.0, 2.0, st.session_state["group_alpha"], 0.1, key="group_alpha_group"
    )
    st.session_state["tau_hours"] = st.slider(
        "Demi-vie temporelle (τ, h)", 1.0, 72.0, float(st.session_state["tau_hours"]), 1.0, key="tau_hours_group"
    )

with st.sidebar.expander("Scores & vues avancées", expanded=False):
    st.session_state["conv_w_mentions"] = st.slider(
        "Poids mentions (score_conviction)", 0.0, 1.0, st.session_state["conv_w_mentions"], 0.05,
        help="0=100% sentiment ; 1=100% mentions."
    )
    st.session_state["hm_top_tokens"] = st.slider(
        "Heatmap: Top N tokens globaux", 5, 50, st.session_state["hm_top_tokens"], 1
    )
    st.session_state["hm_cluster_sort"] = st.checkbox(
        "Heatmap: trier par clusters (Louvain)", value=st.session_state["hm_cluster_sort"]
    )
    st.session_state["consensus_top_k"] = st.slider(
        "Consensus: Top-K par groupe", 3, 10, st.session_state["consensus_top_k"], 1
    )
    st.session_state["consensus_min_groups"] = st.slider(
        "Consensus: min. groupes", 2, 10, st.session_state["consensus_min_groups"], 1
    )

# Bannière de statut (compteur fichiers / messages / uniques)
ui_status_banner(compact=True)


# ---------- Helpers ----------
def _apply_period_filter(df: pd.DataFrame, period_choice: str) -> pd.DataFrame:
    if df.empty or "date" not in df.columns:
        return df
    # Priorité à la plage personnalisée si activée
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


def _compute_group_token_scores(df: pd.DataFrame, w_m: float = 0.6) -> pd.DataFrame:
    """
    Renvoie un DataFrame (group, token, mentions, sentiment, score_conviction [0..10], rank_in_group)
    basé sur df déjà passé dans add_sentiment().
    """
    dt = explode_tokens(df)
    if dt.empty:
        return pd.DataFrame(columns=["group","token","mentions","sentiment","score_conviction","rank_in_group"])
    agg = dt.groupby(["group","token"]).agg(
        mentions=("id","count"),
        sentiment=("w_sentiment","mean")
    ).reset_index()
    # mentions normalisées par groupe
    agg["m_norm"] = agg.groupby("group")["mentions"].transform(lambda s: s / max(1, s.max()))
    agg["s_norm"] = (agg["sentiment"] + 1.0) / 2.0
    agg["score_conviction"] = (w_m*agg["m_norm"] + (1-w_m)*agg["s_norm"]) * 10.0
    agg["score_conviction"] = agg["score_conviction"].round(2)
    agg["rank_in_group"] = agg.groupby("group")["score_conviction"].rank(ascending=False, method="first").astype(int)
    return agg


def _heatmap_order_by_clusters(scores: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Trie groupes & tokens par communautés (Louvain greedy sur graphe biparti),
    fallback: tri par score total.
    """
    try:
        if scores.empty:
            return [], []
        B = nx.Graph()
        # Noeuds marqués pour bipartite
        groups = sorted(scores["group"].unique().tolist())
        tokens = sorted(scores["token"].unique().tolist())
        B.add_nodes_from([("g::"+g) for g in groups], bipartite=0)
        B.add_nodes_from([("t::"+t) for t in tokens], bipartite=1)
        for _, r in scores.iterrows():
            w = float(r["score_conviction"])
            if w <= 0:
                continue
            B.add_edge("g::"+r["group"], "t::"+r["token"], weight=w)

        if B.number_of_edges() == 0:
            raise ValueError("Empty graph")

        comms = list(nx.algorithms.community.greedy_modularity_communities(B, weight="weight"))
        # map cluster id
        cid = {}
        for i, c in enumerate(comms):
            for n in c:
                cid[n] = i
        groups_sorted = sorted(groups, key=lambda g: (cid.get("g::"+g, 1e9), -scores[scores["group"]==g]["score_conviction"].sum()))
        tokens_sorted = sorted(tokens, key=lambda t: (cid.get("t::"+t, 1e9), -scores[scores["token"]==t]["score_conviction"].sum()))
        return groups_sorted, tokens_sorted
    except Exception:
        # fallback: tri par score total
        if scores.empty:
            return [], []
        groups_sorted = (scores.groupby("group")["score_conviction"].sum()
                         .sort_values(ascending=False).index.tolist())
        tokens_sorted = (scores.groupby("token")["score_conviction"].sum()
                         .sort_values(ascending=False).index.tolist())
        return groups_sorted, tokens_sorted


# ---------- Load / parse ----------
if raw_all.empty:
    st.info("Importez d’abord un JSON (ici ou page principale).")
    st.stop()

raw = _apply_period_filter(raw_all, st.session_state["period"])

# ---------- Sentiment ----------
with st.spinner("Calcul du sentiment..."):
    df = add_sentiment(
        raw,
        use_hf=st.session_state["use_hf"],
        w_vader=st.session_state["w_vader"],
        w_crypto=st.session_state["w_crypto"],
        w_hf=st.session_state["w_hf"],
        rule_weight=st.session_state["rule_weight"],
        group_weight_alpha=st.session_state["group_alpha"],
    )
# Patch anti-KeyError: même vide, garantir colonnes
for c in ["sentiment", "sentiment_hf", "w_sentiment"]:
    if c not in df.columns:
        df[c] = pd.Series(dtype=float)

st.session_state["RAW_DF"] = df

# Indicateur résumeur
ready = summarizer_available()
if st.session_state.get("enable_summarizer", False) and ready:
    st.markdown("🟢 **Résumeur DistilBART actif**")
elif st.session_state.get("enable_summarizer", False) and ready is False:
    st.markdown("🟠 **Résumeur demandé mais indisponible** — fallback extractif.")
else:
    st.markdown("🔴 **Résumeur inactif**")


# ---------- Sélecteurs ----------
groups = sorted(df["group"].dropna().unique().tolist()) if "group" in df.columns and not df.empty else []
if not groups:
    st.info("Aucun groupe dans la période sélectionnée. Élargis la fenêtre dans la barre latérale.")
    st.stop()

sel_group = st.selectbox("Groupe", groups, index=0)

sub = df[df["group"] == sel_group].copy()
tokens = ["(Tous)"] + sorted({t for ts in sub.get("tokens", pd.Series([])) for t in (ts or [])})
sel_token = st.selectbox("Token (optionnel)", tokens, index=0)

if sel_token != "(Tous)":
    mask = sub["tokens"].apply(lambda ts: sel_token in (ts or []))
    sub = sub[mask]


# ---------- Messages récents (filtrés) ----------
st.subheader("Messages récents (filtrés)")
cols_msgs = ["date", "group", "conviction", "tokens", "sentiment", "w_sentiment", "text", "remark"]

if sub.empty:
    st.info("Aucun message pour ce filtre.")
else:
    st.dataframe(
        sub.reindex(columns=cols_msgs).sort_values("date", ascending=False).head(300),
        use_container_width=True,
    )

st.divider()

# ---------- Détail par token (groupe sélectionné) ----------
st.subheader("Détail par token (dans ce groupe)")
pt = per_token_view(sub, include_summaries=True, summarizer_enabled=bool(st.session_state.get("enable_summarizer", False)))
st.dataframe(pt, use_container_width=True)

# --- Top conviction (intra-groupe) ---
if not pt.empty:
    w_m = float(st.session_state["conv_w_mentions"])
    denom = max(1, int(pt["mentions"].max()))
    mentions_norm = pt["mentions"] / denom
    sent_norm = (pt["sentiment"] + 1.0) / 2.0
    pt["score_conviction"] = (w_m*mentions_norm + (1-w_m)*sent_norm) * 10.0
    pt["score_conviction"] = pt["score_conviction"].round(2)

    st.subheader("🏆 Top conviction du groupe")
    topN = st.slider("Top N (groupe)", 3, 30, 10, 1, help="Classement intra-groupe (mentions+sentiment).")
    view_cols = ["token","mentions","sentiment","ci95","score_conviction","mot-clés","résumé","Sentiment_HF"]
    st.dataframe(
        pt.sort_values(["score_conviction","mentions"], ascending=[False, False]).head(topN)[view_cols],
        use_container_width=True
    )
else:
    st.info("Aucun token dans ce groupe pour la période sélectionnée.")

st.divider()

# ---------- Heatmap Groupes × Tokens (score_conviction) ----------
st.subheader("📊 Heatmap Groupes × Tokens (score_conviction)")

scores = _compute_group_token_scores(df, w_m=float(st.session_state["conv_w_mentions"]))
if scores.empty:
    st.info("Pas assez de données pour la heatmap.")
else:
    # sélectionner Top N tokens globaux
    tok_global = (scores.groupby("token")["score_conviction"].sum()
                  .sort_values(ascending=False).head(int(st.session_state["hm_top_tokens"])).index.tolist())
    scores_f = scores[scores["token"].isin(tok_global)].copy()

    # Ordre lignes/colonnes
    if st.session_state["hm_cluster_sort"]:
        g_order, t_order = _heatmap_order_by_clusters(scores_f)
        if not g_order:  # fallback si clustering impossible
            g_order = (scores_f.groupby("group")["score_conviction"].sum()
                       .sort_values(ascending=False).index.tolist())
        if not t_order:
            t_order = tok_global
    else:
        g_order = (scores_f.groupby("group")["score_conviction"].sum()
                   .sort_values(ascending=False).index.tolist())
        t_order = tok_global

    pivot = (scores_f.pivot_table(index="group", columns="token", values="score_conviction", aggfunc="mean")
             .reindex(index=g_order, columns=t_order)
             .fillna(0.0))

    fig = px.imshow(
        pivot,
        color_continuous_scale="Blues",
        aspect="auto",
        labels=dict(x="Token", y="Groupe", color="Conviction (0-10)"),
        title="Groupes × Tokens – score_conviction"
    )
    fig.update_layout(margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------- Détecteur de convergence (consensus picks) ----------
st.subheader("🤝 Consensus picks du moment")
if scores.empty:
    st.info("Pas de données pour le consensus.")
else:
    k = int(st.session_state["consensus_top_k"])
    min_g = int(st.session_state["consensus_min_groups"])

    # top-k par groupe
    topk = scores[scores["rank_in_group"] <= k].copy()
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
    consensus = by_tok[by_tok["groups_count"] >= min_g].sort_values(
        ["groups_count","avg_score","mentions_total"], ascending=[False, False, False]
    )

    if consensus.empty:
        st.info(f"Aucun token présent dans le Top-{k} d'au moins {min_g} groupes.")
    else:
        consensus["avg_sent"] = consensus["avg_sent"].round(3)
        consensus["avg_score"] = consensus["avg_score"].round(2)
        st.dataframe(
            consensus[["token","groups_count","groups_list","avg_score","avg_sent","mentions_total"]],
            use_container_width=True
        )
