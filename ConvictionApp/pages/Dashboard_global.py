# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import json
import networkx as nx
import plotly.express as px
from utils import summarize_top_tokens_deepseek, dataset_signature

from utils import (
    available_tokens, load_many_jsons, per_token_view, add_sentiment, explode_tokens, parse_messages_json,
    graph_edges_advanced, summarizer_available,
    ensure_session_dataset, ui_dataset_loader, ui_status_banner,  # <-- ajout UI loader/bannière
    remix_sentiment_weights,
    dataset_signature,
)

st.set_page_config(layout="wide")
st.title("📊 Dashboard global")
ensure_session_dataset()  # <-- garde le dataset en mémoire entre pages

# -----------------------------
# Helpers
# -----------------------------
def ensure_defaults():
    ss = st.session_state
    ss.setdefault("period", "24h")
    ss.setdefault("use_custom_period", False)
    ss.setdefault("custom_start_date", None)
    ss.setdefault("custom_start_time", None)
    ss.setdefault("custom_end_date", None)
    ss.setdefault("custom_end_time", None)

    ss.setdefault("use_hf", False)
    ss.setdefault("w_hf", 0.50)
    ss.setdefault("w_vader", 0.35)
    ss.setdefault("w_crypto", 0.15)
    ss.setdefault("gain_sent", 1.30)

    ss.setdefault("rule_weight", 1.0)
    ss.setdefault("group_alpha", 1.0)
    ss.setdefault("mentions_alpha", 0.6)
    ss.setdefault("tau_hours", 12.0)
    ss.setdefault("alias_no_dollar", True)
    ss.setdefault("enable_summarizer", False)

    ss.setdefault("score_graph_on", True)
    ss.setdefault("gamma_struct", 0.30)
    ss.setdefault("wA", 0.60)
    ss.setdefault("wC", 0.40)
    ss.setdefault("wPRT", 0.20)
    ss.setdefault("npmi_min_sc", 0.10)
    ss.setdefault("min_cooc_sc", 3)

    ss.setdefault("ts_step_h", 1)
    ss.setdefault("flip_win_h", 12)
ensure_defaults()

# après ensure_session_dataset() / ensure_defaults()
st.session_state.setdefault("enable_summarizer", False)

with st.sidebar.expander("Résumé (DeepSeek)", expanded=False):
    st.session_state["enable_summarizer"] = st.checkbox(
        "🧠 Activer résumeur DeepSeek",
        value=st.session_state["enable_summarizer"],
        help="Génère les résumés pour les Top 15 tokens. Désactivé = aucun appel API."
    )


def _period_popover():
    st.session_state["period"] = st.radio(
        "Fenêtre rapide",
        ["2h","6h","12h","24h","48h","Tout"],
        index=["2h","6h","12h","24h","48h","Tout"].index(st.session_state["period"]),
        horizontal=True,
        help="Fenêtre standard. Utilise la plage personnalisée si cochée ci-dessous."
    )
    st.session_state["use_custom_period"] = st.checkbox("Utiliser une plage personnalisée", help="Choisis une fenêtre précise (dates + heures).")
    if st.session_state["use_custom_period"]:
        c1, c2 = st.columns(2)
        st.session_state["custom_start_date"] = c1.date_input("Date début (Europe/Paris)")
        st.session_state["custom_start_time"] = c1.time_input("Heure début")
        st.session_state["custom_end_date"]   = c2.date_input("Date fin (Europe/Paris)")
        st.session_state["custom_end_time"]   = c2.time_input("Heure fin")

def _custom_range_datetimes():
    try:
        sdate = st.session_state.get("custom_start_date")
        stime = st.session_state.get("custom_start_time")
        edate = st.session_state.get("custom_end_date")
        etime = st.session_state.get("custom_end_time")
        if not (sdate and stime and edate and etime):
            return None, None
        start = pd.Timestamp.combine(sdate, stime).tz_localize("Europe/Paris").tz_convert(None)
        end   = pd.Timestamp.combine(edate, etime).tz_localize("Europe/Paris").tz_convert(None)
        return start, end
    except Exception:
        return None, None

def _apply_period_filter(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "date" not in df.columns: return df
    if st.session_state.get("use_custom_period"):
        start, end = _custom_range_datetimes()
        if start and end:
            return df[(df["date"]>=start) & (df["date"]<=end)]
    now = pd.Timestamp.now(tz="Europe/Paris").tz_localize(None)
    if st.session_state["period"] == "Tout": return df
    hours = int(st.session_state["period"].replace("h",""))
    cutoff = now - pd.Timedelta(hours=hours)
    return df[df["date"] >= cutoff]

def _safe_pagerank(G: nx.Graph):
    if G.number_of_edges() == 0:
        return {}
    try:
        return nx.pagerank(G, weight="weight", alpha=0.85, max_iter=500, tol=1.0e-06)
    except Exception:
        try:
            return nx.pagerank_numpy(G, weight="weight")
        except Exception:
            deg = dict(G.degree(weight="weight"))
            s = sum(deg.values()) or 1.0
            return {k: (v/s) for k, v in deg.items()}

# -----------------------------
# Sidebar (unifiée)
# -----------------------------
st.sidebar.title("⚙️ Configuration")

# Loader unifié (multi-JSON + reset + compteur)
raw_all = ui_dataset_loader(page_key="dash")

with st.sidebar.expander("Période d'analyse", expanded=True):
    with st.popover("Choisir la période précisément"):
        _period_popover()

with st.sidebar.expander("Sentiment – canaux & poids (local)", expanded=True):
    st.session_state["use_hf"] = st.checkbox(
        "Utiliser modèle local (HF)", value=st.session_state["use_hf"], key="use_hf_dash",
        help="FinBERT + BERTweet + XLM-R. Si ça rame, décoche (fallback VADER+Lexique). Le calcul HF est mis en cache : changer les poids ne relance pas le modèle."
    )
    st.session_state["w_hf"] = st.slider("Poids modèle local (HF)", 0.0, 1.0, st.session_state["w_hf"], 0.05, key="w_hf_dash")
    st.session_state["w_vader"]  = st.slider("Poids VADER", 0.0, 1.0, st.session_state["w_vader"], 0.05, key="w_vader_dash")
    st.session_state["w_crypto"] = st.slider("Poids Lexique crypto", 0.0, 1.0, st.session_state["w_crypto"], 0.05, key="w_crypto_dash")
    st.session_state["gain_sent"] = st.slider("Gain dynamique (×)", 0.5, 2.0, float(st.session_state["gain_sent"]), 0.05)
with st.sidebar.expander("Ajustements & pondération", expanded=True):
    st.session_state["rule_weight"] = st.slider("Poids règles/boosts lexicaux", 0.0, 1.0, st.session_state["rule_weight"], 0.05)
    st.session_state["group_alpha"] = st.slider("Poids conviction de groupe", 0.0, 2.0, st.session_state["group_alpha"], 0.1)
    st.session_state["mentions_alpha"] = st.slider("Poids mentions (score conviction)", 0.0, 1.0, st.session_state["mentions_alpha"], 0.05)
    st.session_state["tau_hours"] = st.slider("Demi-vie temporelle (τ, h)", 1.0, 72.0, float(st.session_state["tau_hours"]), 1.0)
    st.session_state["alias_no_dollar"] = st.checkbox("Détecter alias sans $ (strict)", value=st.session_state["alias_no_dollar"])
    st.session_state["enable_summarizer"] = st.checkbox("Activer résumeur (DistilBART)", value=st.session_state["enable_summarizer"])

with st.sidebar.expander("Score graphe (bonus)", expanded=False):
    st.session_state["score_graph_on"] = st.checkbox("Activer renfort graphe (γ)", value=st.session_state["score_graph_on"])
    st.session_state["gamma_struct"] = st.slider("γ (poids structure)", 0.0, 1.0, st.session_state["gamma_struct"], 0.05)
    c1, c2, c3 = st.columns(3)
    st.session_state["wA"]   = c1.slider("w AutoritéGroupes", 0.0, 1.0, st.session_state["wA"], 0.05)
    st.session_state["wC"]   = c2.slider("w ConvergenceClusters", 0.0, 1.0, st.session_state["wC"], 0.05)
    st.session_state["wPRT"] = c3.slider("w CentralitéPR", 0.0, 1.0, st.session_state["wPRT"], 0.05)
    st.session_state["npmi_min_sc"] = st.slider("Seuil NPMI co-mentions (tokens)", 0.0, 0.6, st.session_state["npmi_min_sc"], 0.05)
    st.session_state["min_cooc_sc"] = st.slider("Min co-mentions pondérées", 1, 10, int(st.session_state["min_cooc_sc"]))

# -----------------------------
# Bannière de statut dataset
# -----------------------------
ui_status_banner(compact=True)

# -----------------------------
# Data
# -----------------------------
if raw_all.empty:
    st.info("Importe un JSON d’abord.")
    st.stop()

raw = _apply_period_filter(raw_all)
with st.spinner("Calcul du sentiment..."):
    df_raw = add_sentiment(
        raw,
        use_hf=st.session_state["use_hf"],
        w_vader=st.session_state["w_vader"],
        w_crypto=st.session_state["w_crypto"],
        w_hf=st.session_state["w_hf"],
        rule_weight=st.session_state["rule_weight"],
        group_weight_alpha=st.session_state["group_alpha"],
        alias_no_dollar=st.session_state["alias_no_dollar"],
        gain=float(st.session_state["gain_sent"]),
    )
#
# ⚙️ Nouveau : canaux 1× par dataset (pas de ré-inférence quand les poids changent)
#
alias_flag = bool(st.session_state.get("alias_no_dollar", True))
sig = dataset_signature(raw_all)
channels_key = f"CHAN__{sig}__alias{int(alias_flag)}"

if channels_key not in st.session_state:
    with st.spinner("Calcul des canaux de sentiment (1× par dataset)…"):
        # On calcule CryptoBERT/VADER/Lexique sur TOUT le dataset brut
        base = add_sentiment(
            raw_all,
            use_hf=st.session_state["use_hf"],
            # Poids temporaires (n'ont pas d'importance, on remixe ensuite)
            w_vader=0.33, w_crypto=0.34, w_hf=0.33,
            rule_weight=0.0,              # pas de règles ici
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
    gain=float(st.session_state["gain_sent"]),
    group_weight_alpha=float(st.session_state["group_alpha"]),
)

# Appliquer la période APRÈS remix sur le DF complet
df_raw = _apply_period_filter(df_all)

# -----------------------------
# Indicateur résumeur
# -----------------------------
ready = summarizer_available()
if st.session_state["enable_summarizer"] and ready:
    st.markdown("🟢 **Résumeur DistilBART actif** (les résumés sont générés).")
elif st.session_state["enable_summarizer"] and ready is False:
    st.markdown("🟠 **Résumeur demandé mais indisponible** (modèle non chargé) — fallback extractif.")
else:
    st.markdown("🔴 **Résumeur inactif** (citations brutes uniquement).")

# -----------------------------
# Filtres simples
# -----------------------------
groups = ["(Tous)"] + sorted(df_raw["group"].dropna().unique().tolist())
sel_group = st.selectbox("Filtrer par groupe", groups, index=0)
df = df_raw if sel_group == "(Tous)" else df_raw[df_raw["group"] == sel_group]

tokens = ["(Tous)"] + available_tokens(df)
sel_token = st.selectbox("Filtrer par token", tokens, index=0)
if sel_token != "(Tous)":
    mask = df["tokens"].apply(lambda ts: sel_token in (ts or []))
    df = df[mask]

# -----------------------------
# KPIs
# -----------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Messages", f"{len(df):,}")
c2.metric("Groupes", f"{df['group'].nunique():,}")
c3.metric("Tokens uniques", f"{len(available_tokens(df)):,}")
mean_sent = df["sentiment"].mean() if "sentiment" in df.columns and not df["sentiment"].isna().all() else None
c4.metric("Sentiment moyen", f"{mean_sent:.2f}" if mean_sent is not None else "—")
st.divider()

# -----------------------------
# Leaderboard
# -----------------------------
st.subheader("🏆 Classement par token")
pt = per_token_view(df, include_summaries=False, summarizer_enabled=False)
summaries = {}
if st.session_state.get("enable_summarizer", False):
    summaries = summarize_top_tokens_deepseek(
        df,
        top_n=15,
        w_mentions=float(st.session_state.get("mentions_alpha", 0.6)),
        max_tokens=140,            # ajuste ton budget ici
        enabled=True               # <-- nouveau param (voir point 2)
    )

if not pt.empty:
    pt["résumé"] = pt["token"].map(lambda t: summaries.get(t, "—"))

    st.caption(f"🧠 Résumés DeepSeek générés pour {len(summaries)} tokens (Top 15 par score_conviction). Cache par dataset actif.")


# base (mentions + sentiment)
if not pt.empty:
    dt = explode_tokens(df, extra_cols=["sentiment_hf"])
    mentions = dt.groupby("token").size().reset_index(name="mentions")
    breadth = dt.groupby("token")["group"].nunique().reset_index(name="breadth")
    pol = dt.groupby(["token","group"])["w_sentiment"].mean().groupby("token").std().reset_index(name="polarisation")
    pos = dt.assign(pos=(dt["w_sentiment"]>0).astype(int)).groupby("token")["pos"].agg(["sum","count"]).reset_index()
    z=1.96
    pos["p_hat"]=pos["sum"]/pos["count"].replace(0, np.nan)
    n=pos["count"]
    pos["wilson_low"]=((pos["p_hat"] + z*z/(2*n) - z*np.sqrt((pos["p_hat"]*(1-pos["p_hat"]))/n + (z*z)/(4*n*n))) / (1+z*z/n)).fillna(0.0)

    # momentum/accel (pas ts_step_h)
    step_h = int(st.session_state["ts_step_h"])
    df_bins = dt.copy()
    df_bins["bin"] = df_bins["date"].dt.floor(f"{step_h}h")
    agg = df_bins.groupby(["token","bin"]).agg(sent=("w_sentiment","mean"), m=("id","count")).reset_index()
    mm = agg.groupby("token")["m"].transform("max").replace(0,1)
    agg["score_h"] = (st.session_state["mentions_alpha"]*(agg["m"]/mm) + (1-st.session_state["mentions_alpha"])*((agg["sent"]+1)/2))*10.0
    last = agg.sort_values("bin").groupby("token").tail(2).copy()
    mom = last.groupby("token")["score_h"].diff().reset_index(name="momentum")
    last["momentum"]=mom["momentum"]
    acc = last.groupby("token")["momentum"].diff().reset_index(name="accel")
    last["accel"]=acc["accel"]
    ma = last.groupby("token")[["momentum","accel"]].last().reset_index()

    # merge
    pt = pt.drop(columns=["mentions"], errors="ignore").merge(mentions, on="token", how="left")
    pt = pt.merge(breadth, on="token", how="left").merge(pol, on="token", how="left").merge(pos[["token","wilson_low"]], on="token", how="left")
    pt = pt.merge(ma, on="token", how="left")

    # score_conviction (0..10)
    max_m = max(1, pt["mentions"].max())
    norm_m = pt["mentions"] / max_m
    norm_s = (pt["sentiment"] + 1) / 2
    alpha = st.session_state["mentions_alpha"]
    pt["score_conviction"] = (10.0 * (alpha*norm_m + (1-alpha)*norm_s)).round(2)

# Renfort graphe
if st.session_state["score_graph_on"] and not pt.empty:
    edges, node_sent, _ = graph_edges_advanced(df, tau_hours=float(st.session_state["tau_hours"]), group_sent_source="calc")
    if not edges.empty:
        npmi_min = float(st.session_state["npmi_min_sc"])
        min_cooc = float(st.session_state["min_cooc_sc"])
        e = edges.copy()
        e_tt = e[(e["type"]=="token-token") & (e["npmi"].fillna(-1) >= npmi_min) & (e["cooc"].fillna(0) >= min_cooc)]
        e_gt = e[e["type"]=="group-token"]

        G = nx.Graph()
        for _, r in e.iterrows():
            G.add_edge(r["src"], r["dst"], weight=float(r["weight"]))
        pr = _safe_pagerank(G)

        # clusters
        try:
            from networkx.algorithms.community import louvain_communities
            comms = louvain_communities(G, weight="weight", seed=42) if G.number_of_edges()>0 else [{n} for n in G.nodes()]
        except Exception:
            from networkx.algorithms.community import greedy_modularity_communities
            comms = list(greedy_modularity_communities(G, weight="weight")) if G.number_of_edges()>0 else [{n} for n in G.nodes()]
        cluster_id = {}
        for ci, community in enumerate(comms, start=1):
            for n in community:
                cluster_id[n] = ci

        # Autorité groupes (A)
        A={}
        if not e_gt.empty:
            for t, g in e_gt.groupby("dst"):
                s=0.0
                for _, r in g.iterrows():
                    s += float(r["weight"]) * float(pr.get(r["src"], 0.0))
                A[t]=s
            m=max(A.values()) if A else 1.0
            for k in list(A.keys()): A[k]=A[k]/m if m>0 else 0.0

        # Convergence clusters (C)
        C={}
        if not e_gt.empty:
            for t, g in e_gt.groupby("dst"):
                cls = set(cluster_id.get(r["src"], 0) for _, r in g.iterrows())
                C[t]=len([c for c in cls if c!=0])
            m=max(C.values()) if C else 1.0
            for k in list(C.keys()): C[k]=C[k]/m if m>0 else 0.0

        # PR token
        token_nodes = set(node_sent.loc[node_sent["kind"]=="token","node"].tolist())
        PRT = {t: pr.get(t, 0.0) for t in token_nodes}
        if len(PRT)>0:
            m=max(PRT.values())
            if m>0: PRT = {k: v/m for k,v in PRT.items()}
            else:   PRT = {k: 0.0 for k in PRT}

        # mix
        wA=float(st.session_state["wA"]); wC=float(st.session_state["wC"]); wPRT=float(st.session_state["wPRT"])
        wsum=max(1e-9,(wA+wC+wPRT)); wA,wC,wPRT = wA/wsum, wC/wsum, wPRT/wsum
        gamma=float(st.session_state["gamma_struct"])

        pt["AutoritéGroupes"]     = pt["token"].map(A).fillna(0.0)
        pt["ConvergenceClusters"] = pt["token"].map(C).fillna(0.0)
        pt["CentralitéPR"]        = pt["token"].map(PRT).fillna(0.0)

        struct = (wA*pt["AutoritéGroupes"] + wC*pt["ConvergenceClusters"] + wPRT*pt["CentralitéPR"]).clip(0,1)
        base01 = (pt["score_conviction"] / 10.0).clip(0,1)
        pt["score_conviction_graph"] = (10.0 * ((1-gamma)*base01 + gamma*struct)).round(2)
        pt = pt.sort_values(["score_conviction_graph","score_conviction","mentions"], ascending=[False, False, False])
    else:
        st.info("Pas de graphe utilisable pour le renfort structurel — score simple conservé.")

# -----------------------------
# Score “Quick Win” (0..10)
# -----------------------------
if not pt.empty:
    bmax = max(1, pt["breadth"].max())
    pmax = max(1e-6, pt["polarisation"].max())
    m_abs = max(1e-6, pt["momentum"].abs().max(skipna=True))
    a_abs = max(1e-6, pt["accel"].abs().max(skipna=True))

    sent01 = (pt["sentiment"]+1)/2
    wil    = pt["wilson_low"].clip(0,1)
    br01   = (pt["breadth"]/bmax).clip(0,1)
    pol01i = (1 - (pt["polarisation"]/pmax).clip(0,1))
    mom01  = (pt["momentum"]/m_abs/2 + 0.5).fillna(0.5).clip(0,1)
    acc01  = (pt["accel"]/a_abs/2 + 0.5).fillna(0.5).clip(0,1)

    quick = 0.30*sent01 + 0.25*wil + 0.20*br01 + 0.15*mom01 + 0.10*acc01
    quick = quick * (0.7 + 0.3*pol01i)
    pt["score_quick_win"] = (10.0*quick).round(2)

# -----------------------------
# Export + Affichage table
# -----------------------------
if pt is None or isinstance(pt, list):
    st.error("Problème de construction du tableau. Recharge le JSON.")
else:
    show_cols = [c for c in pt.columns if c not in ["remarques","L_prem"]]
    st.caption(f"Tableau : {pt.shape[0]} lignes × {len(show_cols)} colonnes")
    if not pt.empty:
        csv_bytes = pt[show_cols].to_csv(index=False).encode("utf-8")
        st.download_button("📥 Exporter CSV (classement + score)", data=csv_bytes, file_name="classement_tokens.csv", mime="text/csv")
    st.dataframe(pt[show_cols], use_container_width=True)

# -----------------------------
# 📈 Évolution temporelle du score de conviction (token)
# -----------------------------
st.subheader("📈 Évolution temporelle du score de conviction (token)")
tok_opts = ["(Choisir)"] + (pt["token"].tolist() if not pt.empty else available_tokens(df))
tok_sel = st.selectbox("Token", tok_opts, index=0, help="Choisis un token pour tracer son score de conviction dans le temps.")
if tok_sel != "(Choisir)":
    step_h = int(st.session_state["ts_step_h"])
    df_tok = df[df["tokens"].apply(lambda ts: tok_sel in (ts or []))].copy()
    if df_tok.empty:
        st.info("Pas de messages pour ce token dans la période.")
    else:
        df_tok["bin"] = (df_tok["date"].dt.floor(f"{step_h}h"))
        agg = df_tok.groupby("bin").agg(sentiment=("sentiment","mean"), mentions=("id","count")).reset_index().sort_values("bin")
        max_m = max(1, agg["mentions"].max())
        norm_m = agg["mentions"] / max_m
        norm_s = (agg["sentiment"] + 1) / 2
        alpha = st.session_state["mentions_alpha"]
        agg["score"] = 10.0 * (alpha*norm_m + (1-alpha)*norm_s)
        fig = px.line(agg, x="bin", y="score", markers=True, title=f"Score de conviction – {tok_sel}")
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 🧭 Détections : Flips & Newcomers
# -----------------------------
st.subheader("🧭 Détections : Flips & Newcomers")
flip_win = int(st.session_state["flip_win_h"])

def _flip_detector(df_in: pd.DataFrame, win_h: int=12, thr: float=0.1):
    if df_in.empty: return pd.DataFrame(columns=["token","t_flip","s_before","s_after"])
    dt2 = explode_tokens(df_in)
    out=[]
    now = pd.Timestamp.now(tz="Europe/Paris").tz_localize(None)
    t1 = now - pd.Timedelta(hours=win_h*2)
    recent = dt2[dt2["date"]>=t1]
    for tok, g in recent.groupby("token"):
        mid = now - pd.Timestamp.now(tz="Europe/Paris").tz_localize(None).tz_localize(None) if False else now - pd.Timedelta(hours=win_h)
        s1 = g[g["date"]<mid]["w_sentiment"].mean()
        s2 = g[g["date"]>=mid]["w_sentiment"].mean()
        if pd.isna(s1) or pd.isna(s2): continue
        if (s1 < -thr and s2 > +thr) or (s1 > +thr and s2 < -thr):
            out.append((tok, mid, float(s1), float(s2)))
    return pd.DataFrame(out, columns=["token","t_flip","s_before","s_after"]).sort_values("t_flip", ascending=False)

def _newcomers(df_in: pd.DataFrame, hours_recent: int=24, top_k_leaders: int=5, npmi_min: float=0.1):
    if df_in.empty: return pd.DataFrame(columns=["token","first_seen","npmi_with_leader","leader"])
    # leaders = top par score_conviction (ou mentions) dans la période
    dt2 = explode_tokens(df_in)
    leaders = dt2.groupby("token")["id"].count().sort_values(ascending=False).head(top_k_leaders).index.tolist()
    edges, node_sent, _ = graph_edges_advanced(df_in, tau_hours=float(st.session_state["tau_hours"]))
    e_tt = edges[(edges["type"]=="token-token") & (edges["npmi"].fillna(-1) >= npmi_min)]
    if dt2.empty: return pd.DataFrame(columns=["token","first_seen","npmi_with_leader","leader"])
    fs = dt2.groupby("token")["date"].min()
    cutoff = pd.Timestamp.now(tz="Europe/Paris").tz_localize(None) - pd.Timedelta(hours=hours_recent)
    cand = fs[fs>=cutoff].index.tolist()
    rows=[]
    for t in cand:
        rel = e_tt[(e_tt["src"]==t) | (e_tt["dst"]==t)]
        if rel.empty: continue
        best = None
        for _, r in rel.iterrows():
            u = r["dst"] if r["src"]==t else r["src"]
            if u in leaders:
                val = float(r["npmi"])
                if best is None or val>best[0]: best=(val,u)
        if best:
            rows.append((t, fs[t], best[0], best[1]))
    return pd.DataFrame(rows, columns=["token","first_seen","npmi_with_leader","leader"]).sort_values("first_seen", ascending=False)

flips = _flip_detector(df, win_h=flip_win, thr=0.1)
newc = _newcomers(df, hours_recent=24, top_k_leaders=5, npmi_min=0.1)

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Flips de sentiment (±) sur la dernière fenêtre**")
    st.dataframe(flips, use_container_width=True)
with c2:
    st.markdown("**Newcomers (émergents)** — NPMI ↑ avec leaders")
    st.dataframe(newc, use_container_width=True)
