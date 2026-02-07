# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

from utils import (
    parse_messages_json,
    add_sentiment,
    explode_tokens,
    available_tokens,
    graph_edges_advanced,
    summarizer_available,
    load_many_jsons,  # important pour multi-JSON
)

st.set_page_config(layout="wide")
st.title("🔎 Exploration visuelle")

# -----------------------------
# Defaults (IMPORTANT: initialiser toutes les clés)
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
    ss.setdefault("tau_hours", 12.0)
    ss.setdefault("alias_no_dollar", True)
    ss.setdefault("enable_summarizer", False)

    ss.setdefault("ts_step_h", 1)

    # Heatmaps options
    ss.setdefault("heatmap_topN", 20)
    ss.setdefault("heatmap_crit", "Mentions")
    ss.setdefault("heatmap_sig_thr", 2.0)    # z-score
    ss.setdefault("heatmap_annot_flip", True)
    ss.setdefault("heatmap_cluster_rows", True)

    # stockage dataset partagé
    ss.setdefault("RAW_ALL", pd.DataFrame())
    ss.setdefault("RAW_DF", pd.DataFrame())
    ss.setdefault("_files_merged_count", 0)

    # Affichage des graphes (toggles)
    ss.setdefault("show_bubble", True)
    ss.setdefault("show_bump", False)
    ss.setdefault("show_stream", False)
    ss.setdefault("show_ridge", False)
    ss.setdefault("show_volcano", False)
    ss.setdefault("show_rankflow", False)
ensure_defaults()

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

def _apply_period_filter(df: pd.DataFrame, period_choice: str) -> pd.DataFrame:
    if df.empty or "date" not in df.columns:
        return df
    # plage personnalisée
    if st.session_state.get("use_custom_period"):
        try:
            start, end = _custom_range_datetimes()
            if start and end:
                return df[(df["date"] >= start) & (df["date"] <= end)]
        except Exception:
            pass
    # fenêtres rapides
    if period_choice == "Tout":
        return df
    now = pd.Timestamp.now(tz="Europe/Paris").tz_localize(None)
    hours = int(str(period_choice).replace("h", ""))
    cutoff = now - pd.Timedelta(hours=hours)
    return df[df["date"] >= cutoff]

def _safe_louvain(G: nx.Graph):
    try:
        from networkx.algorithms.community import louvain_communities
        return louvain_communities(G, weight="weight", seed=42)
    except Exception:
        from networkx.algorithms.community import greedy_modularity_communities
        return list(greedy_modularity_communities(G, weight="weight"))

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("⚙️ Configuration")

with st.sidebar.expander("Sources & chargement", expanded=True):
    uploads = st.file_uploader(
        "Importer un ou plusieurs JSON",
        type=["json"],
        accept_multiple_files=True,
        key="uploader_explo",
        help="Tu peux fusionner plusieurs exports; déduplication automatique."
    )

    # Bouton reset dataset
    def _reset_dataset():
        st.session_state["RAW_ALL"] = pd.DataFrame()
        st.session_state["RAW_DF"] = pd.DataFrame()
        st.session_state["_files_merged_count"] = 0
        st.success("Dataset réinitialisé.")

    st.button("🗑️ Réinitialiser dataset", on_click=_reset_dataset)

    # Fusion multi-fichiers -> RAW_ALL partagé
    if uploads:
        try:
            objs = []
            for uf in uploads:
                try:
                    objs.append(json.loads(uf.read().decode("utf-8")))
                except Exception:
                    pass
            before_n = len(st.session_state["RAW_ALL"]) if not st.session_state["RAW_ALL"].empty else 0
            st.session_state["RAW_ALL"] = load_many_jsons(objs, base=st.session_state.get("RAW_ALL"))
            after_n = len(st.session_state["RAW_ALL"])
            st.session_state["_files_merged_count"] += len(uploads)
            st.success("✅ Fusion effectuée.")
        except Exception as e:
            st.error(f"Erreur de parsing: {e}")

    # Compteur compact
    total_uniques = len(st.session_state["RAW_ALL"]) if not st.session_state["RAW_ALL"].empty else 0
    st.caption(f"📦 **{st.session_state['_files_merged_count']}** fichiers fusionnés — **{total_uniques:,}** messages **uniques**")

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

with st.sidebar.expander("Sentiment – canaux & poids (local)", expanded=True):
    st.session_state["use_hf"] = st.checkbox("Utiliser modèle local (HF)", value=st.session_state["use_hf"], help="Le modèle local est mis en cache.")
    st.session_state["w_hf"] = st.slider("Poids modèle local (HF)", 0.0, 1.0, st.session_state["w_hf"], 0.05)
    st.session_state["w_vader"]  = st.slider("Poids VADER", 0.0, 1.0, st.session_state["w_vader"], 0.05)
    st.session_state["w_crypto"] = st.slider("Poids Lexique crypto", 0.0, 1.0, st.session_state["w_crypto"], 0.05)
    st.session_state["gain_sent"] = st.slider("Gain dynamique (×)", 0.5, 2.0, float(st.session_state["gain_sent"]), 0.05)
    st.session_state["enable_summarizer"] = st.checkbox("Activer résumeur (DistilBART)", value=st.session_state["enable_summarizer"])

with st.sidebar.expander("Heatmaps – options", expanded=True):
    st.session_state["heatmap_topN"] = st.slider("Top N tokens", 5, 50, int(st.session_state["heatmap_topN"]), 1, help="Nombre de tokens affichés.")
    st.session_state["heatmap_crit"] = st.selectbox("Critère de sélection", ["Mentions","Breadth","Spike récent (z)","Score conviction"], help="Comment choisir le Top N.")
    st.session_state["heatmap_sig_thr"] = st.slider("Seuil z-score (cellules significatives)", 0.0, 5.0, float(st.session_state["heatmap_sig_thr"]), 0.1, help="Spot des pics de mentions.")
    st.session_state["heatmap_annot_flip"] = st.checkbox("Annoter flips (◀▶)", value=st.session_state["heatmap_annot_flip"], help="Indique les inversions de sentiment récentes.")
    st.session_state["heatmap_cluster_rows"] = st.checkbox("Trier par clusters (co-mentions)", value=st.session_state["heatmap_cluster_rows"], help="Regroupe visuellement les tokens liés.")

with st.sidebar.expander("Graphiques à afficher", expanded=True):
    st.session_state["show_bubble"]   = st.checkbox("Bubble Mentions × Sentiment", st.session_state["show_bubble"])
    st.session_state["show_bump"]     = st.checkbox("Bump chart (rangs)", st.session_state["show_bump"])
    st.session_state["show_stream"]   = st.checkbox("Streamgraph/Stacked area par cluster", st.session_state["show_stream"])
    st.session_state["show_ridge"]    = st.checkbox("Ridgeline (distributions)", st.session_state["show_ridge"])
    st.session_state["show_volcano"]  = st.checkbox("Volcano (sentiment vs anomalie)", st.session_state["show_volcano"])
    st.session_state["show_rankflow"] = st.checkbox("Rank-flow (Top N)", st.session_state["show_rankflow"])

# -----------------------------
# Data
# -----------------------------
if "RAW_ALL" not in st.session_state:
    st.session_state["RAW_ALL"] = pd.DataFrame()

raw_all = st.session_state["RAW_ALL"]
if raw_all.empty:
    st.info("Importe un ou plusieurs JSON pour commencer.")
    st.stop()

# 1) Filtre de période sur les messages bruts
raw = _apply_period_filter(raw_all, st.session_state["period"])
if raw.empty:
    st.warning("Aucun message dans la période sélectionnée.")
    st.stop()

# 2) Sentiment (sur la période filtrée)
with st.spinner("Calcul du sentiment..."):
    df = add_sentiment(
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

# Indicateur résumeur
ready = summarizer_available()
if st.session_state["enable_summarizer"] and ready:
    st.markdown("🟢 **Résumeur DistilBART actif**")
elif st.session_state["enable_summarizer"] and ready is False:
    st.markdown("🟠 **Résumeur demandé mais indisponible** — fallback extractif.")
else:
    st.markdown("🔴 **Résumeur inactif**")

# 3) Explode tokens + binning
dt = explode_tokens(df)
if dt.empty:
    st.warning("Pas de tokens détectés.")
    st.stop()
step_h = int(st.session_state["ts_step_h"])
dt["bin"] = dt["date"].dt.floor(f"{step_h}h")

# 4) Communautés (pour clusters & couleurs) — coloration auto par cluster
try:
    edges, _, _ = graph_edges_advanced(df, tau_hours=float(st.session_state["tau_hours"]))
    e_tt = edges[edges["type"]=="token-token"]
except Exception:
    e_tt = pd.DataFrame(columns=["src","dst","type","npmi"])

Gt = nx.Graph()
for _, r in e_tt.iterrows():
    Gt.add_edge(r["src"], r["dst"], weight=float(r.get("npmi", 0.0)))
comms = _safe_louvain(Gt) if Gt.number_of_edges()>0 else [set(dt["token"].unique())]
cluster_id = {}
for i,c in enumerate(comms, start=1):
    for t in c:
        cluster_id[t]=i
dt["cluster"] = dt["token"].map(cluster_id).fillna(0).astype(int)

topk = 5
cl_top = (dt.groupby(['cluster','token'])['id'].count()
            .reset_index()
            .sort_values(['cluster','id'], ascending=[True, False])
            .groupby('cluster').head(topk))
cl_label_map = {}
for cid, g in cl_top.groupby('cluster'):
    tokens_ex = ", ".join(g['token'].head(3).tolist())
    cl_label_map[int(cid)] = f"C{int(cid)}: {tokens_ex}"
dt['cluster_label'] = dt['cluster'].map(cl_label_map).fillna("Autres")

# -----------------------------
# Heatmaps
# -----------------------------
st.subheader("🔥 Heatmaps (mentions & sentiment)")

mentions = dt.groupby("token")["id"].count().rename("mentions")
breadth  = dt.groupby("token")["group"].nunique().rename("breadth")
hist = dt.groupby(["token","bin"])["id"].count().rename("m").reset_index()

def recent_z(g):
    g = g.sort_values("bin")
    if g.empty: return 0.0
    last = g.iloc[-1]["m"]
    mu = g["m"].mean(); sd = g["m"].std(ddof=0) or 1.0
    return (last - mu)/sd

z_recent = hist.groupby("token").apply(recent_z).rename("z_recent")
sent = dt.groupby("token")["w_sentiment"].mean().rename("s")
m_by_t = mentions.reindex(sent.index).fillna(0)
den = max(float(m_by_t.max()), 1.0)
score_conv = (0.6*(m_by_t/den) + 0.4*((sent+1)/2))*10
score_conv = score_conv.rename("score_conv")

sel = pd.concat([mentions, breadth, z_recent, score_conv], axis=1).fillna(0.0)
crit_map = {"Mentions":"mentions","Breadth":"breadth","Spike récent (z)":"z_recent","Score conviction":"score_conv"}
crit = crit_map[st.session_state["heatmap_crit"]]
topN = int(st.session_state["heatmap_topN"])
top_tokens = sel.sort_values(crit, ascending=False).head(topN).index.tolist()

dtt = dt[dt["token"].isin(top_tokens)]
piv_m = dtt.pivot_table(index="token", columns="bin", values="id", aggfunc="count").fillna(0)
piv_s = dtt.pivot_table(index="token", columns="bin", values="w_sentiment", aggfunc="mean")

# tri par clusters (option)
if st.session_state["heatmap_cluster_rows"]:
    order=[]
    for c in sorted(dtt["cluster"].unique()):
        order += sorted(dtt[dtt["cluster"]==c]["token"].unique().tolist())
    piv_m = piv_m.reindex(order)
    piv_s = piv_s.reindex(order)

# annotations flips & signifiance
text_annot = None
if st.session_state["heatmap_annot_flip"] or st.session_state["heatmap_sig_thr"]>0:
    text_annot = []
    s_by = dtt.groupby(["token","bin"])["w_sentiment"].mean().unstack(fill_value=0.0)
    z_thr = float(st.session_state["heatmap_sig_thr"])
    m_by = dtt.groupby(["token","bin"])["id"].count().unstack(fill_value=0.0)
    z_by = (m_by - m_by.mean(axis=1).values[:,None]) / (m_by.std(axis=1).replace(0,1).values[:,None])
    for tok in piv_m.index:
        row=[]
        last_cols = list(s_by.columns)[-2:] if tok in s_by.index and s_by.shape[1]>=2 else []
        flip_idx = None
        if len(last_cols)==2:
            a,b = s_by.loc[tok,last_cols[0]], s_by.loc[tok,last_cols[1]]
            if (a<=-0.1 and b>=0.1) or (a>=0.1 and b<=-0.1):
                flip_idx = -1
        for j, col in enumerate(piv_m.columns):
            mark = ""
            if flip_idx is not None and j==len(piv_m.columns)-1:
                mark = "◀▶"
            if tok in z_by.index and col in z_by.columns and z_by.loc[tok, col] >= z_thr:
                mark = mark + "•"
            row.append(mark)
        text_annot.append(row)

st.markdown(f"**Mentions (Top {topN} — sélection : {st.session_state['heatmap_crit']})**")
fig1 = px.imshow(piv_m, aspect="auto", labels=dict(x="Temps", y="Token", color="Mentions"))
if text_annot:
    fig1.update_traces(text=text_annot, texttemplate="%{text}")
st.plotly_chart(fig1, use_container_width=True)

st.markdown(f"**Sentiment moyen (Top {topN} — sélection : {st.session_state['heatmap_crit']})**")
fig2 = px.imshow(piv_s, aspect="auto", color_continuous_scale="RdYlGn",
                 labels=dict(x="Temps", y="Token", color="Sentiment"))
st.plotly_chart(fig2, use_container_width=True)

st.divider()

# -----------------------------
# Bubble (coloration AUTO par cluster) — affichage conditionnel
# -----------------------------
if st.session_state["show_bubble"]:
    agg = dt.groupby("token").agg(
        mentions=("id","count"),
        sentiment=("w_sentiment","mean"),
        breadth=("group","nunique"),
        cluster=("cluster","max")
    ).reset_index()
    fig = px.scatter(
        agg, x="sentiment", y="mentions", size="breadth", color="cluster",
        hover_data=["token","mentions","sentiment","breadth","cluster"], size_max=40,
        title="Bubble: Sentiment × Mentions (taille=Breadth, couleur=Cluster)"
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Bump chart (rangs) — couleur cluster auto — affichage conditionnel
# -----------------------------
if st.session_state["show_bump"]:
    rank_base = dt.groupby(["token","bin"])["id"].count().rename("m").reset_index()
    def top_rank(df_):
        df_ = df_.sort_values("m", ascending=False)
        df_["rank"] = range(1, len(df_)+1)
        return df_
    rr = rank_base.groupby("bin").apply(top_rank).reset_index(drop=True)
    rr = rr[rr["rank"]<=20]
    rr["cluster"] = rr["token"].map(cluster_id).fillna(0).astype(int)
    fig = px.line(rr, x="bin", y="rank", color="cluster", line_group="token",
                  hover_data=["token","m","cluster"],
                  title="Bump chart (Top 20) – couleur=cluster")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Streamgraph / Stacked area par cluster — affichage conditionnel
# -----------------------------
if st.session_state["show_stream"]:
    area = dt.groupby(["bin","cluster","cluster_label"])["id"].count().reset_index()
    fig = px.area(
        area, x="bin", y="id", color="cluster_label",
        title="Stacked area: mentions par CLUSTER (tokens)",
        hover_data={"cluster":True, "cluster_label":True, "id":True, "bin":True},
    )
    fig.update_layout(
        legend_title_text="Clusters (Top tokens)",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Ridgeline (distributions de sentiment) — affichage conditionnel
# -----------------------------
if st.session_state["show_ridge"]:
    top_tokens = dt.groupby("token")["id"].count().sort_values(ascending=False).head(20).index.tolist()
    ridge = dt[dt["token"].isin(top_tokens)]
    fig = px.violin(ridge, x="w_sentiment", y="token", points=False, box=False, title="Ridgeline (proxy) – distribution des sentiments par token")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Volcano — affichage conditionnel
# -----------------------------
if st.session_state["show_volcano"]:
    base = dt.groupby("token").agg(sent=("w_sentiment","mean"), m=("id","count")).reset_index()
    mu = base["m"].mean(); sd = base["m"].std(ddof=0) or 1.0
    base["z"] = (base["m"] - mu) / sd
    fig = px.scatter(base, x="sent", y="z", hover_name="token", title="Volcano: Sentiment moyen vs z-score(mentions)")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Rank-flow (Alluvial/Sankey) Top N — affichage conditionnel
# -----------------------------
if st.session_state["show_rankflow"]:
    N=10
    rb = dt.groupby(["token","bin"])["id"].count().rename("m").reset_index()
    ranks=[]
    for b, g in rb.groupby("bin"):
        g = g.sort_values("m", ascending=False).head(N).copy()
        g["rank"] = range(1, len(g)+1)
        g["bin"] = b
        ranks.append(g)
    if ranks:
        R = pd.concat(ranks, ignore_index=True)
        bins_sorted = sorted(R["bin"].unique())
        nodes = []
        node_index = {}
        for b in bins_sorted:
            gb = R[R["bin"]==b]
            for _, r in gb.iterrows():
                name = f"{r['token']}@{str(b)}"
                node_index[name] = len(nodes); nodes.append(name)
        links = {"source": [], "target": [], "value": [], "label":[]}
        for i in range(len(bins_sorted)-1):
            b0, b1 = bins_sorted[i], bins_sorted[i+1]
            g0 = R[R["bin"]==b0]; g1 = R[R["bin"]==b1]
            for tok in set(g0["token"]) & set(g1["token"]):
                s = node_index[f"{tok}@{str(b0)}"]
                t = node_index[f"{tok}@{str(b1)}"]
                links["source"].append(s)
                links["target"].append(t)
                links["value"].append(1.0)
                links["label"].append(tok)
        if links["source"]:
            fig = go.Figure(data=[go.Sankey(
                node = dict(label=nodes, pad=10, thickness=8),
                link = dict(source=links["source"], target=links["target"], value=links["value"], label=links["label"])
            )])
            fig.update_layout(title_text="Rank-flow (Top 10) au fil des fenêtres")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Pas assez de recouvrement entre fenêtres pour le rank-flow.")
