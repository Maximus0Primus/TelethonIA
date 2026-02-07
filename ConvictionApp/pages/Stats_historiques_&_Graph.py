# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import json
import networkx as nx
from pyvis.network import Network
from tempfile import NamedTemporaryFile
from pathlib import Path

from utils import (
    graph_edges_advanced,
    add_sentiment,
    load_many_jsons,
    parse_messages_json,
    explode_tokens,
    summarizer_available,
    # --- Ajouts pour dataset unifié ---
    ensure_session_dataset,
    ui_dataset_loader,
    ui_status_banner,
)

st.set_page_config(layout="wide")
st.title("🕸️ Graph: groupes ↔ tokens & co-mentions de tokens (τ, NPMI, sentiment, communautés)")

# ---------- Defaults ----------
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
    ss.setdefault("rule_weight", 1.0)
    ss.setdefault("group_alpha", 1.0)
    ss.setdefault("tau_hours", 12.0)
    ss.setdefault("alias_no_dollar", True)
    # Graph filters
    ss.setdefault("npmi_min", 0.10)
    ss.setdefault("min_cooc", 3)
    ss.setdefault("abs_sent_min", 0.0)
    ss.setdefault("timeline_hours", 0)
    ss.setdefault("size_mode", "Degré")
    ss.setdefault("min_w", 0.0)
    ss.setdefault("max_edges", 1000)
    ss.setdefault("etype_opt", ["group-token","token-token","group-group"])
    ss.setdefault("group_sent_mode", "Calculé")
    ss.setdefault("min_groups_token", 1)

ensure_defaults()
# Dataset persistant (entre pages)
ensure_session_dataset()

# ---------- Sidebar ----------
st.sidebar.title("⚙️ Configuration")

# --- Sources & chargement (UI unifiée : multi JSON + compteur + reset) ---
raw_all = ui_dataset_loader(page_key="graph")

with st.sidebar.expander("Période d'analyse", expanded=True):
    with st.popover("Choisir la période précisément"):
        # Fenêtre rapide + plage custom cohérentes avec les autres pages
        st.session_state["period"] = st.radio(
            "Fenêtre rapide",
            ["2h","6h","12h","24h","48h","Tout"],
            index=["2h","6h","12h","24h","48h","Tout"].index(st.session_state["period"]),
            horizontal=True,
            key="period_graph"
        )
        st.session_state["use_custom_period"] = st.checkbox(
            "Utiliser une plage personnalisée",
            value=st.session_state["use_custom_period"]
        )
        if st.session_state["use_custom_period"]:
            c1, c2 = st.columns(2)
            st.session_state["custom_start_date"] = c1.date_input(
                "Date début (Europe/Paris)",
                value=st.session_state.get("custom_start_date", None)
            )
            st.session_state["custom_start_time"] = c1.time_input(
                "Heure début",
                value=st.session_state.get("custom_start_time", None)
            )
            st.session_state["custom_end_date"]   = c2.date_input(
                "Date fin (Europe/Paris)",
                value=st.session_state.get("custom_end_date", None)
            )
            st.session_state["custom_end_time"]   = c2.time_input(
                "Heure fin",
                value=st.session_state.get("custom_end_time", None)
            )

with st.sidebar.expander("Sentiment – canaux & poids (local)", expanded=True):
    st.session_state["use_hf"] = st.checkbox("Utiliser modèle local (HF)", value=st.session_state["use_hf"], key="use_hf_graph")
    st.session_state["w_hf"] = st.slider("Poids modèle local (HF)", 0.0, 1.0, st.session_state["w_hf"], 0.05, key="w_hf_graph")
    st.session_state["w_vader"]  = st.slider("Poids VADER", 0.0, 1.0, st.session_state["w_vader"], 0.05, key="w_vader_graph")
    st.session_state["w_crypto"] = st.slider("Poids Lexique crypto", 0.0, 1.0, st.session_state["w_crypto"], 0.05, key="w_crypto_graph")

with st.sidebar.expander("Ajustements & pondération", expanded=True):
    st.session_state["rule_weight"] = st.slider("Poids règles/boosts lexicaux", 0.0, 1.0, st.session_state["rule_weight"], 0.05, key="rule_weight_graph")
    st.session_state["group_alpha"] = st.slider("Poids conviction de groupe", 0.0, 2.0, st.session_state["group_alpha"], 0.1, key="group_alpha_graph")
    st.session_state["tau_hours"] = st.slider("Demi-vie temporelle (τ, h)", 1.0, 72.0, float(st.session_state["tau_hours"]), 1.0, key="tau_hours_graph")
    st.session_state["group_sent_mode"] = st.radio("Sentiment des groupes utilisé", ["Calculé","Fourni (conviction→[-1,1])"], horizontal=False)
    st.session_state["alias_no_dollar"] = st.checkbox("Détecter alias sans $ (strict)", value=st.session_state["alias_no_dollar"])

with st.sidebar.expander("Filtrage graphe (anti-bruit) & Timeline", expanded=True):
    st.session_state["npmi_min"] = st.slider("Seuil NPMI token↔token", 0.0, 0.6, st.session_state["npmi_min"], 0.05, key="npmi_min_slider")
    st.session_state["min_cooc"] = st.slider("Min. co-mentions pondérées", 1, 10, int(st.session_state["min_cooc"]), key="min_cooc_slider")
    # timeline: si plage personnalisée active, on laisse la timeline à 0 (la plage prévaut)
    if st.session_state.get("use_custom_period"):
        st.session_state["timeline_hours"] = 0
        st.caption("Timeline désactivée car plage personnalisée active.")
    else:
        max_h = {"2h":2,"6h":6,"12h":12,"24h":24,"48h":48,"Tout":48}[st.session_state["period"]]
        st.session_state["timeline_hours"] = st.slider("Timeline (heures, ≤ période)", 0, max_h, int(st.session_state["timeline_hours"]), key="timeline_hours_slider")
    st.session_state["size_mode"] = st.selectbox(
        "Taille des nœuds", ["Degré","PageRank"],
        index=["Degré","PageRank"].index(st.session_state["size_mode"]),
        key="size_mode_select"
    )
    st.session_state["min_groups_token"] = st.slider("Filtre multi-groupes (≥ K groupes par token)", 1, 10, int(st.session_state["min_groups_token"]))

# Bannière de statut (dataset fusionné / uniques)
ui_status_banner(compact=True)

# Indicateur résumeur
ready = summarizer_available()
if st.session_state.get("use_hf", False):  # page graphe peut ne pas utiliser résumés, mais on informe
    if ready:
        st.caption("🟢 Résumeur DistilBART disponible (utilisé sur les pages qui le demandent).")
    elif ready is False:
        st.caption("🟠 Résumeur demandé mais indisponible sur cette machine.")
    else:
        st.caption("🔴 Résumeur non activé.")

# ---------- Data ----------
def _apply_period_filter(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "date" not in df.columns:
        return df
    # plage personnalisée prioritaire
    if st.session_state.get("use_custom_period"):
        try:
            sdate = st.session_state.get("custom_start_date")
            stime = st.session_state.get("custom_start_time")
            edate = st.session_state.get("custom_end_date")
            etime = st.session_state.get("custom_end_time")
            if sdate and stime and edate and etime:
                start = pd.Timestamp.combine(sdate, stime).tz_localize("Europe/Paris").tz_convert(None)
                end   = pd.Timestamp.combine(edate, etime).tz_localize("Europe/Paris").tz_convert(None)
                return df[(df["date"]>=start) & (df["date"]<=end)]
        except Exception:
            pass
    now = pd.Timestamp.now(tz="Europe/Paris").tz_localize(None)
    base_h = 48 if st.session_state["period"] == "Tout" else int(st.session_state["period"].replace("h",""))
    hours = min(base_h, st.session_state.get("timeline_hours", 0)) if st.session_state.get("timeline_hours", 0) else base_h
    cutoff = now - pd.Timedelta(hours=hours)
    return df[df["date"] >= cutoff]

# Récupère le dataset fusionné (persiste entre pages via session_state)
if raw_all.empty:
    st.info("Importe un JSON d’abord.")
    st.stop()

raw = _apply_period_filter(raw_all)
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
    )

# Debug
dt = explode_tokens(df)
n_msgs_tokens = dt['id'].nunique() if not dt.empty else 0
n_msgs_multi  = (dt.groupby('id')['token'].nunique().ge(2).sum() if not dt.empty else 0)
tokens_uniq   = (dt['token'].nunique() if not dt.empty else 0)
st.info(f"📌 Debug: Msgs ≥1 token: {n_msgs_tokens} | Msgs ≥2 tokens: {n_msgs_multi} | Tokens uniques: {tokens_uniq} | Groupes: {df['group'].nunique()}")

# Build edges (before UI filters)
edges0, node_sent, rel_index = graph_edges_advanced(
    df, tau_hours=float(st.session_state["tau_hours"]),
    group_sent_source=("provided" if st.session_state["group_sent_mode"].startswith("Fourni") else "calc")
)

# Fallback si vide
if edges0.empty:
    st.warning("Pas assez de co-mentions pour générer un graphe. Fallback: affichage minimal group↔token.")
    if dt.empty:
        st.info("Aucun message/tokens. Élargis la période/timeline.")
        st.stop()
    rows=[]; rel_rows=[]
    for mid, g in dt.groupby("id"):
        grp=g['group'].iloc[0]; toks=sorted(set(g['token'].tolist()))
        for t in toks:
            rows.append((grp, t, 1.0, "group-token", 0.0))
            rel_rows.append(("group-token", f"{grp}|{t}", mid))
    edges0 = pd.DataFrame(rows, columns=["src","dst","weight","type","sentiment"])
    rel_index = pd.DataFrame(rel_rows, columns=["rel_type","key","message_id"])

# ---------- UI controls ----------
c1, c2, c3 = st.columns(3)
w_max = float(edges0["weight"].max()) if not edges0.empty else 1.0
st.session_state["min_w"] = c1.slider(
    "Seuil minimum de poids d'arête",
    0.0, max(0.1, w_max), st.session_state["min_w"], 0.1,
    key="min_w_slider",
    help="Filtre les arêtes faibles. Mets 0 pour tout voir."
)
st.session_state["max_edges"] = c2.slider(
    "Limiter le nombre d'arêtes", 100, 5000, int(st.session_state["max_edges"]), 100,
    key="max_edges_slider", help="Affiche au plus N arêtes (les plus ‘fortes’)."
)
st.session_state["etype_opt"] = c3.multiselect(
    "Types d'arêtes à inclure", ["group-token","token-token","group-group"],
    default=st.session_state["etype_opt"], key="etype_opt_mult",
    help="group↔token, token↔token, group↔group"
)

mode = st.radio("Mode de visualisation", ["Global", "Focus sur un token", "Focus sur un groupe"], index=0, horizontal=True)
focus_token = None; focus_group = None
if mode == "Focus sur un token":
    tok_list = sorted(node_sent.loc[node_sent["kind"]=="token","node"].tolist())
    focus_token = st.selectbox("Choisir un token", tok_list, index=0 if tok_list else None)
elif mode == "Focus sur un groupe":
    grp_list = sorted(node_sent.loc[node_sent["kind"]=="group","node"].tolist())
    focus_group = st.selectbox("Choisir un groupe", grp_list, index=0 if grp_list else None)

# Filtrage
e = edges0.copy()
e = e[e["weight"] >= float(st.session_state["min_w"])]
e = e[e["type"].isin(st.session_state["etype_opt"])]
npmi_min = float(st.session_state["npmi_min"])
min_cooc = float(st.session_state["min_cooc"])
mask_tt = (e["type"]!="token-token") | ((e["npmi"].fillna(-1) >= npmi_min) & (e["cooc"].fillna(0) >= min_cooc))
e = e[mask_tt]
abs_s_min = float(st.session_state["abs_sent_min"])
e = e[e["sentiment"].abs() >= abs_s_min]

# Filtre multi-groupes
K = int(st.session_state["min_groups_token"])
if K > 1:
    gt = e[e["type"]=="group-token"].groupby("dst")["src"].nunique().reset_index(name="n_groups")
    keep_tokens = set(gt[gt["n_groups"]>=K]["dst"].tolist())
    if keep_tokens:
        e = e[~((e["type"]=="group-token") & (~e["dst"].isin(keep_tokens)))]
        e = e[~((e["type"]=="token-token") & (~e["src"].isin(keep_tokens)) & (~e["dst"].isin(keep_tokens)))]

if focus_token:
    e = e[(e["src"] == focus_token) | (e["dst"] == focus_token)]
if focus_group:
    e = e[(e["src"] == focus_group) | (e["dst"] == focus_group)]
e = e.sort_values("weight", ascending=False).head(int(st.session_state["max_edges"]))

# Fallback si vide après filtres
if e.empty:
    st.warning("Aucune arête après filtres — fallback group↔token minimal.")
    e_gt = edges0[edges0["type"]=="group-token"].copy()
    if e_gt.empty and not dt.empty:
        rows=[]
        for mid, g in dt.groupby("id"):
            grp=g['group'].iloc[0]; toks=sorted(set(g['token'].tolist()))
            for t in toks: rows.append((grp, t, 1.0, "group-token", 0.0))
        e_gt = pd.DataFrame(rows, columns=["src","dst","weight","type","sentiment"])
    e = e_gt.sort_values("weight", ascending=False).head(1000)

st.caption(f"Arêtes affichées : {len(e)} • Nœuds : {len(pd.unique(e[['src','dst']].values.ravel()))}")
if e.empty:
    st.error("Toujours aucune arête. Clique “🔁 Assouplir les filtres”.")
    st.stop()

# Build graph
G = nx.Graph()
for _, r in e.iterrows():
    G.add_edge(r["src"], r["dst"], weight=float(r["weight"]), etype=r["type"], sentiment=float(r["sentiment"]))

# Node sentiments
sent_map = {}
for _, r in node_sent.iterrows():
    sent_map[r["node"]] = float(r["sentiment"])

# Communautés (safe)
def _safe_communities(G):
    if G.number_of_edges() == 0:
        return [{n} for n in G.nodes()]
    wsum = sum(d.get("weight",0.0) for _,_,d in G.edges(data=True))
    if wsum == 0:
        return [{n} for n in G.nodes()]
    try:
        from networkx.algorithms.community import louvain_communities
        return louvain_communities(G, weight="weight", seed=42)
    except Exception:
        from networkx.algorithms.community import greedy_modularity_communities
        return list(greedy_modularity_communities(G, weight="weight"))

comms = _safe_communities(G)
cluster_id = {}
for ci, community in enumerate(comms, start=1):
    for n in community:
        cluster_id[n] = ci

# Centralités
size_mode = st.session_state["size_mode"]
pr = nx.pagerank(G, weight="weight") if G.number_of_edges()>0 else {}
btw = nx.betweenness_centrality(G, weight="weight", k=100 if G.number_of_edges()>2000 else None) if G.number_of_edges()>0 else {}

# Colors
def color_from_sent(s):
    s = max(-1.0, min(1.0, float(s)))
    if s >= 0:
        r1,g1,b1=(156,163,175); r2,g2,b2=(34,197,94); t=s
    else:
        r1,g1,b1=(239,68,68); r2,g2,b2=(156,163,175); t=s+1
    r=int(r1+(r2-r1)*t); g=int(g1+(r2-g1)*t); b=int(b1+(r2-b1)*t)
    return f"#{r:02x}{g:02x}{b:02x}"

# PyVis
height_px = 900
net = Network(height=f"{height_px}px", width="100%", bgcolor="#0b1220", font_color="#e5e7eb")
net.barnes_hut(gravity=-25000, central_gravity=0.3, spring_length=120, spring_strength=0.007, damping=0.8)

def scale_quad(x, xmin, xmax, out_min=8, out_max=60):
    if xmax == xmin: return (out_min+out_max)/2
    z = (x - xmin) / (xmax - xmin)
    return out_min + (z**2) * (out_max - out_min)

# sizes
if size_mode == "PageRank":
    vals = list(pr.values()) if pr else [0]
else:
    vals = [deg for _,deg in G.degree()] if G.number_of_nodes()>0 else [0]
vmin, vmax = (min(vals), max(vals)) if vals else (0,1)

# nodes
group_nodes = set(node_sent.loc[node_sent["kind"]=="group","node"].tolist())
for n in G.nodes():
    s = sent_map.get(n, 0.0)
    color = color_from_sent(s)
    bw = int(1 + 6*abs(s))
    cid = cluster_id.get(n, 0)
    kind = "group" if n in group_nodes else "token"
    sv = (pr.get(n, 0.0) if size_mode=="PageRank" else G.degree(n))
    size = scale_quad(float(sv), float(vmin), float(vmax))
    title = f"{kind}: {n} — s={s:+.2f}, cluster=c{cid}, PR={pr.get(n,0.0):.4f}, BTWN={btw.get(n,0.0):.4f}"
    net.add_node(n, label=f"{n} (c{cid})", title=title, value=size, color=color,
                 shape=("box" if kind=="group" else "dot"), borderWidth=bw, shadow=True)

# edges
def _scale_w(w, a=1, b=16, w_min=None, w_max=None):
    if w_min is None: w_min = e["weight"].min()
    if w_max is None: w_max = e["weight"].max()
    if w_max == w_min: return (a+b)/2
    return a + (w - w_min) * (b - a) / (w_max - w_min)

for _, r in e.iterrows():
    w=float(r["weight"]); s=float(r["sentiment"]); et=r["type"]
    color=color_from_sent(s)
    ttip=f"{et} — poids={w:.2f}, sentiment={s:+.2f}"
    if et=="token-token" and pd.notna(r.get("npmi", float('nan'))):
        ttip+=f", NPMI={r['npmi']:.3f}, cooc={r.get('cooc',0):.2f}"
    net.add_edge(r["src"], r["dst"], value=_scale_w(w), color=color, title=ttip, physics=True, smooth=True)

st.markdown("**Couleur = sentiment (rouge→gris→vert), halo = |sentiment|, taille = Degré/PageRank.**")

# Messages
st.subheader("🔎 Voir les messages liés à un nœud / à une arête")
col1, col2 = st.columns(2)
with col1:
    all_nodes = sorted(set(pd.unique(e[["src","dst"]].values.ravel())))
    sel_node = st.selectbox("Nœud", all_nodes, index=0 if all_nodes else None)
    if st.button("Voir messages (nœud)") and sel_node:
        if sel_node in df["group"].unique().tolist():
            msgs = df[df["group"] == sel_node].sort_values("date", ascending=False).head(500)
        else:
            msgs = df[df["tokens"].apply(lambda ts: sel_node in (ts or []))].sort_values("date", ascending=False).head(500)
        st.dataframe(msgs[["date","group","tokens","sentiment","w_sentiment","text","remark"]], use_container_width=True)

with col2:
    e_options = e.apply(lambda r: f"{r['type']} | {r['src']} ↔ {r['dst']}", axis=1).tolist()
    sel_edge = st.selectbox("Arête", e_options, index=0 if e_options else None)
    if st.button("Voir messages (arête)") and e_options:
        etype = sel_edge.split(" | ")[0]
        pair = sel_edge.split(" | ")[1]
        src, dst = [x.strip() for x in pair.split("↔")]
        if etype == "group-token":
            key = f"{src}|{dst}"
            ids = set(rel_index[(rel_index["rel_type"]==etype) & (rel_index["key"]==key)]["message_id"].tolist())
            msgs = df[df["id"].isin(ids)]
        elif etype == "token-token":
            key1 = f"{src}|{dst}"; key2 = f"{dst}|{src}"
            ids = set(rel_index[(rel_index["rel_type"]==etype) & (rel_index["key"].isin([key1,key2]))]["message_id"].tolist())
            msgs = df[df["id"].isin(ids)]
        else:  # group-group
            msgs = df[df["group"].isin([src, dst])]
        st.dataframe(msgs.sort_values("date", ascending=False).head(500)[["date","group","tokens","sentiment","w_sentiment","text","remark"]],
                     use_container_width=True)

# ---------- Tops structurels ----------
st.subheader("🏅 Tops structurels")
token_nodes = set(n for n in G.nodes() if n not in df["group"].unique().tolist())
rows_pr = [{"token": n, "pagerank": float(pr.get(n,0.0)), "deg": G.degree(n),
            "sentiment": float(sent_map.get(n,0.0)), "cluster": int(cluster_id.get(n,0))}
           for n in token_nodes]
top_pr = pd.DataFrame(rows_pr, columns=["token","pagerank","deg","sentiment","cluster"]) if rows_pr else pd.DataFrame(columns=["token","pagerank","deg","sentiment","cluster"])
if not top_pr.empty:
    top_pr = top_pr.sort_values("pagerank", ascending=False).head(25)

rows_bw = [{"node": n, "type": ("group" if n in df["group"].unique().tolist() else "token"),
            "betweenness": float(btw.get(n,0.0)), "deg": G.degree(n),
            "sentiment": float(sent_map.get(n,0.0)), "cluster": int(cluster_id.get(n,0))}
           for n in G.nodes()]
top_btw = pd.DataFrame(rows_bw, columns=["node","type","betweenness","deg","sentiment","cluster"]) if rows_bw else pd.DataFrame(columns=["node","type","betweenness","deg","sentiment","cluster"])
if not top_btw.empty:
    top_btw = top_btw.sort_values("betweenness", ascending=False).head(25)

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Top tokens par PageRank**")
    st.dataframe(top_pr, use_container_width=True)
with c2:
    st.markdown("**Top ‘ponts’ (Betweenness)**")
    st.dataframe(top_btw, use_container_width=True)

# Render network
with NamedTemporaryFile(suffix=".html", delete=False) as tmp:
    net.write_html(tmp.name)
    html = Path(tmp.name).read_text(encoding="utf-8")
st.components.v1.html(html, height=900, scrolling=True)
