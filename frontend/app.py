import time
from pathlib import Path

import numpy as np
import httpx
import yaml
import streamlit as st
from sklearn.decomposition import PCA
import plotly.graph_objects as go

API_BASE = "http://localhost:8000"
INTENTS_PATH = Path(__file__).parent.parent / "intents.yaml"

st.set_page_config(page_title="Semantic Search", page_icon="🔍", layout="wide")
st.title("🔍 Semantic Search")

tab_upload, tab_search, tab_viz, tab_intents, tab_files = st.tabs(
    ["Upload", "Search", "Visualize", "Intents", "Indexed Files"]
)


# ---------------------------------------------------------------------------
# Upload tab
# ---------------------------------------------------------------------------
with tab_upload:
    st.subheader("Upload a file to index")
    st.caption("Supported formats: CSV, XLSX, XLS, JSON — one file at a time")

    uploaded = st.file_uploader(
        "Choose a file",
        type=["csv", "xlsx", "xls", "json"],
        label_visibility="collapsed",
    )

    if uploaded and st.button("Upload & Embed", type="primary"):
        with st.spinner("Sending file to API..."):
            try:
                resp = httpx.post(
                    f"{API_BASE}/upload",
                    files={"file": (uploaded.name, uploaded.getvalue(), "application/octet-stream")},
                    timeout=30,
                )
                resp.raise_for_status()
                job_id = resp.json()["job_id"]
            except Exception as e:
                st.error(f"Upload failed: {e}")
                st.stop()

        st.info(f"Job started: `{job_id}`")
        progress_bar = st.progress(0)
        status_text = st.empty()

        while True:
            try:
                status_resp = httpx.get(f"{API_BASE}/status/{job_id}", timeout=10)
                data = status_resp.json()
            except Exception as e:
                st.error(f"Could not poll status: {e}")
                break

            pct = data.get("progress", 0)
            msg = data.get("message", "")
            state = data.get("status", "running")

            progress_bar.progress(pct)
            status_text.text(msg)

            if state == "done":
                st.success(f"Indexed {data['total_rows']} rows from **{uploaded.name}**")
                break
            elif state == "error":
                st.error(f"Error: {data.get('error', 'Unknown error')}")
                break

            time.sleep(1)


# ---------------------------------------------------------------------------
# Search tab
# ---------------------------------------------------------------------------
with tab_search:
    st.subheader("Search across indexed documents")

    query = st.text_input("Enter your search query", placeholder="e.g. high revenue customers in Q3")
    top_k = st.slider("Results to return", min_value=1, max_value=25, value=10)

    if st.button("Search", type="primary") and query.strip():
        with st.spinner("Searching..."):
            try:
                resp = httpx.post(
                    f"{API_BASE}/search",
                    json={"query": query, "top_k": top_k},
                    timeout=30,
                )
                resp.raise_for_status()
                results = resp.json()["results"]
            except Exception as e:
                st.error(f"Search failed: {e}")
                st.stop()

        if not results:
            st.warning("No results found.")
        else:
            st.success(f"{len(results)} results")
            for i, r in enumerate(results, 1):
                score = r.pop("score", None)
                source = r.pop("source_file", "unknown")
                with st.expander(f"#{i} — score: {score}  |  source: {source}"):
                    st.json(r)


# ---------------------------------------------------------------------------
# Visualize tab
# ---------------------------------------------------------------------------
with tab_viz:
    # --- Controls row ---
    q_col, k_col, btn_col = st.columns([4, 1, 1])
    with q_col:
        viz_query = st.text_input(
            "Query",
            placeholder="Type a query and press Enter to see where it lands…",
            key="viz_query",
            label_visibility="collapsed",
        )
    with k_col:
        top_k_viz = st.slider("Top K", 3, 20, 8, key="viz_topk", label_visibility="collapsed")
    with btn_col:
        b1, b2 = st.columns(2)
        refresh_btn = b1.button("↺", help="Reload vector index & refit PCA", use_container_width=True)
        clear_btn   = b2.button("✕", help="Clear query history", use_container_width=True)

    if clear_btn:
        for k in ("query_trajectory", "viz_results", "viz_last_query", "viz_last_topk"):
            st.session_state.pop(k, None)

    # --- Load / refit PCA ---
    need_init = refresh_btn or "pca_coords" not in st.session_state
    if need_init:
        with st.spinner("Fetching vectors and fitting PCA…"):
            try:
                resp = httpx.get(f"{API_BASE}/vectors", timeout=30)
                resp.raise_for_status()
                raw_points = resp.json()["points"]

                vecs     = np.array([p["vector"] for p in raw_points], dtype=np.float32)
                payloads = [p["payload"] for p in raw_points]

                pca    = PCA(n_components=2, random_state=42)
                coords = pca.fit_transform(vecs)

                st.session_state.pca_coords   = coords
                st.session_state.pca_model    = pca
                st.session_state.pca_payloads = payloads
                # reset history when index changes
                st.session_state.query_trajectory = []
                st.session_state.viz_last_query   = ""
            except Exception as exc:
                st.error(f"Failed to load vector index: {exc}")
                st.stop()

    coords   = st.session_state.get("pca_coords")
    payloads = st.session_state.get("pca_payloads", [])
    pca      = st.session_state.get("pca_model")

    if coords is None:
        st.info("Click ↺ to load the vector index.")
        st.stop()

    # --- Embed & search if query changed ---
    prev_query = st.session_state.get("viz_last_query", "")
    prev_topk  = st.session_state.get("viz_last_topk", None)
    q_coord    = None
    result_scores = {}
    results_list  = st.session_state.get("viz_results", [])

    query_changed = viz_query.strip() and (viz_query != prev_query or top_k_viz != prev_topk)

    if query_changed:
        try:
            embed_resp = httpx.post(
                f"{API_BASE}/embed", json={"query": viz_query}, timeout=15
            )
            embed_resp.raise_for_status()
            qvec    = np.array(embed_resp.json()["vector"], dtype=np.float32)
            q_coord = pca.transform(qvec.reshape(1, -1))[0]

            search_resp = httpx.post(
                f"{API_BASE}/search",
                json={"query": viz_query, "top_k": top_k_viz},
                timeout=15,
            )
            search_resp.raise_for_status()
            results_list  = search_resp.json()["results"]
            result_scores = {r["company_name"]: r["score"] for r in results_list}

            traj = st.session_state.get("query_trajectory", [])
            traj.append({"text": viz_query, "x": float(q_coord[0]), "y": float(q_coord[1])})
            st.session_state.query_trajectory = traj[-12:]
            st.session_state.viz_results      = results_list
            st.session_state.viz_last_query   = viz_query
            st.session_state.viz_last_topk    = top_k_viz

        except Exception as exc:
            st.error(f"Query failed: {exc}")

    elif viz_query.strip() and viz_query == prev_query:
        results_list  = st.session_state.get("viz_results", [])
        result_scores = {r["company_name"]: r["score"] for r in results_list}
        traj = st.session_state.get("query_trajectory", [])
        if traj and traj[-1]["text"] == viz_query:
            q_coord = np.array([traj[-1]["x"], traj[-1]["y"]])

    # --- Build Plotly figure ---
    company_names = [p.get("company_name", "?") for p in payloads]
    industries    = [p.get("mapped_industry", "Other") for p in payloads]
    short_descs   = [p.get("short_description", "") for p in payloads]
    websites      = [p.get("website", "") for p in payloads]

    unique_inds = list(dict.fromkeys(industries))
    palette = [
        "#60a5fa", "#34d399", "#f59e0b", "#f87171", "#a78bfa",
        "#38bdf8", "#fb923c", "#4ade80", "#e879f9", "#22d3ee",
        "#fb7185", "#86efac", "#fcd34d", "#c4b5fd", "#67e8f9",
        "#fdba74", "#6ee7b7", "#93c5fd",
    ]
    ind_color = {ind: palette[i % len(palette)] for i, ind in enumerate(unique_inds)}

    result_set = set(result_scores.keys())
    other_idx  = [i for i, n in enumerate(company_names) if n not in result_set]
    result_idx = [i for i, n in enumerate(company_names) if n in result_set]

    fig = go.Figure()

    # 1. Background dots — colored by industry
    for ind in unique_inds:
        idx = [i for i in other_idx if industries[i] == ind]
        if not idx:
            continue
        hex_c = ind_color[ind]
        r, g, b = int(hex_c[1:3], 16), int(hex_c[3:5], 16), int(hex_c[5:7], 16)
        fig.add_trace(go.Scatter(
            x=coords[idx, 0], y=coords[idx, 1],
            mode="markers",
            name=ind,
            legendgroup=ind,
            marker=dict(
                color=f"rgba({r},{g},{b},0.50)",
                size=9,
                line=dict(width=0),
            ),
            text=[company_names[i] for i in idx],
            customdata=[[short_descs[i], websites[i]] for i in idx],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "<i>%{customdata[0]}</i><br>"
                "<extra></extra>"
            ),
        ))

    # 2. Query trajectory (faded history trail)
    traj = st.session_state.get("query_trajectory", [])
    if len(traj) > 1:
        tx = [t["x"] for t in traj]
        ty = [t["y"] for t in traj]
        tt = [t["text"] for t in traj]
        # opacity fades from oldest (0.15) to newest (0.6)
        n = len(traj)
        for j in range(len(traj) - 1):
            alpha_line = 0.12 + 0.5 * (j / max(n - 2, 1))
            alpha_dot  = 0.20 + 0.55 * (j / max(n - 2, 1))
            fig.add_trace(go.Scatter(
                x=[tx[j], tx[j + 1]], y=[ty[j], ty[j + 1]],
                mode="lines",
                line=dict(color=f"rgba(255,140,90,{alpha_line:.2f})", width=1.5, dash="dot"),
                showlegend=False,
                hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=[tx[j]], y=[ty[j]],
                mode="markers",
                marker=dict(symbol="circle", size=8,
                            color=f"rgba(255,140,90,{alpha_dot:.2f})"),
                text=[f'"{tt[j]}"'],
                hovertemplate='<b>Past:</b> %{text}<extra></extra>',
                showlegend=False,
            ))

    # 3. Connector lines — query → each top-k result
    if q_coord is not None:
        for i in result_idx:
            fig.add_trace(go.Scatter(
                x=[q_coord[0], coords[i, 0]],
                y=[q_coord[1], coords[i, 1]],
                mode="lines",
                line=dict(color="rgba(255,80,50,0.20)", width=1.2, dash="dot"),
                showlegend=False,
                hoverinfo="skip",
            ))

    # 4. Top-k results — colored by similarity score
    if result_idx:
        scores = [result_scores[company_names[i]] for i in result_idx]
        sorted_ri = sorted(result_idx, key=lambda i: result_scores[company_names[i]], reverse=True)
        rank_of = {i: r + 1 for r, i in enumerate(sorted_ri)}

        fig.add_trace(go.Scatter(
            x=coords[result_idx, 0], y=coords[result_idx, 1],
            mode="markers+text",
            name="Top-k matches",
            text=[f"#{rank_of[i]} {company_names[i]}" for i in result_idx],
            textposition="top center",
            textfont=dict(size=10, color="#ffffff"),
            marker=dict(
                color=scores,
                colorscale="YlOrRd",
                size=15,
                opacity=0.95,
                line=dict(width=2, color="white"),
                showscale=True,
                colorbar=dict(
                    title=dict(text="Similarity", font=dict(size=11)),
                    thickness=14, len=0.45, x=1.02,
                    tickfont=dict(size=10),
                ),
            ),
            customdata=[[short_descs[i], result_scores[company_names[i]], websites[i]]
                        for i in result_idx],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "<i>%{customdata[0]}</i><br>"
                "Score: <b>%{customdata[1]:.4f}</b>"
                "<extra></extra>"
            ),
        ))

    # 5. Current query — red star
    if q_coord is not None:
        fig.add_trace(go.Scatter(
            x=[q_coord[0]], y=[q_coord[1]],
            mode="markers+text",
            name="Query",
            text=[f'"{viz_query}"'],
            textposition="bottom right",
            textfont=dict(size=11, color="#ff6b6b"),
            marker=dict(
                symbol="star",
                size=22,
                color="#ff3333",
                line=dict(width=2, color="white"),
            ),
            hovertemplate=f'<b>Query:</b> "{viz_query}"<extra></extra>',
        ))

    var = pca.explained_variance_ratio_
    fig.update_layout(
        height=640,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(12,14,21,1)",
        title=dict(
            text=(
                f"Embedding space — PCA 2D  "
                f"<span style='font-size:11px;color:#888'>"
                f"(PC1 {var[0]:.1%} + PC2 {var[1]:.1%} variance explained)</span>"
            ),
            font=dict(size=14),
            x=0.01,
        ),
        xaxis=dict(
            title="Principal Component 1",
            showgrid=True, gridcolor="rgba(255,255,255,0.06)",
            zeroline=False, showticklabels=False,
        ),
        yaxis=dict(
            title="Principal Component 2",
            showgrid=True, gridcolor="rgba(255,255,255,0.06)",
            zeroline=False, showticklabels=False,
        ),
        legend=dict(
            font=dict(size=10),
            bgcolor="rgba(0,0,0,0.45)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,
            tracegroupgap=4,
        ),
        margin=dict(l=40, r=90, t=60, b=40),
        hovermode="closest",
    )

    # --- Layout: chart left, results right ---
    chart_col, results_col = st.columns([5, 2])

    with chart_col:
        st.plotly_chart(fig, use_container_width=True)
        if not viz_query.strip():
            st.caption("💡 Type a query above and press **Enter** — the ★ will appear where your query lands in vector space. Keep refining it to see the trajectory.")

    with results_col:
        if results_list:
            st.markdown(f"**Top {len(results_list)} · _{viz_query}_**")
            st.divider()
            for i, r in enumerate(results_list, 1):
                score = r.get("score", 0)
                name  = r.get("company_name", "?")
                short = r.get("short_description", "")
                url   = r.get("website", "#")
                ind   = r.get("mapped_industry", "")
                bar_filled = round(score * 16)
                bar = "▓" * bar_filled + "░" * (16 - bar_filled)
                st.markdown(
                    f"**{i}. [{name}]({url})**  \n"
                    f"`{bar}` `{score:.3f}`  \n"
                    f"<small style='color:#aaa'>{ind}</small>  \n"
                    f"<small>{short}</small>",
                    unsafe_allow_html=True,
                )
                if i < len(results_list):
                    st.divider()
        else:
            st.markdown("### Results")
            st.caption("Will populate once you run a query.")


# ---------------------------------------------------------------------------
# Intents tab
# ---------------------------------------------------------------------------

# Extra scenario utterances per intent to widen cluster spread
_EXTRA: dict[str, dict[str, list[str]]] = {
    "benevolence": {
        "view_events": [
            "what's happening this weekend",
            "any local events near me",
            "show me upcoming activities",
            "are there community events I can attend",
            "I'm looking for things to do",
        ],
        "view_wishlists": [
            "show me the wish lists",
            "what do people want as gifts",
            "let me see saved wishlists",
        ],
        "buy_gifts": [
            "I need gift ideas for my partner",
            "help me find a present",
            "what should I get for a birthday",
            "I want to send someone a gift",
            "suggest something to buy for a friend",
        ],
    },
    "remembot": {
        "remember": [
            "save this for me",
            "don't let me forget this",
            "make a note of this",
            "jot this down",
            "keep track of this",
        ],
        "recall": [
            "what did I save earlier",
            "remind me what I noted",
            "show me my saved items",
            "what have I been remembering",
            "look up what I stored",
        ],
    },
    "moneyshare": {
        "avoid_fee": [
            "how do I not get charged a fee",
            "help me waive this penalty",
            "I don't want to pay an overdraft fee",
            "can I get this fee removed",
        ],
        "request_loan": [
            "can I borrow some money",
            "I need cash quickly",
            "give me a short term loan",
            "I need to borrow a little to cover something",
            "advance me some funds",
        ],
    },
    "foodshare": {
        "request_food": [
            "I need food",
            "looking for something to eat",
            "where can I find a meal",
            "I haven't eaten and need help",
            "can someone share food with me",
        ],
        "share_food": [
            "I made too much and want to give some away",
            "I want to donate food",
            "I have leftovers to share",
            "someone can have the rest of my food",
        ],
    },
    "billpayshare": {
        "request_bill_help": [
            "can someone help me pay my utilities",
            "I can't cover my electric bill this month",
            "I need help splitting this bill",
            "my bill is overdue and I need assistance",
        ],
    },
    "bloodshare": {
        "request_blood": [
            "I need a blood donor urgently",
            "looking for blood type O positive",
            "can someone donate blood for a patient",
            "blood is needed for surgery",
        ],
        "share_blood": [
            "I want to donate blood",
            "I'm willing to give blood",
            "I can be a blood donor",
            "where can I donate blood",
        ],
    },
    "math": {
        "compute_expression": [
            "calculate 2 plus 2",
            "what is 5 squared",
            "solve this equation for me",
            "evaluate this expression",
            "what does x squared plus one equal",
            "run this calculation",
        ],
    },
    "shopping_assistant": {
        "add_item": [
            "put milk on my grocery list",
            "I need to buy eggs",
            "add bread to my list",
            "throw some coffee on there too",
        ],
        "remove_item": [
            "take eggs off my list",
            "I already have butter, remove it",
            "cross that off",
            "delete that item from my shopping list",
        ],
        "view_list": [
            "read my list to me",
            "what am I supposed to be buying",
            "show my grocery list",
            "what's on the list",
        ],
        "edit_list": [
            "update what's on my shopping list",
            "I want to modify my grocery list",
            "make changes to my list",
        ],
        "mark_purchased": [
            "I got the milk",
            "bought those already",
            "check off the eggs",
            "I picked up most of the items",
        ],
    },
    "bot_store": {
        "add_package": [
            "I want to subscribe to a new feature",
            "activate a module for me",
            "get me access to that add-on",
            "I'd like to try a new package",
        ],
        "remove_package": [
            "cancel my subscription to that package",
            "I don't need that feature anymore",
            "unsubscribe me from this",
            "turn off that module",
        ],
    },
    "taskmaster_ai": {
        "add_task": [
            "put this on my to-do list",
            "new task: call the dentist",
            "remind me to do laundry",
            "add a reminder to my task list",
        ],
        "view_tasks": [
            "what do I have to do today",
            "list all my tasks",
            "show me what's on my agenda",
            "read out my to-dos",
        ],
        "edit_tasks": [
            "update that task",
            "change the details on my to-do item",
            "modify a task in my list",
        ],
        "remove_task": [
            "delete that task",
            "I finished it, take it off the list",
            "clear completed tasks",
            "remove that item from my to-dos",
        ],
    },
    "flow_planner": {
        "create_flow_plan": [
            "plan my day for me",
            "schedule tomorrow",
            "build me a daily plan",
            "organize my week",
            "create a schedule starting tomorrow morning",
        ],
    },
}


def _load_intent_rows() -> list[dict]:
    """Return flat list of {domain, intent, utterance} from YAML + extras."""
    with open(INTENTS_PATH) as f:
        taxonomy = yaml.safe_load(f)["intents"]

    rows = []
    for domain, intents in taxonomy.items():
        for intent, data in intents.items():
            for utt in data.get("utterances", []) + data.get("aliases", []):
                rows.append({"domain": domain, "intent": intent, "utterance": utt, "source": "yaml"})
            for utt in _EXTRA.get(domain, {}).get(intent, []):
                rows.append({"domain": domain, "intent": intent, "utterance": utt, "source": "scenario"})
    return rows


def _embed_batch(utterances: list[str]) -> list[list[float]]:
    vectors = []
    for utt in utterances:
        resp = httpx.post(f"{API_BASE}/embed", json={"query": utt}, timeout=15)
        resp.raise_for_status()
        vectors.append(resp.json()["vector"])
    return vectors


with tab_intents:
    st.subheader("Intent Space")
    st.caption("All intents and scenario utterances projected into 2D embedding space. Colored by domain, shaped by source (● YAML · ✦ scenario).")

    i_col1, i_col2 = st.columns([6, 1])
    with i_col1:
        intent_query = st.text_input(
            "Type a message to see where it lands",
            placeholder="e.g. I need some cash…",
            key="intent_query",
            label_visibility="collapsed",
        )
    with i_col2:
        rebuild_btn = st.button("↺ Rebuild", help="Re-embed all utterances", use_container_width=True)

    if rebuild_btn:
        for k in ("intent_rows", "intent_coords", "intent_pca", "intent_query_history"):
            st.session_state.pop(k, None)

    # --- Embed intent corpus once ---
    if "intent_coords" not in st.session_state:
        rows = _load_intent_rows()
        utterances = [r["utterance"] for r in rows]
        with st.spinner(f"Embedding {len(utterances)} utterances…"):
            try:
                vecs = _embed_batch(utterances)
                pca_i = PCA(n_components=2, random_state=42)
                coords_i = pca_i.fit_transform(np.array(vecs, dtype=np.float32))
                st.session_state.intent_rows = rows
                st.session_state.intent_coords = coords_i
                st.session_state.intent_pca = pca_i
            except Exception as exc:
                st.error(f"Failed to embed intents: {exc}")
                st.stop()

    rows_i = st.session_state.intent_rows
    coords_i = st.session_state.intent_coords
    pca_i = st.session_state.intent_pca

    domains = [r["domain"] for r in rows_i]
    intents_col = [r["intent"] for r in rows_i]
    utterances_col = [r["utterance"] for r in rows_i]
    sources = [r["source"] for r in rows_i]

    unique_domains = list(dict.fromkeys(domains))
    palette = [
        "#60a5fa", "#34d399", "#f59e0b", "#f87171", "#a78bfa",
        "#38bdf8", "#fb923c", "#4ade80", "#e879f9", "#22d3ee",
        "#fb7185", "#86efac",
    ]
    domain_color = {d: palette[i % len(palette)] for i, d in enumerate(unique_domains)}

    fig_i = go.Figure()

    # Domain clusters
    for dom in unique_domains:
        idx = [i for i, d in enumerate(domains) if d == dom]
        yaml_idx = [i for i in idx if sources[i] == "yaml"]
        scen_idx = [i for i in idx if sources[i] == "scenario"]
        hex_c = domain_color[dom]
        r, g, b = int(hex_c[1:3], 16), int(hex_c[3:5], 16), int(hex_c[5:7], 16)

        if yaml_idx:
            fig_i.add_trace(go.Scatter(
                x=coords_i[yaml_idx, 0], y=coords_i[yaml_idx, 1],
                mode="markers",
                name=dom,
                legendgroup=dom,
                marker=dict(symbol="circle", size=10,
                            color=f"rgba({r},{g},{b},0.85)",
                            line=dict(width=1, color="white")),
                text=[utterances_col[i] for i in yaml_idx],
                customdata=[[intents_col[i]] for i in yaml_idx],
                hovertemplate="<b>%{customdata[0]}</b><br>%{text}<extra>" + dom + "</extra>",
            ))

        if scen_idx:
            fig_i.add_trace(go.Scatter(
                x=coords_i[scen_idx, 0], y=coords_i[scen_idx, 1],
                mode="markers",
                name=dom + " (scenario)",
                legendgroup=dom,
                showlegend=False,
                marker=dict(symbol="diamond", size=8,
                            color=f"rgba({r},{g},{b},0.45)",
                            line=dict(width=1, color="white")),
                text=[utterances_col[i] for i in scen_idx],
                customdata=[[intents_col[i]] for i in scen_idx],
                hovertemplate="<b>%{customdata[0]}</b><br>%{text}<extra>" + dom + " · scenario</extra>",
            ))

    # Centroid labels per intent
    seen_intents = set()
    for dom in unique_domains:
        intent_set = dict.fromkeys(intents_col[i] for i, d in enumerate(domains) if d == dom)
        for intent in intent_set:
            idx = [i for i, (d, it) in enumerate(zip(domains, intents_col)) if d == dom and it == intent]
            cx = float(np.mean(coords_i[idx, 0]))
            cy = float(np.mean(coords_i[idx, 1]))
            if intent not in seen_intents:
                fig_i.add_annotation(
                    x=cx, y=cy,
                    text=f"<b>{intent}</b>",
                    showarrow=False,
                    font=dict(size=9, color="rgba(255,255,255,0.65)"),
                    bgcolor="rgba(0,0,0,0.35)",
                    borderpad=2,
                )
                seen_intents.add(intent)

    # Live query overlay
    q_coord_i = None
    classified = None
    prev_iq = st.session_state.get("intent_last_query", "")

    if intent_query.strip() and intent_query != prev_iq:
        try:
            embed_resp = httpx.post(f"{API_BASE}/embed", json={"query": intent_query}, timeout=15)
            embed_resp.raise_for_status()
            qvec_i = np.array(embed_resp.json()["vector"], dtype=np.float32)
            q_coord_i = pca_i.transform(qvec_i.reshape(1, -1))[0]

            cls_resp = httpx.post(f"{API_BASE}/classify", json={"utterance": intent_query}, timeout=15)
            cls_resp.raise_for_status()
            classified = cls_resp.json()

            hist = st.session_state.get("intent_query_history", [])
            hist.append({"text": intent_query, "x": float(q_coord_i[0]), "y": float(q_coord_i[1]),
                         "domain": classified.get("domain", "?"), "intent": classified.get("intent", "?"),
                         "confidence": classified.get("confidence", "?")})
            st.session_state.intent_query_history = hist[-10:]
            st.session_state.intent_last_query = intent_query
            st.session_state.intent_last_classified = classified
            st.session_state.intent_last_coord = q_coord_i.tolist()
        except Exception as exc:
            st.error(f"Query failed: {exc}")

    elif intent_query.strip() and intent_query == prev_iq:
        classified = st.session_state.get("intent_last_classified")
        saved_coord = st.session_state.get("intent_last_coord")
        if saved_coord:
            q_coord_i = np.array(saved_coord)

    # Past query trail
    hist = st.session_state.get("intent_query_history", [])
    for j, h in enumerate(hist[:-1]):
        alpha = 0.15 + 0.55 * (j / max(len(hist) - 2, 1))
        fig_i.add_trace(go.Scatter(
            x=[h["x"]], y=[h["y"]],
            mode="markers",
            marker=dict(symbol="star", size=11,
                        color=f"rgba(255,140,90,{alpha:.2f})"),
            text=[f'"{h["text"]}" → {h["domain"]}.{h["intent"]}'],
            hovertemplate="%{text}<extra>past query</extra>",
            showlegend=False,
        ))

    if q_coord_i is not None:
        fig_i.add_trace(go.Scatter(
            x=[q_coord_i[0]], y=[q_coord_i[1]],
            mode="markers+text",
            name="Your query",
            text=[f'"{intent_query}"'],
            textposition="bottom right",
            textfont=dict(size=11, color="#ff6b6b"),
            marker=dict(symbol="star", size=22, color="#ff3333",
                        line=dict(width=2, color="white")),
            hovertemplate=f'<b>Query:</b> "{intent_query}"<extra></extra>',
        ))

    var_i = pca_i.explained_variance_ratio_
    fig_i.update_layout(
        height=640,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(12,14,21,1)",
        title=dict(
            text=(
                f"Intent Embedding Space — PCA 2D  "
                f"<span style='font-size:11px;color:#888'>"
                f"(PC1 {var_i[0]:.1%} + PC2 {var_i[1]:.1%} variance explained)</span>"
            ),
            font=dict(size=14), x=0.01,
        ),
        xaxis=dict(title="PC 1", showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                   zeroline=False, showticklabels=False),
        yaxis=dict(title="PC 2", showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                   zeroline=False, showticklabels=False),
        legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0.45)",
                    bordercolor="rgba(255,255,255,0.1)", borderwidth=1, tracegroupgap=4),
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode="closest",
    )

    chart_col_i, info_col_i = st.columns([5, 2])

    with chart_col_i:
        st.plotly_chart(fig_i, use_container_width=True)
        st.caption("● YAML utterances · ◆ scenario utterances · ★ your query")

    with info_col_i:
        if classified:
            conf = classified.get("confidence", "?")
            conf_color = {"high": "#34d399", "medium": "#f59e0b", "low": "#f87171"}.get(conf, "#aaa")
            st.markdown(f"### Classification")
            st.markdown(
                f"**Domain:** `{classified.get('domain', '?')}`  \n"
                f"**Intent:** `{classified.get('intent', '?')}`  \n"
                f"**Confidence:** <span style='color:{conf_color}'>{conf}</span>",
                unsafe_allow_html=True,
            )
            st.divider()

        st.markdown("### Intent breakdown")
        domain_counts: dict[str, int] = {}
        for r in rows_i:
            domain_counts[r["domain"]] = domain_counts.get(r["domain"], 0) + 1

        for dom in unique_domains:
            hex_c = domain_color[dom]
            intent_labels = list(dict.fromkeys(
                r["intent"] for r in rows_i if r["domain"] == dom
            ))
            with st.expander(
                f"**{dom}** — {domain_counts[dom]} utterances",
                expanded=classified is not None and classified.get("domain") == dom,
            ):
                for il in intent_labels:
                    count = sum(1 for r in rows_i if r["domain"] == dom and r["intent"] == il)
                    is_match = classified and classified.get("intent") == il and classified.get("domain") == dom
                    prefix = "→ " if is_match else ""
                    weight = "**" if is_match else ""
                    st.markdown(
                        f"{prefix}{weight}`{il}`{weight} <small style='color:#888'>({count})</small>",
                        unsafe_allow_html=True,
                    )


# ---------------------------------------------------------------------------
# Files tab
# ---------------------------------------------------------------------------
with tab_files:
    st.subheader("Indexed source files")

    if st.button("Refresh", type="secondary"):
        st.rerun()

    try:
        resp = httpx.get(f"{API_BASE}/collections", timeout=10)
        resp.raise_for_status()
        files = resp.json()["files"]
    except Exception as e:
        st.error(f"Could not fetch files: {e}")
        files = []

    if not files:
        st.info("No files indexed yet. Upload one in the Upload tab.")
    else:
        for f in files:
            st.markdown(f"- `{f}`")
