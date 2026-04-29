import time
import httpx
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="Semantic Search", page_icon="🔍", layout="wide")
st.title("🔍 Semantic Search")

tab_upload, tab_search, tab_files = st.tabs(["Upload", "Search", "Indexed Files"])


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
