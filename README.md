---
title: DistanceMetrics
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# DistanceMetrics

Semantic search over structured data with live vector-space visualization.

Upload a CSV/XLSX/JSON file, embed it via OpenAI `text-embedding-3-small`, store in Qdrant, then search and watch your query move through the embedding space in real time.

## Stack

- **Backend** — FastAPI + OpenAI Embeddings + Qdrant
- **Frontend** — Streamlit with interactive Plotly PCA scatter

## Required secrets

Set these in the Space **Settings → Variables and secrets**:

| Secret | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key |
| `QDRANT_URL` | Qdrant cluster URL |
| `QDRANT_API_KEY` | Qdrant API key |
| `QDRANT_COLLECTION` | Collection name (default: `documents`) |
