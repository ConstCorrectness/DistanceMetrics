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

Supports **dual embedding modes** depending on the presence of `OPENAI_API_KEY`:
- **OpenAI Mode** (Default): Uses OpenAI `text-embedding-3-small` (1536 dimensions) for premium cloud-based embeddings and GPT-4o-mini for chat intent classification.
- **Local Fallback Mode**: Automatically activates if no `OPENAI_API_KEY` is provided. Uses `sentence-transformers/all-MiniLM-L12-v2` (384 dimensions) running locally, and classifies intents via cosine similarity.

Qdrant collections are segregated dynamically (e.g. `documents_1536` vs `documents_384`) to prevent vector shape conflicts.

## Stack

- **Backend** — FastAPI + OpenAI Embeddings OR local Hugging Face all-MiniLM + Qdrant
- **Frontend** — Streamlit with interactive Plotly PCA scatter (compatible with any embedding size)

## Required secrets

Set these in the Space **Settings → Variables and secrets**:

| Secret | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key |
| `QDRANT_URL` | Qdrant cluster URL |
| `QDRANT_API_KEY` | Qdrant API key |
| `QDRANT_COLLECTION` | Collection name (default: `documents`) |
