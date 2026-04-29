import os
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

VECTOR_SIZE = 1536  # text-embedding-3-small dimension

_client: QdrantClient | None = None


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(
            url=os.environ["QDRANT_URL"],
            api_key=os.environ["QDRANT_API_KEY"],
        )
    return _client


def _collection() -> str:
    return os.environ.get("QDRANT_COLLECTION", "documents")


def ensure_collection() -> None:
    client = _get_client()
    col = _collection()
    existing = [c.name for c in client.get_collections().collections]
    if col not in existing:
        client.create_collection(
            collection_name=col,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


def upsert_points(
    vectors: list[list[float]],
    payloads: list[dict],
    source_file: str,
) -> None:
    client = _get_client()
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload={**payload, "source_file": source_file},
        )
        for vec, payload in zip(vectors, payloads)
    ]
    client.upsert(collection_name=_collection(), points=points)


def search(query_vector: list[float], top_k: int = 10) -> list[dict]:
    client = _get_client()
    results = client.search(
        collection_name=_collection(),
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
    )
    return [
        {"score": round(hit.score, 4), **hit.payload}
        for hit in results
    ]


def list_source_files() -> list[str]:
    client = _get_client()
    seen: set[str] = set()
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=_collection(),
            with_payload=["source_file"],
            limit=256,
            offset=offset,
        )
        for r in records:
            if r.payload and "source_file" in r.payload:
                seen.add(r.payload["source_file"])
        if offset is None:
            break
    return sorted(seen)
