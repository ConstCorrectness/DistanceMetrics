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


def _collection(name: str | None = None) -> str:
    if name:
        return name
    return os.environ.get("QDRANT_COLLECTION", "documents")


def ensure_collection(name: str | None = None) -> None:
    client = _get_client()
    col = _collection(name)
    try:
        existing = [c.name for c in client.get_collections().collections]
    except Exception as e:
        url = os.environ.get("QDRANT_URL", "unknown")
        print(f"Error connecting to Qdrant at {url}: {e}")
        if "404" in str(e):
            print("Hint: A 404 error often means the QDRANT_URL is pointing to a path that doesn't exist or a proxy that doesn't recognize the request. If you are using Hugging Face Spaces, ensure the URL is the direct space URL (e.g., https://user-name.hf.space) and that the Qdrant service is running and reachable.")
        raise e

    if col not in existing:
        client.create_collection(
            collection_name=col,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


def upsert_points(
    vectors: list[list[float]],
    payloads: list[dict],
    source_file: str,
    collection: str | None = None,
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
    client.upsert(collection_name=_collection(collection), points=points)


def search(query_vector: list[float], top_k: int = 10, collection: str | None = None) -> list[dict]:
    client = _get_client()
    results = client.search(
        collection_name=_collection(collection),
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
    )
    return [
        {"score": round(hit.score, 4), **hit.payload}
        for hit in results
    ]


def get_all_vectors(collection: str | None = None) -> list[dict]:
    client = _get_client()
    results = []
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=_collection(collection),
            with_vectors=True,
            with_payload=True,
            limit=256,
            offset=offset,
        )
        for r in records:
            if r.vector is not None:
                results.append({"vector": r.vector, "payload": r.payload or {}})
        if offset is None:
            break
    return results


def clear_collection(name: str | None = None) -> None:
    client = _get_client()
    col = _collection(name)
    try:
        client.delete_collection(col)
    except:
        pass
    ensure_collection(col)


def list_source_files(collection: str | None = None) -> list[str]:
    client = _get_client()
    seen: set[str] = set()
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=_collection(collection),
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
