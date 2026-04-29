import os
from openai import OpenAI

_client: OpenAI | None = None
EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 50


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def embed_texts(texts: list[str]) -> list[list[float]]:
    client = _get_client()
    vectors = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vectors.extend([item.embedding for item in response.data])
    return vectors
