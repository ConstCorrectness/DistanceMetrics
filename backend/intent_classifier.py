import json
import os
from pathlib import Path
from pydantic import BaseModel
import yaml

from typing import cast


_client = None

def _get_openai_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


_INTENTS_PATH = Path(__file__).parent.parent / "intents.yaml"
_taxonomy_embeddings = None


def _get_taxonomy_embeddings() -> list[dict]:
    global _taxonomy_embeddings
    if _taxonomy_embeddings is None:
        from backend.embedder import embed_texts
        taxonomy = _load_taxonomy()
        items = []
        for domain, intents in taxonomy.items():
            for intent, data in intents.items():
                utterances = data.get("utterances", []) + data.get("aliases", [])
                for utt in utterances:
                    if utt.strip():
                        items.append({
                            "domain": domain,
                            "intent": intent,
                            "utterance": utt,
                        })
        if items:
            texts = [item["utterance"] for item in items]
            vectors = embed_texts(texts)
            for item, vec in zip(items, vectors):
                item["vector"] = vec
        _taxonomy_embeddings = items
    return _taxonomy_embeddings


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    import math
    dot_product = sum(x * y for x, y in zip(v1, v2))
    magnitude1 = math.sqrt(sum(x * x for x in v1))
    magnitude2 = math.sqrt(sum(y * y for y in v2))
    if not magnitude1 or not magnitude2:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)



structured_output = {
    "type": "object",
    "properties": {
        "domain": {"type": "string"},
        "intent": {"type": "string"},
        "confidence": {"type": "string", 'enum': ['low', 'medium', 'high']}
    },
    "required": ['domain', 'intent', 'confidence']
}

def _load_taxonomy() -> dict:
    with open(_INTENTS_PATH) as f:
        return yaml.safe_load(f)["intents"]


# TODO: specialize to particular bot package/service specialized handler:
def _build_specialized_handler(taxonomy: dict, handler_name: str) -> str:
    """
    
    """

    return ''


def _build_system_prompt(taxonomy: dict) -> str:
    lines = ["You are an intent classifier. Given a user message, return JSON with keys: domain, intent, confidence (high/medium/low)."]
    lines.append("\nKnown intents (domain → intent: example utterances):\n")
    for domain, intents in taxonomy.items():
        for intent, data in intents.items():
            utterances = data.get("utterances", [])
            examples = "; ".join(u for u in utterances[:2] if u)
            lines.append(f"  {domain}.{intent}: \"{examples}\"")
    lines.append('\nIf nothing matches, return {"domain": "unknown", "intent": "unknown", "confidence": "low"}.')
    lines.append("Respond with JSON only, no prose.")
    return "\n".join(lines)


def classify(utterance: str) -> dict:
    from backend.embedder import has_openai_key
    
    if has_openai_key():
        taxonomy = _load_taxonomy()
        system_prompt = _build_system_prompt(taxonomy)
        client = _get_openai_client()

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": utterance},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )

        return json.loads(cast(str, response.choices[0].message.content))
    else:
        # Similarity-based fallback
        from backend.embedder import embed_texts
        
        # Get query embedding
        query_vectors = embed_texts([utterance])
        if not query_vectors:
            return {"domain": "unknown", "intent": "unknown", "confidence": "low"}
        q_vec = query_vectors[0]
        
        # Get taxonomy embeddings
        tax_embeddings = _get_taxonomy_embeddings()
        if not tax_embeddings:
            return {"domain": "unknown", "intent": "unknown", "confidence": "low"}
            
        best_sim = -1.0
        best_item = None
        
        for item in tax_embeddings:
            sim = cosine_similarity(q_vec, item["vector"])
            if sim > best_sim:
                best_sim = sim
                best_item = item
                
        if best_item is None or best_sim < 0.35:
            return {"domain": "unknown", "intent": "unknown", "confidence": "low"}
            
        if best_sim >= 0.70:
            confidence = "high"
        elif best_sim >= 0.50:
            confidence = "medium"
        else:
            confidence = "low"
            
        return {
            "domain": best_item["domain"],
            "intent": best_item["intent"],
            "confidence": confidence
        }

