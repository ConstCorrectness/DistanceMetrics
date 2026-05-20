import json
import os
from pathlib import Path
from pydantic import BaseModel
import yaml

from typing import cast


from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

_INTENTS_PATH = Path(__file__).parent.parent / "intents.yaml"


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
    taxonomy = _load_taxonomy()
    system_prompt = _build_system_prompt(taxonomy)

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
