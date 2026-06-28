import os
from openai import OpenAI

_client: OpenAI | None = None
EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 50

# Local model configuration
LOCAL_MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"
_tokenizer = None
_model = None


def has_openai_key() -> bool:
    key = os.environ.get("OPENAI_API_KEY")
    return bool(key and key.strip())


def get_embedding_dimension() -> int:
    if has_openai_key():
        return 1536
    else:
        return 384


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def _get_local_model():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        import torch
        from transformers import AutoTokenizer, AutoModel

        print(f"Loading local embedding model '{LOCAL_MODEL_NAME}'...")
        print("Note: If this is the first run, the model files will be downloaded (approx. 120MB).")
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load tokenizer and model
        _tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_NAME)
        _model = AutoModel.from_pretrained(LOCAL_MODEL_NAME).to(device)
        print("Local embedding model loaded successfully.")
    return _tokenizer, _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    if has_openai_key():
        client = _get_client()
        vectors = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            response = client.embeddings.create(model=EMBED_MODEL, input=batch)
            vectors.extend([item.embedding for item in response.data])
        return vectors
    else:
        import torch
        tokenizer, model = _get_local_model()
        device = model.device
        
        vectors = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            
            # Tokenize and process batch
            encoded_input = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)
            
            with torch.no_grad():
                model_output = model(**encoded_input)
            
            # Perform mean pooling
            token_embeddings = model_output[0]
            attention_mask = encoded_input["attention_mask"]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            batch_vectors = (sum_embeddings / sum_mask).cpu().tolist()
            
            vectors.extend(batch_vectors)
        return vectors

