import hashlib, numpy as np
from sentence_transformers import SentenceTransformer
from .config import ModelConfig, OptimizationConfig
from .quantization import quantize_model
from .utils import cache, debug, error  # Add error import

_model = None

def get_model():
    global _model
    if _model is None:
        try:
            _model = SentenceTransformer(
                ModelConfig.EMBEDDING_MODEL, 
                trust_remote_code=True  # ✅ ADD THIS LINE
            )
            _model.max_seq_length = ModelConfig.EMBEDDING_MAX_LENGTH  # Fixed: _model not *model
            if OptimizationConfig.ENABLE_QUANTIZATION:
                _model = quantize_model(_model, OptimizationConfig.QUANTIZATION_BITS)  # Fixed: _model not *model
            debug(f"Loaded model {ModelConfig.EMBEDDING_MODEL}")
        except Exception as e:
            error("Failed to load embedding model", e)
            raise
    return _model

# Rest of the file remains the same...
def encode_texts(texts, show_progress=False) -> np.ndarray:
    key = hashlib.md5("".join(texts).encode()).hexdigest()
    cached = cache.get(key)
    if cached is not None:
        return cached
    model = get_model()
    embs = model.encode(texts,
                        batch_size=ModelConfig.EMBEDDING_BATCH_SIZE,
                        normalize_embeddings=True,
                        show_progress_bar=show_progress,
                        convert_to_numpy=True)
    cache.set(key, embs)
    return embs

def embed_docs(docs: dict) -> dict:
    for d in docs.values():
        embs = encode_texts(d["chunks"])
        d["embeddings"] = embs
        for m,e in zip(d["chunk_metadata"], embs):
            m["embedding"] = e
    return docs

def embed_query(persona: str, job: str, keywords: list) -> np.ndarray:
    q = f"Persona: {persona}. Task: {job}. Keywords: {', '.join(keywords)}"
    debug(f"Embedding query: {q[:80]}...")
    return encode_texts([q])[0]