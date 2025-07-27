import numpy as np
from sentence_transformers import SentenceTransformer
from .config import ModelConfig
from .utils import debug, error, cache_embeddings
import hashlib

# Global model instance
_model = None

def get_model():
    """Get or initialize the embedding model"""
    global _model
    if _model is None:
        try:
            debug(f"Loading embedding model: {ModelConfig.EMBEDDING_MODEL}")
            _model = SentenceTransformer(ModelConfig.EMBEDDING_MODEL)
            _model.max_seq_length = ModelConfig.EMBEDDING_MAX_LENGTH
            debug("Model loaded successfully")
        except Exception as e:
            error("Failed to load embedding model", e)
            raise
    return _model

def embed_docs(docs: dict) -> dict:
    """Embed all document chunks"""
    model = get_model()
    
    for doc_name, doc_data in docs.items():
        if 'embeddings' in doc_data:
            debug(f"Embeddings already exist for {doc_name}")
            continue
            
        chunks = doc_data['chunks']
        debug(f"Embedding {len(chunks)} chunks from {doc_name}")
        
        try:
            # Create cache key
            cache_key = hashlib.md5(str(chunks).encode()).hexdigest()
            
            # Check cache
            if cache_key in cache_embeddings:
                embeddings = cache_embeddings[cache_key]
                debug("Loaded embeddings from cache")
            else:
                # Generate embeddings
                embeddings = model.encode(
                    chunks,
                    batch_size=ModelConfig.EMBEDDING_BATCH_SIZE,
                    normalize_embeddings=True,
                    show_progress_bar=True,
                    convert_to_numpy=True
                )
                cache_embeddings[cache_key] = embeddings
                debug("Generated and cached embeddings")
            
            doc_data['embeddings'] = embeddings
            
            # Update chunk metadata with embeddings
            for meta, emb in zip(doc_data['chunk_metadata'], embeddings):
                meta['embedding'] = emb
                
        except Exception as e:
            error(f"Failed to embed chunks for {doc_name}", e)
            # Continue with empty embeddings to avoid complete failure
            doc_data['embeddings'] = np.zeros((len(chunks), ModelConfig.EMBEDDING_DIMENSION))
    
    return docs

def embed_query(persona: str, job: str, keywords: list) -> np.ndarray:
    """Create query embedding for persona-aware retrieval"""
    model = get_model()
    
    # Enhanced query construction
    query_components = [
        f"User role: {persona}",
        f"Task objective: {job}",
        f"Key focus areas: {', '.join(keywords)}",
        f"Context: document analysis and information retrieval"
    ]
    
    query_text = ". ".join(query_components)
    debug(f"Query constructed: {query_text[:100]}...")
    
    try:
        query_embedding = model.encode(
            [query_text],
            normalize_embeddings=True,
            convert_to_numpy=True
        )[0]
        debug("Query embedding generated successfully")
        return query_embedding
    except Exception as e:
        error("Failed to generate query embedding", e)
        # Return zero vector as fallback
        return np.zeros(ModelConfig.EMBEDDING_DIMENSION)