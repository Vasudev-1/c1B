import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
import pickle
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

from .config import ModelConfig, OptimizationConfig
from .quantization import quantize_model
from .utils import debug_print, cache_embeddings

class AdvancedEmbeddingModel:
    """Advanced embedding model with quantization and optimization"""
    
    def __init__(self, model_name: str = ModelConfig.EMBEDDING_MODEL, 
                 enable_quantization: bool = OptimizationConfig.ENABLE_QUANTIZATION):
        self.model_name = model_name
        self.enable_quantization = enable_quantization
        self.model = None
        self.tokenizer = None
        self._load_model()
        
    def _load_model(self):
        """Load and optionally quantize the embedding model"""
        try:
            # Load the base model
            self.model = SentenceTransformer(self.model_name)
            self.model.max_seq_length = ModelConfig.EMBEDDING_MAX_LENGTH
            
            # Apply quantization if enabled
            if self.enable_quantization:
                self.model = quantize_model(self.model, bits=OptimizationConfig.QUANTIZATION_BITS)
                debug_print(f"Applied {OptimizationConfig.QUANTIZATION_BITS}-bit quantization")
                
            # Move to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                debug_print("Model moved to GPU")
                
            debug_print(f"Loaded model: {self.model_name}")
            
        except Exception as e:
            logging.error(f"Error loading model {self.model_name}: {e}")
            raise
    
    def encode(self, texts: List[str], normalize: bool = True, 
               show_progress: bool = False) -> np.ndarray:
        """Encode texts to embeddings with caching"""
        
        # Check cache first
        cache_key = hashlib.md5(str(texts).encode()).hexdigest()
        cached_embeddings = cache_embeddings.get(cache_key)
        if cached_embeddings is not None:
            debug_print("Loaded embeddings from cache")
            return cached_embeddings
        
        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=ModelConfig.EMBEDDING_BATCH_SIZE,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        # Cache for future use
        cache_embeddings[cache_key] = embeddings
        
        return embeddings
    
    def encode_query(self, persona: str, job: str, keywords: List[str]) -> np.ndarray:
        """Create optimized query embedding for persona-aware retrieval"""
        
        # Enhanced query construction for multilingual support
        query_components = [
            f"User role: {persona}",
            f"Task objective: {job}",
            f"Key focus areas: {', '.join(keywords)}",
            f"Context: document analysis and information retrieval"
        ]
        
        query_text = ". ".join(query_components)
        debug_print(f"Query constructed: {query_text[:100]}...")
        
        return self.encode([query_text])[0]
    
    def get_model_info(self) -> Dict:
        """Get detailed model information"""
        return {
            "model_name": self.model_name,
            "embedding_dimension": ModelConfig.EMBEDDING_DIMENSION,
            "max_sequence_length": ModelConfig.EMBEDDING_MAX_LENGTH,
            "quantized": self.enable_quantization,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "model_size_mb": self._estimate_model_size()
        }
    
    def _estimate_model_size(self) -> float:
        """Estimate model size in MB"""
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        return (param_size + buffer_size) / (1024 * 1024)

def embed_documents(docs: Dict[str, Dict], model: AdvancedEmbeddingModel) -> Dict[str, Dict]:
    """Embed all document chunks with advanced caching and optimization"""
    
    for doc_name, doc_data in docs.items():
        if 'embeddings' in doc_data:
            debug_print(f"Embeddings already exist for {doc_name}")
            continue
            
        debug_print(f"Embedding {len(doc_data['chunks'])} chunks from {doc_name}")
        
        # Generate embeddings in batches
        embeddings = model.encode(doc_data['chunks'], show_progress=True)
        doc_data['embeddings'] = embeddings
        
        # Update metadata with embeddings
        for meta, emb in zip(doc_data['chunk_metadata'], embeddings):
            meta['embedding'] = emb
            
        debug_print(f"Completed embedding for {doc_name}")
    
    return docs
