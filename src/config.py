import os
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class ModelConfig:
    # Best embedding model under 1GB - GTE Multilingual Base
    EMBEDDING_MODEL = "Alibaba-NLP/gte-multilingual-base"  # 305MB, SOTA performance
    EMBEDDING_DIMENSION = 768
    EMBEDDING_MAX_LENGTH = 8192  # Long context support
    EMBEDDING_BATCH_SIZE = 32
    
    # Text processing parameters
    CHUNK_SIZE = 200  # words per chunk
    CHUNK_STRIDE = 50  # overlap between chunks
    
    # Retrieval parameters
    RETRIEVAL_TOP_K = 50  # initial candidates
    FINAL_SECTIONS = 10   # final sections to return
    TOP_SENTENCES = 5     # sentences per refined text
    
    # Scoring weights
    SIMILARITY_WEIGHT = 0.7
    HEADING_BOOST_WEIGHT = 0.3
    
    # Alternative models for comparison
    ALTERNATIVE_MODELS = {
        "gte-multilingual": "Alibaba-NLP/gte-multilingual-base",  # 305MB, best choice
        "mixedbread-large": "mixedbread-ai/mxbai-embed-large-v1",  # 330MB, excellent
        "bge-base-en": "BAAI/bge-base-en-v1.5",  # 109MB, good baseline
        "nomic-embed": "nomic-ai/nomic-embed-text-v1.5",  # 138MB, code-friendly
    }

@dataclass 
class OptimizationConfig:
    # Quantization settings
    ENABLE_QUANTIZATION = True
    QUANTIZATION_BITS = 8  # 8-bit quantization for good balance
    
    # Knowledge distillation
    ENABLE_DISTILLATION = True
    TEACHER_MODEL = "sentence-transformers/all-mpnet-base-v2"  # High quality teacher
    DISTILLATION_TEMPERATURE = 4.0
    DISTILLATION_ALPHA = 0.7
    
    # Fine-tuning
    ENABLE_FINE_TUNING = True
    LEARNING_RATE = 2e-5
    EPOCHS = 3
    BATCH_SIZE = 16

@dataclass
class EvaluationConfig:
    # Evaluation metrics
    METRICS = ["map", "recall_at_k", "precision_at_k", "spearman_correlation", "f1_score"]
    RECALL_K_VALUES = [1, 3, 5, 10]
    
    # Test splits
    TRAIN_SPLIT = 0.7
    VALIDATION_SPLIT = 0.15
    TEST_SPLIT = 0.15
    
    # Evaluation datasets
    BENCHMARK_DATASETS = ["sts-b", "sick-r", "trec-car"]

# Runtime configuration
MAX_PROCESSING_TIME = 55  # seconds
CACHE_DIR = "/tmp/hackathon_cache"
DEBUG_MODE = os.environ.get("DEBUG", "false").lower() == "true"
USE_GPU = True if os.environ.get("CUDA_VISIBLE_DEVICES") else False

def setup_logging(debug: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if debug or DEBUG_MODE else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )
    
    # Suppress verbose third-party logs
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured - Debug: {debug or DEBUG_MODE}")