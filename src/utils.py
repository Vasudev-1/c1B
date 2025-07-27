import time, logging, traceback, pickle, hashlib
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Cache directory from config
CACHE_DIR = "/tmp/hackathon_cache"

def log_time() -> float:
    return time.time()

def debug(msg: str):
    logger.debug(msg)

def debug_print(msg: str):
    """Alternative debug function for compatibility"""
    logger.debug(msg)

def info(msg: str):
    logger.info(msg)

def error(msg: str, exc: Optional[Exception] = None):
    logger.error(msg)
    if exc:
        logger.error(f"Exception details: {str(exc)}")
        logger.error(traceback.format_exc())

class EmbeddingCache:
    def __init__(self, cache_dir: str = CACHE_DIR):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        debug(f"Cache directory: {self.dir}")

    def _path(self, key: str) -> Path:
        h = hashlib.md5(key.encode()).hexdigest()
        return self.dir / f"{h}.pkl"

    def get(self, key: str) -> Optional[Any]:
        p = self._path(key)
        if p.exists():
            try:
                return pickle.loads(p.read_bytes())
            except Exception as e:
                error(f"Failed reading cache {p}", e)
        return None

    def set(self, key: str, value: Any):
        p = self._path(key)
        try:
            p.write_bytes(pickle.dumps(value))
        except Exception as e:
            error(f"Failed writing cache {p}", e)

# Global cache instance
cache = EmbeddingCache()
cache_embeddings = {}  # In-memory cache for embeddings

def validate_input(data: dict) -> bool:
    """Validate input JSON structure"""
    required = ["persona", "job_to_be_done", "documents"]
    for f in required:
        if f not in data:
            error(f"Missing field in input JSON: {f}")
            return False
    
    if not isinstance(data["documents"], list):
        error("'documents' must be a list")
        return False
    
    for doc in data["documents"]:
        if "filename" not in doc:
            error("Each document must include 'filename'")
            return False
    
    return True