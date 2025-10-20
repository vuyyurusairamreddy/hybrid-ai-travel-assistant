import os
import hashlib
from typing import List, Optional
from diskcache import Cache
import config

class EmbeddingCache:
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or config.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache = Cache(self.cache_dir)

    def _key(self, text: str, model_name: str) -> str:
        return hashlib.md5(f"{model_name}:{text}".encode("utf-8")).hexdigest()

    def get(self, text: str, model_name: str) -> Optional[List[float]]:
        return self.cache.get(self._key(text, model_name))

    def set(self, text: str, model_name: str, embedding: List[float]) -> None:
        self.cache.set(self._key(text, model_name), embedding)

    def stats(self) -> dict:
        return {"items": len(self.cache), "size_bytes": self.cache.volume()}

    def clear(self) -> None:
        self.cache.clear()
