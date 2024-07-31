import hashlib
import diskcache as dc


class CachedEngine:
    def __init__(self, cache_path: str):
        super().__init__()
        self.cache_path = cache_path
        self.cache = dc.Cache(cache_path)

    def _hash_prompt(self, prompt: str):
        return hashlib.sha256(f"{prompt}".encode()).hexdigest()

    def _check_cache(self, prompt: str):
        hash_key = self._hash_prompt(prompt)
        if hash_key in self.cache:
            return self.cache[hash_key]
        else:
            return None

    def _save_cache(self, prompt: str, response: str):
        hash_key = self._hash_prompt(prompt)
        self.cache[hash_key] = response

    def __getstate__(self):
        # Remove the cache from the state before pickling
        state = self.__dict__.copy()
        del state["cache"]
        return state

    def __setstate__(self, state):
        # Restore the cache after unpickling
        self.__dict__.update(state)
        self.cache = dc.Cache(self.cache_path)
