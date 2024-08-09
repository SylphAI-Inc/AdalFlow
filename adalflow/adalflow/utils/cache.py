import hashlib
import diskcache as dc


def hash_text(text: str):
    return hashlib.sha256(f"{text}".encode()).hexdigest()


def hash_text_sha1(text: str):  # 160 bits
    return hashlib.sha1(text.encode()).hexdigest()


def direct(text: str):
    return text


class CachedEngine:
    def __init__(self, cache_path: str):
        super().__init__()
        self.cache_path = cache_path
        self.cache = dc.Cache(cache_path)

    def _check_cache(self, prompt: str):
        hash_key = hash_text(prompt)
        if hash_key in self.cache:
            return self.cache[hash_key]
        else:
            return None

    def _save_cache(self, prompt: str, response: str):
        hash_key = hash_text(prompt)
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
