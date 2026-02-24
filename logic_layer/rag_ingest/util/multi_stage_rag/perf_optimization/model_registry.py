# model_registry.py
# Centralized cache for SentenceTransformer models.
# Ensures each model_name is loaded exactly once per process lifetime.

from sentence_transformers import SentenceTransformer


class ModelRegistry:
    _models: dict[str, SentenceTransformer] = {}

    @classmethod
    def get(cls, model_name: str) -> SentenceTransformer:
        """
        Returns the cached SentenceTransformer for model_name.
        Loads it on first call, reuses on subsequent calls.
        """
        if model_name not in cls._models:
            cls._models[model_name] = SentenceTransformer(model_name)
        return cls._models[model_name]

    @classmethod
    def loaded(cls) -> list[str]:
        """Returns list of currently loaded model names."""
        return list(cls._models.keys())

    @classmethod
    def clear(cls) -> None:
        """
        Drops all model references.
        Only call this if you explicitly want to free memory
        and accept the reload cost on next get().
        """
        cls._models.clear()