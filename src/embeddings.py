"""
Text embedding backends: local (offline TF-IDF) and Gemini (live API).

Switch between them with ONE environment variable -- no code changes needed:

    EMBEDDING_BACKEND=local   (default) -- offline, no API key, no network
    EMBEDDING_BACKEND=gemini             -- Google Gemini gemini-embedding-001,
                                             requires GEMINI_API_KEY

Everything else in the pipeline (novelty.py, score.py) calls `get_embedder()`
and only ever uses the shared `fit(corpus)` / `embed(text)` interface, so
neither of them needs to know or care which backend is active.

Local backend design choice: instead of a heavyweight neural embedding model
(e.g. sentence-transformers, which pulls a multi-hundred-MB model + torch),
this uses a TF-IDF vectorizer over character n-grams (3-5 chars). This:

  - runs fully locally with no model download and no API key
  - is robust to the short, informal, sometimes-misspelled text typical of
    user-generated content
  - captures near-duplicate / paraphrase similarity well enough for novelty
    scoring at this scale (~50 pool items)

See README "Switching between local and Gemini embeddings" for setup steps.
"""

from __future__ import annotations

import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TextEmbedder:
    backend = "local"

    def __init__(self):
        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            sublinear_tf=True,
        )
        self._fitted = False

    def fit(self, corpus: list[str]) -> "TextEmbedder":
        """Fit the vectorizer's vocabulary on a corpus of reference texts.
        Must be called once before embed()."""
        self._vectorizer.fit(corpus)
        self._fitted = True
        return self

    def embed(self, text: str) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TextEmbedder.fit(corpus) must be called before embed().")
        vec = self._vectorizer.transform([text])
        return vec.toarray()[0]

    def embed_many(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TextEmbedder.fit(corpus) must be called before embed_many().")
        return self._vectorizer.transform(texts).toarray()


class GeminiEmbedder:
    """
    Live embedding backend using Google's Gemini API.

    Requires:
      1. `pip install google-genai`
      2. A free API key from https://aistudio.google.com/
      3. Set the `GEMINI_API_KEY` environment variable.

    Implements the same `fit(corpus)` / `embed(text)` interface as
    TextEmbedder so it's a drop-in replacement -- `fit()` is a no-op here
    since Gemini's model is already pre-trained and needs no local
    vocabulary, it's kept only for interface compatibility.
    """

    backend = "gemini"
    MODEL = "gemini-embedding-001"

    def __init__(self):
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:
            raise ImportError(
                "EMBEDDING_BACKEND=gemini requires the google-genai "
                'package. Install it with: python -m pip install "google-genai>=1.0"'
            ) from e

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "EMBEDDING_BACKEND=gemini requires GEMINI_API_KEY to be set. "
                "Get a free key at https://aistudio.google.com/ then run: "
                "set GEMINI_API_KEY=your-key-here (Windows cmd)"
            )

        self._client = genai.Client(api_key=api_key)
        self._types = types
        self._cache: dict[str, np.ndarray] = {}

    def fit(self, corpus: list[str]) -> "GeminiEmbedder":
        # No-op: Gemini's embedding model needs no local vocabulary fitting.
        # Kept so this class satisfies the same interface as TextEmbedder.
        return self

    def embed(self, text: str) -> np.ndarray:
        if text in self._cache:
            return self._cache[text]
        result = self._client.models.embed_content(
            model=self.MODEL,
            contents=text,
            config=self._types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
        )
        vec = np.array(result.embeddings[0].values, dtype=float)
        self._cache[text] = vec
        return vec

    def embed_many(self, texts: list[str]) -> np.ndarray:
        return np.array([self.embed(t) for t in texts])


def get_embedder(corpus: list[str] | None = None, backend: str | None = None):
    """
    Factory: returns the embedder to use, selected by the EMBEDDING_BACKEND
    environment variable (defaults to "local" if unset).

    `corpus` is only consumed by the local backend, which needs it to fit a
    TF-IDF vocabulary; the Gemini backend ignores it.
    """
    selected_backend = (
        os.environ.get("EMBEDDING_BACKEND", "local") if backend is None else backend
    )
    backend = selected_backend.strip().lower()

    if backend == "gemini":
        return GeminiEmbedder().fit(corpus or [])
    elif backend == "local":
        embedder = TextEmbedder()
        if corpus is not None:
            embedder.fit(corpus)
        return embedder
    else:
        raise ValueError(
            f"Unknown EMBEDDING_BACKEND={backend!r}. Use 'local' or 'gemini'."
        )


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors, safe against zero vectors."""
    a = a.reshape(1, -1)
    b = b.reshape(1, -1)
    if not np.any(a) or not np.any(b):
        return 0.0
    return float(cosine_similarity(a, b)[0][0])
