"""Local embedders that never call a paid embedding API.

``hash`` is a deterministic feature-hash embedder. It needs no model download
and is the default so a fresh machine can index and retrieve with zero network
access. ``sentence-transformers`` runs a CPU model such as
``BAAI/bge-small-en-v1.5`` when that extra is installed.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import threading
from typing import Protocol

from deeptutor.agent_study.errors import StudyError

_TOKEN = re.compile(r"[a-z0-9]+")
_HASH_MODELS = frozenset({"hash", "hashing", "local-hash", "deeptutor-hash"})
_DEFAULT_ST_MODEL = "BAAI/bge-small-en-v1.5"
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_DEFAULT_DIM = 384


class Embedder(Protocol):
    backend: str
    model: str

    @property
    def dim(self) -> int: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def sentence_transformers_available() -> bool:
    """Return whether the optional sentence-transformers package imports."""

    try:
        import sentence_transformers  # noqa: F401
    except Exception:
        return False
    return True


def resolve_selection() -> tuple[str, str]:
    """Return ``(backend, model)`` for a newly created knowledge base.

    ``DEEPTUTOR_EMBED_BACKEND`` is ``auto`` (default), ``hash``, or
    ``sentence-transformers``. ``auto`` uses sentence-transformers when it is
    installed and the hashing embedder otherwise.
    """

    choice = os.environ.get("DEEPTUTOR_EMBED_BACKEND", "auto").strip().lower()
    model = os.environ.get("DEEPTUTOR_EMBED_MODEL", "").strip()
    if choice in {"hash", "hashing", "local-hash"}:
        return "hash", model or "hashing"
    if choice in {"sentence-transformers", "sentence_transformers", "st", "bge"}:
        return "sentence-transformers", model or _DEFAULT_ST_MODEL
    if sentence_transformers_available():
        return "sentence-transformers", model or _DEFAULT_ST_MODEL
    return "hash", "hashing"


def embedder_for(backend: str, model: str, *, dim: int | None = None) -> Embedder:
    """Build the embedder recorded on a knowledge base."""

    if backend == "hash":
        return HashingEmbedder(dim=dim or _hash_dim(), model=model or "hashing")
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder(model or _DEFAULT_ST_MODEL)
    raise StudyError(f"Unknown embed backend {backend!r}.", code="embed_backend")


def embedder_for_model(model: str | None) -> Embedder:
    """Select an embedder from a model id used by the ``local`` binding."""

    name = (model or "hashing").strip()
    if name.casefold() in _HASH_MODELS:
        return HashingEmbedder(model="hashing")
    return SentenceTransformerEmbedder(name)


def _hash_dim() -> int:
    raw = os.environ.get("DEEPTUTOR_EMBED_DIM", "").strip()
    if not raw:
        return _DEFAULT_DIM
    try:
        dim = int(raw)
    except ValueError as exc:
        raise StudyError("DEEPTUTOR_EMBED_DIM must be an integer.") from exc
    if not 32 <= dim <= 4096:
        raise StudyError("DEEPTUTOR_EMBED_DIM must be between 32 and 4096.")
    return dim


def _l2_normalize(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        return values
    return [value / norm for value in values]


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.casefold())


class HashingEmbedder:
    """Signed feature hashing over word unigrams and adjacent bigrams."""

    backend = "hash"

    def __init__(self, *, dim: int = _DEFAULT_DIM, model: str = "hashing") -> None:
        if dim < 8:
            raise StudyError("Hashing embedder dimension is too small.")
        self.model = model
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dim
        tokens = _tokens(text)
        if not tokens:
            vector[0] = 1.0
            return vector
        features = list(tokens)
        features.extend(f"{left}_{right}" for left, right in zip(tokens, tokens[1:]))
        for feature in features:
            digest = hashlib.sha256(feature.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self._dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        return _l2_normalize(vector)


class SentenceTransformerEmbedder:
    """CPU sentence-transformers embedder, loaded on first use."""

    backend = "sentence-transformers"

    def __init__(self, model: str) -> None:
        self.model = model
        self._model = None
        self._dim: int | None = None
        self._lock = threading.Lock()

    @property
    def dim(self) -> int:
        self._load()
        assert self._dim is not None
        return self._dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        encoded = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [[float(value) for value in row] for row in encoded]

    def embed_query(self, text: str) -> list[float]:
        name = self.model.casefold()
        prefixed = text
        if "bge" in name and "bge-m3" not in name:
            prefixed = _BGE_QUERY_PREFIX + text
        return self.embed_documents([prefixed])[0]

    def _load(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                if not sentence_transformers_available():
                    raise StudyError(
                        "sentence-transformers is not installed. "
                        "Install it with `pip install 'deeptutor[local-embed]'` "
                        "or create knowledge bases with DEEPTUTOR_EMBED_BACKEND=hash.",
                        code="embed_missing",
                    )
                from sentence_transformers import SentenceTransformer

                try:
                    self._model = SentenceTransformer(self.model, device="cpu")
                except Exception as exc:
                    raise StudyError(
                        f"Could not load local embedding model {self.model!r}: {exc}",
                        code="embed_model",
                    ) from exc
                self._dim = int(self._model.get_sentence_embedding_dimension())
        return self._model
