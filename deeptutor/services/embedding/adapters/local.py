"""In-process embeddings for the ``local`` binding.

No HTTP call and no API key. Model id ``hashing`` uses the deterministic
feature-hash embedder. Any other model id is loaded with sentence-transformers
on CPU (``pip install 'deeptutor[local-embed]'``).
"""

from __future__ import annotations

from typing import Any

from deeptutor.agent_study.embed import embedder_for_model

from .base import BaseEmbeddingAdapter, EmbeddingRequest, EmbeddingResponse


class LocalEmbeddingAdapter(BaseEmbeddingAdapter):
    """Text embeddings computed in this process."""

    SUPPORTS_INPUT_TYPE = True

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        if request.contents:
            raise ValueError(
                "The local embedding adapter is text-only. It does not accept multimodal contents."
            )
        embedder = embedder_for_model(request.model or self.model)
        texts = list(request.texts)
        if request.input_type == "search_query":
            vectors = [embedder.embed_query(text) for text in texts]
        else:
            vectors = embedder.embed_documents(texts)
        dim = len(vectors[0]) if vectors else embedder.dim
        return EmbeddingResponse(
            embeddings=vectors,
            model=embedder.model,
            dimensions=dim,
            usage={"provider": "local", "backend": embedder.backend},
        )

    def get_model_info(self) -> dict[str, Any]:
        return {
            "name": self.model or "hashing",
            "provider": "local",
            "multimodal": False,
        }
