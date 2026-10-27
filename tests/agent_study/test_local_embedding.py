"""The in-process embedding binding needs no API key."""

import asyncio

from deeptutor.agent_study.embed import HashingEmbedder
from deeptutor.services.config.provider_runtime import EMBEDDING_PROVIDERS
from deeptutor.services.embedding.adapters.base import EmbeddingRequest
from deeptutor.services.embedding.adapters.local import LocalEmbeddingAdapter


def test_hashing_ranks_shared_terms_above_unrelated_text():
    embedder = HashingEmbedder()
    query = embedder.embed_query("insulation resistance megaohm per kilovolt")
    related = embedder.embed_documents(
        ["IEEE Std 43 insulation resistance is one megaohm per kilovolt."]
    )[0]
    unrelated = embedder.embed_documents(["WACC is the weighted average cost of capital."])[0]

    def dot(left, right):
        return sum(a * b for a, b in zip(left, right))

    assert dot(query, related) > dot(query, unrelated)


def test_local_provider_is_keyless_and_hashes_in_process():
    spec = EMBEDDING_PROVIDERS["local"]
    assert spec.mode == "local"
    assert spec.is_local is True
    assert spec.adapter == "local"
    assert spec.default_model == "hashing"

    adapter = LocalEmbeddingAdapter(
        {"model": "hashing", "api_key": "", "base_url": "local://embeddings"}
    )
    response = asyncio.run(
        adapter.embed(
            EmbeddingRequest(
                texts=["creepage distance"],
                model="hashing",
                input_type="search_query",
            )
        )
    )
    assert len(response.embeddings) == 1
    assert len(response.embeddings[0]) == 384
    assert response.usage["provider"] == "local"
