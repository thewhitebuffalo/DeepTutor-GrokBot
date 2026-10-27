"""Headless study backend for an external assistant.

The assistant (for example Grok Bot) teaches and writes questions. This package
stores the learner's documents, question bank, attempts, spaced-repetition
state, memory notes, and task board. Indexing uses a local embedder and does
not call a cloud LLM.

Importing a submodule such as ``deeptutor.agent_study.embed`` does not load the
database. ``StudyService`` is resolved on first attribute access.

See ``GROKBOT.md`` at the repository root.
"""

from __future__ import annotations

__all__ = ["StudyError", "StudyService"]


def __getattr__(name: str):
    if name == "StudyError":
        from deeptutor.agent_study.errors import StudyError

        return StudyError
    if name == "StudyService":
        from deeptutor.agent_study.service import StudyService

        return StudyService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
