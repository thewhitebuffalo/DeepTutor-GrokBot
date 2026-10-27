"""Grade a learner attempt against an assistant-authored answer.

When the caller supplies ``correct`` or ``score``, that judgment is kept.
Otherwise the stored answer is compared locally: normalized equality, then
token overlap. No model is called.
"""

from __future__ import annotations

from collections import Counter
import re

_NON_WORD = re.compile(r"[^a-z0-9\s]+")
_SPACES = re.compile(r"\s+")
_PASSING_SCORE = 0.8


def normalize_answer(text: str) -> str:
    folded = _NON_WORD.sub(" ", str(text or "").casefold())
    return _SPACES.sub(" ", folded).strip()


def token_f1(expected: str, given: str) -> float:
    expected_tokens = normalize_answer(expected).split()
    given_tokens = normalize_answer(given).split()
    if not expected_tokens or not given_tokens:
        return 0.0
    expected_counts = Counter(expected_tokens)
    given_counts = Counter(given_tokens)
    overlap = sum(min(expected_counts[token], given_counts[token]) for token in expected_counts)
    if overlap == 0:
        return 0.0
    precision = overlap / len(given_tokens)
    recall = overlap / len(expected_tokens)
    return (2 * precision * recall) / (precision + recall)


def grade_attempt(
    expected: str,
    given: str,
    *,
    correct: bool | None = None,
    score: float | None = None,
) -> tuple[bool, float, str]:
    """Return ``(correct, score, graded_by)``.

    ``graded_by`` is ``agent`` when the caller judged the attempt and
    ``exact`` or ``overlap`` when this module did.
    """

    if correct is not None or score is not None:
        if score is None:
            judged = 1.0 if correct else 0.0
        else:
            judged = float(score)
        if judged < 0.0 or judged > 1.0:
            raise ValueError("score must be between 0 and 1")
        passed = bool(correct) if correct is not None else judged >= _PASSING_SCORE
        return passed, judged, "agent"
    if not str(given or "").strip():
        return False, 0.0, "exact"
    if normalize_answer(expected) == normalize_answer(given):
        return True, 1.0, "exact"
    overlap = token_f1(expected, given)
    expected_norm = normalize_answer(expected)
    given_norm = normalize_answer(given)
    if len(expected_norm) >= 8 and (expected_norm in given_norm or given_norm in expected_norm):
        overlap = max(overlap, 0.85)
    return overlap >= _PASSING_SCORE, round(overlap, 4), "overlap"
