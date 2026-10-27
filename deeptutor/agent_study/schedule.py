"""SM-2 style scheduling for a study topic.

A failing attempt stays due immediately so the assistant can quiz it again in
the same session. A passing attempt follows the classic 1 day, 6 day, then
``interval * ease`` ladder. This is intentionally smaller than the upstream
mastery-path scheduler, which still drives the in-app tutor.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TypedDict


class TopicState(TypedDict):
    ease: float
    interval_days: float
    repetitions: int
    lapses: int
    mastery: float
    next_review_at: str
    last_review_at: str | None
    attempt_count: int
    correct_count: int


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.isoformat()


def fresh_topic(now: datetime | None = None) -> TopicState:
    moment = now or utcnow()
    return {
        "ease": 2.5,
        "interval_days": 0.0,
        "repetitions": 0,
        "lapses": 0,
        "mastery": 0.0,
        "next_review_at": _iso(moment),
        "last_review_at": None,
        "attempt_count": 0,
        "correct_count": 0,
    }


def quality_from_score(score: float, correct: bool) -> int:
    if not correct:
        if score >= 0.4:
            return 2
        if score > 0:
            return 1
        return 0
    if score >= 0.95:
        return 5
    if score >= 0.85:
        return 4
    return 3


def apply_review(
    state: TopicState,
    *,
    score: float,
    correct: bool,
    now: datetime | None = None,
) -> TopicState:
    """Return the next topic state after one graded attempt."""

    moment = now or utcnow()
    quality = quality_from_score(score, correct)
    ease = float(state["ease"]) + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    ease = max(1.3, ease)
    repetitions = int(state["repetitions"])
    lapses = int(state["lapses"])
    if quality < 3:
        repetitions = 0
        interval = 0.0
        lapses += 1
    elif repetitions <= 0:
        interval = 1.0
        repetitions = 1
    elif repetitions == 1:
        interval = 6.0
        repetitions = 2
    else:
        interval = max(1.0, float(state["interval_days"]) * ease)
        repetitions += 1
    attempts = int(state["attempt_count"]) + 1
    correct_count = int(state["correct_count"]) + (1 if correct else 0)
    previous = float(state["mastery"])
    mastery = score if int(state["attempt_count"]) == 0 else (0.6 * previous + 0.4 * score)
    return {
        "ease": round(ease, 4),
        "interval_days": round(interval, 4),
        "repetitions": repetitions,
        "lapses": lapses,
        "mastery": round(mastery, 4),
        "next_review_at": _iso(moment + timedelta(days=interval)),
        "last_review_at": _iso(moment),
        "attempt_count": attempts,
        "correct_count": correct_count,
    }


def is_due(next_review_at: str | None, now: datetime | None = None) -> bool:
    if not next_review_at:
        return True
    return next_review_at <= _iso(now or utcnow())
