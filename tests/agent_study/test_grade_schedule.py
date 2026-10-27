"""Local grading and spaced repetition, with no model and no database."""

from datetime import datetime, timedelta, timezone

from deeptutor.agent_study.grade import grade_attempt
from deeptutor.agent_study.schedule import apply_review, fresh_topic, is_due


def test_exact_and_overlap_grading_does_not_need_a_model():
    correct, score, graded_by = grade_attempt("1 megaohm per kilovolt", "1 megaohm per kilovolt")
    assert correct is True
    assert score == 1.0
    assert graded_by == "exact"

    correct, score, graded_by = grade_attempt(
        "one megaohm per kilovolt",
        "the minimum is one megaohm per kilovolt of rating",
    )
    assert correct is True
    assert graded_by == "overlap"
    assert score >= 0.8

    correct, score, _graded_by = grade_attempt("insulation resistance", "wacc")
    assert correct is False
    assert score < 0.8


def test_agent_judgment_overrides_string_match():
    correct, score, graded_by = grade_attempt(
        "a long canonical explanation",
        "the learner explained the idea in their own words",
        correct=True,
        score=0.9,
    )
    assert correct is True
    assert score == 0.9
    assert graded_by == "agent"


def test_failure_stays_due_and_a_pass_schedules_the_next_day():
    start = datetime(2026, 10, 27, tzinfo=timezone.utc)
    state = fresh_topic(start)
    failed = apply_review(state, score=0.0, correct=False, now=start)
    assert failed["interval_days"] == 0.0
    assert failed["lapses"] == 1
    assert is_due(failed["next_review_at"], start)

    passed = apply_review(failed, score=1.0, correct=True, now=start)
    assert passed["interval_days"] == 1.0
    assert passed["repetitions"] == 1
    assert is_due(passed["next_review_at"], start) is False
    assert is_due(passed["next_review_at"], start + timedelta(days=1, seconds=1)) is True

    second = apply_review(passed, score=1.0, correct=True, now=start + timedelta(days=1))
    assert second["interval_days"] == 6.0
