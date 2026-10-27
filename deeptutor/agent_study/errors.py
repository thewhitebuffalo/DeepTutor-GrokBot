"""Errors the study backend expects callers to handle."""

from __future__ import annotations


class StudyError(Exception):
    """A caller-facing failure with a stable ``code``."""

    def __init__(self, message: str, *, code: str = "invalid") -> None:
        super().__init__(message)
        self.code = code
