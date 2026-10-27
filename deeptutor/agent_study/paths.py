"""Where the headless study backend stores its data."""

from __future__ import annotations

import os
from pathlib import Path


def study_root(explicit: str | Path | None = None) -> Path:
    """Return the study data directory.

    Priority: explicit argument, ``DEEPTUTOR_STUDY_DIR``, then
    ``<DEEPTUTOR_HOME or cwd>/data/study``.
    """

    if explicit is not None and str(explicit).strip():
        return Path(explicit).expanduser().resolve()
    raw = os.environ.get("DEEPTUTOR_STUDY_DIR", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    from deeptutor.runtime.home import get_runtime_data_root

    return (get_runtime_data_root() / "study").resolve()
