"""Study operations shared by the CLI and the MCP server."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
import struct
import uuid

from deeptutor.agent_study.embed import (
    Embedder,
    embedder_for,
    resolve_selection,
    sentence_transformers_available,
)
from deeptutor.agent_study.errors import StudyError
from deeptutor.agent_study.grade import grade_attempt
from deeptutor.agent_study.parse import SUPPORTED_EXTENSIONS, passages_for
from deeptutor.agent_study.paths import study_root
from deeptutor.agent_study.schedule import apply_review, fresh_topic, is_due, utcnow
from deeptutor.utils.secret_files import ensure_private_directory

_KB_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_TASK_STATUS = frozenset({"todo", "doing", "done"})
_SCHEMA = """
CREATE TABLE IF NOT EXISTS knowledge_bases (
    name TEXT PRIMARY KEY,
    description TEXT NOT NULL DEFAULT '',
    embed_backend TEXT NOT NULL,
    embed_model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    kb TEXT NOT NULL,
    original_path TEXT NOT NULL,
    stored_path TEXT NOT NULL,
    display_name TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    UNIQUE(kb, original_path)
);
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    kb TEXT NOT NULL,
    document_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    text TEXT NOT NULL,
    display_name TEXT NOT NULL,
    stored_path TEXT NOT NULL,
    original_path TEXT NOT NULL,
    page INTEGER,
    section TEXT,
    vector BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_kb ON chunks(kb);
CREATE TABLE IF NOT EXISTS questions (
    id TEXT PRIMARY KEY,
    bank TEXT NOT NULL,
    topic TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    explanation TEXT NOT NULL DEFAULT '',
    tags TEXT NOT NULL DEFAULT '[]',
    sources TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS questions_topic ON questions(topic);
CREATE TABLE IF NOT EXISTS attempts (
    id TEXT PRIMARY KEY,
    question_id TEXT NOT NULL,
    learner_answer TEXT NOT NULL,
    correct INTEGER NOT NULL,
    score REAL NOT NULL,
    feedback TEXT NOT NULL DEFAULT '',
    graded_by TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS attempts_question ON attempts(question_id);
CREATE TABLE IF NOT EXISTS topics (
    name TEXT PRIMARY KEY,
    ease REAL NOT NULL,
    interval_days REAL NOT NULL,
    repetitions INTEGER NOT NULL,
    lapses INTEGER NOT NULL,
    mastery REAL NOT NULL,
    next_review_at TEXT NOT NULL,
    last_review_at TEXT,
    attempt_count INTEGER NOT NULL,
    correct_count INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS memory_notes (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    tags TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL,
    archived INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def _new_id(prefix: str) -> str:
    return prefix + uuid.uuid4().hex[:12]


def _now() -> str:
    return utcnow().isoformat()


def _json_loads(raw: str) -> list:
    try:
        loaded = json.loads(raw or "[]")
    except json.JSONDecodeError:
        return []
    return loaded if isinstance(loaded, list) else []


class StudyService:
    """Local study store. Safe to construct once per process."""

    def __init__(self, root: str | Path | None = None, embedder: Embedder | None = None) -> None:
        self.root = study_root(root)
        ensure_private_directory(self.root)
        self._embedder_override = embedder
        self._db_path = self.root / "study.sqlite"
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def status(self) -> dict:
        backend, model = self._create_selection()
        return {
            "ok": True,
            "data_dir": str(self.root),
            "embed_backend": backend,
            "embed_model": model,
            "sentence_transformers_installed": sentence_transformers_available(),
            "api_key_required": False,
            "llm_required": False,
        }

    def kb_create(self, name: str, description: str = "") -> dict:
        kb = _kb_name(name)
        embedder = self._new_embedder()
        created = _now()
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT name FROM knowledge_bases WHERE name = ?", (kb,)
            ).fetchone()
            if existing is not None:
                raise StudyError(f"Knowledge base {kb!r} already exists.", code="exists")
            conn.execute(
                """
                INSERT INTO knowledge_bases
                    (name, description, embed_backend, embed_model, dim, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (kb, description.strip(), embedder.backend, embedder.model, embedder.dim, created),
            )
            conn.commit()
        return {
            "ok": True,
            "name": kb,
            "description": description.strip(),
            "embed_backend": embedder.backend,
            "embed_model": embedder.model,
            "dim": embedder.dim,
            "created_at": created,
        }

    def kb_list(self) -> dict:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM knowledge_bases ORDER BY created_at, name"
            ).fetchall()
            bases = [self._kb_summary(conn, row) for row in rows]
        return {"ok": True, "knowledge_bases": bases}

    def kb_ingest(self, name: str, paths: list[str] | str) -> dict:
        kb = _kb_name(name)
        inputs = _collect_inputs(paths)
        with self._connect() as conn:
            row = self._require_kb(conn, kb)
            embedder = embedder_for(
                row["embed_backend"], row["embed_model"], dim=int(row["dim"]) or None
            )
            if int(row["dim"]) and embedder.dim != int(row["dim"]):
                raise StudyError(
                    "The embedding model dimension does not match this knowledge base. "
                    "Create a new knowledge base for the new model.",
                    code="embed_mismatch",
                )
        added: list[str] = []
        updated: list[str] = []
        unchanged: list[str] = []
        failed: list[dict[str, str]] = []
        for source, display in inputs:
            try:
                outcome = self._ingest_file(kb, source, display, embedder)
            except StudyError as exc:
                failed.append({"path": str(source), "error": str(exc)})
                continue
            except Exception as exc:
                failed.append({"path": str(source), "error": str(exc)})
                continue
            bucket = {"added": added, "updated": updated, "unchanged": unchanged}[outcome]
            bucket.append(display)
        with self._connect() as conn:
            chunk_count = conn.execute(
                "SELECT COUNT(*) AS n FROM chunks WHERE kb = ?", (kb,)
            ).fetchone()["n"]
        ok = bool(added or updated or unchanged) or not failed
        return {
            "ok": ok,
            "kb": kb,
            "added": added,
            "updated": updated,
            "unchanged": unchanged,
            "failed": failed,
            "chunks": int(chunk_count),
            "embed_backend": embedder.backend,
            "embed_model": embedder.model,
            "error": failed[0]["error"] if not ok and failed else "",
        }

    def kb_retrieve(self, name: str, query: str, k: int = 5) -> dict:
        kb = _kb_name(name)
        text = " ".join(str(query or "").split())
        if not text:
            raise StudyError("query is required.")
        limit = _bounded_int(k, default=5, maximum=20, field="k")
        with self._connect() as conn:
            row = self._require_kb(conn, kb)
            embedder = embedder_for(
                row["embed_backend"], row["embed_model"], dim=int(row["dim"]) or None
            )
            chunks = conn.execute(
                "SELECT * FROM chunks WHERE kb = ? ORDER BY ordinal, id", (kb,)
            ).fetchall()
        if not chunks:
            return {"ok": True, "kb": kb, "query": text, "passages": []}
        query_vector = embedder.embed_query(text)
        scored = [(_cosine(query_vector, _unpack(item["vector"])), item) for item in chunks]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        passages = [_citation(score, item) for score, item in scored[:limit]]
        return {"ok": True, "kb": kb, "query": text, "passages": passages}

    def question_save(
        self,
        *,
        question: str,
        answer: str,
        bank: str = "default",
        topic: str = "general",
        explanation: str = "",
        tags: list | str | None = None,
        sources: list | dict | None = None,
        question_id: str | None = None,
    ) -> dict:
        prompt = _bounded_text(question, field="question", limit=8000)
        key = _bounded_text(answer, field="answer", limit=8000)
        bank_name = _label(bank or "default", field="bank", limit=80)
        topic_name = _label(topic or "general", field="topic", limit=120)
        explained = _bounded_text(
            explanation or "", field="explanation", limit=16000, required=False
        )
        tag_list = _tags(tags)
        source_list = _sources(sources)
        stamp = _now()
        with self._connect() as conn:
            if question_id:
                existing = conn.execute(
                    "SELECT id FROM questions WHERE id = ?", (question_id,)
                ).fetchone()
                if existing is None:
                    raise StudyError(f"Question {question_id!r} was not found.", code="not_found")
                conn.execute(
                    """
                    UPDATE questions
                    SET bank = ?, topic = ?, question = ?, answer = ?, explanation = ?,
                        tags = ?, sources = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        bank_name,
                        topic_name,
                        prompt,
                        key,
                        explained,
                        json.dumps(tag_list),
                        json.dumps(source_list),
                        stamp,
                        question_id,
                    ),
                )
                saved_id = question_id
            else:
                saved_id = _new_id("q_")
                conn.execute(
                    """
                    INSERT INTO questions (
                        id, bank, topic, question, answer, explanation, tags, sources,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        saved_id,
                        bank_name,
                        topic_name,
                        prompt,
                        key,
                        explained,
                        json.dumps(tag_list),
                        json.dumps(source_list),
                        stamp,
                        stamp,
                    ),
                )
            conn.commit()
            saved = conn.execute("SELECT * FROM questions WHERE id = ?", (saved_id,)).fetchone()
        return {"ok": True, "question": _question_payload(saved, include_answers=True)}

    def question_list(
        self,
        *,
        bank: str | None = None,
        topic: str | None = None,
        tag: str | None = None,
        limit: int = 50,
        include_answers: bool = True,
    ) -> dict:
        capped = _bounded_int(limit, default=50, maximum=200, field="limit")
        clauses = ["1 = 1"]
        params: list = []
        if bank:
            clauses.append("bank = ?")
            params.append(_label(bank, field="bank", limit=80))
        if topic:
            clauses.append("topic = ?")
            params.append(_label(topic, field="topic", limit=120))
        sql = f"SELECT * FROM questions WHERE {' AND '.join(clauses)} ORDER BY updated_at DESC, id"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        wanted = (tag or "").strip()
        matched = []
        for row in rows:
            tags = _json_loads(row["tags"])
            if wanted and wanted not in tags:
                continue
            matched.append(_question_payload(row, include_answers=include_answers))
            if len(matched) >= capped:
                break
        return {"ok": True, "questions": matched, "count": len(matched)}

    def attempt_record(
        self,
        question_id: str,
        learner_answer: str,
        *,
        correct: bool | None = None,
        score: float | None = None,
        feedback: str = "",
    ) -> dict:
        if not str(question_id or "").strip():
            raise StudyError("question_id is required.")
        answer = _bounded_text(learner_answer, field="learner_answer", limit=8000, required=False)
        judged_score = _optional_score(score)
        judged_correct = _optional_bool(correct)
        note = _bounded_text(feedback or "", field="feedback", limit=4000, required=False)
        moment = utcnow()
        with self._connect() as conn:
            question = conn.execute(
                "SELECT * FROM questions WHERE id = ?", (question_id,)
            ).fetchone()
            if question is None:
                raise StudyError(f"Question {question_id!r} was not found.", code="not_found")
            try:
                passed, graded_score, graded_by = grade_attempt(
                    question["answer"],
                    answer,
                    correct=judged_correct,
                    score=judged_score,
                )
            except ValueError as exc:
                raise StudyError(str(exc)) from exc
            if not note:
                note = (
                    "Matches the stored answer." if passed else "Does not match the stored answer."
                )
            attempt_id = _new_id("a_")
            conn.execute(
                """
                INSERT INTO attempts (
                    id, question_id, learner_answer, correct, score, feedback, graded_by, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    attempt_id,
                    question_id,
                    answer,
                    1 if passed else 0,
                    graded_score,
                    note,
                    graded_by,
                    moment.isoformat(),
                ),
            )
            topic_row = conn.execute(
                "SELECT * FROM topics WHERE name = ?", (question["topic"],)
            ).fetchone()
            state = _topic_state(topic_row) if topic_row else fresh_topic(moment)
            updated = apply_review(state, score=graded_score, correct=passed, now=moment)
            _save_topic(conn, question["topic"], updated)
            conn.commit()
        return {
            "ok": True,
            "attempt": {
                "id": attempt_id,
                "question_id": question_id,
                "correct": passed,
                "score": graded_score,
                "graded_by": graded_by,
                "feedback": note,
                "learner_answer": answer,
                "expected_answer": question["answer"],
                "explanation": question["explanation"],
                "created_at": moment.isoformat(),
            },
            "topic": _public_topic(question["topic"], updated, question_count=None),
        }

    def mastery_status(self, topic: str | None = None) -> dict:
        wanted = _label(topic, field="topic", limit=120) if topic else None
        with self._connect() as conn:
            question_rows = conn.execute(
                "SELECT topic, COUNT(*) AS n FROM questions GROUP BY topic"
            ).fetchall()
            topic_rows = {
                row["name"]: row for row in conn.execute("SELECT * FROM topics").fetchall()
            }
            banks: dict[str, list[str]] = {}
            for row in conn.execute("SELECT topic, bank FROM questions").fetchall():
                banks.setdefault(row["topic"], [])
                if row["bank"] not in banks[row["topic"]]:
                    banks[row["topic"]].append(row["bank"])
        counts = {row["topic"]: int(row["n"]) for row in question_rows}
        names = set(counts) | set(topic_rows)
        topics = []
        for name in sorted(names):
            if name in topic_rows:
                state = _topic_state(topic_rows[name])
            else:
                state = fresh_topic()
                state["next_review_at"] = ""
            topics.append(
                _public_topic(
                    name,
                    state,
                    question_count=counts.get(name, 0),
                    banks=banks.get(name, []),
                )
            )
        if wanted is not None:
            topics = [item for item in topics if item["topic"] == wanted]
        topics.sort(key=lambda item: (not item["due"], item["mastery"], item["topic"]))
        return {"ok": True, "topics": topics}

    def mastery_next(self, limit: int = 3) -> dict:
        capped = _bounded_int(limit, default=3, maximum=20, field="limit")
        status = self.mastery_status()
        due = [item for item in status["topics"] if item["due"] and item["question_count"]]
        upcoming = [item for item in status["topics"] if not item["due"]]
        suggestions = []
        with self._connect() as conn:
            for item in due:
                if len(suggestions) >= capped:
                    break
                question = self._next_question(conn, item["topic"])
                if question is None:
                    continue
                suggestions.append(
                    {
                        "topic": item["topic"],
                        "mastery": item["mastery"],
                        "due": True,
                        "reason": item["reason"],
                        "next_review_at": item["next_review_at"],
                        "question": question,
                    }
                )
        return {
            "ok": True,
            "suggestions": suggestions,
            "due_count": len(due),
            "upcoming": upcoming[:capped],
        }

    def memory_list(self) -> dict:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memory_notes ORDER BY updated_at DESC, id"
            ).fetchall()
        notes = [_memory_payload(row, preview=True) for row in rows]
        return {"ok": True, "notes": notes}

    def memory_read(self, note_id: str) -> dict:
        with self._connect() as conn:
            row = self._require_note(conn, note_id)
        return {"ok": True, "note": _memory_payload(row, preview=False)}

    def memory_write(
        self,
        *,
        title: str,
        body: str,
        note_id: str | None = None,
        tags: list | str | None = None,
    ) -> dict:
        heading = _bounded_text(title, field="title", limit=200)
        text = _bounded_text(body, field="body", limit=100_000, required=False)
        tag_list = _tags(tags)
        stamp = _now()
        with self._connect() as conn:
            if note_id:
                existing = conn.execute(
                    "SELECT id, created_at FROM memory_notes WHERE id = ?", (note_id,)
                ).fetchone()
                if existing is None:
                    raise StudyError(f"Memory note {note_id!r} was not found.", code="not_found")
                conn.execute(
                    """
                    UPDATE memory_notes
                    SET title = ?, body = ?, tags = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (heading, text, json.dumps(tag_list), stamp, note_id),
                )
                saved_id = note_id
            else:
                saved_id = _new_id("m_")
                conn.execute(
                    """
                    INSERT INTO memory_notes (id, title, body, tags, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (saved_id, heading, text, json.dumps(tag_list), stamp, stamp),
                )
            conn.commit()
            saved = conn.execute("SELECT * FROM memory_notes WHERE id = ?", (saved_id,)).fetchone()
        return {"ok": True, "note": _memory_payload(saved, preview=False)}

    def memory_delete(self, note_id: str) -> dict:
        with self._connect() as conn:
            self._require_note(conn, note_id)
            conn.execute("DELETE FROM memory_notes WHERE id = ?", (note_id,))
            conn.commit()
        return {"ok": True, "deleted": note_id}

    def task_list(self, *, status: str | None = None, include_archived: bool = False) -> dict:
        wanted = _task_status(status) if status else None
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM tasks ORDER BY created_at, id").fetchall()
        tasks = []
        for row in rows:
            if not include_archived and row["archived"]:
                continue
            if wanted and row["status"] != wanted:
                continue
            tasks.append(_task_payload(row))
        return {"ok": True, "tasks": tasks}

    def task_write(
        self,
        *,
        title: str | None = None,
        note: str | None = None,
        status: str | None = None,
        task_id: str | None = None,
        archived: bool | None = None,
    ) -> dict:
        stamp = _now()
        with self._connect() as conn:
            if not task_id:
                heading = _bounded_text(title or "", field="title", limit=160)
                body = _bounded_text(note or "", field="note", limit=2000, required=False)
                task_status = _task_status(status or "todo")
                saved_id = _new_id("t_")
                conn.execute(
                    """
                    INSERT INTO tasks (id, title, note, status, archived, created_at, updated_at)
                    VALUES (?, ?, ?, ?, 0, ?, ?)
                    """,
                    (saved_id, heading, body, task_status, stamp, stamp),
                )
            else:
                existing = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
                if existing is None:
                    raise StudyError(f"Task {task_id!r} was not found.", code="not_found")
                heading = (
                    _bounded_text(title, field="title", limit=160)
                    if title is not None
                    else existing["title"]
                )
                body = (
                    _bounded_text(note, field="note", limit=2000, required=False)
                    if note is not None
                    else existing["note"]
                )
                task_status = _task_status(status) if status is not None else existing["status"]
                is_archived = (
                    int(bool(archived)) if archived is not None else int(existing["archived"])
                )
                conn.execute(
                    """
                    UPDATE tasks
                    SET title = ?, note = ?, status = ?, archived = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (heading, body, task_status, is_archived, stamp, task_id),
                )
                saved_id = task_id
            conn.commit()
            saved = conn.execute("SELECT * FROM tasks WHERE id = ?", (saved_id,)).fetchone()
        return {"ok": True, "task": _task_payload(saved)}

    def _ingest_file(self, kb: str, source: Path, display: str, embedder: Embedder) -> str:
        if _is_relative_to(source, self.root):
            raise StudyError(
                "Refusing to ingest files from the study data directory.",
                code="refused",
            )
        digest = _sha256(source)
        stored = self._store_copy(kb, source, display)
        passages = passages_for(source)
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM documents WHERE kb = ? AND original_path = ?",
                (kb, str(source)),
            ).fetchone()
            if existing is not None and existing["sha256"] == digest:
                return "unchanged"
            vectors = embedder.embed_documents([passage.text for passage in passages])
            if len(vectors) != len(passages):
                raise StudyError("Embedder returned the wrong number of vectors.", code="embed")
            stamp = _now()
            if existing is None:
                document_id = _new_id("d_")
                conn.execute(
                    """
                    INSERT INTO documents (
                        id, kb, original_path, stored_path, display_name, sha256, ingested_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (document_id, kb, str(source), str(stored), display, digest, stamp),
                )
                outcome = "added"
            else:
                document_id = existing["id"]
                conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
                conn.execute(
                    """
                    UPDATE documents
                    SET stored_path = ?, display_name = ?, sha256 = ?, ingested_at = ?
                    WHERE id = ?
                    """,
                    (str(stored), display, digest, stamp, document_id),
                )
                outcome = "updated"
            for ordinal, (passage, vector) in enumerate(zip(passages, vectors)):
                conn.execute(
                    """
                    INSERT INTO chunks (
                        id, kb, document_id, ordinal, text, display_name, stored_path,
                        original_path, page, section, vector
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        _new_id("c_"),
                        kb,
                        document_id,
                        ordinal,
                        passage.text,
                        display,
                        str(stored),
                        str(source),
                        passage.page,
                        passage.section,
                        _pack(vector),
                    ),
                )
            conn.commit()
        return outcome

    def _store_copy(self, kb: str, source: Path, display: str) -> Path:
        destination = self.root / "files" / kb / display
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.resolve() != source.resolve():
            shutil.copy2(source, destination)
        return destination

    def _next_question(self, conn: sqlite3.Connection, topic: str) -> dict | None:
        rows = conn.execute(
            "SELECT * FROM questions WHERE topic = ? ORDER BY created_at, id",
            (topic,),
        ).fetchall()
        if not rows:
            return None
        latest: dict[str, dict] = {}
        attempts = conn.execute(
            """
            SELECT a.* FROM attempts a
            JOIN questions q ON q.id = a.question_id
            WHERE q.topic = ?
            ORDER BY a.created_at DESC, a.id DESC
            """,
            (topic,),
        ).fetchall()
        for attempt in attempts:
            latest.setdefault(attempt["question_id"], attempt)
        chosen = None
        for row in rows:
            attempt = latest.get(row["id"])
            if attempt is None or not attempt["correct"]:
                chosen = row
                break
        if chosen is None:
            chosen = rows[0]
        return _question_payload(chosen, include_answers=False)

    def _new_embedder(self) -> Embedder:
        if self._embedder_override is not None:
            return self._embedder_override
        backend, model = resolve_selection()
        if backend == "sentence-transformers" and not sentence_transformers_available():
            raise StudyError(
                "sentence-transformers is not installed. "
                "pip install 'deeptutor[local-embed]', or set DEEPTUTOR_EMBED_BACKEND=hash.",
                code="embed_missing",
            )
        return embedder_for(backend, model)

    def _create_selection(self) -> tuple[str, str]:
        if self._embedder_override is not None:
            return self._embedder_override.backend, self._embedder_override.model
        return resolve_selection()

    def _require_kb(self, conn: sqlite3.Connection, name: str) -> dict:
        row = conn.execute("SELECT * FROM knowledge_bases WHERE name = ?", (name,)).fetchone()
        if row is None:
            raise StudyError(f"Knowledge base {name!r} was not found.", code="not_found")
        return row

    def _require_note(self, conn: sqlite3.Connection, note_id: str) -> dict:
        if not str(note_id or "").strip():
            raise StudyError("note_id is required.")
        row = conn.execute("SELECT * FROM memory_notes WHERE id = ?", (note_id,)).fetchone()
        if row is None:
            raise StudyError(f"Memory note {note_id!r} was not found.", code="not_found")
        return row

    def _kb_summary(self, conn: sqlite3.Connection, row: dict) -> dict:
        documents = conn.execute(
            "SELECT COUNT(*) AS n FROM documents WHERE kb = ?", (row["name"],)
        ).fetchone()["n"]
        chunks = conn.execute(
            "SELECT COUNT(*) AS n FROM chunks WHERE kb = ?", (row["name"],)
        ).fetchone()["n"]
        return {
            "name": row["name"],
            "description": row["description"],
            "documents": int(documents),
            "chunks": int(chunks),
            "embed_backend": row["embed_backend"],
            "embed_model": row["embed_model"],
            "dim": int(row["dim"]),
            "created_at": row["created_at"],
        }

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.row_factory = _dict_row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 5000")
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


def _dict_row(cursor, row) -> dict:
    return {column[0]: row[index] for index, column in enumerate(cursor.description)}


def _kb_name(name: str) -> str:
    text = str(name or "").strip()
    if not _KB_NAME.fullmatch(text):
        raise StudyError(
            "Knowledge base names must start with a letter or digit and contain only "
            "letters, digits, dots, underscores, and hyphens (max 64 characters)."
        )
    return text


def _label(value: str, *, field: str, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        raise StudyError(f"{field} is required.")
    if len(text) > limit:
        raise StudyError(f"{field} must be at most {limit} characters.")
    return text


def _bounded_text(value: str, *, field: str, limit: int, required: bool = True) -> str:
    text = str(value or "")
    if required:
        text = text.strip()
        if not text:
            raise StudyError(f"{field} is required.")
    if len(text) > limit:
        raise StudyError(f"{field} must be at most {limit} characters.")
    return text


def _bounded_int(value, *, default: int, maximum: int, field: str) -> int:
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise StudyError(f"{field} must be an integer.") from exc
    if parsed < 1:
        raise StudyError(f"{field} must be at least 1.")
    return min(parsed, maximum)


def _optional_score(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise StudyError("score must be a number between 0 and 1.") from exc
    if score < 0.0 or score > 1.0:
        raise StudyError("score must be between 0 and 1.")
    return score


def _optional_bool(value) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    raise StudyError("correct must be a boolean.")


def _tags(raw) -> list[str]:
    if raw is None or raw == "":
        return []
    if isinstance(raw, str):
        items = raw.split(",")
    elif isinstance(raw, list):
        items = raw
    else:
        raise StudyError("tags must be a list of strings.")
    tags: list[str] = []
    for item in items:
        text = " ".join(str(item).split())
        if text and text not in tags:
            tags.append(text)
    return tags[:32]


def _sources(raw) -> list[dict]:
    if raw is None or raw == "":
        return []
    if isinstance(raw, dict):
        raw = [raw]
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise StudyError("sources must be JSON objects.") from exc
        if isinstance(raw, dict):
            raw = [raw]
    if not isinstance(raw, list):
        raise StudyError("sources must be a list of objects.")
    cleaned = []
    for item in raw:
        if not isinstance(item, dict):
            raise StudyError("each source must be an object with file, page, and section.")
        page = item.get("page")
        if page in (None, ""):
            page_value = None
        else:
            try:
                page_value = int(page)
            except (TypeError, ValueError) as exc:
                raise StudyError("source page must be an integer.") from exc
        entry = {
            "file": _optional_short(item.get("file")),
            "page": page_value,
            "section": _optional_short(item.get("section")),
            "kb": _optional_short(item.get("kb")),
            "chunk_id": _optional_short(item.get("chunk_id")),
            "quote": _optional_short(item.get("quote"), limit=500),
        }
        cleaned.append({key: value for key, value in entry.items() if value not in (None, "")})
    return cleaned


def _optional_short(value, *, limit: int = 240) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    if not text:
        return None
    return text[:limit]


def _task_status(value: str | None) -> str:
    text = str(value or "").strip().lower()
    if text not in _TASK_STATUS:
        raise StudyError("status must be todo, doing, or done.")
    return text


def _collect_inputs(paths: list[str] | str) -> list[tuple[Path, str]]:
    if isinstance(paths, str):
        raw_paths = [paths]
    else:
        raw_paths = [str(item) for item in paths or []]
    raw_paths = [item for item in raw_paths if str(item).strip()]
    if not raw_paths:
        raise StudyError("At least one file or folder path is required.")
    found: list[tuple[Path, str]] = []
    seen: set[str] = set()
    for raw in raw_paths:
        path = Path(raw).expanduser().resolve()
        if path.is_dir():
            supported = [
                child
                for child in sorted(path.rglob("*"))
                if child.is_file()
                and not child.name.startswith(".")
                and child.suffix.lower() in SUPPORTED_EXTENSIONS
                and "__pycache__" not in child.parts
            ]
            if not supported:
                raise StudyError(f"No supported documents under {path}.", code="empty")
            for child in supported:
                _add_input(found, seen, child, path)
            continue
        if not path.is_file():
            raise StudyError(f"Path not found: {raw}", code="not_found")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise StudyError(
                f"Unsupported file type for {path.name}.",
                code="unsupported",
            )
        _add_input(found, seen, path, None)
    return found


def _add_input(
    found: list[tuple[Path, str]], seen: set[str], path: Path, root: Path | None
) -> None:
    key = str(path.resolve())
    if key in seen:
        return
    seen.add(key)
    if root is None:
        display = path.name
    else:
        display = path.resolve().relative_to(root.resolve()).as_posix()
    display = _unique_display(found, display)
    found.append((path.resolve(), display))


def _unique_display(found: list[tuple[Path, str]], display: str) -> str:
    used = {item[1] for item in found}
    if display not in used:
        return display
    path = Path(display)
    parent = "" if path.parent.as_posix() in {"", "."} else path.parent.as_posix()
    counter = 2
    while True:
        renamed = f"{path.stem}-{counter}{path.suffix}"
        candidate = f"{parent}/{renamed}" if parent else renamed
        if candidate not in used:
            return candidate
        counter += 1


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pack(vector: list[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *[float(value) for value in vector])


def _unpack(blob: bytes) -> list[float]:
    count = len(blob) // 4
    return list(struct.unpack(f"<{count}f", blob))


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    return float(sum(a * b for a, b in zip(left, right)))


def _citation(score: float, row: dict) -> dict:
    return {
        "chunk_id": row["id"],
        "text": row["text"],
        "score": round(float(score), 4),
        "file": row["display_name"],
        "path": row["stored_path"],
        "original_path": row["original_path"],
        "page": row["page"],
        "section": row["section"],
        "kb": row["kb"],
    }


def _question_payload(row: dict, *, include_answers: bool) -> dict:
    payload = {
        "id": row["id"],
        "bank": row["bank"],
        "topic": row["topic"],
        "question": row["question"],
        "tags": _json_loads(row["tags"]),
        "sources": _json_loads(row["sources"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }
    if include_answers:
        payload["answer"] = row["answer"]
        payload["explanation"] = row["explanation"]
    return payload


def _topic_state(row: dict | None) -> dict:
    if row is None:
        return fresh_topic()
    return {
        "ease": float(row["ease"]),
        "interval_days": float(row["interval_days"]),
        "repetitions": int(row["repetitions"]),
        "lapses": int(row["lapses"]),
        "mastery": float(row["mastery"]),
        "next_review_at": row["next_review_at"],
        "last_review_at": row["last_review_at"],
        "attempt_count": int(row["attempt_count"]),
        "correct_count": int(row["correct_count"]),
    }


def _save_topic(conn: sqlite3.Connection, name: str, state: dict) -> None:
    conn.execute(
        """
        INSERT INTO topics (
            name, ease, interval_days, repetitions, lapses, mastery, next_review_at,
            last_review_at, attempt_count, correct_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
            ease = excluded.ease,
            interval_days = excluded.interval_days,
            repetitions = excluded.repetitions,
            lapses = excluded.lapses,
            mastery = excluded.mastery,
            next_review_at = excluded.next_review_at,
            last_review_at = excluded.last_review_at,
            attempt_count = excluded.attempt_count,
            correct_count = excluded.correct_count
        """,
        (
            name,
            state["ease"],
            state["interval_days"],
            state["repetitions"],
            state["lapses"],
            state["mastery"],
            state["next_review_at"],
            state["last_review_at"],
            state["attempt_count"],
            state["correct_count"],
        ),
    )


def _public_topic(
    name: str,
    state: dict,
    question_count: int | None,
    banks: list[str] | None = None,
) -> dict:
    attempts = int(state["attempt_count"])
    due = is_due(state["next_review_at"] or None)
    if attempts == 0:
        reason = "unseen"
        due = True
    elif float(state["interval_days"]) == 0.0 and attempts:
        reason = "failed"
    elif due:
        reason = "due"
    else:
        reason = "scheduled"
    payload = {
        "topic": name,
        "mastery": float(state["mastery"]),
        "accuracy": (int(state["correct_count"]) / attempts) if attempts else 0.0,
        "attempt_count": attempts,
        "correct_count": int(state["correct_count"]),
        "ease": float(state["ease"]),
        "interval_days": float(state["interval_days"]),
        "repetitions": int(state["repetitions"]),
        "lapses": int(state["lapses"]),
        "next_review_at": state["next_review_at"] or None,
        "last_review_at": state["last_review_at"],
        "due": due,
        "reason": reason,
        "banks": banks or [],
    }
    if question_count is not None:
        payload["question_count"] = question_count
    return payload


def _memory_payload(row: dict, *, preview: bool) -> dict:
    body = row["body"]
    return {
        "id": row["id"],
        "title": row["title"],
        "body": body if not preview else body[:280],
        "tags": _json_loads(row["tags"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _task_payload(row: dict) -> dict:
    return {
        "id": row["id"],
        "title": row["title"],
        "note": row["note"],
        "status": row["status"],
        "archived": bool(row["archived"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }
