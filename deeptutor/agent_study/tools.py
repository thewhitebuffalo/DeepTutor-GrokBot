"""Tool schemas and dispatch shared by the MCP server and tests."""

from __future__ import annotations

from collections.abc import Callable

from deeptutor.agent_study.errors import StudyError
from deeptutor.agent_study.service import StudyService

ToolHandler = Callable[[StudyService, dict], dict]

_OBJECT = "object"


def _schema(properties: dict, required: list[str] | None = None) -> dict:
    return {
        "type": _OBJECT,
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


def _string(description: str) -> dict:
    return {"type": "string", "description": description}


def _int(description: str) -> dict:
    return {"type": "integer", "description": description}


def _bool(description: str) -> dict:
    return {"type": "boolean", "description": description}


def _string_array(description: str) -> dict:
    return {"type": "array", "items": {"type": "string"}, "description": description}


TOOLS: list[dict] = [
    {
        "name": "study_status",
        "description": (
            "Show the study data directory and local embedding backend. "
            "Indexing does not use a cloud API key or an LLM."
        ),
        "inputSchema": _schema({}),
    },
    {
        "name": "kb_create",
        "description": "Create an empty knowledge base for later ingestion.",
        "inputSchema": _schema(
            {
                "name": _string("Short id, for example pe-exam or mba."),
                "description": _string("What this knowledge base is for."),
            },
            ["name"],
        ),
    },
    {
        "name": "kb_list",
        "description": "List knowledge bases with document and chunk counts.",
        "inputSchema": _schema({}),
    },
    {
        "name": "kb_ingest",
        "description": (
            "Copy files or a folder into a knowledge base and index them with the "
            "local embedder. PDF, Markdown, text, Word, PowerPoint, and Excel are supported. "
            "No LLM and no API key."
        ),
        "inputSchema": _schema(
            {
                "name": _string("Knowledge base name."),
                "paths": _string_array("Files or folders to ingest."),
            },
            ["name", "paths"],
        ),
    },
    {
        "name": "kb_retrieve",
        "description": (
            "Retrieve passages for a query. Each hit cites file, page, and section "
            "when the source has them. Use the passages to draft questions; do not "
            "treat this tool as a tutor."
        ),
        "inputSchema": _schema(
            {
                "name": _string("Knowledge base name."),
                "query": _string("What to look up."),
                "k": _int("How many passages to return, 1-20. Default 5."),
            },
            ["name", "query"],
        ),
    },
    {
        "name": "question_save",
        "description": (
            "Save an assistant-authored question, answer, explanation, tags, and "
            "source citations. Pass question_id to update an existing question."
        ),
        "inputSchema": _schema(
            {
                "question": _string("The question the learner will see."),
                "answer": _string("The answer key. Keep this from the learner until grading."),
                "bank": _string("Question bank name. Default 'default'."),
                "topic": _string("Mastery topic. Default 'general'."),
                "explanation": _string("Why the answer is right, shown after an attempt."),
                "tags": _string_array("Short tags."),
                "sources": {
                    "type": "array",
                    "description": "Citations: file, page, section, kb, chunk_id, quote.",
                    "items": {"type": "object"},
                },
                "question_id": _string("Existing question id to update."),
            },
            ["question", "answer"],
        ),
    },
    {
        "name": "question_list",
        "description": "List saved questions. Answers are included unless include_answers is false.",
        "inputSchema": _schema(
            {
                "bank": _string("Only this bank."),
                "topic": _string("Only this topic."),
                "tag": _string("Only questions carrying this tag."),
                "limit": _int("Maximum rows, 1-200. Default 50."),
                "include_answers": _bool("Include answer and explanation. Default true."),
            }
        ),
    },
    {
        "name": "attempt_record",
        "description": (
            "Record the learner's answer. Omit correct and score to grade against the "
            "stored answer locally. Pass correct or score when you judged a free-text "
            "answer yourself. Returns the expected answer, explanation, and updated mastery."
        ),
        "inputSchema": _schema(
            {
                "question_id": _string("Question id from question_save or mastery_next."),
                "learner_answer": _string("What the learner answered."),
                "correct": _bool("Your judgment. Omit to grade locally."),
                "score": {
                    "type": "number",
                    "description": "Score from 0 to 1. Omit to grade locally.",
                },
                "feedback": _string("Optional note stored with the attempt."),
            },
            ["question_id", "learner_answer"],
        ),
    },
    {
        "name": "mastery_status",
        "description": "Mastery, accuracy, and spaced-repetition schedule for each topic.",
        "inputSchema": _schema({"topic": _string("Only this topic.")}),
    },
    {
        "name": "mastery_next",
        "description": (
            "Suggest what to review next. Due and unseen topics come first. "
            "The suggested question omits the answer key."
        ),
        "inputSchema": _schema({"limit": _int("How many suggestions, 1-20. Default 3.")}),
    },
    {
        "name": "memory_list",
        "description": "List learner memory notes written by the assistant.",
        "inputSchema": _schema({}),
    },
    {
        "name": "memory_read",
        "description": "Read one memory note, including its full body.",
        "inputSchema": _schema({"note_id": _string("Note id.")}, ["note_id"]),
    },
    {
        "name": "memory_write",
        "description": "Create or replace a learner memory note. Pass note_id to update.",
        "inputSchema": _schema(
            {
                "title": _string("Note title."),
                "body": _string("Note body."),
                "note_id": _string("Existing note id to replace."),
                "tags": _string_array("Tags."),
            },
            ["title", "body"],
        ),
    },
    {
        "name": "memory_delete",
        "description": "Delete one memory note.",
        "inputSchema": _schema({"note_id": _string("Note id.")}, ["note_id"]),
    },
    {
        "name": "task_list",
        "description": "Read the agent task board (separate from the web UI task board).",
        "inputSchema": _schema(
            {
                "status": _string("Filter to todo, doing, or done."),
                "include_archived": _bool("Include archived cards."),
            }
        ),
    },
    {
        "name": "task_write",
        "description": "Create a task card, or update one when task_id is set.",
        "inputSchema": _schema(
            {
                "title": _string("Card title. Required when creating."),
                "note": _string("Longer note."),
                "status": _string("todo, doing, or done."),
                "task_id": _string("Existing card id to update."),
                "archived": _bool("Archive or restore the card."),
            }
        ),
    },
]

_TOOL_NAMES = frozenset(tool["name"] for tool in TOOLS)


def call_tool(service: StudyService, name: str, arguments: dict | None) -> dict:
    """Run one study tool. Unknown names and arguments raise StudyError."""

    if name not in _HANDLERS:
        raise StudyError(
            f"Unknown tool {name!r}. Known tools: {', '.join(sorted(_TOOL_NAMES))}.",
            code="unknown_tool",
        )
    args = arguments or {}
    if not isinstance(args, dict):
        raise StudyError("Tool arguments must be an object.")
    return _HANDLERS[name](service, args)


def _flag(value, *, default: bool) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"false", "0", "no"}
    return bool(value)


def _check(args: dict, allowed: set[str]) -> None:
    unknown = sorted(set(args) - allowed)
    if unknown:
        raise StudyError(f"Unknown arguments: {', '.join(unknown)}.")


def _status(service: StudyService, args: dict) -> dict:
    _check(args, set())
    return service.status()


def _kb_create(service: StudyService, args: dict) -> dict:
    _check(args, {"name", "description"})
    return service.kb_create(args.get("name", ""), args.get("description") or "")


def _kb_list(service: StudyService, args: dict) -> dict:
    _check(args, set())
    return service.kb_list()


def _kb_ingest(service: StudyService, args: dict) -> dict:
    _check(args, {"name", "paths", "path"})
    paths = args.get("paths", args.get("path"))
    return service.kb_ingest(args.get("name", ""), paths)


def _kb_retrieve(service: StudyService, args: dict) -> dict:
    _check(args, {"name", "query", "k"})
    return service.kb_retrieve(args.get("name", ""), args.get("query", ""), args.get("k", 5))


def _question_save(service: StudyService, args: dict) -> dict:
    _check(
        args,
        {"question", "answer", "bank", "topic", "explanation", "tags", "sources", "question_id"},
    )
    return service.question_save(
        question=args.get("question", ""),
        answer=args.get("answer", ""),
        bank=args.get("bank") or "default",
        topic=args.get("topic") or "general",
        explanation=args.get("explanation") or "",
        tags=args.get("tags"),
        sources=args.get("sources"),
        question_id=args.get("question_id"),
    )


def _question_list(service: StudyService, args: dict) -> dict:
    _check(args, {"bank", "topic", "tag", "limit", "include_answers"})
    return service.question_list(
        bank=args.get("bank"),
        topic=args.get("topic"),
        tag=args.get("tag"),
        limit=args.get("limit", 50),
        include_answers=_flag(args.get("include_answers"), default=True),
    )


def _attempt(service: StudyService, args: dict) -> dict:
    _check(args, {"question_id", "learner_answer", "correct", "score", "feedback"})
    return service.attempt_record(
        args.get("question_id", ""),
        args.get("learner_answer", ""),
        correct=args.get("correct"),
        score=args.get("score"),
        feedback=args.get("feedback") or "",
    )


def _mastery_status(service: StudyService, args: dict) -> dict:
    _check(args, {"topic"})
    return service.mastery_status(args.get("topic"))


def _mastery_next(service: StudyService, args: dict) -> dict:
    _check(args, {"limit"})
    return service.mastery_next(args.get("limit", 3))


def _memory_list(service: StudyService, args: dict) -> dict:
    _check(args, set())
    return service.memory_list()


def _memory_read(service: StudyService, args: dict) -> dict:
    _check(args, {"note_id"})
    return service.memory_read(args.get("note_id", ""))


def _memory_write(service: StudyService, args: dict) -> dict:
    _check(args, {"title", "body", "note_id", "tags"})
    return service.memory_write(
        title=args.get("title", ""),
        body=args.get("body", ""),
        note_id=args.get("note_id"),
        tags=args.get("tags"),
    )


def _memory_delete(service: StudyService, args: dict) -> dict:
    _check(args, {"note_id"})
    return service.memory_delete(args.get("note_id", ""))


def _task_list(service: StudyService, args: dict) -> dict:
    _check(args, {"status", "include_archived"})
    return service.task_list(
        status=args.get("status"),
        include_archived=bool(args.get("include_archived")),
    )


def _task_write(service: StudyService, args: dict) -> dict:
    _check(args, {"title", "note", "status", "task_id", "archived"})
    return service.task_write(
        title=args.get("title"),
        note=args.get("note"),
        status=args.get("status"),
        task_id=args.get("task_id"),
        archived=args.get("archived"),
    )


_HANDLERS: dict[str, ToolHandler] = {
    "study_status": _status,
    "kb_create": _kb_create,
    "kb_list": _kb_list,
    "kb_ingest": _kb_ingest,
    "kb_retrieve": _kb_retrieve,
    "question_save": _question_save,
    "question_list": _question_list,
    "attempt_record": _attempt,
    "mastery_status": _mastery_status,
    "mastery_next": _mastery_next,
    "memory_list": _memory_list,
    "memory_read": _memory_read,
    "memory_write": _memory_write,
    "memory_delete": _memory_delete,
    "task_list": _task_list,
    "task_write": _task_write,
}
