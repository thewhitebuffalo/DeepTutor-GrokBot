"""CLI for the headless study backend.

JSON on stdout is the default so an assistant can pipe the commands. The same
operations are exposed over stdio MCP by ``deeptutor study mcp``.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import traceback

import typer

from deeptutor.agent_study.errors import StudyError
from deeptutor.agent_study.mcp_server import serve_stdio
from deeptutor.agent_study.service import StudyService


def _format_option():
    return typer.Option("json", "--format", "-f", help="Output format: json | text.")


def register(app: typer.Typer) -> None:
    kb_app = typer.Typer(help="Create, list, ingest, and search knowledge bases.")
    question_app = typer.Typer(help="Save and list assistant-authored questions.")
    attempt_app = typer.Typer(help="Record a learner attempt.")
    mastery_app = typer.Typer(help="Read mastery and the next review.")
    memory_app = typer.Typer(help="Read and write learner memory notes.")
    task_app = typer.Typer(help="Read and write the agent task board.")
    app.add_typer(kb_app, name="kb")
    app.add_typer(question_app, name="question")
    app.add_typer(attempt_app, name="attempt")
    app.add_typer(mastery_app, name="mastery")
    app.add_typer(memory_app, name="memory")
    app.add_typer(task_app, name="task")

    @app.callback()
    def _root(
        ctx: typer.Context,
        data_dir: Path | None = typer.Option(
            None,
            "--data-dir",
            help="Study data directory. Overrides DEEPTUTOR_STUDY_DIR for this command.",
        ),
    ) -> None:
        ctx.obj = {"data_dir": data_dir}

    @app.command("status")
    def status(ctx: typer.Context, fmt: str = _format_option()) -> None:
        """Show the data directory and local embedding backend."""
        _run(ctx, fmt, lambda service: service.status())

    @app.command("mcp")
    def mcp(ctx: typer.Context) -> None:
        """Serve the study tools over MCP on stdin/stdout."""
        serve_stdio(_service(ctx))

    @kb_app.command("create")
    def kb_create(
        ctx: typer.Context,
        name: str = typer.Argument(..., help="Knowledge base name."),
        description: str = typer.Option(
            "", "--description", help="What the knowledge base is for."
        ),
        fmt: str = _format_option(),
    ) -> None:
        """Create an empty knowledge base."""
        _run(ctx, fmt, lambda service: service.kb_create(name, description))

    @kb_app.command("list")
    def kb_list(ctx: typer.Context, fmt: str = _format_option()) -> None:
        """List knowledge bases."""
        _run(ctx, fmt, lambda service: service.kb_list())

    @kb_app.command("ingest")
    def kb_ingest(
        ctx: typer.Context,
        name: str = typer.Argument(..., help="Knowledge base name."),
        paths: list[Path] = typer.Argument(..., help="Files or folders to index."),
        fmt: str = _format_option(),
    ) -> None:
        """Index files or a folder into a knowledge base."""
        _run(ctx, fmt, lambda service: service.kb_ingest(name, [str(path) for path in paths]))

    @kb_app.command("retrieve")
    def kb_retrieve(
        ctx: typer.Context,
        name: str = typer.Argument(..., help="Knowledge base name."),
        query: str = typer.Argument(..., help="Search query."),
        k: int = typer.Option(5, "--k", help="Number of passages."),
        fmt: str = _format_option(),
    ) -> None:
        """Retrieve cited passages."""
        _run(ctx, fmt, lambda service: service.kb_retrieve(name, query, k))

    @question_app.command("save")
    def question_save(
        ctx: typer.Context,
        question: str = typer.Option(..., "--question", help="Question text."),
        answer: str = typer.Option(..., "--answer", help="Answer key."),
        bank: str = typer.Option("default", "--bank", help="Question bank name."),
        topic: str = typer.Option("general", "--topic", help="Mastery topic."),
        explanation: str = typer.Option("", "--explanation", help="Shown after grading."),
        tag: list[str] = typer.Option([], "--tag", help="Tag. Repeat for several."),
        sources_json: str = typer.Option(
            "", "--sources-json", help="JSON list of source citations."
        ),
        question_id: str | None = typer.Option(None, "--id", help="Update this question."),
        fmt: str = _format_option(),
    ) -> None:
        """Save an assistant-authored question."""

        def _save(service: StudyService) -> dict:
            sources = json.loads(sources_json) if sources_json else None
            return service.question_save(
                question=question,
                answer=answer,
                bank=bank,
                topic=topic,
                explanation=explanation,
                tags=tag,
                sources=sources,
                question_id=question_id,
            )

        _run(ctx, fmt, _save)

    @question_app.command("list")
    def question_list(
        ctx: typer.Context,
        bank: str | None = typer.Option(None, "--bank"),
        topic: str | None = typer.Option(None, "--topic"),
        tag: str | None = typer.Option(None, "--tag"),
        limit: int = typer.Option(50, "--limit"),
        include_answers: bool = typer.Option(True, "--answers/--no-answers"),
        fmt: str = _format_option(),
    ) -> None:
        """List saved questions."""
        _run(
            ctx,
            fmt,
            lambda service: service.question_list(
                bank=bank,
                topic=topic,
                tag=tag,
                limit=limit,
                include_answers=include_answers,
            ),
        )

    @attempt_app.command("record")
    def attempt_record(
        ctx: typer.Context,
        question_id: str = typer.Argument(..., help="Question id."),
        learner_answer: str = typer.Option(..., "--answer", help="The learner's answer."),
        correct: bool = typer.Option(False, "--correct", help="Mark the attempt correct."),
        incorrect: bool = typer.Option(False, "--incorrect", help="Mark the attempt incorrect."),
        score: float | None = typer.Option(None, "--score", help="Score from 0 to 1."),
        feedback: str = typer.Option("", "--feedback"),
        fmt: str = _format_option(),
    ) -> None:
        """Record an attempt and update mastery."""
        if correct and incorrect:
            _emit(
                {
                    "ok": False,
                    "error": "Pass only one of --correct and --incorrect.",
                    "code": "invalid",
                },
                fmt,
            )
            raise typer.Exit(1)
        judged: bool | None = True if correct else False if incorrect else None

        def _record(service: StudyService) -> dict:
            return service.attempt_record(
                question_id,
                learner_answer,
                correct=judged,
                score=score,
                feedback=feedback,
            )

        _run(ctx, fmt, _record)

    @mastery_app.command("status")
    def mastery_status(
        ctx: typer.Context,
        topic: str | None = typer.Option(None, "--topic"),
        fmt: str = _format_option(),
    ) -> None:
        """Show mastery and review schedule."""
        _run(ctx, fmt, lambda service: service.mastery_status(topic))

    @mastery_app.command("next")
    def mastery_next(
        ctx: typer.Context,
        limit: int = typer.Option(3, "--limit"),
        fmt: str = _format_option(),
    ) -> None:
        """Suggest the next topics and questions to review."""
        _run(ctx, fmt, lambda service: service.mastery_next(limit))

    @memory_app.command("list")
    def memory_list(ctx: typer.Context, fmt: str = _format_option()) -> None:
        """List memory notes."""
        _run(ctx, fmt, lambda service: service.memory_list())

    @memory_app.command("read")
    def memory_read(
        ctx: typer.Context,
        note_id: str = typer.Argument(..., help="Note id."),
        fmt: str = _format_option(),
    ) -> None:
        """Read one memory note."""
        _run(ctx, fmt, lambda service: service.memory_read(note_id))

    @memory_app.command("write")
    def memory_write(
        ctx: typer.Context,
        title: str = typer.Option(..., "--title"),
        body: str = typer.Option(..., "--body"),
        note_id: str | None = typer.Option(None, "--id"),
        tag: list[str] = typer.Option([], "--tag"),
        fmt: str = _format_option(),
    ) -> None:
        """Create or replace a memory note."""
        _run(
            ctx,
            fmt,
            lambda service: service.memory_write(title=title, body=body, note_id=note_id, tags=tag),
        )

    @memory_app.command("delete")
    def memory_delete(
        ctx: typer.Context,
        note_id: str = typer.Argument(...),
        fmt: str = _format_option(),
    ) -> None:
        """Delete a memory note."""
        _run(ctx, fmt, lambda service: service.memory_delete(note_id))

    @task_app.command("list")
    def task_list(
        ctx: typer.Context,
        status: str | None = typer.Option(None, "--status"),
        include_archived: bool = typer.Option(False, "--archived"),
        fmt: str = _format_option(),
    ) -> None:
        """List task cards."""
        _run(
            ctx,
            fmt,
            lambda service: service.task_list(status=status, include_archived=include_archived),
        )

    @task_app.command("write")
    def task_write(
        ctx: typer.Context,
        title: str | None = typer.Option(None, "--title"),
        note: str | None = typer.Option(None, "--note"),
        status: str | None = typer.Option(None, "--status"),
        task_id: str | None = typer.Option(None, "--id"),
        archived: bool | None = typer.Option(None, "--archived/--no-archived"),
        fmt: str = _format_option(),
    ) -> None:
        """Create a task card, or update one with --id."""
        _run(
            ctx,
            fmt,
            lambda service: service.task_write(
                title=title,
                note=note,
                status=status,
                task_id=task_id,
                archived=archived,
            ),
        )


def _service(ctx: typer.Context) -> StudyService:
    data_dir = None if ctx.obj is None else ctx.obj.get("data_dir")
    return StudyService(data_dir)


def _run(ctx: typer.Context, fmt: str, action) -> None:
    try:
        payload = action(_service(ctx))
    except StudyError as exc:
        _emit({"ok": False, "error": str(exc), "code": exc.code}, fmt)
        raise typer.Exit(code=1) from exc
    except json.JSONDecodeError as exc:
        _emit({"ok": False, "error": f"Invalid JSON: {exc}", "code": "invalid"}, fmt)
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        _emit({"ok": False, "error": str(exc), "code": "internal"}, fmt)
        raise typer.Exit(code=1) from exc
    _emit(payload, fmt)
    if not payload.get("ok", True):
        raise typer.Exit(code=1)


def _emit(payload: dict, fmt: str) -> None:
    if fmt == "text":
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return
    typer.echo(json.dumps(payload, ensure_ascii=False, default=str))
