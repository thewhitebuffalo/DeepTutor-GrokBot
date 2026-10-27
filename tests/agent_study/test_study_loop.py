"""Question bank, attempts, mastery, memory, and tasks without an LLM."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from deeptutor.agent_study.embed import HashingEmbedder
from deeptutor.agent_study.service import StudyService
from deeptutor.agent_study.tools import TOOLS, call_tool


def _service(tmp_path, monkeypatch) -> StudyService:
    monkeypatch.setenv("DEEPTUTOR_EMBED_BACKEND", "hash")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return StudyService(tmp_path / "study", embedder=HashingEmbedder())


def test_study_loop_saves_grades_and_schedules(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    notes = tmp_path / "note.md"
    notes.write_text(
        "# Creepage\n\nCreepage distance depends on pollution degree.\n", encoding="utf-8"
    )
    service.kb_create("pe")
    service.kb_ingest("pe", [str(notes)])
    hit = service.kb_retrieve("pe", "creepage pollution degree")["passages"][0]

    saved = service.question_save(
        question="What does creepage distance depend on?",
        answer="pollution degree",
        bank="pe-bank",
        topic="Creepage",
        explanation="Pollution degree sets the required creepage.",
        tags=["ieee", "insulation"],
        sources=[
            {
                "file": hit["file"],
                "page": hit["page"],
                "section": hit["section"],
                "kb": hit["kb"],
                "chunk_id": hit["chunk_id"],
                "quote": hit["text"][:120],
            }
        ],
    )
    question_id = saved["question"]["id"]
    listed = service.question_list(bank="pe-bank", tag="ieee")
    assert listed["questions"][0]["id"] == question_id
    assert listed["questions"][0]["sources"][0]["section"] == "Creepage"

    hidden = service.question_list(include_answers=False)
    assert "answer" not in hidden["questions"][0]

    wrong = service.attempt_record(question_id, "conductor size")
    assert wrong["attempt"]["correct"] is False
    assert wrong["topic"]["reason"] == "failed"
    assert wrong["attempt"]["explanation"].startswith("Pollution")

    nxt = service.mastery_next(limit=3)
    assert nxt["suggestions"][0]["topic"] == "Creepage"
    assert "answer" not in nxt["suggestions"][0]["question"]

    right = service.attempt_record(question_id, "It depends on the pollution degree.")
    assert right["attempt"]["correct"] is True
    assert right["topic"]["interval_days"] == 1.0
    status = service.mastery_status(topic="Creepage")
    assert status["topics"][0]["attempt_count"] == 2
    assert status["topics"][0]["due"] is False
    assert service.mastery_next()["suggestions"] == []

    note = service.memory_write(
        title="Creepage",
        body="Ask about pollution degree before clearance.",
        tags=["pe"],
    )
    note_id = note["note"]["id"]
    assert "pollution" in service.memory_read(note_id)["note"]["body"]
    assert service.memory_list()["notes"][0]["id"] == note_id
    assert service.memory_delete(note_id)["deleted"] == note_id

    card = service.task_write(title="Read IEEE 43 section 5", note="Before the next quiz")
    task_id = card["task"]["id"]
    service.task_write(task_id=task_id, status="doing")
    tasks = service.task_list(status="doing")
    assert tasks["tasks"][0]["title"] == "Read IEEE 43 section 5"
    service.task_write(task_id=task_id, archived=True)
    assert service.task_list()["tasks"] == []
    assert service.task_list(include_archived=True)["tasks"][0]["archived"] is True


def test_tool_dispatch_rejects_unknown_arguments(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    names = [tool["name"] for tool in TOOLS]
    assert "kb_retrieve" in names
    assert "attempt_record" in names
    with pytest.raises(Exception, match="Unknown arguments"):
        call_tool(service, "kb_create", {"name": "ok", "extra": 1})


def test_mcp_stdio_round_trip(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    notes = tmp_path / "lesson.md"
    notes.write_text("# Torque\n\nTorque is force times the moment arm.\n", encoding="utf-8")
    env = os.environ.copy()
    env["DEEPTUTOR_STUDY_DIR"] = str(tmp_path / "study")
    env["DEEPTUTOR_EMBED_BACKEND"] = "hash"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path(__file__).resolve().parents[2]), env.get("PYTHONPATH", "")]
    )
    env.pop("OPENAI_API_KEY", None)
    process = subprocess.Popen(
        [sys.executable, "-m", "deeptutor.agent_study"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=tmp_path,
    )
    assert process.stdin is not None
    assert process.stdout is not None

    def send(message: dict, *, expect: bool = True) -> dict | None:
        process.stdin.write((json.dumps(message) + "\n").encode())
        process.stdin.flush()
        if not expect:
            return None
        line = process.stdout.readline()
        assert line, process.stderr.read().decode() if process.stderr else "no MCP response"
        return json.loads(line)

    try:
        initialized = send(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2024-11-05", "capabilities": {}},
            }
        )
        assert initialized is not None
        assert initialized["result"]["serverInfo"]["name"] == "deeptutor-study"
        send({"jsonrpc": "2.0", "method": "notifications/initialized"}, expect=False)
        listed = send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        assert listed is not None
        tool_names = {tool["name"] for tool in listed["result"]["tools"]}
        assert {
            "kb_ingest",
            "kb_retrieve",
            "question_save",
            "mastery_next",
            "task_write",
        } <= tool_names

        created = _call(send, 3, "kb_create", {"name": "machines"})
        assert created["embed_backend"] == "hash"
        ingested = _call(send, 4, "kb_ingest", {"name": "machines", "paths": [str(notes)]})
        assert ingested["ok"] is True
        retrieved = _call(
            send,
            5,
            "kb_retrieve",
            {"name": "machines", "query": "torque moment arm"},
        )
        assert "moment arm" in retrieved["passages"][0]["text"]
        assert retrieved["passages"][0]["section"] == "Torque"
    finally:
        process.stdin.close()
        process.wait(timeout=10)
        if process.returncode not in (0, None):
            stderr = process.stderr.read().decode() if process.stderr else ""
            raise AssertionError(stderr)


def _call(send, msg_id: int, name: str, arguments: dict) -> dict:
    response = send(
        {
            "jsonrpc": "2.0",
            "id": msg_id,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
    )
    assert response is not None
    assert response["result"]["isError"] is False, response
    return response["result"]["structuredContent"]
