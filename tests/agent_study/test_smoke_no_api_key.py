"""Ingest a PDF and a Markdown note and retrieve cited passages with no API keys."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from deeptutor.agent_study.service import StudyService

_CLOUD_KEYS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "DASHSCOPE_API_KEY",
    "COHERE_API_KEY",
    "JINA_API_KEY",
    "DEEPSEEK_API_KEY",
    "MISTRAL_API_KEY",
    "GROQ_API_KEY",
    "TOGETHER_API_KEY",
    "PERPLEXITY_API_KEY",
)


def _strip_cloud_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name in _CLOUD_KEYS or name.endswith("_API_KEY") or name.endswith("_API_TOKEN"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DEEPTUTOR_EMBED_BACKEND", "hash")
    monkeypatch.delenv("DEEPTUTOR_EMBED_MODEL", raising=False)


def _write_pdf(path: Path, text: str) -> None:
    """Write a one-page PDF using only the standard library."""

    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET\n"
    objects = [
        "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n",
        "2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n",
        "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        "/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\n",
        f"4 0 obj << /Length {len(stream.encode('latin-1'))} >> stream\n{stream}endstream\nendobj\n",
        "5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n",
    ]
    header = "%PDF-1.4\n"
    offsets = [0]
    cursor = len(header.encode("latin-1"))
    body = ""
    for obj in objects:
        offsets.append(cursor)
        body += obj
        cursor += len(obj.encode("latin-1"))
    xref = ["xref\n", "0 6\n", "0000000000 65535 f \n"]
    xref.extend(f"{offset:010d} 00000 n \n" for offset in offsets[1:])
    trailer = f"trailer << /Size 6 /Root 1 0 R >>\nstartxref\n{cursor}\n%%EOF\n"
    path.write_bytes((header + body + "".join(xref) + trailer).encode("latin-1"))


def test_smoke_ingests_pdf_and_markdown_and_cites_the_passage(tmp_path, monkeypatch):
    pytest.importorskip("fitz")
    _strip_cloud_keys(monkeypatch)
    assert "OPENAI_API_KEY" not in os.environ

    pdf = tmp_path / "ieee-43.pdf"
    _write_pdf(
        pdf,
        "The insulation resistance of a field winding measured per IEEE Std 43 "
        "shall be at least one megaohm plus one megaohm per kilovolt of machine rating.",
    )
    notes = tmp_path / "mba.md"
    notes.write_text(
        "# MBA coursework\n\n"
        "## Weighted average cost of capital\n\n"
        "WACC is the weighted average cost of capital used to discount free cash flow.\n",
        encoding="utf-8",
    )

    service = StudyService(tmp_path / "study")
    created = service.kb_create("pe-mba", "PE standards and MBA notes")
    assert created["embed_backend"] == "hash"
    assert created["embed_model"] == "hashing"

    ingested = service.kb_ingest("pe-mba", [str(pdf), str(notes)])
    assert ingested["ok"] is True, ingested
    assert ingested["failed"] == []
    assert ingested["chunks"] >= 2

    standard = service.kb_retrieve(
        "pe-mba",
        "IEEE Std 43 insulation resistance megaohm per kilovolt",
        k=3,
    )
    top = standard["passages"][0]
    assert "megaohm" in top["text"]
    assert top["file"] == "ieee-43.pdf"
    assert top["page"] == 1
    assert top["kb"] == "pe-mba"
    assert Path(top["path"]).is_file()

    finance = service.kb_retrieve("pe-mba", "weighted average cost of capital free cash flow")
    finance_top = finance["passages"][0]
    assert finance_top["file"] == "mba.md"
    assert finance_top["section"] == "Weighted average cost of capital"
    assert finance_top["page"] is None

    again = service.kb_ingest("pe-mba", [str(notes)])
    assert "mba.md" in again["unchanged"]


def test_cli_smoke_matches_the_service(tmp_path, monkeypatch):
    pytest.importorskip("fitz")
    pytest.importorskip("typer")
    from typer.testing import CliRunner

    from deeptutor_cli.study import register

    _strip_cloud_keys(monkeypatch)
    notes = tmp_path / "standards.md"
    notes.write_text(
        "# Insulation\n\nIEEE Std 43 calls for one megaohm per kilovolt.\n",
        encoding="utf-8",
    )
    app = __import__("typer").Typer()
    register(app)
    runner = CliRunner()
    data = str(tmp_path / "study")
    created = runner.invoke(app, ["--data-dir", data, "kb", "create", "std"])
    assert created.exit_code == 0, created.stdout
    assert json.loads(created.stdout)["embed_backend"] == "hash"

    ingested = runner.invoke(app, ["--data-dir", data, "kb", "ingest", "std", str(notes)])
    assert ingested.exit_code == 0, ingested.stdout
    found = runner.invoke(
        app,
        ["--data-dir", data, "kb", "retrieve", "std", "IEEE Std 43 megaohm per kilovolt"],
    )
    assert found.exit_code == 0, found.stdout
    payload = json.loads(found.stdout)
    assert payload["passages"][0]["section"] == "Insulation"
    assert "megaohm" in payload["passages"][0]["text"]
    assert "OPENAI_API_KEY" not in os.environ
