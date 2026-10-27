"""Stdio MCP server for the headless study backend.

The protocol is newline-delimited JSON-RPC. Logs go to stderr so a client
reading stdout only sees MCP messages.

Run ``python -m deeptutor.agent_study`` or ``deeptutor study mcp``.
"""

from __future__ import annotations

import json
import logging
import sys

from deeptutor.agent_study.errors import StudyError
from deeptutor.agent_study.service import StudyService
from deeptutor.agent_study.tools import TOOLS, call_tool

logger = logging.getLogger(__name__)
_PROTOCOL = "2024-11-05"
_SERVER = "deeptutor-study"


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(message)s")
    serve_stdio()


def serve_stdio(service: StudyService | None = None) -> None:
    """Read MCP messages from stdin and write responses to stdout."""

    study = service or StudyService()
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            _write(
                {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}
            )
            continue
        response = handle_message(message, study)
        if response is not None:
            _write(response)


def handle_message(message: dict, service: StudyService) -> dict | None:
    """Return a JSON-RPC response, or None when the message is a notification."""

    if not isinstance(message, dict):
        return _error(None, -32600, "Invalid request")
    method = message.get("method")
    if not isinstance(method, str) or not method:
        if "id" not in message:
            return None
        return _error(message.get("id"), -32600, "Missing method")
    if "id" not in message:
        return None
    msg_id = message.get("id")
    params = message.get("params") or {}
    if params is None:
        params = {}
    if not isinstance(params, dict):
        return _error(msg_id, -32602, "Params must be an object")
    try:
        result = dispatch(method, params, service)
    except StudyError as exc:
        return _error(msg_id, -32000, str(exc))
    except Exception as exc:
        logger.exception("MCP method %s failed", method)
        return _error(msg_id, -32603, str(exc))
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def dispatch(method: str, params: dict, service: StudyService) -> dict:
    if method == "initialize":
        return {
            "protocolVersion": _PROTOCOL,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": _server_info(),
            "instructions": (
                "DeepTutor study tools store documents, questions, attempts, mastery, "
                "memory notes, and tasks. You write the teaching and the questions. "
                "These tools do not call an LLM and do not need an API key."
            ),
        }
    if method == "ping":
        return {}
    if method == "tools/list":
        return {"tools": TOOLS}
    if method == "tools/call":
        return _tools_call(params, service)
    raise StudyError(f"Method not found: {method}", code="method_not_found")


def _tools_call(params: dict, service: StudyService) -> dict:
    name = params.get("name")
    arguments = params.get("arguments") or {}
    if not isinstance(name, str) or not name:
        raise StudyError("tools/call requires a tool name.")
    if not isinstance(arguments, dict):
        raise StudyError("Tool arguments must be an object.")
    try:
        payload = call_tool(service, name, arguments)
    except StudyError as exc:
        return _tool_result({"ok": False, "error": str(exc), "code": exc.code}, is_error=True)
    return _tool_result(payload, is_error=not payload.get("ok", True))


def _tool_result(payload: dict, *, is_error: bool) -> dict:
    return {
        "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False, default=str)}],
        "isError": is_error,
        "structuredContent": payload,
    }


def _server_info() -> dict:
    try:
        from deeptutor.__version__ import __version__
    except Exception:
        __version__ = "0"
    return {"name": _SERVER, "version": __version__}


def _error(msg_id, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def _write(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    sys.stdout.flush()
