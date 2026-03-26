from __future__ import annotations

import json
import os
from pathlib import Path
import threading
from datetime import datetime, timezone
from collections.abc import Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit


SAFE_TEXT_FIELDS = {
    "action",
    "actor",
    "approval_id",
    "category",
    "child_id",
    "domain_hint",
    "kind",
    "model",
    "nick",
    "outcome",
    "owner",
    "path",
    "ref",
    "repo",
    "requested_by",
    "requested_in",
    "scope",
    "subject",
    "target",
    "tool_call_id",
    "tool_name",
}
SAFE_LIST_FIELDS = {"tool_names"}
SENSITIVE_FIELD_TOKENS = (
    "authorization",
    "cookie",
    "credential",
    "key",
    "password",
    "secret",
    "session",
    "token",
)
URL_FIELDS = {"preferred_direct_url", "url"}
MAX_SUMMARY_ITEMS = 5
MAX_SUMMARY_DEPTH = 4


class AuditLogger:
    def __init__(self, log_path: str | Path) -> None:
        self._log_path = Path(log_path)
        self._lock = threading.Lock()

    @property
    def log_path(self) -> Path:
        return self._log_path

    def log_approval_request(
        self,
        *,
        approval_id: str,
        tool_name: str,
        arguments: Mapping[str, object],
        requested_by: str,
        requested_in: str,
        summary: str | None = None,
        created_at: float | None = None,
        expires_at: float | None = None,
    ) -> dict[str, object]:
        return self._append(
            "approval_requested",
            {
                "approval_id": self._require_text(approval_id, "approval_id"),
                "tool_name": self._require_text(tool_name, "tool_name"),
                "arguments": dict(arguments),
                "requested_by": self._require_text(requested_by, "requested_by"),
                "requested_in": self._require_text(requested_in, "requested_in"),
                "summary": self._clean_optional_text(summary),
                "created_at": created_at,
                "expires_at": expires_at,
            },
        )

    def log_approval(
        self,
        *,
        approval_id: str,
        actor: str,
        tool_name: str,
        summary: str | None = None,
    ) -> dict[str, object]:
        return self._append(
            "approval_granted",
            {
                "approval_id": self._require_text(approval_id, "approval_id"),
                "actor": self._require_text(actor, "actor"),
                "tool_name": self._require_text(tool_name, "tool_name"),
                "summary": self._clean_optional_text(summary),
            },
        )

    def log_rejection(
        self,
        *,
        approval_id: str,
        actor: str,
        tool_name: str,
        summary: str | None = None,
    ) -> dict[str, object]:
        return self._append(
            "approval_rejected",
            {
                "approval_id": self._require_text(approval_id, "approval_id"),
                "actor": self._require_text(actor, "actor"),
                "tool_name": self._require_text(tool_name, "tool_name"),
                "summary": self._clean_optional_text(summary),
            },
        )

    def log_dangerous_action_result(
        self,
        *,
        tool_name: str,
        arguments: Mapping[str, object],
        ok: bool,
        approval_id: str | None = None,
        actor: str | None = None,
        summary: str | None = None,
        result: object | None = None,
        error: str | None = None,
    ) -> dict[str, object]:
        return self._append(
            "dangerous_action_result",
            {
                "approval_id": self._clean_optional_text(approval_id),
                "actor": self._clean_optional_text(actor),
                "tool_name": self._require_text(tool_name, "tool_name"),
                "arguments": dict(arguments),
                "summary": self._clean_optional_text(summary),
                "ok": bool(ok),
                "result": result,
                "error": self._clean_optional_text(error),
            },
        )

    def log_request_start(
        self,
        *,
        request_id: str,
        nick: str,
        target: str,
        is_private: bool,
        prompt: str,
        github_scope: str | None = None,
        domain_hint: str | None = None,
        preferred_direct_url: str | None = None,
        requires_web_lookup: bool | None = None,
    ) -> dict[str, object]:
        return self._append(
            "request_start",
            {
                "request_id": self._require_text(request_id, "request_id"),
                "nick": self._require_text(nick, "nick"),
                "target": self._require_text(target, "target"),
                "is_private": bool(is_private),
                "prompt_len": len(self._require_text(prompt, "prompt")),
                "github_scope": self._clean_optional_text(github_scope),
                "domain_hint": self._clean_optional_text(domain_hint),
                "preferred_direct_url": self._clean_optional_url(preferred_direct_url),
                "requires_web_lookup": requires_web_lookup,
            },
        )

    def log_request_tool_call(
        self,
        *,
        request_id: str,
        tool_name: str,
        arguments: Mapping[str, object],
        tool_call_id: str | None = None,
        category: str | None = None,
        round_index: int | None = None,
    ) -> dict[str, object]:
        return self._append(
            "request_tool_call",
            {
                "request_id": self._require_text(request_id, "request_id"),
                "tool_name": self._require_text(tool_name, "tool_name"),
                "tool_call_id": self._clean_optional_text(tool_call_id),
                "category": self._clean_optional_text(category),
                "round_index": round_index,
                "arguments_summary": self._summarize_mapping(arguments),
            },
        )

    def log_request_tool_result(
        self,
        *,
        request_id: str,
        tool_name: str,
        ok: bool,
        tool_call_id: str | None = None,
        category: str | None = None,
        round_index: int | None = None,
        approval_required: bool | None = None,
        duration_ms: int | None = None,
        result: object | None = None,
        error: object | None = None,
    ) -> dict[str, object]:
        return self._append(
            "request_tool_result",
            {
                "request_id": self._require_text(request_id, "request_id"),
                "tool_name": self._require_text(tool_name, "tool_name"),
                "tool_call_id": self._clean_optional_text(tool_call_id),
                "category": self._clean_optional_text(category),
                "round_index": round_index,
                "ok": bool(ok),
                "approval_required": approval_required,
                "duration_ms": duration_ms,
                "result_summary": self._summarize_value(result, field_name="result") if result is not None else None,
                "error_summary": self._summarize_value(error, field_name="error") if error is not None else None,
            },
        )

    def log_request_finish(
        self,
        *,
        request_id: str,
        outcome: str,
        rounds: int | None = None,
        tools_used: int | None = None,
        tool_names: Sequence[str] | None = None,
        response: str | None = None,
        error: object | None = None,
    ) -> dict[str, object]:
        return self._append(
            "request_finish",
            {
                "request_id": self._require_text(request_id, "request_id"),
                "outcome": self._require_text(outcome, "outcome"),
                "rounds": rounds,
                "tools_used": tools_used,
                "tool_names": self._clean_text_sequence(tool_names),
                "response_len": len(response) if response is not None else None,
                "error_summary": self._summarize_value(error, field_name="error") if error is not None else None,
            },
        )

    def log_child_bot_event(
        self,
        *,
        child_id: str,
        action: str,
        status: str,
        nick: str | None = None,
        channels: Sequence[str] | None = None,
        model: str | None = None,
        pid: int | None = None,
        exit_code: int | None = None,
        error: str | None = None,
    ) -> dict[str, object]:
        return self._append(
            "child_bot_event",
            {
                "child_id": self._require_text(child_id, "child_id"),
                "action": self._require_text(action, "action"),
                "status": self._require_text(status, "status"),
                "nick": self._clean_optional_text(nick),
                "channels": self._clean_text_sequence(channels),
                "model": self._clean_optional_text(model),
                "pid": pid,
                "exit_code": exit_code,
                "error": self._clean_optional_text(error),
            },
        )

    def _append(self, event: str, payload: Mapping[str, object]) -> dict[str, object]:
        record: dict[str, object] = {
            "event": self._require_text(event, "event"),
            "timestamp": self._timestamp(),
        }
        for key, value in payload.items():
            if value is not None:
                record[key] = value

        encoded = (json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")

        with self._lock:
            self._ensure_parent_directory()
            flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
            file_descriptor = os.open(self._log_path, flags, 0o600)
            try:
                os.write(file_descriptor, encoded)
                os.fsync(file_descriptor)
            finally:
                os.close(file_descriptor)

        return record

    def _ensure_parent_directory(self) -> None:
        parent = self._log_path.parent
        if parent.exists():
            if not parent.is_dir():
                raise NotADirectoryError(f"audit log parent is not a directory: {parent}")
        else:
            parent.mkdir(parents=True, exist_ok=True)

        if self._log_path.exists() and self._log_path.is_dir():
            raise IsADirectoryError(f"audit log path is a directory: {self._log_path}")

    def _timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    def _summarize_mapping(self, value: Mapping[str, object], depth: int = 0) -> dict[str, object]:
        summary: dict[str, object] = {}
        for key, item in value.items():
            summary[str(key)] = self._summarize_value(item, field_name=str(key), depth=depth + 1)
        return summary

    def _summarize_value(self, value: object, field_name: str | None = None, depth: int = 0) -> object:
        lowered = field_name.lower() if field_name is not None else ""
        if lowered and any(token in lowered for token in SENSITIVE_FIELD_TOKENS):
            return "<redacted>"
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            if lowered in URL_FIELDS:
                return self._sanitize_url(value)
            if lowered in SAFE_TEXT_FIELDS:
                return value.strip()
            return {"type": "text", "length": len(value)}
        if isinstance(value, Mapping):
            if depth >= MAX_SUMMARY_DEPTH:
                return {"type": "mapping", "length": len(value)}
            return self._summarize_mapping(value, depth)
        if isinstance(value, (bytes, bytearray)):
            return {"type": type(value).__name__, "length": len(value)}
        if isinstance(value, Sequence) and not isinstance(value, str):
            if lowered in SAFE_LIST_FIELDS:
                return self._clean_text_sequence(value)
            items = [self._summarize_value(item, depth=depth + 1) for item in value[:MAX_SUMMARY_ITEMS]]
            return {"type": "list", "length": len(value), "items": items}
        if isinstance(value, (set, frozenset)):
            items = [self._summarize_value(item, depth=depth + 1) for item in list(value)[:MAX_SUMMARY_ITEMS]]
            return {"type": "set", "length": len(value), "items": items}
        return {"type": type(value).__name__}

    def _clean_text_sequence(self, value: Sequence[str] | None) -> list[str] | None:
        if value is None:
            return None
        cleaned: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                cleaned.append(text)
        return cleaned

    def _sanitize_url(self, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            return stripped
        split = urlsplit(stripped)
        hostname = split.hostname
        if split.scheme and hostname:
            netloc = hostname
            if split.port is not None:
                netloc = f"{netloc}:{split.port}"
            return urlunsplit((split.scheme, netloc, split.path, "", ""))
        return stripped.split("?", 1)[0].split("#", 1)[0]

    def _require_text(self, value: str, field_name: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError(f"{field_name} must not be blank")
        return cleaned

    def _clean_optional_text(self, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    def _clean_optional_url(self, value: str | None) -> str | None:
        cleaned = self._clean_optional_text(value)
        if cleaned is None:
            return None
        return self._sanitize_url(cleaned)
