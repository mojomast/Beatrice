from __future__ import annotations

import json
import os
from pathlib import Path
import threading
from datetime import datetime, timezone
from collections.abc import Mapping


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
