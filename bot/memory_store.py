from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
import sqlite3


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    id: int
    scope: str
    kind: str
    subject: str | None
    content: str
    created_at: str


class MemoryStore:
    def __init__(self, database_path: str | Path) -> None:
        self._database_path = Path(database_path)
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def initialize(self) -> None:
        if self._initialized:
            return

        async with self._init_lock:
            if self._initialized:
                return
            await asyncio.to_thread(self._initialize_sync)
            self._initialized = True

    async def store_memory(
        self,
        scope: str,
        content: str,
        *,
        kind: str = "note",
        subject: str | None = None,
    ) -> MemoryRecord:
        await self.initialize()
        normalized_scope = self._require_text(scope, "scope")
        normalized_content = self._require_text(content, "content")
        normalized_kind = self._normalize_kind(kind)
        normalized_subject = self._normalize_optional_text(subject, "subject")
        return await asyncio.to_thread(
            self._store_memory_sync,
            normalized_scope,
            normalized_content,
            normalized_kind,
            normalized_subject,
        )

    async def search_recent_memories(
        self,
        scope: str,
        query: str | None = None,
        limit: int = 8,
        *,
        kind: str | None = None,
        subject: str | None = None,
    ) -> list[MemoryRecord]:
        await self.initialize()
        normalized_scope = self._require_text(scope, "scope")
        if limit < 1:
            raise ValueError("limit must be at least 1")

        normalized_query = query.strip() if query is not None else None
        if normalized_query == "":
            normalized_query = None
        normalized_kind = self._normalize_kind(kind) if kind is not None else None
        normalized_subject = self._normalize_optional_text(subject, "subject")
        return await asyncio.to_thread(
            self._search_recent_memories_sync,
            normalized_scope,
            normalized_query,
            limit,
            normalized_kind,
            normalized_subject,
        )

    async def get_summary(self, scope: str) -> str | None:
        await self.initialize()
        normalized_scope = self._require_text(scope, "scope")
        return await asyncio.to_thread(self._get_summary_sync, normalized_scope)

    async def update_summary(self, scope: str, summary: str | None) -> None:
        await self.initialize()
        normalized_scope = self._require_text(scope, "scope")
        normalized_summary = summary.strip() if summary is not None else None
        if normalized_summary == "":
            normalized_summary = None
        await asyncio.to_thread(self._update_summary_sync, normalized_scope, normalized_summary)

    async def get_profile(self, scope: str, subject: str) -> str | None:
        await self.initialize()
        normalized_scope = self._require_text(scope, "scope")
        normalized_subject = self._require_text(subject, "subject")
        return await asyncio.to_thread(self._get_profile_sync, normalized_scope, normalized_subject)

    async def update_profile(self, scope: str, subject: str, profile: str | None) -> None:
        await self.initialize()
        normalized_scope = self._require_text(scope, "scope")
        normalized_subject = self._require_text(subject, "subject")
        normalized_profile = profile.strip() if profile is not None else None
        if normalized_profile == "":
            normalized_profile = None
        await asyncio.to_thread(self._update_profile_sync, normalized_scope, normalized_subject, normalized_profile)

    def _initialize_sync(self) -> None:
        with self._connect() as connection:
            self._ensure_memories_table(connection)
            self._ensure_summaries_table(connection)
            self._ensure_profiles_table(connection)
            connection.commit()

    def _store_memory_sync(
        self,
        scope: str,
        content: str,
        kind: str,
        subject: str | None,
    ) -> MemoryRecord:
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO memories (scope, kind, subject, content) VALUES (?, ?, ?, ?)",
                (scope, kind, subject, content),
            )
            row = connection.execute(
                "SELECT id, scope, kind, subject, content, created_at FROM memories WHERE id = ?",
                (cursor.lastrowid,),
            ).fetchone()
            connection.commit()
        return self._memory_from_row(row)

    def _search_recent_memories_sync(
        self,
        scope: str,
        query: str | None,
        limit: int,
        kind: str | None,
        subject: str | None,
    ) -> list[MemoryRecord]:
        sql = "SELECT id, scope, kind, subject, content, created_at FROM memories WHERE scope = ?"
        parameters: list[object] = [scope]
        if kind is not None:
            sql += " AND kind = ?"
            parameters.append(kind)
        if subject is not None:
            sql += " AND subject = ?"
            parameters.append(subject)
        if query is not None:
            sql += " AND content LIKE ?"
            parameters.append(f"%{query}%")
        sql += " ORDER BY id DESC LIMIT ?"
        parameters.append(limit)

        with self._connect() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        return [self._memory_from_row(row) for row in rows]

    def _get_summary_sync(self, scope: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT summary FROM summaries WHERE scope = ?",
                (scope,),
            ).fetchone()
            if row is None:
                row = connection.execute(
                    """
                    SELECT content AS summary FROM memories
                    WHERE scope = ? AND kind = 'summary' AND subject IS NULL
                    ORDER BY id DESC
                    LIMIT 1
                    """,
                    (scope,),
                ).fetchone()
        if row is None:
            return None
        return str(row["summary"])

    def _update_summary_sync(self, scope: str, summary: str | None) -> None:
        with self._connect() as connection:
            if summary is None:
                connection.execute("DELETE FROM summaries WHERE scope = ?", (scope,))
            else:
                connection.execute(
                    """
                    INSERT INTO summaries (scope, summary)
                    VALUES (?, ?)
                    ON CONFLICT(scope) DO UPDATE SET
                        summary = excluded.summary,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (scope, summary),
                )
            connection.commit()

    def _get_profile_sync(self, scope: str, subject: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT profile FROM profiles WHERE scope = ? AND subject = ?",
                (scope, subject),
            ).fetchone()
        if row is None:
            return None
        return str(row["profile"])

    def _update_profile_sync(self, scope: str, subject: str, profile: str | None) -> None:
        with self._connect() as connection:
            if profile is None:
                connection.execute(
                    "DELETE FROM profiles WHERE scope = ? AND subject = ?",
                    (scope, subject),
                )
            else:
                connection.execute(
                    """
                    INSERT INTO profiles (scope, subject, profile)
                    VALUES (?, ?, ?)
                    ON CONFLICT(scope, subject) DO UPDATE SET
                        profile = excluded.profile,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (scope, subject, profile),
                )
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self._database_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    def _memory_from_row(self, row: sqlite3.Row | None) -> MemoryRecord:
        if row is None:
            raise RuntimeError("memory row missing after insert")
        return MemoryRecord(
            id=int(row["id"]),
            scope=str(row["scope"]),
            kind=str(row["kind"]),
            subject=str(row["subject"]) if row["subject"] is not None else None,
            content=str(row["content"]),
            created_at=str(row["created_at"]),
        )

    def _ensure_memories_table(self, connection: sqlite3.Connection) -> None:
        if not self._table_exists(connection, "memories"):
            connection.execute(
                """
                CREATE TABLE memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scope TEXT NOT NULL,
                    kind TEXT NOT NULL DEFAULT 'note',
                    subject TEXT,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
        else:
            columns = self._table_columns(connection, "memories")
            if "kind" not in columns:
                connection.execute("ALTER TABLE memories ADD COLUMN kind TEXT NOT NULL DEFAULT 'note'")
            if "subject" not in columns:
                connection.execute("ALTER TABLE memories ADD COLUMN subject TEXT")

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memories_scope_id
            ON memories (scope, id DESC)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memories_scope_kind_id
            ON memories (scope, kind, id DESC)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memories_scope_subject_id
            ON memories (scope, subject, id DESC)
            """
        )

    def _ensure_summaries_table(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS summaries (
                scope TEXT PRIMARY KEY,
                summary TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

    def _ensure_profiles_table(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS profiles (
                scope TEXT NOT NULL,
                subject TEXT NOT NULL,
                profile TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (scope, subject)
            )
            """
        )

    def _table_exists(self, connection: sqlite3.Connection, table_name: str) -> bool:
        row = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone()
        return row is not None

    def _table_columns(self, connection: sqlite3.Connection, table_name: str) -> set[str]:
        rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        return {str(row["name"]) for row in rows}

    def _require_text(self, value: str, field_name: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"{field_name} must not be empty")
        return normalized

    def _normalize_optional_text(self, value: str | None, field_name: str) -> str | None:
        if value is None:
            return None
        return self._require_text(value, field_name)

    def _normalize_kind(self, kind: str) -> str:
        return self._require_text(kind, "kind").lower()


__all__ = ["MemoryRecord", "MemoryStore"]
