from pathlib import Path
import sqlite3
import tempfile
import unittest

from bot.memory_store import MemoryStore


class MemoryStoreTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.database_path = Path(self.temp_dir.name) / "state" / "memory.sqlite3"
        self.store = MemoryStore(self.database_path)

    async def test_store_memory_persists_and_returns_recent_first(self) -> None:
        first = await self.store.store_memory("#ussycode", "first memory")
        second = await self.store.store_memory("#ussycode", "second memory")

        reopened = MemoryStore(self.database_path)
        memories = await reopened.search_recent_memories("#ussycode")

        self.assertEqual(first.content, "first memory")
        self.assertEqual(first.kind, "note")
        self.assertIsNone(first.subject)
        self.assertTrue(first.created_at)
        self.assertEqual([memory.content for memory in memories], ["second memory", "first memory"])
        self.assertEqual(memories[0].id, second.id)
        self.assertTrue(self.database_path.exists())

    async def test_search_recent_memories_filters_by_scope_query_kind_subject_and_limit(self) -> None:
        await self.store.store_memory("#ussycode", "docker timeout on bridge", kind="fact", subject="ops")
        await self.store.store_memory("#ussycode", "irc reconnect worked", kind="observation", subject="ops")
        await self.store.store_memory("#ussycode", "docker logs show retry loop", kind="fact", subject="ops")
        await self.store.store_memory("#ussycode", "docker summary", kind="summary")
        await self.store.store_memory("alice", "docker issue in private chat", kind="fact", subject="ops")

        memories = await self.store.search_recent_memories(
            "#ussycode",
            query="docker",
            limit=2,
            kind="fact",
            subject="ops",
        )

        self.assertEqual(
            [memory.content for memory in memories],
            ["docker logs show retry loop", "docker timeout on bridge"],
        )
        self.assertTrue(all(memory.kind == "fact" for memory in memories))
        self.assertTrue(all(memory.subject == "ops" for memory in memories))

    async def test_get_and_update_summary_supports_replace_and_clear(self) -> None:
        self.assertIsNone(await self.store.get_summary("#ussycode"))

        await self.store.update_summary("#ussycode", "first summary")
        self.assertEqual(await self.store.get_summary("#ussycode"), "first summary")

        await self.store.update_summary("#ussycode", "updated summary")
        self.assertEqual(await self.store.get_summary("#ussycode"), "updated summary")

        await self.store.update_summary("#ussycode", None)
        self.assertIsNone(await self.store.get_summary("#ussycode"))

    async def test_get_summary_falls_back_to_latest_summary_memory(self) -> None:
        await self.store.store_memory("#ussycode", "older summary", kind="summary")
        await self.store.store_memory("#ussycode", "latest summary", kind="summary")

        self.assertEqual(await self.store.get_summary("#ussycode"), "latest summary")

    async def test_get_and_update_profile_supports_replace_and_clear(self) -> None:
        self.assertIsNone(await self.store.get_profile("#ussycode", "alice"))

        await self.store.update_profile("#ussycode", "alice", "alice likes detailed postmortems")
        self.assertEqual(
            await self.store.get_profile("#ussycode", "alice"),
            "alice likes detailed postmortems",
        )

        await self.store.update_profile("#ussycode", "alice", "alice prefers concise summaries")
        self.assertEqual(
            await self.store.get_profile("#ussycode", "alice"),
            "alice prefers concise summaries",
        )

        await self.store.update_profile("#ussycode", "alice", None)
        self.assertIsNone(await self.store.get_profile("#ussycode", "alice"))

    async def test_initialize_migrates_existing_memory_table(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                CREATE TABLE memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scope TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                "INSERT INTO memories (scope, content) VALUES (?, ?)",
                ("#ussycode", "legacy memory"),
            )
            connection.commit()

        reopened = MemoryStore(self.database_path)
        memories = await reopened.search_recent_memories("#ussycode")

        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0].content, "legacy memory")
        self.assertEqual(memories[0].kind, "note")
        self.assertIsNone(memories[0].subject)

    async def test_initialize_enables_wal_mode(self) -> None:
        await self.store.initialize()

        with sqlite3.connect(self.database_path) as connection:
            journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]

        self.assertEqual(journal_mode.lower(), "wal")


if __name__ == "__main__":
    unittest.main()
