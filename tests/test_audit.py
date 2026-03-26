from pathlib import Path
import json
import tempfile
import unittest

from bot.audit import AuditLogger


class AuditLoggerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.log_path = Path(self.temp_dir.name) / "state" / "audit" / "beatrice.jsonl"
        self.logger = AuditLogger(self.log_path)

    def test_log_approval_request_appends_jsonl_and_creates_directories(self) -> None:
        record = self.logger.log_approval_request(
            approval_id="abc123",
            tool_name="set_runtime_config",
            arguments={"temperature": 1.2},
            requested_by="alice",
            requested_in="#ussycode",
            summary="set runtime temperature=1.2",
            created_at=123.0,
            expires_at=456.0,
        )

        self.assertTrue(self.log_path.exists())
        self.assertEqual(record["event"], "approval_requested")
        self.assertEqual(record["approval_id"], "abc123")

        lines = self.log_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1)
        payload = json.loads(lines[0])
        self.assertEqual(payload["arguments"], {"temperature": 1.2})
        self.assertEqual(payload["requested_by"], "alice")
        self.assertEqual(payload["requested_in"], "#ussycode")

    def test_log_methods_append_multiple_records(self) -> None:
        self.logger.log_approval(
            approval_id="abc123",
            actor="bea-admin",
            tool_name="persist_runtime_config",
            summary="persist runtime config",
        )
        self.logger.log_rejection(
            approval_id="def456",
            actor="bea-admin",
            tool_name="set_runtime_config",
            summary="set runtime model=test/model",
        )
        self.logger.log_dangerous_action_result(
            approval_id="abc123",
            actor="bea-admin",
            tool_name="persist_runtime_config",
            arguments={},
            summary="persist runtime config",
            ok=False,
            error="disk full",
        )

        lines = self.log_path.read_text(encoding="utf-8").splitlines()
        events = [json.loads(line)["event"] for line in lines]
        self.assertEqual(events, ["approval_granted", "approval_rejected", "dangerous_action_result"])

        payload = json.loads(lines[-1])
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"], "disk full")

    def test_blank_required_fields_raise_value_error(self) -> None:
        with self.assertRaises(ValueError):
            self.logger.log_approval_request(
                approval_id=" ",
                tool_name="set_runtime_config",
                arguments={},
                requested_by="alice",
                requested_in="#ussycode",
            )

    def test_raises_when_parent_path_is_not_directory(self) -> None:
        bad_parent = Path(self.temp_dir.name) / "blocked"
        bad_parent.write_text("nope", encoding="utf-8")
        logger = AuditLogger(bad_parent / "audit.jsonl")

        with self.assertRaises(NotADirectoryError):
            logger.log_approval(
                approval_id="abc123",
                actor="alice",
                tool_name="persist_runtime_config",
            )


if __name__ == "__main__":
    unittest.main()
