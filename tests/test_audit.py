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

    def test_request_trace_events_capture_safe_structured_summaries(self) -> None:
        self.logger.log_request_start(
            request_id="req123",
            nick="alice",
            target="#ussycode",
            is_private=False,
            prompt="search for current deploy docs",
            github_scope="octo/repo",
            domain_hint="example.com",
            preferred_direct_url="https://user:pass@example.com/docs?q=secret#frag",
            requires_web_lookup=True,
        )
        self.logger.log_request_tool_call(
            request_id="req123",
            tool_name="web_fetch",
            tool_call_id="call_1",
            category="web",
            round_index=1,
            arguments={
                "url": "https://user:pass@example.com/docs?q=secret#frag",
                "authorization": "Bearer super-secret",
                "query": "deploy token rollout",
                "limit": 3,
                "nested": {"api_key": "abc123", "path": "docs/index.md"},
            },
        )
        self.logger.log_request_tool_result(
            request_id="req123",
            tool_name="web_fetch",
            tool_call_id="call_1",
            category="web",
            round_index=1,
            ok=True,
            approval_required=False,
            duration_ms=42,
            result={
                "url": "https://user:pass@example.com/docs?q=secret#frag",
                "content": "top secret page contents",
                "headers": {"authorization": "Bearer nope"},
            },
        )
        self.logger.log_request_finish(
            request_id="req123",
            outcome="answered",
            rounds=1,
            tools_used=1,
            tool_names=["web_fetch"],
            response="Here is the deploy documentation.",
        )

        lines = self.log_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual([json.loads(line)["event"] for line in lines], [
            "request_start",
            "request_tool_call",
            "request_tool_result",
            "request_finish",
        ])

        start_payload = json.loads(lines[0])
        self.assertEqual(start_payload["prompt_len"], len("search for current deploy docs"))
        self.assertEqual(start_payload["preferred_direct_url"], "https://example.com/docs")
        self.assertNotIn("prompt", start_payload)

        call_payload = json.loads(lines[1])
        self.assertEqual(call_payload["arguments_summary"]["url"], "https://example.com/docs")
        self.assertEqual(call_payload["arguments_summary"]["authorization"], "<redacted>")
        self.assertEqual(call_payload["arguments_summary"]["query"], {"type": "text", "length": 20})
        self.assertEqual(call_payload["arguments_summary"]["nested"]["api_key"], "<redacted>")
        self.assertEqual(call_payload["arguments_summary"]["nested"]["path"], "docs/index.md")
        self.assertNotIn("super-secret", lines[1])

        result_payload = json.loads(lines[2])
        self.assertEqual(result_payload["result_summary"]["url"], "https://example.com/docs")
        self.assertEqual(result_payload["result_summary"]["content"], {"type": "text", "length": 24})
        self.assertEqual(result_payload["result_summary"]["headers"]["authorization"], "<redacted>")

        finish_payload = json.loads(lines[3])
        self.assertEqual(finish_payload["tool_names"], ["web_fetch"])
        self.assertEqual(finish_payload["response_len"], len("Here is the deploy documentation."))
        self.assertNotIn("response", finish_payload)

    def test_request_trace_records_safe_result_payloads(self) -> None:
        self.logger.log_request_tool_result(
            request_id="req200",
            tool_name="github_get_repository",
            ok=True,
            result={
                "full_name": "mojomast/ussynet",
                "html_url": "https://github.com/mojomast/ussynet?tab=readme#top",
                "description": "repo details",
            },
        )

        payload = json.loads(self.log_path.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(payload["result_summary"]["html_url"], {"type": "text", "length": 50})
        self.assertEqual(payload["result_summary"]["full_name"], {"type": "text", "length": 16})

    def test_child_bot_event_logs_safe_fields(self) -> None:
        self.logger.log_child_bot_event(
            child_id="helper",
            action="start",
            status="running",
            nick="HelperBot",
            channels=["#ussycode"],
            model="google/gemini-2.5-flash-lite",
            pid=1234,
        )

        payload = json.loads(self.log_path.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(payload["event"], "child_bot_event")
        self.assertEqual(payload["child_id"], "helper")
        self.assertEqual(payload["action"], "start")
        self.assertEqual(payload["channels"], ["#ussycode"])

    def test_request_trace_error_summary_redacts_secrets(self) -> None:
        self.logger.log_request_tool_result(
            request_id="req999",
            tool_name="persist_runtime_config",
            ok=False,
            error={
                "message": "could not persist config",
                "openrouter_key": "sk-live-secret",
                "details": ["token leaked", {"password": "hunter2"}],
            },
        )

        payload = json.loads(self.log_path.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(payload["error_summary"]["openrouter_key"], "<redacted>")
        self.assertEqual(payload["error_summary"]["message"], {"type": "text", "length": 24})
        self.assertEqual(
            payload["error_summary"]["details"],
            {
                "type": "list",
                "length": 2,
                "items": [
                    {"type": "text", "length": 12},
                    {"password": "<redacted>"},
                ],
            },
        )
        self.assertNotIn("sk-live-secret", self.log_path.read_text(encoding="utf-8"))

    def test_blank_required_fields_raise_value_error(self) -> None:
        with self.assertRaises(ValueError):
            self.logger.log_approval_request(
                approval_id=" ",
                tool_name="set_runtime_config",
                arguments={},
                requested_by="alice",
                requested_in="#ussycode",
            )

        with self.assertRaises(ValueError):
            self.logger.log_request_start(
                request_id="req123",
                nick="alice",
                target="#ussycode",
                is_private=False,
                prompt="   ",
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
