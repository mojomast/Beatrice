import unittest

from bot.evidence import EvidenceLedger, normalize_github_result, normalize_web_fetch_result, normalize_web_search_result


class EvidenceLedgerTests(unittest.TestCase):
    def test_add_tool_result_uses_stable_ids_and_deduplicates(self) -> None:
        ledger = EvidenceLedger()

        first = ledger.add_tool_result(
            "web_search",
            {
                "query": " current   events ",
                "results": [
                    {
                        "title": "Example Story",
                        "url": "https://example.com/story",
                        "snippet": "Fresh news here",
                    }
                ],
            },
        )[0]
        second = ledger.add_tool_result(
            "web_search",
            {
                "query": "current events",
                "results": [
                    {
                        "title": "Example Story",
                        "url": "https://example.com/story",
                        "snippet": "Fresh news here",
                    }
                ],
            },
        )[0]

        self.assertEqual(first.id, second.id)
        self.assertEqual(len(ledger), 1)
        self.assertEqual(first.tools, ("web_search",))

    def test_compact_summaries_render_source_kind_and_title(self) -> None:
        ledger = EvidenceLedger()
        ledger.add_tool_result(
            "github_get_repository",
            {
                "full_name": "mojomast/ussynet",
                "description": "network repo",
                "html_url": "https://github.com/mojomast/ussynet",
                "stargazers_count": 1,
                "language": "Python",
                "updated_at": "2026-03-26T00:00:00Z",
            },
        )

        rendered = ledger.render_compact_summaries()

        self.assertIn("github/repository", rendered)
        self.assertIn("mojomast/ussynet", rendered)
        self.assertIn("structured", rendered)


class EvidenceNormalizationTests(unittest.TestCase):
    def test_normalize_web_fetch_result_extracts_metadata_and_summary(self) -> None:
        notes = normalize_web_fetch_result(
            "URL: https://example.com/story\n"
            "Content-Type: text/plain; charset=utf-8\n\n"
            "Example headline\n"
            "Fresh news here from the public web."
        )

        self.assertEqual(len(notes), 1)
        note = notes[0]
        self.assertEqual(note.source, "web")
        self.assertEqual(note.kind, "page")
        self.assertEqual(note.trust, "untrusted")
        self.assertEqual(note.locator, "https://example.com/story")
        self.assertEqual(note.metadata["content_type"], "text/plain; charset=utf-8")
        self.assertIn("Example headline", note.summary)

    def test_normalize_web_search_result_creates_ranked_notes(self) -> None:
        notes = normalize_web_search_result(
            {
                "query": "current events",
                "results": [
                    {
                        "title": "Example Story",
                        "url": "https://example.com/story",
                        "snippet": "Fresh news here",
                    },
                    {
                        "title": "Second Story",
                        "url": "https://example.com/second",
                        "snippet": "More context",
                    },
                ],
            }
        )

        self.assertEqual([note.metadata["rank"] for note in notes], [1, 2])
        self.assertEqual(notes[0].kind, "search_result")
        self.assertEqual(notes[0].metadata["query"], "current events")

    def test_normalize_github_repository_results_marks_structured_trust(self) -> None:
        notes = normalize_github_result(
            "github_list_owner_repositories",
            {
                "owner": "mojomast",
                "repositories": [
                    {
                        "full_name": "mojomast/ussynet",
                        "description": "network repo",
                        "html_url": "https://github.com/mojomast/ussynet",
                        "stargazers_count": 1,
                        "language": "Python",
                        "updated_at": "2026-03-26T00:00:00Z",
                    }
                ],
            },
        )

        self.assertEqual(len(notes), 1)
        note = notes[0]
        self.assertEqual(note.source, "github")
        self.assertEqual(note.kind, "repository")
        self.assertEqual(note.trust, "structured")
        self.assertEqual(note.metadata["owner"], "mojomast")

    def test_normalize_github_file_and_readme_results(self) -> None:
        readme = normalize_github_result(
            "github_read_repository_readme",
            {
                "owner": "mojomast",
                "repo": "ussynet",
                "path": "README",
                "content": "# Ussynet\n\nIRC bot and network tooling.",
            },
        )[0]
        source_file = normalize_github_result(
            "github_read_repository_file",
            {
                "owner": "mojomast",
                "repo": "ussynet",
                "path": "bot/evidence.py",
                "ref": "master",
                "content": "from dataclasses import dataclass\n",
            },
        )[0]

        self.assertEqual(readme.kind, "readme")
        self.assertEqual(readme.locator, "mojomast/ussynet:README")
        self.assertEqual(source_file.kind, "file")
        self.assertEqual(source_file.locator, "mojomast/ussynet:bot/evidence.py@master")

    def test_normalize_github_directory_listing(self) -> None:
        directory = normalize_github_result(
            "github_list_repository_directory",
            {
                "owner": "mojomast",
                "repo": "ussynet",
                "path": "bot",
                "ref": "main",
                "entries": [
                    {"name": "app.py", "type": "file"},
                    {"name": "irc.py", "type": "file"},
                ],
            },
        )[0]

        self.assertEqual(directory.kind, "directory")
        self.assertEqual(directory.trust, "structured")
        self.assertEqual(directory.locator, "mojomast/ussynet:bot@main")
        self.assertIn("file app.py", directory.summary)


if __name__ == "__main__":
    unittest.main()
