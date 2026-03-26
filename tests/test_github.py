import unittest
from urllib.parse import parse_qs, urlsplit

import httpx

from bot.github import GitHubClient, GitHubError


class GitHubClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_owner_repositories_scopes_query_to_owner(self) -> None:
        requests: list[httpx.Request] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "full_name": "mojomast/ussyverse",
                            "description": "ussyverse repo",
                            "html_url": "https://github.com/mojomast/ussyverse",
                            "stargazers_count": 3,
                            "language": "Python",
                            "updated_at": "2026-03-26T00:00:00Z",
                        }
                    ]
                },
                request=request,
            )

        client = GitHubClient(httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://api.github.com"))
        self.addAsyncCleanup(client.aclose)

        result = await client.search_owner_repositories("mojomast", "ussyverse")

        self.assertEqual(result["repositories"][0]["full_name"], "mojomast/ussyverse")
        query = parse_qs(urlsplit(str(requests[0].url)).query)
        self.assertEqual(query["q"][0], "user:mojomast ussyverse")

    async def test_readme_raises_for_missing_repo(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, request=request)

        client = GitHubClient(httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://api.github.com"))
        self.addAsyncCleanup(client.aclose)

        with self.assertRaisesRegex(GitHubError, "not found"):
            await client.read_repository_readme("mojomast", "missing")

    async def test_list_owner_repositories_returns_repo_metadata(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json=[
                    {
                        "full_name": "mojomast/ussynet",
                        "description": "network repo",
                        "html_url": "https://github.com/mojomast/ussynet",
                        "stargazers_count": 1,
                        "language": "Python",
                        "updated_at": "2026-03-26T00:00:00Z",
                    }
                ],
                request=request,
            )

        client = GitHubClient(httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://api.github.com"))
        self.addAsyncCleanup(client.aclose)

        result = await client.list_owner_repositories("mojomast")

        self.assertEqual(result["repositories"][0]["full_name"], "mojomast/ussynet")

    async def test_list_repository_directory_returns_compact_entries(self) -> None:
        requests: list[httpx.Request] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                200,
                json=[
                    {
                        "name": "bot",
                        "path": "bot",
                        "type": "dir",
                        "size": 0,
                        "sha": "dirsha",
                        "html_url": "https://github.com/mojomast/ussynet/tree/main/bot",
                        "download_url": None,
                        "url": "https://api.github.com/repos/mojomast/ussynet/contents/bot",
                        "git_url": "https://api.github.com/repos/mojomast/ussynet/git/trees/dirsha",
                        "_links": {"self": "ignored"},
                    },
                    {
                        "name": "README.md",
                        "path": "README.md",
                        "type": "file",
                        "size": 42,
                        "sha": "filesha",
                        "html_url": "https://github.com/mojomast/ussynet/blob/main/README.md",
                        "download_url": "https://raw.githubusercontent.com/mojomast/ussynet/main/README.md",
                    },
                ],
                request=request,
            )

        client = GitHubClient(httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://api.github.com"))
        self.addAsyncCleanup(client.aclose)

        result = await client.list_repository_directory("mojomast", "ussynet", path="/", ref="main")

        self.assertEqual(result["path"], "")
        self.assertEqual(result["ref"], "main")
        self.assertEqual(
            result["entries"],
            [
                {
                    "name": "bot",
                    "path": "bot",
                    "type": "dir",
                    "size": 0,
                    "sha": "dirsha",
                    "html_url": "https://github.com/mojomast/ussynet/tree/main/bot",
                    "download_url": None,
                },
                {
                    "name": "README.md",
                    "path": "README.md",
                    "type": "file",
                    "size": 42,
                    "sha": "filesha",
                    "html_url": "https://github.com/mojomast/ussynet/blob/main/README.md",
                    "download_url": "https://raw.githubusercontent.com/mojomast/ussynet/main/README.md",
                },
            ],
        )
        self.assertEqual(requests[0].url.path, "/repos/mojomast/ussynet/contents")
        query = parse_qs(urlsplit(str(requests[0].url)).query)
        self.assertEqual(query["ref"][0], "main")

    async def test_list_repository_directory_rejects_parent_traversal(self) -> None:
        client = GitHubClient(httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request)), base_url="https://api.github.com"))
        self.addAsyncCleanup(client.aclose)

        with self.assertRaisesRegex(GitHubError, "invalid repository file path"):
            await client.list_repository_directory("mojomast", "ussynet", path="../secrets")

    async def test_list_repository_directory_rejects_file_payload(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "name": "README.md",
                    "path": "README.md",
                    "type": "file",
                },
                request=request,
            )

        client = GitHubClient(httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://api.github.com"))
        self.addAsyncCleanup(client.aclose)

        with self.assertRaisesRegex(GitHubError, "not a directory"):
            await client.list_repository_directory("mojomast", "ussynet", path="README.md")
