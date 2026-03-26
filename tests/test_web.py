import unittest

import httpx

from bot.web import WebFetchError, WebFetcher


class WebFetcherTests(unittest.IsolatedAsyncioTestCase):
    async def test_fetch_text_allows_public_get_and_supported_content(self) -> None:
        requests: list[httpx.Request] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                200,
                headers={"Content-Type": "text/plain; charset=utf-8"},
                content=b"hello web",
                request=request,
            )

        fetcher = self._make_fetcher(handler)
        self.addAsyncCleanup(fetcher.aclose)

        text = await fetcher.fetch_text("https://example.com/path")

        self.assertEqual(text, "hello web")
        self.assertEqual([request.method for request in requests], ["GET"])

    async def test_fetch_text_rejects_embedded_credentials(self) -> None:
        fetcher = self._make_fetcher(self._ok_handler)
        self.addAsyncCleanup(fetcher.aclose)

        with self.assertRaisesRegex(WebFetchError, "embedded credentials"):
            await fetcher.fetch_text("https://user:pass@example.com/")

    async def test_fetch_text_rejects_localhost_and_private_targets(self) -> None:
        fetcher = self._make_fetcher(self._ok_handler)
        self.addAsyncCleanup(fetcher.aclose)

        with self.assertRaisesRegex(WebFetchError, "localhost"):
            await fetcher.fetch_text("https://localhost/")

        with self.assertRaisesRegex(WebFetchError, "non-public IP"):
            await fetcher.fetch_text("https://127.0.0.1/")

    async def test_fetch_text_rejects_non_default_ports(self) -> None:
        fetcher = self._make_fetcher(self._ok_handler)
        self.addAsyncCleanup(fetcher.aclose)

        with self.assertRaisesRegex(WebFetchError, "non-default ports"):
            await fetcher.fetch_text("https://example.com:444/")

    async def test_fetch_text_rejects_hostname_resolving_to_private_ip(self) -> None:
        async def resolve(_host: str) -> list[str]:
            return ["10.0.0.5"]

        fetcher = self._make_fetcher(self._ok_handler, resolver=resolve)
        self.addAsyncCleanup(fetcher.aclose)

        with self.assertRaisesRegex(WebFetchError, "non-public IP"):
            await fetcher.fetch_text("https://example.com/")

    async def test_fetch_text_follows_safe_redirects(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            if str(request.url) == "https://example.com/start":
                return httpx.Response(302, headers={"Location": "/final"}, request=request)
            return httpx.Response(
                200,
                headers={"Content-Type": "application/json"},
                content=b'{"ok": true}',
                request=request,
            )

        fetcher = self._make_fetcher(handler)
        self.addAsyncCleanup(fetcher.aclose)

        text = await fetcher.fetch_text("https://example.com/start")

        self.assertEqual(text, '{"ok": true}')

    async def test_fetch_text_rejects_unsafe_redirect_targets(self) -> None:
        async def downgrade_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(302, headers={"Location": "http://example.com/final"}, request=request)

        async def localhost_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(302, headers={"Location": "https://localhost/final"}, request=request)

        fetcher = self._make_fetcher(downgrade_handler)
        self.addAsyncCleanup(fetcher.aclose)
        with self.assertRaisesRegex(WebFetchError, "https to http"):
            await fetcher.fetch_text("https://example.com/start")

        fetcher_localhost = self._make_fetcher(localhost_handler)
        self.addAsyncCleanup(fetcher_localhost.aclose)
        with self.assertRaisesRegex(WebFetchError, "localhost"):
            await fetcher_localhost.fetch_text("https://example.com/start")

    async def test_fetch_text_caps_redirects(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path == "/1":
                return httpx.Response(302, headers={"Location": "/2"}, request=request)
            if path == "/2":
                return httpx.Response(302, headers={"Location": "/3"}, request=request)
            if path == "/3":
                return httpx.Response(302, headers={"Location": "/4"}, request=request)
            return httpx.Response(200, headers={"Content-Type": "text/plain"}, content=b"done", request=request)

        fetcher = self._make_fetcher(handler, max_redirects=2)
        self.addAsyncCleanup(fetcher.aclose)

        with self.assertRaisesRegex(WebFetchError, "too many redirects"):
            await fetcher.fetch_text("https://example.com/1")

    async def test_fetch_text_rejects_unsupported_content_types(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, headers={"Content-Type": "image/png"}, content=b"png", request=request)

        fetcher = self._make_fetcher(handler)
        self.addAsyncCleanup(fetcher.aclose)

        with self.assertRaisesRegex(WebFetchError, "unsupported content type"):
            await fetcher.fetch_text("https://example.com/")

    async def test_fetch_text_caps_body_size(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"Content-Type": "text/plain", "Content-Length": "50"},
                content=b"x" * 50,
                request=request,
            )

        fetcher = self._make_fetcher(handler, max_body_bytes=16)
        self.addAsyncCleanup(fetcher.aclose)

        with self.assertRaisesRegex(WebFetchError, "exceeds 16 bytes"):
            await fetcher.fetch_text("https://example.com/")

    async def test_tool_result_formats_success_and_failure(self) -> None:
        fetcher = self._make_fetcher(self._ok_handler)
        self.addAsyncCleanup(fetcher.aclose)

        success = await fetcher.tool_result("https://example.com/")
        failure = await fetcher.tool_result("https://localhost/")

        self.assertIn("URL: https://example.com/", success)
        self.assertIn("Content-Type: text/plain", success)
        self.assertTrue(success.endswith("ok"))
        self.assertEqual(failure, "Web fetch failed: localhost URLs are not allowed")

    def _make_fetcher(
        self,
        handler,
        *,
        resolver=None,
        max_redirects: int = 3,
        max_body_bytes: int = 1024,
    ) -> WebFetcher:
        async def resolve(host: str) -> list[str]:
            if resolver is not None:
                return await resolver(host)
            return ["93.184.216.34"]

        client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            follow_redirects=False,
            trust_env=False,
        )
        return WebFetcher(
            client,
            resolver=resolve,
            max_redirects=max_redirects,
            max_body_bytes=max_body_bytes,
        )

    @staticmethod
    async def _ok_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "text/plain"},
            content=b"ok",
            request=request,
        )
