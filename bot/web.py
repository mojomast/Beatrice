from __future__ import annotations

import asyncio
import base64
import ipaddress
import socket
from dataclasses import dataclass
import xml.etree.ElementTree as ET
import html
import re
from typing import Awaitable, Callable
from urllib.parse import unquote
from urllib.parse import urljoin, urlsplit

import httpx


ALLOWED_CONTENT_TYPES = frozenset(
    {
        "text/html",
        "text/plain",
        "text/xml",
        "application/json",
        "application/xml",
        "application/xhtml+xml",
    }
)
DEFAULT_MAX_REDIRECTS = 3
DEFAULT_MAX_BODY_BYTES = 1024 * 1024
DEFAULT_TIMEOUT = httpx.Timeout(20.0, connect=5.0)
SEARCH_RESULT_RE = re.compile(r'<a[^>]+class="result__a"[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>', re.IGNORECASE | re.DOTALL)
SEARCH_SNIPPET_RE = re.compile(r'<a[^>]+class="result__snippet"[^>]*>(?P<snippet>.*?)</a>|<div[^>]+class="result__snippet"[^>]*>(?P<snippet_div>.*?)</div>', re.IGNORECASE | re.DOTALL)

Resolver = Callable[[str], Awaitable[list[str]]]


class WebFetchError(RuntimeError):
    """Raised when a URL cannot be fetched safely."""


@dataclass(frozen=True)
class WebFetchResult:
    url: str
    content_type: str
    text: str


class WebFetcher:
    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        *,
        resolver: Resolver | None = None,
        max_redirects: int = DEFAULT_MAX_REDIRECTS,
        max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
        timeout: httpx.Timeout | None = None,
    ) -> None:
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            follow_redirects=False,
            timeout=timeout or DEFAULT_TIMEOUT,
            trust_env=False,
        )
        self._resolver = resolver or self._default_resolver
        self._max_redirects = max(0, int(max_redirects))
        self._max_body_bytes = max(1, int(max_body_bytes))

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def fetch_text(self, url: str) -> str:
        result = await self._fetch(url)
        return result.text

    async def tool_result(self, url: str) -> str:
        try:
            result = await self._fetch(url)
        except WebFetchError as exc:
            return f"Web fetch failed: {exc}"
        return f"URL: {result.url}\nContent-Type: {result.content_type}\n\n{result.text}"

    async def search(self, query: str, limit: int = 5) -> list[dict[str, str]]:
        cleaned = " ".join(str(query).split()).strip()
        if not cleaned:
            raise WebFetchError("search query is required")
        capped_limit = max(1, min(int(limit), 5))
        results = await self._search_duckduckgo_html(cleaned, capped_limit)
        if not results:
            results = await self._search_bing_rss(cleaned, capped_limit)
        if not results:
            raise WebFetchError("search returned no results")
        return results

    async def search_tool_result(self, query: str, limit: int = 5) -> dict[str, object]:
        results = await self.search(query, limit=limit)
        return {"query": " ".join(str(query).split()).strip(), "results": results}

    async def _search_duckduckgo_html(self, query: str, limit: int) -> list[dict[str, str]]:
        response = await self._fetch(f"https://html.duckduckgo.com/html/?q={httpx.QueryParams({'q': query})['q']}")
        return self._parse_search_results(response.text, limit)

    async def _search_bing_rss(self, query: str, limit: int) -> list[dict[str, str]]:
        response = await self._fetch(f"https://www.bing.com/search?q={httpx.QueryParams({'q': query})['q']}&format=rss")
        try:
            root = ET.fromstring(response.text)
        except ET.ParseError as exc:
            raise WebFetchError("search provider returned invalid XML") from exc
        results: list[dict[str, str]] = []
        for item in root.findall('./channel/item')[:limit]:
            title = ' '.join((item.findtext('title') or '').split()).strip()
            url = ' '.join((item.findtext('link') or '').split()).strip()
            snippet = ' '.join((item.findtext('description') or '').split()).strip()
            if not title or not url:
                continue
            results.append({'title': title, 'url': url, 'snippet': snippet})
        return results

    def _parse_search_results(self, html_text: str, limit: int) -> list[dict[str, str]]:
        results: list[dict[str, str]] = []
        seen_urls: set[str] = set()

        snippet_matches = list(SEARCH_SNIPPET_RE.finditer(html_text))
        snippet_index = 0
        for match in SEARCH_RESULT_RE.finditer(html_text):
            href = self._clean_search_href(match.group('href'))
            title = self._strip_html(match.group('title'))
            if not href or not title or href in seen_urls:
                continue
            snippet = ''
            if snippet_index < len(snippet_matches):
                snippet_match = snippet_matches[snippet_index]
                snippet_raw = snippet_match.group('snippet') if snippet_match.group('snippet') else snippet_match.group('snippet_div')
                snippet = self._strip_html(snippet_raw)
                snippet_index += 1
            seen_urls.add(href)
            results.append({'title': title, 'url': href, 'snippet': snippet})
            if len(results) >= limit:
                break
        return results

    @staticmethod
    def _strip_html(value: str) -> str:
        text = re.sub(r'<[^>]+>', ' ', value)
        return html.unescape(' '.join(text.split())).strip()

    @staticmethod
    def _clean_search_href(value: str) -> str:
        href = html.unescape(value).strip()
        if not href:
            return ''
        if href.startswith('//'):
            return 'https:' + href
        if href.startswith('/l/?uddg='):
            encoded = href.split('/l/?uddg=', 1)[1].split('&', 1)[0]
            if encoded.startswith('http'):
                return unquote(encoded)
            decoded = base64.urlsafe_b64decode(encoded + '=' * (-len(encoded) % 4)).decode('utf-8', errors='ignore')
            return decoded.strip()
        return href

    async def _fetch(self, url: str) -> WebFetchResult:
        current_url = await self._validate_url(url)
        self._reject_github_hosts(current_url)
        redirects = 0

        while True:
            request = self._client.build_request("GET", current_url)
            try:
                response = await self._client.send(request, stream=True)
            except httpx.HTTPError as exc:
                raise WebFetchError(f"request failed: {exc}") from exc

            try:
                if response.is_redirect:
                    location = response.headers.get("Location")
                    if not location:
                        raise WebFetchError("redirect response is missing a Location header")
                    if redirects >= self._max_redirects:
                        raise WebFetchError(f"too many redirects (max {self._max_redirects})")
                    current_url = await self._validate_url(
                        urljoin(str(current_url), location),
                        previous_url=current_url,
                    )
                    self._reject_github_hosts(current_url)
                    redirects += 1
                    continue

                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    raise WebFetchError(f"request failed with HTTP {response.status_code}") from exc

                content_type = self._validate_content_type(response)
                body = await self._read_body(response)
                return WebFetchResult(
                    url=str(response.url),
                    content_type=content_type,
                    text=self._decode_body(body, response.headers.get("Content-Type", "")),
                )
            finally:
                await response.aclose()

    async def _validate_url(self, raw_url: str, previous_url: httpx.URL | None = None) -> httpx.URL:
        try:
            split = urlsplit(raw_url)
        except ValueError as exc:
            raise WebFetchError("invalid URL") from exc

        scheme = split.scheme.lower()
        if scheme not in {"http", "https"}:
            raise WebFetchError("only http and https URLs are allowed")

        if previous_url is not None and previous_url.scheme == "https" and scheme != "https":
            raise WebFetchError("redirect from https to http is not allowed")

        if split.username is not None or split.password is not None:
            raise WebFetchError("URLs with embedded credentials are not allowed")

        host = split.hostname
        if not host:
            raise WebFetchError("URL must include a hostname")
        if host.lower() == "localhost" or host.lower().endswith(".localhost"):
            raise WebFetchError("localhost URLs are not allowed")

        try:
            port = split.port
        except ValueError as exc:
            raise WebFetchError("invalid URL port") from exc
        if port is not None and port != self._default_port_for_scheme(scheme):
            raise WebFetchError("non-default ports are not allowed")

        await self._validate_host(host)

        try:
            return httpx.URL(raw_url)
        except Exception as exc:  # pragma: no cover - defensive httpx parsing guard
            raise WebFetchError("invalid URL") from exc

    async def _validate_host(self, host: str) -> None:
        ip = self._parse_ip(host)
        if ip is not None:
            self._ensure_public_ip(ip)
            return

        try:
            addresses = await self._resolver(host)
        except OSError as exc:
            raise WebFetchError("hostname could not be resolved") from exc
        if not addresses:
            raise WebFetchError("hostname did not resolve to a public IP address")

        for address in addresses:
            resolved_ip = self._parse_ip(address)
            if resolved_ip is None:
                raise WebFetchError("hostname resolution returned an invalid IP address")
            self._ensure_public_ip(resolved_ip)

    async def _default_resolver(self, host: str) -> list[str]:
        loop = asyncio.get_running_loop()
        results = await loop.getaddrinfo(host, None, type=socket.SOCK_STREAM)
        addresses: list[str] = []
        seen: set[str] = set()

        for family, _, _, _, sockaddr in results:
            if family not in {socket.AF_INET, socket.AF_INET6}:
                continue
            address = sockaddr[0]
            if address not in seen:
                seen.add(address)
                addresses.append(address)

        return addresses

    def _validate_content_type(self, response: httpx.Response) -> str:
        content_type_header = response.headers.get("Content-Type", "")
        content_type = content_type_header.split(";", 1)[0].strip().lower()
        if content_type not in ALLOWED_CONTENT_TYPES:
            raise WebFetchError(f"unsupported content type: {content_type or 'missing'}")
        return content_type

    async def _read_body(self, response: httpx.Response) -> bytes:
        content_length = response.headers.get("Content-Length")
        if content_length:
            try:
                if int(content_length) > self._max_body_bytes:
                    raise WebFetchError(f"response body exceeds {self._max_body_bytes} bytes")
            except ValueError:
                pass

        chunks: list[bytes] = []
        total = 0
        async for chunk in response.aiter_bytes():
            total += len(chunk)
            if total > self._max_body_bytes:
                raise WebFetchError(f"response body exceeds {self._max_body_bytes} bytes")
            chunks.append(chunk)
        return b"".join(chunks)

    @staticmethod
    def _decode_body(body: bytes, content_type_header: str) -> str:
        encoding = "utf-8"
        for part in content_type_header.split(";")[1:]:
            name, _, value = part.partition("=")
            if name.strip().lower() == "charset" and value.strip():
                encoding = value.strip().strip('"')
                break
        try:
            return body.decode(encoding, errors="replace")
        except LookupError:
            return body.decode("utf-8", errors="replace")

    @staticmethod
    def _default_port_for_scheme(scheme: str) -> int:
        return 80 if scheme == "http" else 443

    @staticmethod
    def _parse_ip(value: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
        try:
            ip = ipaddress.ip_address(value)
        except ValueError:
            return None
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
            return ip.ipv4_mapped
        return ip

    @staticmethod
    def _ensure_public_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> None:
        if not ip.is_global:
            raise WebFetchError(f"non-public IP addresses are not allowed: {ip}")

    @staticmethod
    def _reject_github_hosts(url: httpx.URL) -> None:
        host = (url.host or "").lower()
        if host in {"api.github.com", "github.com", "raw.githubusercontent.com"}:
            raise WebFetchError("GitHub URLs must be accessed through github_* tools")
