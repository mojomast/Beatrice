from __future__ import annotations

import asyncio
import ipaddress
import socket
from dataclasses import dataclass
from typing import Awaitable, Callable
from urllib.parse import urljoin, urlsplit

import httpx


ALLOWED_CONTENT_TYPES = frozenset(
    {
        "text/html",
        "text/plain",
        "application/json",
        "application/xml",
        "application/xhtml+xml",
    }
)
DEFAULT_MAX_REDIRECTS = 3
DEFAULT_MAX_BODY_BYTES = 1024 * 1024
DEFAULT_TIMEOUT = httpx.Timeout(20.0, connect=5.0)

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

    async def _fetch(self, url: str) -> WebFetchResult:
        current_url = await self._validate_url(url)
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
