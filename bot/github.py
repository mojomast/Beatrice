from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote

import httpx


DEFAULT_TIMEOUT = httpx.Timeout(20.0, connect=5.0)


class GitHubError(RuntimeError):
    """Raised when a GitHub request cannot be completed safely."""


@dataclass(frozen=True)
class GitHubScope:
    owner: str
    repo: str | None = None


class GitHubClient:
    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url="https://api.github.com",
            timeout=DEFAULT_TIMEOUT,
            trust_env=False,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "beatrice-bot/1.0",
            },
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def search_owner_repositories(self, owner: str, query: str, limit: int = 5) -> dict[str, object]:
        owner = self._clean_name(owner, "owner")
        query = " ".join(str(query).split()).strip()
        if not query:
            raise GitHubError("query is required")
        limit = max(1, min(int(limit), 10))

        response = await self._request(
            "GET",
            "/search/repositories",
            params={"q": f"user:{owner} {query}", "per_page": str(limit)},
        )
        payload = response.json()
        items = payload.get("items", []) if isinstance(payload, dict) else []
        repositories: list[dict[str, object]] = []
        for item in items[:limit]:
            if not isinstance(item, dict):
                continue
            repositories.append(
                {
                    "full_name": item.get("full_name"),
                    "description": item.get("description"),
                    "html_url": item.get("html_url"),
                    "stargazers_count": item.get("stargazers_count"),
                    "language": item.get("language"),
                    "updated_at": item.get("updated_at"),
                }
            )
        return {"owner": owner, "query": query, "repositories": repositories}

    async def list_owner_repositories(self, owner: str, limit: int = 8) -> dict[str, object]:
        owner = self._clean_name(owner, "owner")
        limit = max(1, min(int(limit), 10))

        response = await self._request(
            "GET",
            f"/users/{quote(owner)}/repos",
            params={"sort": "updated", "per_page": str(limit)},
        )
        payload = response.json()
        items = payload if isinstance(payload, list) else []
        repositories: list[dict[str, object]] = []
        for item in items[:limit]:
            if not isinstance(item, dict):
                continue
            repositories.append(
                {
                    "full_name": item.get("full_name"),
                    "description": item.get("description"),
                    "html_url": item.get("html_url"),
                    "stargazers_count": item.get("stargazers_count"),
                    "language": item.get("language"),
                    "updated_at": item.get("updated_at"),
                }
            )
        return {"owner": owner, "repositories": repositories}

    async def get_repository(self, owner: str, repo: str) -> dict[str, object]:
        owner = self._clean_name(owner, "owner")
        repo = self._clean_name(repo, "repo")
        response = await self._request("GET", f"/repos/{quote(owner)}/{quote(repo)}")
        payload = response.json()
        if not isinstance(payload, dict):
            raise GitHubError("unexpected GitHub repository response")
        return {
            "full_name": payload.get("full_name"),
            "description": payload.get("description"),
            "html_url": payload.get("html_url"),
            "default_branch": payload.get("default_branch"),
            "language": payload.get("language"),
            "topics": payload.get("topics") if isinstance(payload.get("topics"), list) else [],
            "stargazers_count": payload.get("stargazers_count"),
            "updated_at": payload.get("updated_at"),
        }

    async def read_repository_readme(self, owner: str, repo: str) -> dict[str, object]:
        owner = self._clean_name(owner, "owner")
        repo = self._clean_name(repo, "repo")
        response = await self._request(
            "GET",
            f"/repos/{quote(owner)}/{quote(repo)}/readme",
            headers={"Accept": "application/vnd.github.raw+json"},
        )
        return {
            "owner": owner,
            "repo": repo,
            "path": "README",
            "content": response.text,
        }

    async def list_repository_directory(
        self,
        owner: str,
        repo: str,
        path: str | None = None,
        ref: str | None = None,
    ) -> dict[str, object]:
        owner = self._clean_name(owner, "owner")
        repo = self._clean_name(repo, "repo")
        normalized_path = self._normalize_repository_path(path)

        params = {"ref": ref} if ref else None
        response = await self._request(
            "GET",
            self._repository_contents_path(owner, repo, normalized_path),
            params=params,
        )
        payload = response.json()
        if not isinstance(payload, list):
            raise GitHubError("repository path is not a directory")

        entries: list[dict[str, object]] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            entries.append(
                {
                    "name": item.get("name"),
                    "path": item.get("path"),
                    "type": item.get("type"),
                    "size": item.get("size"),
                    "sha": item.get("sha"),
                    "html_url": item.get("html_url"),
                    "download_url": item.get("download_url"),
                }
            )
        return {
            "owner": owner,
            "repo": repo,
            "path": normalized_path,
            "ref": ref,
            "entries": entries,
        }

    async def read_repository_file(self, owner: str, repo: str, path: str, ref: str | None = None) -> dict[str, object]:
        owner = self._clean_name(owner, "owner")
        repo = self._clean_name(repo, "repo")
        normalized_path = self._normalize_repository_path(path)
        if not normalized_path:
            raise GitHubError("invalid repository file path")

        params = {"ref": ref} if ref else None
        response = await self._request(
            "GET",
            self._repository_contents_path(owner, repo, normalized_path),
            params=params,
            headers={"Accept": "application/vnd.github.raw+json"},
        )
        return {
            "owner": owner,
            "repo": repo,
            "path": normalized_path,
            "ref": ref,
            "content": response.text,
        }

    async def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        try:
            response = await self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise GitHubError(f"request failed: {exc}") from exc
        if response.status_code == 404:
            raise GitHubError("GitHub resource not found")
        if response.status_code >= 400:
            raise GitHubError(f"GitHub request failed with HTTP {response.status_code}")
        return response

    @staticmethod
    def _clean_name(value: str, field_name: str) -> str:
        cleaned = str(value).strip()
        if not cleaned:
            raise GitHubError(f"{field_name} is required")
        return cleaned

    @staticmethod
    def _normalize_repository_path(path: str | None) -> str:
        normalized_path = "/".join(part for part in str(path or "").split("/") if part and part != ".")
        if normalized_path.startswith("..") or "/../" in f"/{normalized_path}/":
            raise GitHubError("invalid repository file path")
        return normalized_path

    @staticmethod
    def _repository_contents_path(owner: str, repo: str, path: str) -> str:
        base_path = f"/repos/{quote(owner)}/{quote(repo)}/contents"
        if not path:
            return base_path
        return f"{base_path}/{quote(path, safe='/')}"
