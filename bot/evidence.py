from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Iterable, Mapping


_WEB_FETCH_HEADER_RE = re.compile(
    r"^URL:\s*(?P<url>[^\n]+)\nContent-Type:\s*(?P<content_type>[^\n]+)\n\n(?P<body>.*)$",
    re.DOTALL,
)


@dataclass(frozen=True)
class EvidenceNote:
    tool: str
    source: str
    kind: str
    trust: str
    title: str
    summary: str
    locator: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceRecord:
    id: str
    tools: tuple[str, ...]
    source: str
    kind: str
    trust: str
    title: str
    summary: str
    locator: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def compact_summary(self, max_chars: int = 180) -> str:
        text = f"[{self.id}] {self.source}/{self.kind} {self.trust}: {self.title}"
        if self.summary and self.summary != self.title:
            text = f"{text} - {self.summary}"
        return _truncate(text, max_chars)


class EvidenceLedger:
    def __init__(self) -> None:
        self._records: dict[str, EvidenceRecord] = {}
        self._order: list[str] = []

    def __len__(self) -> int:
        return len(self._order)

    def __iter__(self):
        for evidence_id in self._order:
            yield self._records[evidence_id]

    def get(self, evidence_id: str) -> EvidenceRecord | None:
        return self._records.get(evidence_id)

    def add(self, note: EvidenceNote) -> EvidenceRecord:
        evidence_id = _stable_evidence_id(note)
        existing = self._records.get(evidence_id)
        if existing is None:
            record = EvidenceRecord(
                id=evidence_id,
                tools=(note.tool,),
                source=note.source,
                kind=note.kind,
                trust=note.trust,
                title=note.title,
                summary=note.summary,
                locator=note.locator,
                metadata=dict(note.metadata),
            )
            self._records[evidence_id] = record
            self._order.append(evidence_id)
            return record

        merged_tools = tuple(dict.fromkeys(existing.tools + (note.tool,)))
        merged_summary = existing.summary
        if len(note.summary) > len(existing.summary):
            merged_summary = note.summary
        merged_title = existing.title if len(existing.title) >= len(note.title) else note.title
        merged_locator = existing.locator or note.locator
        merged_metadata = dict(existing.metadata)
        merged_metadata.update(note.metadata)
        record = EvidenceRecord(
            id=existing.id,
            tools=merged_tools,
            source=existing.source,
            kind=existing.kind,
            trust=existing.trust,
            title=merged_title,
            summary=merged_summary,
            locator=merged_locator,
            metadata=merged_metadata,
        )
        self._records[evidence_id] = record
        return record

    def add_all(self, notes: Iterable[EvidenceNote]) -> tuple[EvidenceRecord, ...]:
        return tuple(self.add(note) for note in notes)

    def add_tool_result(self, tool_name: str, result: object) -> tuple[EvidenceRecord, ...]:
        return self.add_all(normalize_tool_evidence(tool_name, result))

    def compact_summaries(self, *, limit: int | None = None, max_chars: int = 180) -> list[str]:
        records = list(self)
        if limit is not None:
            records = records[: max(0, int(limit))]
        return [record.compact_summary(max_chars=max_chars) for record in records]

    def render_compact_summaries(self, *, limit: int | None = None, max_chars: int = 180) -> str:
        return "\n".join(self.compact_summaries(limit=limit, max_chars=max_chars))


def normalize_tool_evidence(tool_name: str, result: object) -> tuple[EvidenceNote, ...]:
    if tool_name == "web_fetch":
        return normalize_web_fetch_result(result)
    if tool_name == "web_search":
        return normalize_web_search_result(result)
    if tool_name in {
        "github_search_owner_repositories",
        "github_list_owner_repositories",
        "github_get_repository",
        "github_read_repository_readme",
        "github_read_repository_file",
        "github_list_repository_directory",
    }:
        return normalize_github_result(tool_name, result)
    raise ValueError(f"unsupported evidence tool: {tool_name}")


def normalize_web_fetch_result(result: object) -> tuple[EvidenceNote, ...]:
    if isinstance(result, str):
        match = _WEB_FETCH_HEADER_RE.match(result.strip())
        if match is None:
            raise ValueError("web fetch result is not in the expected tool format")
        url = match.group("url").strip()
        content_type = match.group("content_type").strip()
        text = match.group("body").strip()
    elif isinstance(result, Mapping):
        url = str(result.get("url", "")).strip()
        content_type = str(result.get("content_type", "")).strip()
        text = str(result.get("text", "")).strip()
    else:
        raise ValueError("web fetch result must be a string or mapping")

    title = _pick_title(text, fallback=url or "web fetch")
    return (
        EvidenceNote(
            tool="web_fetch",
            source="web",
            kind="page",
            trust="untrusted",
            title=title,
            summary=_summarize_text(text, fallback=title),
            locator=url or None,
            metadata={"url": url, "content_type": content_type},
        ),
    )


def normalize_web_search_result(result: object) -> tuple[EvidenceNote, ...]:
    if not isinstance(result, Mapping):
        raise ValueError("web search result must be a mapping")
    query = _collapse_whitespace(str(result.get("query", "")))
    items = result.get("results")
    if not isinstance(items, list):
        raise ValueError("web search result is missing results")

    notes: list[EvidenceNote] = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, Mapping):
            continue
        url = str(item.get("url", "")).strip()
        title = _collapse_whitespace(str(item.get("title", ""))) or url or f"search result {index}"
        snippet = _collapse_whitespace(str(item.get("snippet", "")))
        notes.append(
            EvidenceNote(
                tool="web_search",
                source="web",
                kind="search_result",
                trust="untrusted",
                title=title,
                summary=_truncate(snippet or title, 220),
                locator=url or None,
                metadata={"query": query, "rank": index, "url": url},
            )
        )
    return tuple(notes)


def normalize_github_result(tool_name: str, result: object) -> tuple[EvidenceNote, ...]:
    if not isinstance(result, Mapping):
        raise ValueError("GitHub result must be a mapping")
    if tool_name in {"github_search_owner_repositories", "github_list_owner_repositories"}:
        return _normalize_github_repository_list(tool_name, result)
    if tool_name == "github_get_repository":
        return (_github_repository_note(tool_name, result, rank=None),)
    if tool_name == "github_read_repository_readme":
        owner = str(result.get("owner", "")).strip()
        repo = str(result.get("repo", "")).strip()
        content = str(result.get("content", "")).strip()
        locator = f"{owner}/{repo}:README".strip(":")
        return (
            EvidenceNote(
                tool=tool_name,
                source="github",
                kind="readme",
                trust="untrusted",
                title=f"{owner}/{repo} README".strip(),
                summary=_summarize_text(content, fallback="repository README"),
                locator=locator or None,
                metadata={"owner": owner, "repo": repo, "path": "README"},
            ),
        )
    if tool_name == "github_read_repository_file":
        owner = str(result.get("owner", "")).strip()
        repo = str(result.get("repo", "")).strip()
        path = str(result.get("path", "")).strip()
        ref = str(result.get("ref", "")).strip()
        content = str(result.get("content", "")).strip()
        locator = f"{owner}/{repo}:{path}"
        if ref:
            locator = f"{locator}@{ref}"
        return (
            EvidenceNote(
                tool=tool_name,
                source="github",
                kind="file",
                trust="untrusted",
                title=f"{owner}/{repo}:{path}",
                summary=_summarize_text(content, fallback=path or "repository file"),
                locator=locator or None,
                metadata={"owner": owner, "repo": repo, "path": path, "ref": ref or None},
            ),
        )
    if tool_name == "github_list_repository_directory":
        owner = str(result.get("owner", "")).strip()
        repo = str(result.get("repo", "")).strip()
        path = str(result.get("path", "")).strip() or "/"
        ref = str(result.get("ref", "")).strip()
        entries = result.get("entries")
        if not isinstance(entries, list):
            raise ValueError("GitHub directory listing is missing entries")
        summary_items: list[str] = []
        for item in entries[:5]:
            if not isinstance(item, Mapping):
                continue
            item_name = _collapse_whitespace(str(item.get("name", "")))
            item_type = _collapse_whitespace(str(item.get("type", "")))
            if item_name:
                summary_items.append(f"{item_type or 'entry'} {item_name}".strip())
        locator = f"{owner}/{repo}:{path}"
        if ref:
            locator = f"{locator}@{ref}"
        return (
            EvidenceNote(
                tool=tool_name,
                source="github",
                kind="directory",
                trust="structured",
                title=f"{owner}/{repo}:{path}",
                summary=_truncate('; '.join(summary_items) or 'repository directory listing', 220),
                locator=locator or None,
                metadata={"owner": owner, "repo": repo, "path": path, "ref": ref or None, "count": len(entries)},
            ),
        )
    raise ValueError(f"unsupported GitHub evidence tool: {tool_name}")


def _normalize_github_repository_list(tool_name: str, result: Mapping[str, object]) -> tuple[EvidenceNote, ...]:
    items = result.get("repositories")
    if not isinstance(items, list):
        raise ValueError("GitHub repository list is missing repositories")
    notes: list[EvidenceNote] = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, Mapping):
            continue
        notes.append(_github_repository_note(tool_name, item, rank=index, context=result))
    return tuple(notes)


def _github_repository_note(
    tool_name: str,
    item: Mapping[str, object],
    *,
    rank: int | None,
    context: Mapping[str, object] | None = None,
) -> EvidenceNote:
    full_name = _collapse_whitespace(str(item.get("full_name", ""))) or "unknown repository"
    html_url = str(item.get("html_url", "")).strip()
    description = _collapse_whitespace(str(item.get("description", "")))
    language = _collapse_whitespace(str(item.get("language", "")))
    updated_at = _collapse_whitespace(str(item.get("updated_at", "")))
    stars = item.get("stargazers_count")
    summary_parts: list[str] = []
    if description:
        summary_parts.append(description)
    if language:
        summary_parts.append(f"language={language}")
    if isinstance(stars, int):
        summary_parts.append(f"stars={stars}")
    if updated_at:
        summary_parts.append(f"updated={updated_at}")
    metadata: dict[str, object] = {"url": html_url}
    if context is not None:
        owner = _collapse_whitespace(str(context.get("owner", "")))
        query = _collapse_whitespace(str(context.get("query", "")))
        if owner:
            metadata["owner"] = owner
        if query:
            metadata["query"] = query
    if rank is not None:
        metadata["rank"] = rank
    return EvidenceNote(
        tool=tool_name,
        source="github",
        kind="repository",
        trust="structured",
        title=full_name,
        summary=_truncate("; ".join(summary_parts) or full_name, 220),
        locator=html_url or full_name,
        metadata=metadata,
    )


def _stable_evidence_id(note: EvidenceNote) -> str:
    fingerprint = {
        "source": note.source,
        "kind": note.kind,
        "locator": note.locator or "",
        "title": _collapse_whitespace(note.title),
    }
    digest = hashlib.sha1(_canonical_json(fingerprint).encode("utf-8")).hexdigest()[:12]
    return f"ev_{digest}"


def _canonical_json(value: object) -> str:
    return json.dumps(_canonicalize(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _canonicalize(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _canonicalize(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _pick_title(text: str, *, fallback: str) -> str:
    for line in text.splitlines():
        candidate = _collapse_whitespace(line)
        if candidate:
            return _truncate(candidate, 100)
    return _truncate(_collapse_whitespace(fallback) or "evidence", 100)


def _summarize_text(text: str, *, fallback: str) -> str:
    compact = _collapse_whitespace(text)
    if not compact:
        return _truncate(_collapse_whitespace(fallback), 220)
    return _truncate(compact, 220)


def _collapse_whitespace(value: str) -> str:
    return " ".join(str(value).split()).strip()


def _truncate(value: str, max_chars: int) -> str:
    cleaned = _collapse_whitespace(value)
    if len(cleaned) <= max_chars:
        return cleaned
    if max_chars <= 3:
        return cleaned[:max_chars]
    return cleaned[: max_chars - 3].rstrip() + "..."
