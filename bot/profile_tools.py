from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Mapping, Sequence


TOPIC_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_+./-]{2,}")
TOPIC_STOP_WORDS = frozenset(
    {
        "about",
        "after",
        "again",
        "anyone",
        "because",
        "been",
        "being",
        "could",
        "does",
        "from",
        "have",
        "just",
        "know",
        "like",
        "maybe",
        "more",
        "really",
        "should",
        "some",
        "someone",
        "something",
        "than",
        "that",
        "their",
        "there",
        "these",
        "thing",
        "think",
        "this",
        "the",
        "those",
        "today",
        "using",
        "what",
        "when",
        "where",
        "which",
        "while",
        "with",
        "would",
        "why",
        "your",
    }
)
ROLE_KEYWORDS = frozenset(
    {
        "admin",
        "artist",
        "developer",
        "designer",
        "engineer",
        "maintainer",
        "moderator",
        "musician",
        "operator",
        "programmer",
        "researcher",
        "student",
        "sysadmin",
        "teacher",
        "writer",
    }
)
DISALLOWED_FACT_WORDS = frozenset(
    {
        "because",
        "but",
        "if",
        "maybe",
        "not",
        "now",
        "right",
        "since",
        "than",
        "that",
        "then",
        "today",
        "tomorrow",
        "tonight",
        "when",
        "while",
    }
)
PRONOUNS_RE = re.compile(
    r"\b(?:my\s+pronouns\s+are|pronouns\s*[:=-])\s*(?P<value>[a-z]+/[a-z]+(?:/[a-z]+)?)\b",
    re.IGNORECASE,
)
ROLE_RE = re.compile(
    r"\b(?:i\s+am|i'm)\s+(?:an?\s+)?(?P<value>[a-z][a-z0-9_+/-]*(?:\s+[a-z][a-z0-9_+/-]*){0,3})\b",
    re.IGNORECASE,
)
USES_RE = re.compile(r"\bi\s+(?:mostly\s+|usually\s+)?use\s+(?P<value>[^.!?;,]+)", re.IGNORECASE)
PREFER_RE = re.compile(r"\bi\s+prefer\s+(?P<value>[^.!?;,]+)", re.IGNORECASE)
LIKE_RE = re.compile(r"\bi\s+(?:really\s+)?(?:like|love|enjoy)\s+(?P<value>[^.!?;,]+)", re.IGNORECASE)
WORK_RE = re.compile(r"\bi\s+(?:work\s+on|maintain|run)\s+(?P<value>[^.!?;,]+)", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class IRCActivity:
    nick: str
    text: str


def extract_topic_keywords(text: str, limit: int = 4) -> list[str]:
    keywords: list[str] = []
    seen: set[str] = set()
    for match in TOPIC_WORD_RE.finditer(text.lower()):
        token = match.group(0).strip("_+./-'")
        if len(token) < 3 or token in TOPIC_STOP_WORDS or token.isdigit():
            continue
        if token in seen:
            continue
        seen.add(token)
        keywords.append(token)
        if len(keywords) >= limit:
            break
    return keywords


def extract_profile_facts(text: str, subject: str, *, max_facts: int = 3) -> list[str]:
    normalized_subject = _require_text(subject, "subject")
    compact = _compact(text)
    if not compact or "?" in compact:
        return []

    lowered = compact.lower()
    if re.search(r"\bi\s+(?:do\s+not|don't|cant|can't|cannot|won't|never)\b", lowered):
        return []

    facts: list[str] = []
    seen: set[str] = set()

    pronouns = PRONOUNS_RE.search(compact)
    if pronouns is not None:
        _append_fact(facts, seen, f"{normalized_subject}'s pronouns are {pronouns.group('value').lower()}", max_facts)

    role_match = ROLE_RE.search(compact)
    if role_match is not None:
        role = _clean_fact_value(role_match.group("value"), allow_articles=True)
        if role is not None and _looks_like_role(role):
            _append_fact(facts, seen, f"{normalized_subject} is {role}", max_facts)

    for pattern, verb in ((USES_RE, "uses"), (PREFER_RE, "prefers"), (LIKE_RE, "likes"), (WORK_RE, "works on")):
        for match in pattern.finditer(compact):
            value = _clean_fact_value(match.group("value"))
            if value is None:
                continue
            _append_fact(facts, seen, f"{normalized_subject} {verb} {value}", max_facts)
            if len(facts) >= max_facts:
                return facts

    return facts


def build_user_profile_fragment(
    subject: str,
    *,
    remembered_profile: str | None = None,
    remembered_facts: Sequence[str] = (),
    recent_activity: Sequence[IRCActivity | str] = (),
    max_items: int = 4,
) -> str | None:
    normalized_subject = _require_text(subject, "subject")
    fragments: list[str] = []
    seen: set[str] = set()
    topic_counts: Counter[str] = Counter()

    if remembered_profile:
        _append_fragment(fragments, seen, _fact_to_fragment(normalized_subject, remembered_profile), max_items)

    for fact in remembered_facts:
        _append_fragment(fragments, seen, _fact_to_fragment(normalized_subject, fact), max_items)
        if len(fragments) >= max_items:
            break

    for activity in recent_activity:
        nick, text = _activity_parts(activity, normalized_subject)
        if nick.casefold() != normalized_subject.casefold():
            continue
        extracted_facts = extract_profile_facts(text, normalized_subject)
        for fact in extracted_facts:
            _append_fragment(fragments, seen, _fact_to_fragment(normalized_subject, fact), max_items)
            if len(fragments) >= max_items:
                break
        if len(fragments) < max_items and not extracted_facts:
            topic_counts.update(extract_topic_keywords(text))
        if len(fragments) >= max_items:
            break

    if len(fragments) < max_items and topic_counts:
        topics = [topic for topic, _count in topic_counts.most_common(3)]
        _append_fragment(fragments, seen, f"recent topics: {', '.join(topics)}", max_items)

    if not fragments:
        return None
    return f"{normalized_subject}: {'; '.join(fragments[:max_items])}"


def format_channel_member_prompt(
    members: Sequence[str],
    *,
    profiles: Mapping[str, str] | None = None,
    active_nicks: Sequence[str] = (),
    max_members: int = 8,
) -> str | None:
    ordered_members = _ordered_members(members, active_nicks)
    if not ordered_members:
        return None

    profile_map = {nick.casefold(): profile for nick, profile in (profiles or {}).items() if profile.strip()}
    items: list[str] = []
    visible_members = ordered_members[:max_members]
    for member in visible_members:
        profile = profile_map.get(member.casefold())
        if profile:
            fragment = _fact_to_fragment(member, profile)
            if fragment:
                items.append(f"{member} ({_truncate(fragment, 80)})")
                continue
        items.append(member)

    remaining = len(ordered_members) - len(visible_members)
    members_part = ", ".join(items)
    if remaining > 0:
        members_part = f"{members_part}, +{remaining} more"

    parts = [f"Channel members: {members_part}."]
    active = [nick for nick in _unique_preserve_order(active_nicks) if nick.casefold() in {member.casefold() for member in ordered_members}]
    if active:
        parts.append(f"Recently active: {', '.join(active[:4])}.")
    return " ".join(parts)


def format_channel_topic_prompt(
    channel: str,
    *,
    topic: str | None = None,
    recent_topic_keywords: Sequence[str] = (),
) -> str | None:
    normalized_channel = _require_text(channel, "channel")
    parts: list[str] = []
    compact_topic = _compact(topic or "")
    if compact_topic:
        parts.append(f"Channel topic for {normalized_channel}: {compact_topic}.")
    keywords = [keyword for keyword in _unique_preserve_order(recent_topic_keywords) if keyword.strip()]
    if keywords:
        parts.append(f"Recent channel topics: {', '.join(keywords[:6])}.")
    if not parts:
        return None
    return " ".join(parts)


def build_channel_prompt_context(
    channel: str,
    *,
    members: Sequence[str] = (),
    member_profiles: Mapping[str, str] | None = None,
    active_nicks: Sequence[str] = (),
    topic: str | None = None,
    recent_topic_keywords: Sequence[str] = (),
) -> list[str]:
    snippets: list[str] = []
    topic_prompt = format_channel_topic_prompt(
        channel,
        topic=topic,
        recent_topic_keywords=recent_topic_keywords,
    )
    if topic_prompt:
        snippets.append(topic_prompt)

    member_prompt = format_channel_member_prompt(
        members,
        profiles=member_profiles,
        active_nicks=active_nicks,
    )
    if member_prompt:
        snippets.append(member_prompt)
    return snippets


def _activity_parts(activity: IRCActivity | str, default_nick: str) -> tuple[str, str]:
    if isinstance(activity, IRCActivity):
        return activity.nick.strip(), activity.text
    return default_nick, str(activity)


def _ordered_members(members: Sequence[str], active_nicks: Sequence[str]) -> list[str]:
    member_map = {member.casefold(): member.strip() for member in members if member.strip()}
    ordered: list[str] = []
    for nick in active_nicks:
        key = nick.strip().casefold()
        if key in member_map and member_map[key] not in ordered:
            ordered.append(member_map[key])
    for member in sorted(member_map.values(), key=str.casefold):
        if member not in ordered:
            ordered.append(member)
    return ordered


def _append_fact(facts: list[str], seen: set[str], fact: str, max_facts: int) -> None:
    compact = _compact(fact)
    if not compact:
        return
    key = compact.casefold()
    if key in seen or len(facts) >= max_facts:
        return
    seen.add(key)
    facts.append(compact)


def _append_fragment(fragments: list[str], seen: set[str], fragment: str | None, max_items: int) -> None:
    if fragment is None:
        return
    compact = _truncate(_compact(fragment), 120)
    if not compact:
        return
    key = compact.casefold()
    if key in seen or len(fragments) >= max_items:
        return
    seen.add(key)
    fragments.append(compact)


def _clean_fact_value(value: str, *, allow_articles: bool = False) -> str | None:
    compact = _compact(value)
    compact = re.split(r"\b(?:because|but|if|so|when|while)\b", compact, maxsplit=1, flags=re.IGNORECASE)[0].strip(" .!?,;:-")
    if not allow_articles:
        compact = re.sub(r"^(?:a|an|the)\s+", "", compact, flags=re.IGNORECASE)
    if not compact or len(compact) > 48:
        return None
    words = compact.lower().split()
    if len(words) > 5 or any(word in DISALLOWED_FACT_WORDS for word in words):
        return None
    return compact


def _looks_like_role(value: str) -> bool:
    tokens = {token.lower() for token in value.split()}
    return any(token in ROLE_KEYWORDS for token in tokens)


def _fact_to_fragment(subject: str, fact: str) -> str | None:
    compact = _compact(fact)
    if not compact:
        return None
    lowered = compact.casefold()
    subject_lower = subject.casefold()
    if lowered.startswith(f"{subject_lower}'s "):
        return compact[len(subject) + 3 :]
    if lowered.startswith(f"{subject_lower}: "):
        return compact[len(subject) + 2 :]
    if lowered.startswith(f"{subject_lower} "):
        return compact[len(subject) + 1 :]
    return compact


def _compact(text: str) -> str:
    return " ".join(str(text).split())


def _require_text(value: str, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3].rstrip()}..."


def _unique_preserve_order(values: Sequence[str]) -> list[str]:
    items: list[str] = []
    seen: set[str] = set()
    for value in values:
        compact = value.strip()
        if not compact:
            continue
        key = compact.casefold()
        if key in seen:
            continue
        seen.add(key)
        items.append(compact)
    return items


__all__ = [
    "IRCActivity",
    "build_channel_prompt_context",
    "build_user_profile_fragment",
    "extract_profile_facts",
    "extract_topic_keywords",
    "format_channel_member_prompt",
    "format_channel_topic_prompt",
]
