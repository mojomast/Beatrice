from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re

from .child_bots import CHILD_RESPONSE_MODE_ADDRESSED_ONLY, normalize_response_mode


VOICE_ARCHETYPES = {
    "switchboard": {
        "label": "switchboard concierge",
        "voice": "Treat the channel like a busy front desk. Route attention cleanly, greet quickly, and keep your wording polished and social.",
        "habits": "Prefer short welcoming sentences. Ask one tidy follow-up when useful. Keep energy warm and professional.",
    },
    "gremlin": {
        "label": "playful gremlin",
        "voice": "Sound mischievous, chatty, and internet-native without becoming chaotic or obnoxious.",
        "habits": "Use compact slang sparingly. Tease lightly. Keep replies punchy and high-energy.",
    },
    "archivist": {
        "label": "quiet archivist",
        "voice": "Speak like a calm keeper of notes who prefers precision over excitement.",
        "habits": "Answer with compact factual lines. Avoid filler. Be reserved, steady, and helpful.",
    },
    "foreman": {
        "label": "shop-floor foreman",
        "voice": "Act direct, practical, and task-focused. Treat chat like a place to get things moving.",
        "habits": "Lead with the useful point. Use procedural language. Do not ramble.",
    },
    "host": {
        "label": "party host",
        "voice": "Sound socially attentive, inviting, and quick to keep conversation flowing.",
        "habits": "Encourage participation. Keep the vibe light. Prefer upbeat, human-sounding responses.",
    },
    "critic": {
        "label": "dry critic",
        "voice": "Be dry, skeptical, and sharp-edged while still cooperative and non-hostile.",
        "habits": "Correct gently but directly. Favor concise observations over reassurance.",
    },
}

MAX_CHILD_BATCH = 8
SAFE_ID_RE = re.compile(r"[^a-z0-9_-]+")
SAFE_NICK_RE = re.compile(r"[^A-Za-z0-9_\-\[\]\\`^{}|]+")


@dataclass(frozen=True)
class ConcreteChildPlan:
    action: str
    child_id: str
    nick: str | None = None
    channels: tuple[str, ...] = ()
    system_prompt: str | None = None
    model: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    reply_interval_seconds: float | None = None
    start_after_create: bool = False
    response_mode: str | None = None
    enabled: bool | None = None
    purpose: str | None = None
    variation: tuple[str, ...] = ()


def normalize_child_id(value: str) -> str:
    cleaned = SAFE_ID_RE.sub("", value.strip().lower())
    if not cleaned:
        raise ValueError("child_id must contain letters, numbers, '-' or '_'")
    return cleaned[:40]


def _derive_child_id_from_nick_or_purpose(nick: str | None, purpose: str | None) -> str:
    """Auto-derive a child_id from nick or purpose when the LLM omits child_id."""
    if nick and nick.strip():
        candidate = normalize_child_id(nick.strip())
        if candidate:
            return candidate
    if purpose and purpose.strip():
        words = purpose.strip().split()
        candidate_text = "-".join(words[:2])
        candidate = normalize_child_id(candidate_text)
        if candidate:
            return candidate
    import time
    return normalize_child_id(f"bot-{int(time.time())}")


def normalize_nick(value: str) -> str:
    cleaned = SAFE_NICK_RE.sub("", value.strip())
    if not cleaned:
        raise ValueError("nick must contain IRC-safe characters")
    return cleaned[:32]


def _suffix(prefix: str, index: int, total: int) -> str:
    width = max(1, len(str(total)))
    separator = "-" if prefix and prefix[-1].isdigit() else ""
    return f"{prefix}{separator}{index + 1:0{width}d}"


def _choice(seed: str, axis: str, values: tuple[str, ...]) -> str:
    digest = hashlib.sha256(f"{seed}:{axis}".encode("utf-8")).digest()
    return values[digest[0] % len(values)]


def _variation_bundle(seed: str, requested_tone: str | None = None, style_tags: tuple[str, ...] = ()) -> tuple[str, ...]:
    archetype_name = _choice(seed, 'archetype', tuple(VOICE_ARCHETYPES.keys()))
    archetype = VOICE_ARCHETYPES[archetype_name]
    selected = [
        f"archetype: {archetype['label']}",
        f"voice: {archetype['voice']}",
        f"habits: {archetype['habits']}",
    ]
    if requested_tone and requested_tone.strip():
        selected.append(f"requested tone: {requested_tone.strip()}")
    for tag in style_tags[:4]:
        compact = " ".join(str(tag).split()).strip()
        if compact:
            selected.append(f"style tag: {compact[:48]}")
    return tuple(selected)


def _response_mode_clause(response_mode: str) -> str:
    if response_mode == CHILD_RESPONSE_MODE_ADDRESSED_ONLY:
        return "In channels, only respond when directly addressed by nick or when someone is clearly talking to you. Stay quiet otherwise."
    return "In channels, you may respond naturally when conversation presents a strong opening such as a question, topic overlap, or clear invitation. Do not dominate the room."


def render_child_system_prompt(
    *,
    nick: str,
    purpose: str,
    persona: str | None = None,
    requested_tone: str | None = None,
    style_tags: tuple[str, ...] = (),
    avoid: tuple[str, ...] = (),
    seed: str,
    response_mode: str = CHILD_RESPONSE_MODE_ADDRESSED_ONLY,
) -> tuple[str, tuple[str, ...]]:
    variation = _variation_bundle(seed, requested_tone=requested_tone, style_tags=style_tags)
    lines = [
        f"You are {nick}, a simple managed IRC chatbot.",
        f"Purpose: {' '.join(purpose.split())[:320]}",
        "Stay in your role, keep replies concise, and act like a lightweight chat-only bot with a distinct fixed personality.",
        "Do not claim to use tools, browse the web, inspect GitHub, change config, or manage other bots.",
        _response_mode_clause(response_mode),
        "In private messages, you can be a bit more detailed.",
        "Distinctive behavior contract:",
    ]
    for item in variation:
        lines.append(f"- {item}")
    if persona and persona.strip():
        lines.append(f"Persona notes: {' '.join(persona.split())[:320]}")
    if avoid:
        trimmed = [" ".join(str(item).split())[:80] for item in avoid if str(item).strip()]
        if trimmed:
            lines.append("Avoid: " + "; ".join(trimmed[:5]))
    prompt = "\n".join(lines)
    return prompt[:4000], variation


def expand_child_bot_operations(arguments: dict[str, object], default_model: str) -> list[ConcreteChildPlan]:
    raw_operations = arguments.get("operations")
    if not isinstance(raw_operations, list) or not raw_operations:
        raise ValueError("operations must contain at least one child bot operation")
    plans: list[ConcreteChildPlan] = []
    for raw in raw_operations:
        if not isinstance(raw, dict):
            raise ValueError("each child bot operation must be an object")
        action = str(raw.get("action", "")).strip().lower()
        if action not in {"create", "update", "start", "stop", "enable", "disable", "remove"}:
            raise ValueError(f"unknown child bot action '{action}'")
        if action == "create":
            plans.extend(_expand_create_operation(raw, default_model))
            continue
        child_id_raw = str(raw.get("child_id", "")).strip()
        if not child_id_raw:
            raise ValueError(f"child_id is required for {action} actions")
        child_id = normalize_child_id(child_id_raw)
        plans.append(ConcreteChildPlan(action=action, child_id=child_id))
    return plans


def _expand_create_operation(raw: dict[str, object], default_model: str) -> list[ConcreteChildPlan]:
    count = int(raw.get("count", 1))
    if count < 1 or count > MAX_CHILD_BATCH:
        raise ValueError(f"count must be between 1 and {MAX_CHILD_BATCH}")
    channels_raw = raw.get("channels")
    if not isinstance(channels_raw, list) or not channels_raw:
        raise ValueError("create operations require channels")
    channels = tuple(str(channel).strip() for channel in channels_raw if str(channel).strip())
    if not channels:
        raise ValueError("create operations require at least one non-empty channel")
    purpose = " ".join(str(raw.get("purpose", "")).split()).strip()
    if not purpose:
        raise ValueError("create operations require purpose")
    persona = " ".join(str(raw.get("persona", "")).split()).strip() or None
    requested_tone = " ".join(str(raw.get("tone", "")).split()).strip() or None
    style_tags_raw = raw.get("style_tags", ())
    style_tags = tuple(str(item).strip() for item in style_tags_raw if str(item).strip()) if isinstance(style_tags_raw, list) else ()
    avoid_raw = raw.get("avoid", ())
    avoid = tuple(str(item).strip() for item in avoid_raw if str(item).strip()) if isinstance(avoid_raw, list) else ()
    start_after_create = bool(raw.get("start_after_create", True))
    response_mode = normalize_response_mode(str(raw.get("response_mode", CHILD_RESPONSE_MODE_ADDRESSED_ONLY)))
    enabled = bool(raw.get("enabled", True))
    model = str(raw.get("model", default_model)).strip() or default_model
    temperature = float(raw.get("temperature", 0.7))
    top_p = float(raw.get("top_p", 1.0))
    max_tokens = int(raw.get("max_tokens", 180))
    reply_interval_seconds = float(raw.get("reply_interval_seconds", 4.0))
    child_id_value = str(raw.get("child_id", "")).strip()
    nick_value = str(raw.get("nick", "")).strip()
    id_prefix = str(raw.get("id_prefix", child_id_value)).strip()
    nick_prefix = str(raw.get("nick_prefix", nick_value or id_prefix)).strip()
    if count == 1:
        child_id = normalize_child_id(child_id_value or id_prefix) if (child_id_value or id_prefix) else _derive_child_id_from_nick_or_purpose(nick_value, purpose)
        nick = normalize_nick(nick_value or nick_prefix or child_id)
        prompt, variation = render_child_system_prompt(
            nick=nick,
            purpose=purpose,
            persona=persona,
            requested_tone=requested_tone,
            style_tags=style_tags,
            avoid=avoid,
            seed=child_id,
            response_mode=response_mode,
        )
        return [
            ConcreteChildPlan(
                action="create",
                child_id=child_id,
                nick=nick,
                channels=channels,
                system_prompt=prompt,
                model=model,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                reply_interval_seconds=reply_interval_seconds,
                enabled=enabled,
                start_after_create=start_after_create,
                response_mode=response_mode,
                purpose=purpose,
                variation=variation,
            )
        ]
    if not id_prefix:
        raise ValueError("multi-bot create operations require id_prefix or child_id")
    if not nick_prefix:
        raise ValueError("multi-bot create operations require nick_prefix or nick")
    plans: list[ConcreteChildPlan] = []
    for index in range(count):
        child_id = normalize_child_id(_suffix(id_prefix, index, count))
        nick = normalize_nick(_suffix(nick_prefix, index, count))
        prompt, variation = render_child_system_prompt(
            nick=nick,
            purpose=purpose,
            persona=persona,
            requested_tone=requested_tone,
            style_tags=style_tags,
            avoid=avoid,
            seed=f"{id_prefix}:{index + 1}:{purpose}",
            response_mode=response_mode,
        )
        plans.append(
            ConcreteChildPlan(
                action="create",
                child_id=child_id,
                nick=nick,
                channels=channels,
                system_prompt=prompt,
                model=model,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                reply_interval_seconds=reply_interval_seconds,
                enabled=enabled,
                start_after_create=start_after_create,
                response_mode=response_mode,
                purpose=purpose,
                variation=variation,
            )
        )
    return plans


def summarize_child_bot_operations(plans: list[ConcreteChildPlan]) -> str:
    if not plans:
        return "no child bot changes requested"
    counts: dict[str, int] = {}
    previews: list[str] = []
    for plan in plans[:6]:
        counts[plan.action] = counts.get(plan.action, 0) + 1
        if plan.action == "create":
            previews.append(
                f"{plan.child_id}/{plan.nick} channels={','.join(plan.channels)} purpose={' '.join((plan.purpose or '').split())[:60]}"
            )
        else:
            previews.append(f"{plan.action} {plan.child_id}")
    for plan in plans[6:]:
        counts[plan.action] = counts.get(plan.action, 0) + 1
    action_summary = ", ".join(f"{action}={count}" for action, count in sorted(counts.items()))
    return f"child bot changes {action_summary}: {' | '.join(previews)}"
