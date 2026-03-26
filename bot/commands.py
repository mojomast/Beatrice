from __future__ import annotations

import shlex
import re
from typing import Callable

from .config import RuntimeStore, SecretStore


CHANNEL_REQUEST_RE = re.compile(
    r"^(?P<prompt>.+?)\s+(?:in|to)\s+(?P<channel>#[A-Za-z0-9_\-\[\]\\`^{}|]+)[.!?]*$",
    re.IGNORECASE,
)

LEADING_POLITENESS_RE = re.compile(
    r"^(?:please\s+)?(?:can you\s+|could you\s+|would you\s+)?",
    re.IGNORECASE,
)

DIRECT_POST_RE = re.compile(
    r"^(?:(?:please\s+)?(?:can you\s+|could you\s+|would you\s+|can u\s+|could u\s+|would u\s+)?)?"
    r"(?:say|post|send|message|announce)\s+"
    r"(?P<message>.+?)\s+(?:in|to)\s+(?P<channel>#[A-Za-z0-9_\-\[\]\\`^{}|]+)"
    r"(?:\s+please)?[.!?]*$",
    re.IGNORECASE,
)

CHANNEL_CHAT_RE = re.compile(
    r"^(?:(?:please\s+)?(?:can you\s+|could you\s+|would you\s+|can u\s+|could u\s+|would u\s+)?)?"
    r"(?:talk|chat|respond)\s+(?:in|to)\s+(?P<channel>#[A-Za-z0-9_\-\[\]\\`^{}|]+)"
    r"(?:\s+and\s+respond)?(?:\s+please)?[.!?]*$",
    re.IGNORECASE,
)

NATURAL_SET_SYSTEM_RE = re.compile(
    r"^(?:please\s+)?(?:change|set|update)\s+(?:your|ur)\s+system\s+prompt\s+to\s+(?P<value>.+?)[.!?]*$",
    re.IGNORECASE,
)

NATURAL_SHOW_SYSTEM_RE = re.compile(
    r"^(?:please\s+)?(?:what(?:'s|s)?|what\s+is|show(?:\s+me)?|shwo(?:\s+me)?)\s+(?:your|ur)\s+system\s+prompt[!?]*$",
    re.IGNORECASE,
)

NATURAL_SET_MODEL_RE = re.compile(
    r"^(?:please\s+)?(?:change|set|update)\s+(?:your|ur)\s+model\s+to\s+(?P<value>.+?)[.!?]*$",
    re.IGNORECASE,
)


def tokenize_control_command(text: str, prefix: str) -> tuple[list[str] | None, str | None]:
    stripped = text.strip()
    if not stripped.startswith(prefix):
        return None, None
    if len(stripped) > len(prefix) and not stripped[len(prefix)].isspace():
        return None, None

    remainder = stripped[len(prefix):].strip()
    if not remainder:
        return [], None

    try:
        return shlex.split(remainder), None
    except ValueError:
        return None, "could not parse command arguments"


def strip_admin_password(tokens: list[str], expected_password: str) -> tuple[list[str], bool]:
    cleaned: list[str] = []
    password_ok = False

    for token in tokens:
        if token.startswith("password="):
            password_ok = password_ok or token.partition("=")[2] == expected_password
            continue
        cleaned.append(token)

    if not password_ok and cleaned and cleaned[-1] == expected_password:
        password_ok = True
        cleaned = cleaned[:-1]

    return cleaned, password_ok


def extract_prompt(text: str, bot_nick: str, prefix: str, is_private: bool) -> str | None:
    stripped = text.strip()
    if not stripped:
        return None

    tokens, error = tokenize_control_command(stripped, prefix)
    if tokens is not None and error is None:
        if tokens and tokens[0].lower() == "ask":
            prompt = " ".join(tokens[1:]).strip()
            return prompt or None
        return None

    lowered = stripped.lower()
    nick_lower = bot_nick.lower()
    for marker in (f"{nick_lower}:", f"{nick_lower},", f"{nick_lower} "):
        if lowered.startswith(marker):
            prompt = stripped[len(marker):].strip()
            return prompt or None

    if is_private:
        return stripped
    return None


def extract_channel_request(text: str, is_private: bool) -> tuple[str, str] | None:
    if not is_private:
        return None

    stripped = text.strip()
    if not stripped:
        return None

    match = CHANNEL_REQUEST_RE.match(stripped)
    if not match:
        return None

    prompt = match.group("prompt").strip()
    channel = match.group("channel")
    prompt = LEADING_POLITENESS_RE.sub("", prompt).strip()
    if not prompt:
        return None

    return channel, prompt


def extract_direct_post_request(text: str, is_private: bool) -> tuple[str, str] | None:
    if not is_private:
        return None

    stripped = text.strip()
    if not stripped:
        return None

    match = DIRECT_POST_RE.match(stripped)
    if not match:
        return None

    channel = match.group("channel")
    message = match.group("message").strip().strip('"').strip("'")
    if not message:
        return None

    return channel, message


def extract_channel_chat_request(text: str, is_private: bool) -> str | None:
    if not is_private:
        return None

    stripped = text.strip()
    if not stripped:
        return None

    match = CHANNEL_CHAT_RE.match(stripped)
    if not match:
        return None

    return match.group("channel")


def extract_natural_admin_command(text: str, admin_password: str, bot_nick: str) -> list[str] | None:
    stripped = text.strip()
    if not stripped:
        return None

    password_pattern = re.compile(rf"\b{re.escape(admin_password)}\b", re.IGNORECASE)
    if not password_pattern.search(stripped):
        return None

    cleaned = re.sub(
        rf"\b{re.escape(admin_password)}\b",
        " ",
        stripped,
        count=1,
        flags=re.IGNORECASE,
    ).strip()

    for _ in range(4):
        previous = cleaned
        cleaned = re.sub(r"^magic\s+word\s*:\s*", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(rf"^{re.escape(bot_nick)}[,:]?\s*", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(rf"^{re.escape(admin_password)}[,:]?\s*", "", cleaned, flags=re.IGNORECASE).strip()
        if cleaned == previous:
            break

    cleaned = re.sub(rf"^{re.escape(bot_nick)}[,:]?\s*", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(rf"\s+{re.escape(admin_password)}[.!?]*$", "", cleaned, flags=re.IGNORECASE).strip()

    match = NATURAL_SHOW_SYSTEM_RE.match(cleaned)
    if match:
        return ["show", "system"]

    match = NATURAL_SET_SYSTEM_RE.match(cleaned)
    if match:
        value = match.group("value").strip().rstrip(".!?")
        if value:
            return ["set", "system", value, admin_password]

    match = NATURAL_SET_MODEL_RE.match(cleaned)
    if match:
        value = match.group("value").strip().rstrip(".!?")
        if value:
            return ["set", "model", value, admin_password]

    return None


class CommandProcessor:
    def __init__(
        self,
        store: RuntimeStore,
        secrets: SecretStore,
        prefix: str,
        admin_password: str,
        bot_nick: str,
        set_api_key: Callable[[str | None], None],
        reset_history: Callable[[str | None], str],
        context_status: Callable[[str | None], str],
        persist_runtime: Callable[[], str],
        list_approvals: Callable[[], str],
        approve_action: Callable[[str, str, bool], str],
        reject_action: Callable[[str, str, bool], str],
        child_command: Callable[[list[str], str | None, bool], list[str]] | None = None,
    ) -> None:
        self.store = store
        self.secrets = secrets
        self.prefix = prefix
        self.admin_password = admin_password
        self.bot_nick = bot_nick
        self.set_api_key = set_api_key
        self.reset_history = reset_history
        self.context_status = context_status
        self.persist_runtime = persist_runtime
        self.list_approvals = list_approvals
        self.approve_action = approve_action
        self.reject_action = reject_action
        self.child_command = child_command or (lambda _tokens, _actor, _is_private: ["child bot control unavailable"])

    def handle(self, tokens: list[str], actor: str | None = None, is_private: bool = False) -> list[str]:
        if not tokens:
            return self._help_lines()

        command = tokens[0].lower()
        if command == "help":
            return self._help_lines()
        if command == "status":
            return [self._status_line()]
        if command == "show":
            return self._handle_show(tokens[1:])
        if command == "context":
            return self._handle_context(tokens[1:])
        if command == "set":
            return self._handle_set(tokens[1:])
        if command == "clear":
            return self._handle_clear(tokens[1:])
        if command == "save":
            return self._handle_save(tokens[1:])
        if command == "approvals":
            return self._handle_approvals(tokens[1:])
        if command == "approve":
            return self._handle_approve(tokens[1:], actor, is_private)
        if command == "reject":
            return self._handle_reject(tokens[1:], actor, is_private)
        if command == "child":
            return self.child_command(tokens[1:], actor, is_private)
        if command == "reset":
            return self._handle_reset(tokens[1:])

        return [f"unknown command '{command}'. Try {self.prefix} help."]

    def _help_lines(self) -> list[str]:
        return [
            f"Queries: '{self.prefix} ask <prompt>', '{self.bot_nick}: <prompt>', or send a private message.",
            f"Commands: {self.prefix} help | {self.prefix} status | {self.prefix} show system | {self.prefix} show params | {self.prefix} show models | {self.prefix} context [status|reset] | {self.prefix} approvals",
            "Private features: IRC awareness (server/channels/users/topics/nick changes), WHOIS lookups, safe web fetch, typed memories, and subject profiles.",
            "Private chat requests: ask about users/server state, fetch a public URL, remember facts/notes, recall memories, or ask what Beatrice knows about someone.",
            "Admin commands: set system/model/temperature/top_p/max_tokens/reply_interval_seconds/stream/openrouter_key, clear openrouter_key, save runtime, approve <id>, reject <id>, context reset, reset, and child list/create/start/stop/enable/disable/remove; append the admin password or password=<value>.",
            "Dangerous autonomous actions never self-apply: they create approval IDs, and an admin must approve them in a private message with approve/reject.",
        ]

    def _status_line(self) -> str:
        current = self.store.current()
        return (
            f"{current.params_summary()} openrouter_key={self.secrets.openrouter_status()} "
            f"system={current.system_excerpt(80)}"
        )

    def _handle_show(self, tokens: list[str]) -> list[str]:
        if not tokens:
            return [f"usage: {self.prefix} show <system|params|models>"]

        current = self.store.current()
        subject = tokens[0].lower()
        if subject == "system":
            return [f"system={current.system_excerpt(350)}"]
        if subject == "params":
            return [f"{current.params_summary()} openrouter_key={self.secrets.openrouter_status()}"]
        if subject == "models":
            routes = current.models.to_mapping()
            return [
                f"models default={current.model} chat={routes['chat'] or current.model} research={routes['research'] or current.model} code={routes['code'] or current.model}"
            ]
        return [f"unknown show target '{subject}'. Use system, params, or models."]

    def _handle_context(self, tokens: list[str]) -> list[str]:
        if not tokens or tokens[0].lower() == "status":
            scope = tokens[1] if len(tokens) > 1 else None
            return [self.context_status(scope)]

        action = tokens[0].lower()
        if action != "reset":
            return [f"usage: {self.prefix} context [status|reset] [channel|nick] [beans]"]

        remaining, password_ok = strip_admin_password(tokens[1:], self.admin_password)
        if not password_ok:
            return ["admin password required"]
        scope = remaining[0] if remaining else None
        return [self.reset_history(scope)]

    def _handle_set(self, tokens: list[str]) -> list[str]:
        tokens, password_ok = strip_admin_password(tokens, self.admin_password)
        if not password_ok:
            return ["admin password required"]
        if len(tokens) < 2:
            return [f"usage: {self.prefix} set <system|model|temperature|top_p|max_tokens|stream|openrouter_key> <value> beans"]

        current = self.store.current()
        setting = tokens[0].lower()
        value_tokens = tokens[1:]

        if setting == "system":
            prompt = " ".join(value_tokens).strip()
            if not prompt:
                return ["system prompt cannot be empty"]
            current.set_system_prompt(prompt)
            return [f"system prompt updated: {current.system_excerpt(120)}"]

        if setting == "openrouter_key":
            secret = " ".join(value_tokens).strip()
            if not secret:
                return ["openrouter_key cannot be empty"]
            self.secrets.set_openrouter_api_key(secret)
            self.set_api_key(self.secrets.openrouter_api_key)
            return ["OpenRouter API key updated"]

        value = value_tokens[0]
        if setting == "model":
            current.set_model(value)
            return [f"model set to {current.model}"]

        if setting == "temperature":
            try:
                actual = current.set_temperature(float(value))
            except ValueError:
                return ["temperature must be a number"]
            return [f"temperature set to {actual:.2f}"]

        if setting == "top_p":
            try:
                actual = current.set_top_p(float(value))
            except ValueError:
                return ["top_p must be a number"]
            return [f"top_p set to {actual:.2f}"]

        if setting == "max_tokens":
            try:
                actual = current.set_max_tokens(int(value))
            except ValueError:
                return ["max_tokens must be an integer"]
            return [f"max_tokens set to {actual}"]

        if setting == "stream":
            lowered = value.lower()
            if lowered not in {"on", "off"}:
                return ["stream must be 'on' or 'off'"]
            current.set_stream(lowered == "on")
            return [f"stream set to {lowered}"]

        if setting == "reply_interval_seconds":
            try:
                actual = current.set_reply_interval_seconds(float(value))
            except ValueError:
                return ["reply_interval_seconds must be a number"]
            return [f"reply_interval_seconds set to {actual:.0f}"]

        return [f"unknown setting '{setting}'"]

    def _handle_clear(self, tokens: list[str]) -> list[str]:
        tokens, password_ok = strip_admin_password(tokens, self.admin_password)
        if not password_ok:
            return ["admin password required"]
        if len(tokens) != 1:
            return [f"usage: {self.prefix} clear openrouter_key beans"]
        if tokens[0].lower() != "openrouter_key":
            return [f"unknown clear target '{tokens[0]}'"]

        self.secrets.clear_openrouter_api_key()
        self.set_api_key(None)
        return ["OpenRouter API key cleared"]

    def _handle_reset(self, tokens: list[str]) -> list[str]:
        tokens, password_ok = strip_admin_password(tokens, self.admin_password)
        if not password_ok:
            return ["admin password required"]
        if tokens:
            return [f"usage: {self.prefix} reset beans"]

        current = self.store.reset()
        return [f"runtime config reset. {current.params_summary()} openrouter_key={self.secrets.openrouter_status()}"]

    def _handle_save(self, tokens: list[str]) -> list[str]:
        tokens, password_ok = strip_admin_password(tokens, self.admin_password)
        if not password_ok:
            return ["admin password required"]
        if len(tokens) != 1 or tokens[0].lower() != "runtime":
            return [f"usage: {self.prefix} save runtime beans"]
        return [self.persist_runtime()]

    def _handle_approvals(self, tokens: list[str]) -> list[str]:
        if tokens:
            return [f"usage: {self.prefix} approvals"]
        return [self.list_approvals()]

    def _handle_approve(self, tokens: list[str], actor: str | None, is_private: bool) -> list[str]:
        tokens, password_ok = strip_admin_password(tokens, self.admin_password)
        if not password_ok:
            return ["admin password required"]
        if len(tokens) != 1:
            return [f"usage: {self.prefix} approve <id> beans"]
        return [self.approve_action(tokens[0], actor or "", is_private)]

    def _handle_reject(self, tokens: list[str], actor: str | None, is_private: bool) -> list[str]:
        tokens, password_ok = strip_admin_password(tokens, self.admin_password)
        if not password_ok:
            return ["admin password required"]
        if len(tokens) != 1:
            return [f"usage: {self.prefix} reject <id> beans"]
        return [self.reject_action(tokens[0], actor or "", is_private)]
