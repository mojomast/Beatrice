from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None


DEFAULT_MODEL = "deepseek/deepseek-v3.2"
DEFAULT_SYSTEM_PROMPT = (
    "You are Beatrice, a helpful IRC bot on irc.ussyco.de. "
    "Keep replies concise, accurate, and suitable for IRC conversations."
)
DEFAULT_COMMAND_PREFIX = "!bot"
DEFAULT_ADMIN_PASSWORD = "beans"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_SETTINGS_FILE = "bot/settings.json"
DEFAULT_SECRETS_FILE = "bot/secrets.json"
DEFAULT_RUNTIME_FILE = "data/beatrice_runtime.json"
DEFAULT_MEMORY_DB_FILE = "data/beatrice_memory.sqlite3"
DEFAULT_AUDIT_LOG_FILE = "data/beatrice_audit.jsonl"
DEFAULT_IRC_SERVER = "irc.ussyco.de"
DEFAULT_IRC_PORT = 6667
DEFAULT_IRC_NICK = "Beatrice"
DEFAULT_IRC_USER = "beatrice"
DEFAULT_IRC_REALNAME = "Beatrice OpenRouter IRC Bot"
DEFAULT_IRC_CHANNELS = ("#ussycode",)
DEFAULT_OPENROUTER_TITLE = "Beatrice IRC Bot"
DEFAULT_IRC_MESSAGE_LENGTH = 900
DEFAULT_IRC_MAX_LINE_BYTES = 2048
DEFAULT_HISTORY_TURNS = 4
DEFAULT_REPLY_INTERVAL_SECONDS = 8.0
DEFAULT_APPROVAL_TIMEOUT_SECONDS = 900.0

MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0
MIN_TOP_P = 0.0
MAX_TOP_P = 1.0
MIN_MAX_TOKENS = 1
MAX_MAX_TOKENS = 16384


class ConfigError(ValueError):
    """Raised when required runtime configuration is missing or invalid."""


def clamp_float(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def clamp_int(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, int(value)))


def parse_channels(raw_value: str) -> tuple[str, ...]:
    return tuple(channel.strip() for channel in raw_value.split(",") if channel.strip())


def parse_csv_values(raw_value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw_value.split(",") if part.strip())


def load_settings_file(path: str) -> dict:
    settings_path = Path(path)
    if not settings_path.is_absolute():
        settings_path = Path.cwd() / settings_path

    if not settings_path.exists():
        return {}

    with settings_path.open(encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ConfigError(f"Settings file must contain a JSON object: {settings_path}")
    return data


def settings_section(data: dict, key: str) -> dict:
    section = data.get(key, {})
    if not isinstance(section, dict):
        raise ConfigError(f"Settings section '{key}' must be an object")
    return section


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate


def load_json_object(path: str) -> dict:
    resolved = resolve_path(path)
    if not resolved.exists():
        return {}

    with resolved.open(encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ConfigError(f"JSON file must contain an object: {resolved}")
    return data


def write_json_object(path: str, payload: dict) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    temporary.replace(resolved)


def coerce_channels(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return parse_channels(value)
    if isinstance(value, (list, tuple)):
        channels = [str(channel).strip() for channel in value if str(channel).strip()]
        return tuple(channels)
    return ()


def coerce_text_values(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return parse_csv_values(value)
    if isinstance(value, (list, tuple)):
        values = [str(item).strip() for item in value if str(item).strip()]
        return tuple(values)
    return ()


def default_irc_user(nick: str) -> str:
    normalized = re.sub(r"[^a-z0-9_\-\[\]\\`^{}|]", "", nick.lower())
    return normalized or "beatrice"


@dataclass(frozen=True)
class RuntimeDefaults:
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    model: str = DEFAULT_MODEL
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 700
    stream: bool = False
    reply_interval_seconds: float = DEFAULT_REPLY_INTERVAL_SECONDS

    @classmethod
    def from_mapping(cls, data: dict | None) -> "RuntimeDefaults":
        if not data:
            return cls()
        return cls(
            system_prompt=str(data.get("system_prompt", DEFAULT_SYSTEM_PROMPT)).strip() or DEFAULT_SYSTEM_PROMPT,
            model=str(data.get("model", DEFAULT_MODEL)).strip() or DEFAULT_MODEL,
            temperature=clamp_float(data.get("temperature", 0.7), MIN_TEMPERATURE, MAX_TEMPERATURE),
            top_p=clamp_float(data.get("top_p", 1.0), MIN_TOP_P, MAX_TOP_P),
            max_tokens=clamp_int(data.get("max_tokens", 700), MIN_MAX_TOKENS, MAX_MAX_TOKENS),
            stream=bool(data.get("stream", False)),
            reply_interval_seconds=clamp_float(data.get("reply_interval_seconds", DEFAULT_REPLY_INTERVAL_SECONDS), 0.0, 3600.0),
        )


@dataclass
class RuntimeConfig:
    system_prompt: str
    model: str
    temperature: float
    top_p: float
    max_tokens: int
    stream: bool
    reply_interval_seconds: float

    @classmethod
    def from_defaults(cls, defaults: RuntimeDefaults | None = None) -> "RuntimeConfig":
        baseline = defaults or RuntimeDefaults()
        return cls(
            system_prompt=baseline.system_prompt,
            model=baseline.model,
            temperature=baseline.temperature,
            top_p=baseline.top_p,
            max_tokens=baseline.max_tokens,
            stream=baseline.stream,
            reply_interval_seconds=baseline.reply_interval_seconds,
        )

    def snapshot(self) -> "RuntimeConfig":
        return RuntimeConfig(
            system_prompt=self.system_prompt,
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stream=self.stream,
            reply_interval_seconds=self.reply_interval_seconds,
        )

    def reset(self, defaults: RuntimeDefaults) -> None:
        fresh = RuntimeConfig.from_defaults(defaults)
        self.system_prompt = fresh.system_prompt
        self.model = fresh.model
        self.temperature = fresh.temperature
        self.top_p = fresh.top_p
        self.max_tokens = fresh.max_tokens
        self.stream = fresh.stream
        self.reply_interval_seconds = fresh.reply_interval_seconds

    def set_system_prompt(self, value: str) -> str:
        self.system_prompt = value.strip()
        return self.system_prompt

    def set_model(self, value: str) -> str:
        self.model = value.strip()
        return self.model

    def set_temperature(self, value: float) -> float:
        self.temperature = clamp_float(value, MIN_TEMPERATURE, MAX_TEMPERATURE)
        return self.temperature

    def set_top_p(self, value: float) -> float:
        self.top_p = clamp_float(value, MIN_TOP_P, MAX_TOP_P)
        return self.top_p

    def set_max_tokens(self, value: int) -> int:
        self.max_tokens = clamp_int(value, MIN_MAX_TOKENS, MAX_MAX_TOKENS)
        return self.max_tokens

    def set_stream(self, enabled: bool) -> bool:
        self.stream = bool(enabled)
        return self.stream

    def set_reply_interval_seconds(self, value: float) -> float:
        self.reply_interval_seconds = clamp_float(value, 0.0, 3600.0)
        return self.reply_interval_seconds

    def system_excerpt(self, limit: int = 80) -> str:
        compact = " ".join(self.system_prompt.split())
        if len(compact) <= limit:
            return compact
        return f"{compact[: limit - 3]}..."

    def params_summary(self) -> str:
        stream_flag = "on" if self.stream else "off"
        return (
            f"model={self.model} temperature={self.temperature:.2f} "
            f"top_p={self.top_p:.2f} max_tokens={self.max_tokens} stream={stream_flag} "
            f"reply_interval_seconds={self.reply_interval_seconds:.0f}"
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "system_prompt": self.system_prompt,
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stream": self.stream,
            "reply_interval_seconds": self.reply_interval_seconds,
        }

    def apply_updates(self, updates: dict[str, object]) -> dict[str, object]:
        changed: dict[str, object] = {}
        for key, value in updates.items():
            if key == "system_prompt":
                actual = self.set_system_prompt(str(value))
            elif key == "model":
                actual = self.set_model(str(value))
            elif key == "temperature":
                actual = self.set_temperature(float(value))
            elif key == "top_p":
                actual = self.set_top_p(float(value))
            elif key == "max_tokens":
                actual = self.set_max_tokens(int(value))
            elif key == "stream":
                actual = self.set_stream(bool(value))
            elif key == "reply_interval_seconds":
                actual = self.set_reply_interval_seconds(float(value))
            else:
                raise ValueError(f"unknown runtime setting '{key}'")
            changed[key] = actual
        return changed


class RuntimeStore:
    def __init__(self, defaults: RuntimeDefaults | None = None, overrides: dict[str, object] | None = None) -> None:
        self.defaults = defaults or RuntimeDefaults()
        self._current = RuntimeConfig.from_defaults(self.defaults)
        if overrides:
            self._current.apply_updates(overrides)

    def current(self) -> RuntimeConfig:
        return self._current

    def snapshot(self) -> RuntimeConfig:
        return self._current.snapshot()

    def reset(self) -> RuntimeConfig:
        self._current.reset(self.defaults)
        return self._current

    def apply_updates(self, updates: dict[str, object]) -> dict[str, object]:
        return self._current.apply_updates(updates)

    def persist(self, path: str) -> None:
        write_json_object(path, self._current.to_mapping())


@dataclass
class SecretStore:
    openrouter_api_key: str | None = None
    secrets_file: str = DEFAULT_SECRETS_FILE

    @classmethod
    def from_file(cls, secrets_file: str, api_key: str | None = None) -> "SecretStore":
        data = load_json_object(secrets_file)
        stored_key = data.get("openrouter_api_key")
        if not isinstance(stored_key, str):
            stored_key = None
        return cls(openrouter_api_key=api_key or stored_key, secrets_file=secrets_file)

    def has_openrouter_api_key(self) -> bool:
        return bool(self.openrouter_api_key)

    def set_openrouter_api_key(self, value: str) -> None:
        cleaned = value.strip()
        self.openrouter_api_key = cleaned or None
        self._persist()

    def clear_openrouter_api_key(self) -> None:
        self.openrouter_api_key = None
        self._persist()

    def openrouter_status(self) -> str:
        return "configured" if self.has_openrouter_api_key() else "missing"

    def _persist(self) -> None:
        payload: dict[str, str] = {}
        if self.openrouter_api_key:
            payload["openrouter_api_key"] = self.openrouter_api_key
        write_json_object(self.secrets_file, payload)


@dataclass(frozen=True)
class BotSettings:
    openrouter_api_key: str | None = None
    irc_server: str = DEFAULT_IRC_SERVER
    irc_port: int = DEFAULT_IRC_PORT
    irc_nick: str = DEFAULT_IRC_NICK
    irc_user: str = DEFAULT_IRC_USER
    irc_realname: str = DEFAULT_IRC_REALNAME
    irc_channels: tuple[str, ...] = DEFAULT_IRC_CHANNELS
    irc_password: str | None = None
    irc_message_length: int = DEFAULT_IRC_MESSAGE_LENGTH
    irc_max_line_bytes: int = DEFAULT_IRC_MAX_LINE_BYTES
    command_prefix: str = DEFAULT_COMMAND_PREFIX
    admin_password: str = DEFAULT_ADMIN_PASSWORD
    admin_nicks: tuple[str, ...] = ()
    approval_timeout_seconds: float = DEFAULT_APPROVAL_TIMEOUT_SECONDS
    openrouter_base_url: str = DEFAULT_OPENROUTER_BASE_URL
    openrouter_http_referer: str | None = None
    openrouter_title: str = DEFAULT_OPENROUTER_TITLE
    settings_file: str = DEFAULT_SETTINGS_FILE
    secrets_file: str = DEFAULT_SECRETS_FILE
    runtime_file: str = DEFAULT_RUNTIME_FILE
    memory_db_file: str = DEFAULT_MEMORY_DB_FILE
    audit_log_file: str = DEFAULT_AUDIT_LOG_FILE
    history_turns: int = DEFAULT_HISTORY_TURNS
    runtime_defaults: RuntimeDefaults = RuntimeDefaults()

    @classmethod
    def from_env(cls) -> "BotSettings":
        if load_dotenv is not None:
            load_dotenv()

        settings_file = os.getenv("BOT_SETTINGS_FILE", DEFAULT_SETTINGS_FILE).strip() or DEFAULT_SETTINGS_FILE
        secrets_file = os.getenv("BOT_SECRETS_FILE", DEFAULT_SECRETS_FILE).strip() or DEFAULT_SECRETS_FILE
        data = load_settings_file(settings_file)
        irc = settings_section(data, "irc")
        bot = settings_section(data, "bot")
        openrouter = settings_section(data, "openrouter")
        runtime_defaults = RuntimeDefaults.from_mapping(settings_section(data, "defaults"))

        api_key = os.getenv("OPENROUTER_API_KEY", "").strip() or None

        nick = os.getenv("IRC_NICK", str(irc.get("nick", DEFAULT_IRC_NICK))).strip() or DEFAULT_IRC_NICK
        user = os.getenv("IRC_USER", "").strip() or default_irc_user(nick)
        realname = (
            os.getenv("IRC_REALNAME", str(irc.get("realname", DEFAULT_IRC_REALNAME))).strip()
            or DEFAULT_IRC_REALNAME
        )
        prefix = os.getenv("BOT_COMMAND_PREFIX", str(bot.get("command_prefix", DEFAULT_COMMAND_PREFIX))).strip() or DEFAULT_COMMAND_PREFIX

        if "IRC_CHANNEL" in os.environ:
            channels = parse_channels(os.getenv("IRC_CHANNEL", ""))
        else:
            channels = coerce_channels(irc.get("channels", DEFAULT_IRC_CHANNELS)) or DEFAULT_IRC_CHANNELS

        try:
            port = int(os.getenv("IRC_PORT", str(irc.get("port", DEFAULT_IRC_PORT))))
        except ValueError as exc:
            raise ConfigError("IRC_PORT must be an integer") from exc

        try:
            irc_message_length = int(os.getenv("IRC_MESSAGE_LENGTH", str(irc.get("message_length", DEFAULT_IRC_MESSAGE_LENGTH))))
            irc_max_line_bytes = int(os.getenv("IRC_MAX_LINE_BYTES", str(irc.get("max_line_bytes", DEFAULT_IRC_MAX_LINE_BYTES))))
            history_turns = int(os.getenv("BOT_HISTORY_TURNS", str(bot.get("history_turns", DEFAULT_HISTORY_TURNS))))
        except ValueError as exc:
            raise ConfigError("IRC message length and history settings must be integers") from exc

        try:
            approval_timeout_seconds = clamp_float(
                os.getenv(
                    "BOT_APPROVAL_TIMEOUT_SECONDS",
                    str(bot.get("approval_timeout_seconds", DEFAULT_APPROVAL_TIMEOUT_SECONDS)),
                ),
                30.0,
                86400.0,
            )
        except ValueError as exc:
            raise ConfigError("BOT_APPROVAL_TIMEOUT_SECONDS must be a number") from exc

        if "BOT_ADMIN_NICKS" in os.environ:
            admin_nicks = parse_csv_values(os.getenv("BOT_ADMIN_NICKS", ""))
        else:
            admin_nicks = coerce_text_values(bot.get("admin_nicks", ()))

        return cls(
            openrouter_api_key=api_key,
            irc_server=os.getenv("IRC_SERVER", str(irc.get("server", DEFAULT_IRC_SERVER))).strip() or DEFAULT_IRC_SERVER,
            irc_port=port,
            irc_nick=nick,
            irc_user=user,
            irc_realname=realname,
            irc_channels=channels,
            irc_password=os.getenv("IRC_PASSWORD", str(irc.get("password", ""))).strip() or None,
            irc_message_length=irc_message_length,
            irc_max_line_bytes=irc_max_line_bytes,
            command_prefix=prefix,
            admin_password=os.getenv("BOT_ADMIN_PASSWORD", str(bot.get("admin_password", DEFAULT_ADMIN_PASSWORD))).strip() or DEFAULT_ADMIN_PASSWORD,
            admin_nicks=admin_nicks,
            approval_timeout_seconds=approval_timeout_seconds,
            openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", str(openrouter.get("base_url", DEFAULT_OPENROUTER_BASE_URL))).strip() or DEFAULT_OPENROUTER_BASE_URL,
            openrouter_http_referer=os.getenv("OPENROUTER_HTTP_REFERER", str(openrouter.get("http_referer", ""))).strip() or None,
            openrouter_title=os.getenv("OPENROUTER_TITLE", str(openrouter.get("title", DEFAULT_OPENROUTER_TITLE))).strip() or DEFAULT_OPENROUTER_TITLE,
            settings_file=settings_file,
            secrets_file=secrets_file,
            runtime_file=os.getenv("BOT_RUNTIME_FILE", str(bot.get("runtime_file", DEFAULT_RUNTIME_FILE))).strip() or DEFAULT_RUNTIME_FILE,
            memory_db_file=os.getenv("BOT_MEMORY_DB_FILE", str(bot.get("memory_db_file", DEFAULT_MEMORY_DB_FILE))).strip() or DEFAULT_MEMORY_DB_FILE,
            audit_log_file=os.getenv("BOT_AUDIT_LOG_FILE", str(bot.get("audit_log_file", DEFAULT_AUDIT_LOG_FILE))).strip() or DEFAULT_AUDIT_LOG_FILE,
            history_turns=history_turns,
            runtime_defaults=runtime_defaults,
        )
