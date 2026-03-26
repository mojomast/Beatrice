from __future__ import annotations

import asyncio
from dataclasses import dataclass
import os
from pathlib import Path
import signal
import time

from .audit import AuditLogger
from .config import BotSettings, default_irc_user, load_json_object, resolve_path, write_json_object


CHILD_STATE_RUNNING = "running"
CHILD_STATE_STOPPED = "stopped"
CHILD_STATE_FAILED = "failed"
CHILD_STATE_STARTING = "starting"
CHILD_STATE_STOPPING = "stopping"
CHILD_RESPONSE_MODE_ADDRESSED_ONLY = "addressed_only"
CHILD_RESPONSE_MODE_AMBIENT = "ambient"


def _now() -> float:
    return time.time()


def _normalize_child_id(value: str) -> str:
    cleaned = "".join(ch for ch in value.strip().lower() if ch.isalnum() or ch in {"-", "_"})
    if not cleaned:
        raise ValueError("child bot id must contain letters, numbers, '-' or '_'")
    return cleaned[:40]


def _normalize_prompt(value: str) -> str:
    compact = " ".join(value.split())
    if not compact:
        raise ValueError("child bot system prompt cannot be empty")
    return compact[:4000]


def normalize_response_mode(value: str) -> str:
    cleaned = "_".join(value.strip().lower().split())
    if cleaned in {CHILD_RESPONSE_MODE_ADDRESSED_ONLY, "reply_when_addressed", "addressed", "direct_only"}:
        return CHILD_RESPONSE_MODE_ADDRESSED_ONLY
    if cleaned in {CHILD_RESPONSE_MODE_AMBIENT, "natural", "natural_chat", "ambient_chat"}:
        return CHILD_RESPONSE_MODE_AMBIENT
    raise ValueError("response_mode must be 'addressed_only' or 'ambient'")


@dataclass(frozen=True)
class ChildBotSpec:
    child_id: str
    nick: str
    channels: tuple[str, ...]
    system_prompt: str
    model: str
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 180
    reply_interval_seconds: float = 4.0
    response_mode: str = CHILD_RESPONSE_MODE_ADDRESSED_ONLY
    enabled: bool = True

    @classmethod
    def from_mapping(cls, child_id: str, payload: object, default_model: str) -> "ChildBotSpec":
        if not isinstance(payload, dict):
            raise ValueError(f"child bot '{child_id}' must be an object")
        nick = str(payload.get("nick", child_id)).strip()
        if not nick:
            raise ValueError(f"child bot '{child_id}' must define a nick")
        raw_channels = payload.get("channels", ())
        if isinstance(raw_channels, str):
            channels = tuple(part.strip() for part in raw_channels.split(",") if part.strip())
        elif isinstance(raw_channels, (list, tuple)):
            channels = tuple(str(item).strip() for item in raw_channels if str(item).strip())
        else:
            channels = ()
        if not channels:
            raise ValueError(f"child bot '{child_id}' must define at least one channel")
        return cls(
            child_id=_normalize_child_id(str(payload.get("id", child_id))),
            nick=nick,
            channels=channels,
            system_prompt=_normalize_prompt(str(payload.get("system_prompt", "")).strip()),
            model=str(payload.get("model", default_model)).strip() or default_model,
            temperature=float(payload.get("temperature", 0.7)),
            top_p=float(payload.get("top_p", 1.0)),
            max_tokens=int(payload.get("max_tokens", 180)),
            reply_interval_seconds=float(payload.get("reply_interval_seconds", 4.0)),
            response_mode=normalize_response_mode(str(payload.get("response_mode", CHILD_RESPONSE_MODE_ADDRESSED_ONLY))),
            enabled=bool(payload.get("enabled", True)),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "id": self.child_id,
            "nick": self.nick,
            "channels": list(self.channels),
            "system_prompt": self.system_prompt,
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "reply_interval_seconds": self.reply_interval_seconds,
            "response_mode": self.response_mode,
            "enabled": self.enabled,
        }


@dataclass(frozen=True)
class ChildBotRuntimeState:
    child_id: str
    status: str = CHILD_STATE_STOPPED
    pid: int | None = None
    started_at: float | None = None
    stopped_at: float | None = None
    exit_code: int | None = None
    last_error: str | None = None

    @classmethod
    def from_mapping(cls, child_id: str, payload: object) -> "ChildBotRuntimeState":
        if not isinstance(payload, dict):
            return cls(child_id=child_id)
        return cls(
            child_id=child_id,
            status=str(payload.get("status", CHILD_STATE_STOPPED)).strip() or CHILD_STATE_STOPPED,
            pid=int(payload["pid"]) if payload.get("pid") is not None else None,
            started_at=float(payload["started_at"]) if payload.get("started_at") is not None else None,
            stopped_at=float(payload["stopped_at"]) if payload.get("stopped_at") is not None else None,
            exit_code=int(payload["exit_code"]) if payload.get("exit_code") is not None else None,
            last_error=str(payload.get("last_error")).strip() or None if payload.get("last_error") is not None else None,
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "status": self.status,
            "pid": self.pid,
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
            "exit_code": self.exit_code,
            "last_error": self.last_error,
        }


class ChildBotRegistry:
    def __init__(self, path: str, default_model: str) -> None:
        self.path = path
        self.default_model = default_model

    def load(self) -> dict[str, ChildBotSpec]:
        payload = load_json_object(self.path)
        raw_children = payload.get("children", {})
        if not isinstance(raw_children, dict):
            return {}
        children: dict[str, ChildBotSpec] = {}
        for child_id, child_payload in raw_children.items():
            spec = ChildBotSpec.from_mapping(str(child_id), child_payload, self.default_model)
            children[spec.child_id] = spec
        return children

    def save(self, children: dict[str, ChildBotSpec]) -> None:
        write_json_object(self.path, {"children": {child_id: spec.to_mapping() for child_id, spec in sorted(children.items())}})


class ChildBotStateStore:
    def __init__(self, path: str) -> None:
        self.path = path

    def load(self) -> dict[str, ChildBotRuntimeState]:
        payload = load_json_object(self.path)
        raw_states = payload.get("children", {})
        if not isinstance(raw_states, dict):
            return {}
        return {
            str(child_id): ChildBotRuntimeState.from_mapping(str(child_id), child_payload)
            for child_id, child_payload in raw_states.items()
        }

    def save(self, states: dict[str, ChildBotRuntimeState]) -> None:
        write_json_object(self.path, {"children": {child_id: state.to_mapping() for child_id, state in sorted(states.items())}})


class ChildBotManager:
    def __init__(self, settings: BotSettings, audit: AuditLogger) -> None:
        self.settings = settings
        self.audit = audit
        self.registry = ChildBotRegistry(settings.child_bots_file, settings.child_default_model)
        self.state_store = ChildBotStateStore(settings.child_state_file)
        self._children = self.registry.load()
        self._states = self.state_store.load()
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._watch_tasks: dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()

    def list_specs(self) -> list[ChildBotSpec]:
        return [self._children[key] for key in sorted(self._children)]

    def get_spec(self, child_id: str) -> ChildBotSpec | None:
        return self._children.get(child_id)

    def get_state(self, child_id: str) -> ChildBotRuntimeState:
        return self._states.get(child_id, ChildBotRuntimeState(child_id=child_id))

    def create_child(
        self,
        *,
        child_id: str,
        nick: str,
        channels: tuple[str, ...],
        system_prompt: str,
        model: str | None = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 180,
        reply_interval_seconds: float = 4.0,
        response_mode: str = CHILD_RESPONSE_MODE_ADDRESSED_ONLY,
        enabled: bool = True,
    ) -> ChildBotSpec:
        normalized_id = _normalize_child_id(child_id)
        if normalized_id in self._children:
            raise ValueError(f"child bot '{normalized_id}' already exists")
        spec = ChildBotSpec(
            child_id=normalized_id,
            nick=nick.strip(),
            channels=tuple(channel.strip() for channel in channels if channel.strip()),
            system_prompt=_normalize_prompt(system_prompt),
            model=(model or self.settings.child_default_model).strip() or self.settings.child_default_model,
            temperature=float(temperature),
            top_p=float(top_p),
            max_tokens=int(max_tokens),
            reply_interval_seconds=float(reply_interval_seconds),
            response_mode=normalize_response_mode(response_mode),
            enabled=bool(enabled),
        )
        if not spec.nick:
            raise ValueError("child bot nick cannot be empty")
        if not spec.channels:
            raise ValueError("child bot must define at least one channel")
        self._children[normalized_id] = spec
        self._states.setdefault(normalized_id, ChildBotRuntimeState(child_id=normalized_id))
        self.registry.save(self._children)
        self.state_store.save(self._states)
        return spec

    def remove_child(self, child_id: str) -> ChildBotSpec:
        normalized_id = _normalize_child_id(child_id)
        if normalized_id in self._processes:
            raise ValueError(f"child bot '{normalized_id}' is still running")
        spec = self._children.pop(normalized_id, None)
        if spec is None:
            raise ValueError(f"child bot '{normalized_id}' not found")
        self._states.pop(normalized_id, None)
        self.registry.save(self._children)
        self.state_store.save(self._states)
        return spec

    def set_enabled(self, child_id: str, enabled: bool) -> ChildBotSpec:
        normalized_id = _normalize_child_id(child_id)
        spec = self._children.get(normalized_id)
        if spec is None:
            raise ValueError(f"child bot '{normalized_id}' not found")
        updated = ChildBotSpec(
            child_id=spec.child_id,
            nick=spec.nick,
            channels=spec.channels,
            system_prompt=spec.system_prompt,
            model=spec.model,
            temperature=spec.temperature,
            top_p=spec.top_p,
            max_tokens=spec.max_tokens,
            reply_interval_seconds=spec.reply_interval_seconds,
            response_mode=spec.response_mode,
            enabled=enabled,
        )
        self._children[normalized_id] = updated
        self.registry.save(self._children)
        return updated

    def update_child(self, child_id: str, **updates: object) -> ChildBotSpec:
        normalized_id = _normalize_child_id(child_id)
        spec = self._children.get(normalized_id)
        if spec is None:
            raise ValueError(f"child bot '{normalized_id}' not found")
        channels_value = updates.get("channels", spec.channels)
        if isinstance(channels_value, str):
            channels = tuple(part.strip() for part in channels_value.split(",") if part.strip())
        else:
            channels = tuple(str(part).strip() for part in channels_value if str(part).strip())
        updated = ChildBotSpec(
            child_id=spec.child_id,
            nick=str(updates.get("nick", spec.nick)).strip() or spec.nick,
            channels=channels or spec.channels,
            system_prompt=_normalize_prompt(str(updates.get("system_prompt", spec.system_prompt))),
            model=str(updates.get("model", spec.model)).strip() or spec.model,
            temperature=float(updates.get("temperature", spec.temperature)),
            top_p=float(updates.get("top_p", spec.top_p)),
            max_tokens=int(updates.get("max_tokens", spec.max_tokens)),
            reply_interval_seconds=float(updates.get("reply_interval_seconds", spec.reply_interval_seconds)),
            response_mode=normalize_response_mode(str(updates.get("response_mode", spec.response_mode))),
            enabled=bool(updates.get("enabled", spec.enabled)),
        )
        self._children[normalized_id] = updated
        self.registry.save(self._children)
        return updated

    async def start_enabled_children(self) -> None:
        for spec in self.list_specs():
            if spec.enabled:
                await self.start_child(spec.child_id)

    async def stop_all(self) -> None:
        for child_id in list(self._processes):
            await self.stop_child(child_id)

    async def start_child(self, child_id: str) -> ChildBotRuntimeState:
        normalized_id = _normalize_child_id(child_id)
        async with self._lock:
            spec = self._children.get(normalized_id)
            if spec is None:
                raise ValueError(f"child bot '{normalized_id}' not found")
            process = self._processes.get(normalized_id)
            if process is not None and process.returncode is None:
                return self.get_state(normalized_id)
            process = await asyncio.create_subprocess_exec(
                *self._child_argv(normalized_id),
                cwd=str(resolve_path(".")),
                env=self._child_env(spec),
                start_new_session=True,
            )
            state = ChildBotRuntimeState(
                child_id=normalized_id,
                status=CHILD_STATE_RUNNING,
                pid=process.pid,
                started_at=_now(),
                stopped_at=None,
                exit_code=None,
                last_error=None,
            )
            self._processes[normalized_id] = process
            self._states[normalized_id] = state
            self.state_store.save(self._states)
            self.audit.log_child_bot_event(
                child_id=normalized_id,
                action="start",
                status=state.status,
                nick=spec.nick,
                channels=spec.channels,
                model=spec.model,
                pid=process.pid,
            )
            task = asyncio.create_task(self._watch_child(normalized_id, process))
            self._watch_tasks[normalized_id] = task
            task.add_done_callback(lambda _task, key=normalized_id: self._watch_tasks.pop(key, None))
            return state

    async def stop_child(self, child_id: str) -> ChildBotRuntimeState:
        normalized_id = _normalize_child_id(child_id)
        async with self._lock:
            spec = self._children.get(normalized_id)
            if spec is None:
                raise ValueError(f"child bot '{normalized_id}' not found")
            process = self._processes.get(normalized_id)
            if process is None or process.returncode is not None:
                state = ChildBotRuntimeState(
                    child_id=normalized_id,
                    status=CHILD_STATE_STOPPED,
                    pid=None,
                    started_at=self.get_state(normalized_id).started_at,
                    stopped_at=_now(),
                    exit_code=process.returncode if process is not None else None,
                    last_error=None,
                )
                self._states[normalized_id] = state
                self.state_store.save(self._states)
                return state
            self._states[normalized_id] = ChildBotRuntimeState(
                child_id=normalized_id,
                status=CHILD_STATE_STOPPING,
                pid=process.pid,
                started_at=self.get_state(normalized_id).started_at,
                stopped_at=None,
                exit_code=None,
                last_error=None,
            )
            self.state_store.save(self._states)
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        try:
            await asyncio.wait_for(process.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()
        return self._mark_stopped(normalized_id, process.returncode, expected=True)

    async def _watch_child(self, child_id: str, process: asyncio.subprocess.Process) -> None:
        exit_code = await process.wait()
        expected = self._states.get(child_id, ChildBotRuntimeState(child_id=child_id)).status == CHILD_STATE_STOPPING
        self._mark_stopped(child_id, exit_code, expected=expected)

    def _mark_stopped(self, child_id: str, exit_code: int | None, expected: bool) -> ChildBotRuntimeState:
        self._processes.pop(child_id, None)
        status = CHILD_STATE_STOPPED if expected or exit_code == 0 else CHILD_STATE_FAILED
        error = None if status == CHILD_STATE_STOPPED else f"exit code {exit_code}"
        state = ChildBotRuntimeState(
            child_id=child_id,
            status=status,
            pid=None,
            started_at=self.get_state(child_id).started_at,
            stopped_at=_now(),
            exit_code=exit_code,
            last_error=error,
        )
        self._states[child_id] = state
        self.state_store.save(self._states)
        spec = self._children.get(child_id)
        self.audit.log_child_bot_event(
            child_id=child_id,
            action="stop" if expected else "exit",
            status=state.status,
            nick=spec.nick if spec else None,
            channels=spec.channels if spec else None,
            model=spec.model if spec else None,
            pid=None,
            exit_code=exit_code,
            error=error,
        )
        return state

    def describe_child(self, child_id: str) -> str:
        spec = self.get_spec(child_id)
        if spec is None:
            raise ValueError(f"child bot '{child_id}' not found")
        state = self.get_state(child_id)
        channels = ",".join(spec.channels)
        enabled = "on" if spec.enabled else "off"
        return (
            f"child {spec.child_id} nick={spec.nick} status={state.status} mode={spec.response_mode} enabled={enabled} "
            f"model={spec.model} channels={channels}"
        )

    def list_summary(self) -> str:
        specs = self.list_specs()
        if not specs:
            return "no child bots configured"
        return " | ".join(self.describe_child(spec.child_id) for spec in specs)

    def child_runtime_paths(self, child_id: str) -> dict[str, Path]:
        base = resolve_path(self.settings.child_data_dir) / child_id
        return {
            "base": base,
            "runtime": base / "runtime.json",
            "memory": base / "memory.sqlite3",
            "audit": base / "audit.jsonl",
            "log": base / "bot.log",
        }

    def _child_env(self, spec: ChildBotSpec) -> dict[str, str]:
        env = os.environ.copy()
        paths = self.child_runtime_paths(spec.child_id)
        paths["base"].mkdir(parents=True, exist_ok=True)
        env.update(
            {
                "OPENROUTER_API_KEY": self.settings.openrouter_api_key or "",
                "IRC_SERVER": self.settings.irc_server,
                "IRC_PORT": str(self.settings.irc_port),
                "IRC_PASSWORD": self.settings.irc_password or "",
                "IRC_NICK": spec.nick,
                "IRC_USER": default_irc_user(spec.nick),
                "IRC_REALNAME": f"{spec.nick} managed child bot",
                "IRC_CHANNEL": ",".join(spec.channels),
                "BOT_RUNTIME_FILE": str(paths["runtime"]),
                "BOT_MEMORY_DB_FILE": str(paths["memory"]),
                "BOT_AUDIT_LOG_FILE": str(paths["audit"]),
                "BOT_SECRETS_FILE": self.settings.secrets_file,
                "BOT_SETTINGS_FILE": self.settings.settings_file,
                "BOT_CHILD_MODE": "1",
                "BOT_COMMAND_PREFIX": self.settings.command_prefix,
                "BOT_ADMIN_PASSWORD": self.settings.admin_password,
                "BOT_CHILD_ID": spec.child_id,
                "BOT_CHILD_SYSTEM_PROMPT": spec.system_prompt,
                "BOT_CHILD_MODEL": spec.model,
                "BOT_CHILD_TEMPERATURE": str(spec.temperature),
                "BOT_CHILD_TOP_P": str(spec.top_p),
                "BOT_CHILD_MAX_TOKENS": str(spec.max_tokens),
                "BOT_CHILD_REPLY_INTERVAL_SECONDS": str(spec.reply_interval_seconds),
                "BOT_CHILD_RESPONSE_MODE": spec.response_mode,
            }
        )
        return env

    def _child_argv(self, child_id: str) -> tuple[str, ...]:
        python = os.environ.get("VIRTUAL_ENV")
        if python:
            python_path = Path(python) / "bin" / "python"
            if python_path.exists():
                executable = str(python_path)
            else:
                executable = os.environ.get("PYTHON", "python3")
        else:
            executable = os.environ.get("PYTHON", "python3")
        return (executable, "-m", "bot.child_runner", child_id)
