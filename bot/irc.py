from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass, field
import inspect
import logging
import ssl as ssl_module
import time
from typing import Callable, Literal


LOGGER = logging.getLogger("beatrice.irc")

DEFAULT_MAX_LINE_BYTES = 512
DEFAULT_MESSAGE_LENGTH = 400
FLOOD_RATE = 3
FLOOD_WINDOW = 1.0
RECONNECT_INITIAL = 5
RECONNECT_MAX = 300
DEFAULT_WHOIS_TIMEOUT = 10.0
RECENT_NICK_CHANGE_LIMIT = 20
NICK_PREFIXES = "~&@%+"


@dataclass(frozen=True)
class WhoisInfo:
    nick: str
    user: str = ""
    host: str = ""
    realname: str = ""
    server: str = ""
    server_info: str = ""
    channels: tuple[str, ...] = ()
    idle_seconds: int | None = None
    signon_time: int | None = None
    is_operator: bool = False


@dataclass(frozen=True)
class WhoisResult:
    status: Literal["ok", "not_found", "error"]
    nick: str
    info: WhoisInfo | None = None
    error: str | None = None


@dataclass(frozen=True)
class NickChange:
    old_nick: str
    new_nick: str
    changed_at: float


@dataclass
class _WhoisState:
    requested_nick: str
    future: asyncio.Future[WhoisResult]
    nick: str = ""
    user: str = ""
    host: str = ""
    realname: str = ""
    server: str = ""
    server_info: str = ""
    channels: list[str] = field(default_factory=list)
    idle_seconds: int | None = None
    signon_time: int | None = None
    is_operator: bool = False

    def build_info(self) -> WhoisInfo:
        return WhoisInfo(
            nick=self.nick or self.requested_nick,
            user=self.user,
            host=self.host,
            realname=self.realname,
            server=self.server,
            server_info=self.server_info,
            channels=tuple(self.channels),
            idle_seconds=self.idle_seconds,
            signon_time=self.signon_time,
            is_operator=self.is_operator,
        )


def split_message(message: str, limit: int = DEFAULT_MESSAGE_LENGTH) -> list[str]:
    cleaned = " ".join(message.split())
    if not cleaned:
        return [""]
    if len(cleaned) <= limit:
        return [cleaned]

    chunks: list[str] = []
    remaining = cleaned
    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break
        split_at = remaining.rfind(" ", 0, limit)
        if split_at == -1:
            split_at = limit
        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:].lstrip()
    return chunks


def fit_message_bytes(message: str, limit: int) -> str:
    if limit <= 0:
        return ""
    encoded = message.encode("utf-8")
    if len(encoded) <= limit:
        return message

    trimmed = encoded[:limit]
    while trimmed:
        try:
            return trimmed.decode("utf-8")
        except UnicodeDecodeError:
            trimmed = trimmed[:-1]
    return ""


class IRCClient:
    def __init__(
        self,
        host: str,
        port: int,
        nick: str,
        user: str,
        realname: str,
        password: str | None = None,
        use_ssl: bool = False,
        message_length: int = DEFAULT_MESSAGE_LENGTH,
        max_line_bytes: int = DEFAULT_MAX_LINE_BYTES,
    ) -> None:
        self.host = host
        self.port = port
        self.nick = nick
        self._desired_nick = nick
        self.user = user
        self.realname = realname
        self.password = password
        self.use_ssl = use_ssl
        self.message_length = message_length
        self.max_line_bytes = max_line_bytes

        self.server_name = ""
        self.connected = False
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._handlers: dict[str, list[Callable]] = defaultdict(list)
        self._shutdown = False
        self._reconnect_delay = RECONNECT_INITIAL
        self._send_times: list[float] = []
        self._send_lock = asyncio.Lock()
        self._whois_lock = asyncio.Lock()
        self._pending_whois: _WhoisState | None = None
        self._channel_names: dict[str, str] = {}
        self._joined_channels: dict[str, str] = {}
        self._channel_users: dict[str, dict[str, str]] = {}
        self._channel_topics: dict[str, str] = {}
        self._recent_nick_changes: deque[NickChange] = deque(maxlen=RECENT_NICK_CHANGE_LIMIT)
        self._pending_names: dict[str, dict[str, str]] = {}

    def on(self, event: str, callback: Callable) -> None:
        self._handlers[event].append(callback)

    async def emit(self, event: str, *args) -> None:
        for callback in self._handlers.get(event, []):
            result = callback(*args)
            if inspect.isawaitable(result):
                await result

    def joined_channels(self) -> tuple[str, ...]:
        return tuple(sorted(self._joined_channels.values(), key=str.casefold))

    def known_channels(self) -> tuple[str, ...]:
        return tuple(sorted(self._channel_names.values(), key=str.casefold))

    def channel_users(self, channel: str) -> tuple[str, ...]:
        users = self._channel_users.get(self._normalize_channel(channel), {})
        return tuple(sorted(users.values(), key=str.casefold))

    def channel_topic(self, channel: str) -> str | None:
        return self._channel_topics.get(self._normalize_channel(channel))

    def recent_nick_changes(self, limit: int | None = None) -> tuple[NickChange, ...]:
        changes = tuple(self._recent_nick_changes)
        if limit is None or limit >= len(changes):
            return changes
        return changes[-limit:]

    def environment_state(self) -> dict[str, object]:
        channels = []
        for channel in self.known_channels():
            channel_key = self._normalize_channel(channel)
            channels.append(
                {
                    "name": channel,
                    "joined": channel_key in self._joined_channels,
                    "topic": self.channel_topic(channel),
                    "users": list(self.channel_users(channel)),
                }
            )
        return {
            "joined_channels": list(self.joined_channels()),
            "channels": channels,
            "recent_nick_changes": [
                {
                    "old_nick": change.old_nick,
                    "new_nick": change.new_nick,
                    "changed_at": change.changed_at,
                }
                for change in self._recent_nick_changes
            ],
        }

    async def connect(self) -> None:
        ssl_context = ssl_module.create_default_context() if self.use_ssl else None
        self._reader, self._writer = await asyncio.open_connection(self.host, self.port, ssl=ssl_context)
        self.connected = True
        self._send_times.clear()

        if self.password:
            await self.send_raw(f"PASS {self.password}")
        await self.send_raw(f"NICK {self._desired_nick}")
        await self.send_raw(f"USER {self.user} 0 * :{self.realname}")

    async def disconnect(self, quit_message: str | None = None) -> None:
        if quit_message is None:
            quit_message = f"{self.nick} signing off"
        self._shutdown = True
        self._fail_pending_whois("disconnected")
        if self._writer is not None and not self._writer.is_closing():
            try:
                await self.send_raw(f"QUIT :{quit_message}")
            except Exception:  # pragma: no cover - best effort
                pass
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except Exception:  # pragma: no cover - best effort
                pass
        self._reset_environment_state()
        self.server_name = ""
        self.connected = False

    async def run(self) -> None:
        self._shutdown = False
        await self.connect()

        while not self._shutdown:
            try:
                line = await self._read_line()
                if line is None:
                    if self._shutdown:
                        return
                    await self._reconnect()
                    continue
                await self._process_line(line)
            except asyncio.CancelledError:
                await self.disconnect()
                raise
            except Exception:
                LOGGER.exception("IRC loop error")
                if self._shutdown:
                    return
                await self._reconnect()

    async def join(self, channel: str) -> None:
        await self.send_raw(f"JOIN {channel}")

    async def whois(self, nick: str, timeout: float = DEFAULT_WHOIS_TIMEOUT) -> WhoisResult:
        nick = nick.strip()
        if not nick:
            return WhoisResult(status="error", nick="", error="nick is required")
        if not self.connected:
            return WhoisResult(status="error", nick=nick, error="not connected")

        async with self._whois_lock:
            state = _WhoisState(
                requested_nick=nick,
                future=asyncio.get_running_loop().create_future(),
                nick=nick,
            )
            self._pending_whois = state

            try:
                await self.send_raw(f"WHOIS {nick}")
                return await asyncio.wait_for(state.future, timeout=timeout)
            except asyncio.CancelledError:
                raise
            except asyncio.TimeoutError:
                return WhoisResult(status="error", nick=nick, error="WHOIS timed out")
            except Exception as exc:
                message = str(exc) or exc.__class__.__name__
                return WhoisResult(status="error", nick=nick, error=message)
            finally:
                if self._pending_whois is state:
                    self._pending_whois = None

    async def send_privmsg(self, target: str, message: str) -> None:
        for chunk in split_message(message, limit=self.message_length):
            await self.send_raw(f"PRIVMSG {target} :{chunk}")

    async def send_raw(self, line: str) -> None:
        if self._writer is None or self._writer.is_closing():
            return

        encoded = line.encode("utf-8")
        if len(encoded) > self.max_line_bytes - 2:
            if " :" in line:
                prefix, trailing = line.split(" :", 1)
                prefix_bytes = len((prefix + " :").encode("utf-8"))
                trailing_limit = max(0, self.max_line_bytes - 2 - prefix_bytes)
                line = prefix + " :" + fit_message_bytes(trailing, trailing_limit)
                encoded = line.encode("utf-8")
            else:
                encoded = encoded[: self.max_line_bytes - 2]

        async with self._send_lock:
            now = time.monotonic()
            self._send_times = [stamp for stamp in self._send_times if now - stamp < FLOOD_WINDOW]
            if len(self._send_times) >= FLOOD_RATE:
                wait_for = self._send_times[0] + FLOOD_WINDOW - now
                if wait_for > 0:
                    await asyncio.sleep(wait_for)

            self._send_times.append(time.monotonic())
            self._writer.write(encoded + b"\r\n")
            await self._writer.drain()

    async def _reconnect(self) -> None:
        self.connected = False
        self._reset_environment_state()
        self.server_name = ""
        self._fail_pending_whois("connection lost")
        if self._writer is not None and not self._writer.is_closing():
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except Exception:  # pragma: no cover - best effort
                pass

        delay = self._reconnect_delay
        LOGGER.warning("Connection lost, reconnecting in %ss", int(delay))
        await asyncio.sleep(delay)
        self._reconnect_delay = min(self._reconnect_delay * 2, RECONNECT_MAX)
        await self.connect()

    async def _read_line(self) -> str | None:
        if self._reader is None:
            return None
        data = await self._reader.readline()
        if not data:
            return None
        return data.decode("utf-8", errors="replace").strip()

    async def _process_line(self, line: str) -> None:
        prefix, source_nick, command, params = self.parse_line(line)
        if command == "PING":
            token = params[0] if params else ""
            await self.send_raw(f"PONG :{token}")
            return

        if command == "001":
            self.nick = params[0] if params else self._desired_nick
            self._reset_environment_state()
            self.server_name = prefix
            self._reconnect_delay = RECONNECT_INITIAL
            await self.emit("connected", self.server_name)
            return

        if command == "331" and len(params) >= 2:
            channel = params[1]
            self._remember_channel(channel)
            self._channel_topics.pop(self._normalize_channel(channel), None)
            return

        if command == "332" and len(params) >= 3:
            channel = params[1]
            topic = params[2]
            self._remember_channel(channel)
            self._set_channel_topic(channel, topic)
            await self.emit("topic", source_nick, prefix, channel, topic)
            return

        if command == "353" and len(params) >= 4:
            channel = params[2]
            self._remember_channel(channel)
            names = self._pending_names.setdefault(self._normalize_channel(channel), {})
            for raw_nick in params[3].split():
                nick = raw_nick.lstrip(NICK_PREFIXES)
                if nick:
                    names[self._normalize_nick(nick)] = nick
            return

        if command == "366" and len(params) >= 2:
            channel = params[1]
            channel_key = self._normalize_channel(channel)
            self._remember_channel(channel)
            names = self._pending_names.pop(channel_key, {})
            self._channel_users[channel_key] = names
            if self._normalize_nick(self.nick) in names:
                self._joined_channels[channel_key] = channel
            await self.emit("names", channel, tuple(sorted(names.values(), key=str.casefold)))
            return

        if command in {"311", "312", "313", "317", "319", "318", "401"}:
            self._handle_whois_numeric(command, params)
            return

        if command == "433":
            alternate = f"{self._desired_nick}_"
            self.nick = alternate
            await self.send_raw(f"NICK {alternate}")
            return

        if command == "JOIN" and params:
            channel = params[0]
            self._remember_channel(channel)
            self._add_channel_user(channel, source_nick)
            if source_nick.casefold() == self.nick.casefold():
                self._joined_channels[self._normalize_channel(channel)] = channel
            await self.emit("join", source_nick, prefix, channel)
            return

        if command == "PART" and params:
            channel = params[0]
            reason = params[1] if len(params) >= 2 else ""
            if source_nick.casefold() == self.nick.casefold():
                self._drop_channel(channel)
            else:
                self._discard_channel_user(channel, source_nick)
            await self.emit("part", source_nick, prefix, channel, reason)
            return

        if command == "QUIT":
            reason = params[0] if params else ""
            self._remove_user_everywhere(source_nick)
            await self.emit("quit", source_nick, prefix, reason)
            return

        if command == "NICK" and params:
            new_nick = params[0]
            old_nick = source_nick
            self._recent_nick_changes.append(NickChange(old_nick=old_nick, new_nick=new_nick, changed_at=time.time()))
            self._rename_user_everywhere(old_nick, new_nick)
            if old_nick.casefold() == self.nick.casefold():
                self.nick = new_nick
            await self.emit("nick", old_nick, prefix, new_nick)
            return

        if command == "TOPIC" and len(params) >= 2:
            channel = params[0]
            topic = params[1]
            self._remember_channel(channel)
            self._set_channel_topic(channel, topic)
            await self.emit("topic", source_nick, prefix, channel, topic)
            return

        if command == "PRIVMSG" and len(params) >= 2:
            await self.emit("privmsg", source_nick, prefix, params[0], params[1])

    @staticmethod
    def parse_line(line: str) -> tuple[str, str, str, list[str]]:
        if line.startswith("@"):
            _, _, line = line.partition(" ")
            line = line.lstrip()

        prefix = ""
        source_nick = ""
        if line.startswith(":"):
            prefix_end = line.index(" ")
            prefix = line[1:prefix_end]
            line = line[prefix_end + 1:]
            source_nick = prefix.split("!", 1)[0]

        if " :" in line:
            head, trailing = line.split(" :", 1)
            parts = head.split()
            command = parts[0].upper() if parts else ""
            params = parts[1:] + [trailing]
        else:
            parts = line.split()
            command = parts[0].upper() if parts else ""
            params = parts[1:]

        return prefix, source_nick, command, params

    def _handle_whois_numeric(self, command: str, params: list[str]) -> None:
        state = self._pending_whois
        if state is None or len(params) < 2:
            return

        response_nick = params[1]
        if response_nick.casefold() != state.requested_nick.casefold():
            return

        state.nick = response_nick

        if command == "311":
            if len(params) >= 4:
                state.user = params[2]
                state.host = params[3]
            if params:
                state.realname = params[-1]
            return

        if command == "312":
            if len(params) >= 3:
                state.server = params[2]
            if params:
                state.server_info = params[-1]
            return

        if command == "313":
            state.is_operator = True
            return

        if command == "317":
            if len(params) >= 3:
                try:
                    state.idle_seconds = int(params[2])
                except ValueError:
                    pass
            if len(params) >= 4:
                try:
                    state.signon_time = int(params[3])
                except ValueError:
                    pass
            return

        if command == "319":
            if len(params) >= 3:
                for channel in params[2].split():
                    if channel and channel not in state.channels:
                        state.channels.append(channel)
            return

        if command == "401":
            self._finish_pending_whois(
                WhoisResult(status="not_found", nick=response_nick, error=params[-1] if params else None)
            )
            return

        if command == "318":
            self._finish_pending_whois(
                WhoisResult(status="ok", nick=state.nick or state.requested_nick, info=state.build_info())
            )

    def _finish_pending_whois(self, result: WhoisResult) -> None:
        state = self._pending_whois
        if state is None:
            return
        if not state.future.done():
            state.future.set_result(result)
        self._pending_whois = None

    def _fail_pending_whois(self, message: str) -> None:
        state = self._pending_whois
        if state is None:
            return
        self._finish_pending_whois(
            WhoisResult(status="error", nick=state.requested_nick, error=message)
        )

    @staticmethod
    def _normalize_channel(channel: str) -> str:
        return channel.casefold()

    @staticmethod
    def _normalize_nick(nick: str) -> str:
        return nick.casefold()

    def _reset_environment_state(self) -> None:
        self._channel_names.clear()
        self._joined_channels.clear()
        self._channel_users.clear()
        self._channel_topics.clear()
        self._recent_nick_changes.clear()
        self._pending_names.clear()

    def _remember_channel(self, channel: str) -> None:
        channel_key = self._normalize_channel(channel)
        self._channel_names[channel_key] = channel
        self._channel_users.setdefault(channel_key, {})

    def _channel_name(self, channel_key: str) -> str:
        return self._channel_names.get(channel_key, channel_key)

    def _add_channel_user(self, channel: str, nick: str) -> None:
        if not nick:
            return
        channel_key = self._normalize_channel(channel)
        users = self._channel_users.setdefault(channel_key, {})
        users[self._normalize_nick(nick)] = nick

    def _discard_channel_user(self, channel: str, nick: str) -> None:
        channel_key = self._normalize_channel(channel)
        users = self._channel_users.get(channel_key)
        if users is None:
            return
        users.pop(self._normalize_nick(nick), None)

    def _remove_user_everywhere(self, nick: str) -> None:
        if not nick:
            return
        nick_key = self._normalize_nick(nick)
        for users in self._channel_users.values():
            users.pop(nick_key, None)
        for users in self._pending_names.values():
            users.pop(nick_key, None)

    def _rename_user_everywhere(self, old_nick: str, new_nick: str) -> None:
        if not old_nick or not new_nick:
            return
        old_key = self._normalize_nick(old_nick)
        new_key = self._normalize_nick(new_nick)
        for users in self._channel_users.values():
            if old_key in users:
                users.pop(old_key, None)
                users[new_key] = new_nick
        for users in self._pending_names.values():
            if old_key in users:
                users.pop(old_key, None)
                users[new_key] = new_nick

    def _set_channel_topic(self, channel: str, topic: str) -> None:
        channel_key = self._normalize_channel(channel)
        if topic:
            self._channel_topics[channel_key] = topic
        else:
            self._channel_topics.pop(channel_key, None)

    def _drop_channel(self, channel: str) -> None:
        channel_key = self._normalize_channel(channel)
        self._channel_names.pop(channel_key, None)
        self._joined_channels.pop(channel_key, None)
        self._channel_users.pop(channel_key, None)
        self._channel_topics.pop(channel_key, None)
        self._pending_names.pop(channel_key, None)
