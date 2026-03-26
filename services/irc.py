"""
ussynet IRC connection layer.

Raw-socket asyncio IRC client for channel services on irc.ussy.net
(InspIRCd 4.8.0). No external libraries — stdlib only.

Usage:
    client = IRCClient("irc.ussy.host", 6667, "dancussy", "services",
                        "dancussy Channel Services")
    client.on("connected", my_on_connect)
    client.on("privmsg", my_on_privmsg)
    await client.run()
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import ssl as _ssl
import time
from collections import defaultdict
from typing import Callable

logger = logging.getLogger("ussynet.irc")

# IRC protocol limits
_MAX_LINE_BYTES = 512          # RFC 2812 max line length (including CRLF)
_MAX_MSG_SPLIT = 400           # safe payload limit for PRIVMSG/NOTICE bodies
_FLOOD_RATE = 3                # max messages per second
_FLOOD_WINDOW = 1.0            # window in seconds for rate limiting
_RECONNECT_INITIAL = 10        # initial reconnect delay (seconds)
_RECONNECT_MAX = 300           # max reconnect delay (seconds)


class IRCClient:
    """Asynchronous IRC client using raw sockets and asyncio streams.

    Provides connection management, IRC protocol sending/parsing, a simple
    event/callback system, flood protection, and automatic reconnection
    with exponential backoff.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        host: str,
        port: int,
        nick: str,
        ident: str,
        realname: str,
        oper_name: str | None = None,
        oper_password: str | None = None,
        use_ssl: bool = False,
    ) -> None:
        # Server / identity
        self.host = host
        self.port = port
        self.nick = nick
        self._desired_nick = nick
        self.ident = ident
        self.realname = realname
        self.oper_name = oper_name
        self.oper_password = oper_password
        self.use_ssl = use_ssl

        # Connection state
        self.connected: bool = False
        self._registered: bool = False  # True after 001 (RPL_WELCOME)
        self.channels: set[str] = set()
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._reconnect_delay: float = _RECONNECT_INITIAL

        # Event system — event name -> list of callbacks
        self._handlers: dict[str, list[Callable]] = defaultdict(list)

        # Shutdown flag — when True, the run() loop exits instead of reconnecting
        self._shutdown: bool = False

        # Flood protection state
        self._send_times: list[float] = []
        self._send_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Core connection
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open a TCP connection to the IRC server and register (NICK/USER)."""
        logger.info("Connecting to %s:%d (SSL=%s) …", self.host, self.port, self.use_ssl)

        ssl_ctx: _ssl.SSLContext | None = None
        if self.use_ssl:
            ssl_ctx = _ssl.create_default_context()

        self._reader, self._writer = await asyncio.open_connection(
            self.host, self.port, ssl=ssl_ctx,
        )

        self.connected = True
        self._send_times.clear()

        # IRC registration sequence
        await self.send_raw(f"NICK {self._desired_nick}")
        await self.send_raw(
            f"USER {self.ident} 0 * :{self.realname}"
        )

        logger.info("Registration sent as %s", self._desired_nick)

    async def disconnect(self, quit_msg: str = "ussynet services shutting down") -> None:
        """Send QUIT and close the transport.  Sets the shutdown flag so
        :meth:`run` will not attempt to reconnect."""
        self._shutdown = True
        if self._writer is not None and not self._writer.is_closing():
            try:
                await self.send_raw(f"QUIT :{quit_msg}")
            except Exception:
                pass  # best-effort
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass

        self.connected = False
        self._registered = False
        self.channels.clear()
        self._reader = None
        self._writer = None
        logger.info("Disconnected from %s", self.host)

    async def reconnect(self, delay: float | None = None) -> None:
        """Disconnect, wait *delay* seconds, then reconnect.

        Uses exponential backoff (doubling up to ``_RECONNECT_MAX``) when
        *delay* is ``None``.  Does nothing if shutdown was requested.
        """
        if self._shutdown:
            return

        # Disconnect without setting the shutdown flag
        if self._writer is not None and not self._writer.is_closing():
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
        self.connected = False
        self._registered = False
        self.channels.clear()
        self._reader = None
        self._writer = None
        await self.emit("disconnected")

        wait = delay if delay is not None else self._reconnect_delay
        logger.info("Reconnecting in %.0f s …", wait)
        await asyncio.sleep(wait)

        # Exponential backoff for the *next* reconnect attempt
        if delay is None:
            self._reconnect_delay = min(self._reconnect_delay * 2, _RECONNECT_MAX)

        await self.connect()

    async def run(self) -> None:
        """Main loop: connect, read lines, parse & dispatch.  Reconnect on drop."""
        self._shutdown = False
        await self.connect()

        while not self._shutdown:
            try:
                line = await self._read_line()
                if line is None:
                    # Connection lost (EOF)
                    logger.warning("Connection lost (EOF)")
                    self.connected = False
                    await self.emit("disconnected")
                    if self._shutdown:
                        return
                    await self.reconnect()
                    continue

                await self._process_line(line)

            except asyncio.CancelledError:
                logger.info("Run loop cancelled — shutting down")
                await self.disconnect()
                return

            except ConnectionError as exc:
                logger.warning("Connection error: %s", exc)
                self.connected = False
                await self.emit("disconnected")
                if self._shutdown:
                    return
                await self.reconnect()

            except Exception:
                logger.exception("Unhandled exception in run loop")
                self.connected = False
                await self.emit("disconnected")
                if self._shutdown:
                    return
                await self.reconnect()

    # ------------------------------------------------------------------
    # Sending — raw & helpers
    # ------------------------------------------------------------------

    async def send_raw(self, line: str) -> None:
        """Send a raw IRC line.  Appends ``\\r\\n``, encodes UTF-8, and
        enforces flood protection (≤ 3 msgs/s) and 512-byte max.
        """
        if self._writer is None or self._writer.is_closing():
            logger.warning("Cannot send — not connected: %s", line)
            return

        # Truncate to 510 bytes (+ CRLF = 512) if necessary
        encoded = line.encode("utf-8")
        if len(encoded) > _MAX_LINE_BYTES - 2:
            logger.warning("Line exceeds 510 bytes, truncating: %s", line)
            encoded = encoded[: _MAX_LINE_BYTES - 2]

        # Flood protection — sliding window
        async with self._send_lock:
            now = time.monotonic()
            # Prune timestamps older than the window
            self._send_times = [t for t in self._send_times if now - t < _FLOOD_WINDOW]

            if len(self._send_times) >= _FLOOD_RATE:
                # Wait until the oldest timestamp falls out of the window
                wait_until = self._send_times[0] + _FLOOD_WINDOW
                sleep_for = wait_until - now
                if sleep_for > 0:
                    logger.debug("Flood protection: sleeping %.3f s", sleep_for)
                    await asyncio.sleep(sleep_for)

            self._send_times.append(time.monotonic())

            self._writer.write(encoded + b"\r\n")
            await self._writer.drain()

        logger.debug(">> %s", line)

    async def send_privmsg(self, target: str, message: str) -> None:
        """Send a PRIVMSG, splitting long messages into chunks."""
        for chunk in _split_message(message, _MAX_MSG_SPLIT):
            await self.send_raw(f"PRIVMSG {target} :{chunk}")

    async def send_notice(self, target: str, message: str) -> None:
        """Send a NOTICE, splitting long messages into chunks."""
        for chunk in _split_message(message, _MAX_MSG_SPLIT):
            await self.send_raw(f"NOTICE {target} :{chunk}")

    async def join(self, channel: str) -> None:
        """Send JOIN for *channel*."""
        await self.send_raw(f"JOIN {channel}")

    async def part(self, channel: str, reason: str = "") -> None:
        """Send PART for *channel* with optional reason."""
        if reason:
            await self.send_raw(f"PART {channel} :{reason}")
        else:
            await self.send_raw(f"PART {channel}")

    async def kick(self, channel: str, nick: str, reason: str = "") -> None:
        """KICK *nick* from *channel*."""
        if reason:
            await self.send_raw(f"KICK {channel} {nick} :{reason}")
        else:
            await self.send_raw(f"KICK {channel} {nick}")

    async def set_mode(self, target: str, modes: str) -> None:
        """Send MODE *target* *modes*."""
        await self.send_raw(f"MODE {target} {modes}")

    async def set_topic(self, channel: str, topic: str) -> None:
        """Set the TOPIC of *channel*."""
        await self.send_raw(f"TOPIC {channel} :{topic}")

    async def oper_up(self, name: str, password: str) -> None:
        """Send OPER to gain IRC operator privileges."""
        await self.send_raw(f"OPER {name} {password}")

    async def sajoin(self, nick: str, channel: str) -> None:
        """InspIRCd SAJOIN — force *nick* into *channel*."""
        await self.send_raw(f"SAJOIN {nick} {channel}")

    async def sakick(self, channel: str, nick: str, reason: str = "") -> None:
        """InspIRCd SAKICK — force kick *nick* from *channel*."""
        if reason:
            await self.send_raw(f"SAKICK {channel} {nick} :{reason}")
        else:
            await self.send_raw(f"SAKICK {channel} {nick}")

    async def samode(self, target: str, modes: str) -> None:
        """InspIRCd SAMODE — set modes bypassing channel access checks."""
        await self.send_raw(f"SAMODE {target} {modes}")

    async def invite(self, nick: str, channel: str) -> None:
        """INVITE *nick* to *channel*."""
        await self.send_raw(f"INVITE {nick} {channel}")

    async def chghost(self, nick: str, new_host: str) -> None:
        """InspIRCd CHGHOST — change the displayed hostname of *nick*.

        Requires oper privileges and the ``chghost`` module loaded.
        """
        await self.send_raw(f"CHGHOST {nick} {new_host}")

    async def chgident(self, nick: str, new_ident: str) -> None:
        """InspIRCd CHGIDENT — change the ident (username) of *nick*.

        Requires oper privileges and the ``chgident`` module loaded.
        """
        await self.send_raw(f"CHGIDENT {nick} {new_ident}")

    async def change_nick(self, new_nick: str) -> None:
        """Change the bot's own nickname."""
        self._desired_nick = new_nick
        await self.send_raw(f"NICK {new_nick}")

    async def who(self, target: str) -> None:
        """Send WHO *target*."""
        await self.send_raw(f"WHO {target}")

    async def whois(self, nick: str) -> None:
        """Send WHOIS *nick*."""
        await self.send_raw(f"WHOIS {nick}")

    # ------------------------------------------------------------------
    # Event system
    # ------------------------------------------------------------------

    def on(self, event: str, callback: Callable) -> None:
        """Register *callback* for *event*.

        Multiple callbacks per event are allowed.  Coroutine callbacks
        will be awaited; regular functions are called normally.
        """
        self._handlers[event].append(callback)
        logger.debug("Registered handler for '%s': %s", event, callback.__qualname__)

    async def emit(self, event: str, *args, **kwargs) -> None:
        """Invoke all registered callbacks for *event*.

        Coroutines are awaited; plain functions are called directly.
        Exceptions in callbacks are logged but do not interrupt other
        handlers.
        """
        for handler in self._handlers.get(event, []):
            try:
                result = handler(*args, **kwargs)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.exception(
                    "Error in handler %s for event '%s'",
                    handler.__qualname__,
                    event,
                )

    # ------------------------------------------------------------------
    # IRC protocol — parsing
    # ------------------------------------------------------------------

    @staticmethod
    def parse_line(line: str) -> tuple[str, str, str, list[str]]:
        """Parse a raw IRC line into ``(prefix, source_nick, command, params)``.

        Handles IRCv3 message tags (lines starting with ``@``).

        * *prefix* — the full ``nick!user@host`` prefix (or server name),
          empty string if absent.
        * *source_nick* — the nick part extracted from the prefix
          (``nick!user@host`` → ``nick``), empty string if absent.
        * *command* — the IRC command or numeric (uppercased).
        * *params* — list of parameters; the trailing parameter (after
          ``:`` in the raw line) is the last element.
        """
        # Strip IRCv3 message tags
        if line.startswith("@"):
            # Tags section ends at the first space
            _tags, _, line = line.partition(" ")
            line = line.lstrip()

        prefix = ""
        source_nick = ""

        # Extract prefix
        if line.startswith(":"):
            prefix_end = line.index(" ")
            prefix = line[1:prefix_end]
            line = line[prefix_end + 1:]

            # Extract nick from prefix
            if "!" in prefix:
                source_nick = prefix.split("!", 1)[0]
            else:
                source_nick = prefix

        # Split command and params
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

    @staticmethod
    def parse_hostmask(prefix: str) -> tuple[str, str, str]:
        """Split ``nick!user@host`` into ``(nick, user, host)``.

        Returns ``(prefix, "", "")`` if the prefix doesn't contain ``!``
        and ``@``.
        """
        if "!" in prefix and "@" in prefix:
            nick, rest = prefix.split("!", 1)
            user, host = rest.split("@", 1)
            return nick, user, host
        return prefix, "", ""

    # ------------------------------------------------------------------
    # Nick reclaim — internal
    # ------------------------------------------------------------------

    async def _reclaim_nick(self) -> None:
        """Try to reclaim the desired nick every 30 seconds."""
        while self.connected and self.nick != self._desired_nick:
            await asyncio.sleep(30)
            if self._shutdown or not self.connected:
                return
            if self.nick != self._desired_nick:
                logger.info("Attempting to reclaim nick %s", self._desired_nick)
                await self.send_raw(f"NICK {self._desired_nick}")

    # ------------------------------------------------------------------
    # Line processing — internal
    # ------------------------------------------------------------------

    async def _read_line(self) -> str | None:
        """Read a single line from the server.

        Returns ``None`` on EOF / connection drop.
        """
        if self._reader is None:
            return None
        try:
            data = await self._reader.readline()
            if not data:
                return None  # EOF
            line = data.decode("utf-8", errors="replace").strip()
            if line:
                logger.debug("<< %s", line)
            return line if line else await self._read_line()
        except (ConnectionError, asyncio.IncompleteReadError):
            return None

    async def _process_line(self, line: str) -> None:
        """Parse an IRC line and dispatch to the appropriate event handlers."""
        prefix, source_nick, command, params = self.parse_line(line)

        # Always emit the raw event
        await self.emit("raw", prefix, command, params)

        # --- Server messages / numerics --------------------------------

        if command == "PING":
            # Respond to PING immediately (params[0] is the token)
            token = params[0] if params else ""
            await self.send_raw(f"PONG :{token}")
            return

        if command == "001":
            # RPL_WELCOME — registration successful
            # The nick we actually got is in params[0]
            actual_nick = params[0] if params else self._desired_nick
            self.nick = actual_nick
            self._registered = True
            self._reconnect_delay = _RECONNECT_INITIAL  # reset backoff
            server_name = prefix  # the prefix of 001 is the server name
            logger.info("Connected to %s as %s", server_name, self.nick)

            # OPER up if credentials are configured
            if self.oper_name and self.oper_password:
                await self.oper_up(self.oper_name, self.oper_password)

            await self.emit("connected", server_name)
            return

        if command == "396":
            # RPL_HOSTHIDDEN — displayed host has been set
            if params:
                vhost = params[1] if len(params) > 1 else params[0]
                logger.info("Virtual host set: %s", vhost)
            return

        if command == "381":
            # RPL_YOUREOPER — we are now an IRC operator
            logger.info("Successfully opered up as %s", self.oper_name or "?")
            return

        if command == "433":
            # ERR_NICKNAMEINUSE — nick already taken
            attempted = params[1] if len(params) > 1 else self._desired_nick
            if not self._registered:
                # During initial registration — use alt nick to complete connect
                alt_nick = self._desired_nick + "_"
                logger.warning("Nick %s is in use during registration, trying %s", attempted, alt_nick)
                self.nick = alt_nick
                await self.send_raw(f"NICK {alt_nick}")
                # Schedule reclaim attempt after connecting
                asyncio.ensure_future(self._reclaim_nick())
            else:
                # During runtime (reclaim attempt failed) — just log it
                logger.info("Nick %s is still in use, will retry later", attempted)
            return

        # --- Channel / user events -------------------------------------

        if command == "PRIVMSG" and len(params) >= 2:
            target = params[0]
            message = params[1]
            await self.emit("privmsg", source_nick, prefix, target, message)
            return

        if command == "NOTICE" and len(params) >= 2:
            target = params[0]
            message = params[1]
            await self.emit("notice", source_nick, prefix, target, message)
            return

        if command == "JOIN" and params:
            channel = params[0]
            # Track our own joins
            if source_nick.lower() == self.nick.lower():
                self.channels.add(channel.lower())
                logger.info("Joined %s", channel)
            await self.emit("join", source_nick, prefix, channel)
            return

        if command == "PART" and params:
            channel = params[0]
            reason = params[1] if len(params) > 1 else ""
            # Track our own parts
            if source_nick.lower() == self.nick.lower():
                self.channels.discard(channel.lower())
                logger.info("Parted %s", channel)
            await self.emit("part", source_nick, prefix, channel, reason)
            return

        if command == "QUIT":
            reason = params[0] if params else ""
            await self.emit("quit", source_nick, prefix, reason)
            return

        if command == "KICK" and len(params) >= 2:
            channel = params[0]
            kicked_nick = params[1]
            reason = params[2] if len(params) > 2 else ""
            # Track if we were kicked
            if kicked_nick.lower() == self.nick.lower():
                self.channels.discard(channel.lower())
                logger.warning("Kicked from %s by %s: %s", channel, source_nick, reason)
            await self.emit("kick", source_nick, prefix, channel, kicked_nick, reason)
            return

        if command == "NICK" and params:
            new_nick = params[0]
            # Track our own nick changes
            if source_nick.lower() == self.nick.lower():
                self.nick = new_nick
                logger.info("Nick changed to %s", new_nick)
            await self.emit("nick", source_nick, prefix, new_nick)
            return

        if command == "MODE" and len(params) >= 2:
            target = params[0]
            mode_string = params[1]
            mode_params = params[2:]
            await self.emit("mode", source_nick, prefix, target, mode_string, mode_params)
            return

        if command == "TOPIC" and len(params) >= 2:
            channel = params[0]
            new_topic = params[1]
            await self.emit("topic", source_nick, prefix, channel, new_topic)
            return

        if command == "INVITE" and len(params) >= 2:
            # INVITE <target_nick> <channel>
            channel = params[1]
            await self.emit("invite", source_nick, prefix, channel)
            return

        # --- WHO replies ------------------------------------------------

        if command == "352" and len(params) >= 7:
            # RPL_WHOREPLY: <client> <channel> <user> <host> <server> <nick> <flags> :<hopcount> <realname>
            channel = params[1]
            user = params[2]
            host = params[3]
            server = params[4]
            nick = params[5]
            flags = params[6]
            # The trailing param is "<hopcount> <realname>"
            realname = params[7] if len(params) > 7 else ""
            # Strip the hopcount from realname if present
            if realname and " " in realname:
                realname = realname.split(" ", 1)[1]
            await self.emit("who_reply", channel, user, host, server, nick, flags, realname)
            return

        if command == "315" and len(params) >= 2:
            # RPL_ENDOFWHO
            channel = params[1]
            await self.emit("end_of_who", channel)
            return


# ----------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------

def _split_message(message: str, max_len: int) -> list[str]:
    """Split *message* into chunks of at most *max_len* characters.

    Tries to split on word boundaries when possible.
    """
    if len(message) <= max_len:
        return [message]

    chunks: list[str] = []
    while message:
        if len(message) <= max_len:
            chunks.append(message)
            break

        # Try to find a space near the limit to split on
        split_idx = message.rfind(" ", 0, max_len)
        if split_idx == -1:
            # No space found — hard split
            split_idx = max_len

        chunks.append(message[:split_idx])
        message = message[split_idx:].lstrip()

    return chunks
