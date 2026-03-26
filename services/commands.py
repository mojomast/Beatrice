"""
ussynet.commands — Command dispatcher and implementations for channel services.

Users interact via ``/msg ussynet COMMAND [args]``.  All responses are sent
as NOTICEs (Undernet X convention).

Modeled after Undernet's X/CService command set with Dancer heritage.
Designed for InspIRCd 4.8.0 on irc.ussy.host.

Python 3.13+, stdlib only.
"""

from __future__ import annotations

import fnmatch
import logging
import re
from datetime import datetime, timezone
from typing import Callable, Optional

from .database import Database
from .irc import IRCClient

logger = logging.getLogger("ussynet.commands")

# ---------------------------------------------------------------------------
# Duration parsing
# ---------------------------------------------------------------------------

_DURATION_RE = re.compile(r"^(\d+)\s*([smhdw]?)$", re.IGNORECASE)

_DURATION_MULTIPLIERS = {
    "":  1,
    "s": 1,
    "m": 60,
    "h": 3600,
    "d": 86400,
    "w": 604800,
}


def parse_duration(s: str) -> int:
    """Parse a human-readable duration string into seconds.

    Accepted formats:
        "0"   -> 0  (permanent)
        "30"  -> 30 (plain number = seconds)
        "30s" -> 30
        "5m"  -> 300
        "1h"  -> 3600
        "1d"  -> 86400
        "1w"  -> 604800

    Raises:
        ValueError: On unrecognised format.
    """
    s = s.strip()
    if s == "0":
        return 0

    m = _DURATION_RE.match(s)
    if not m:
        raise ValueError(f"Invalid duration format: {s!r}")

    amount = int(m.group(1))
    unit = m.group(2).lower()
    return amount * _DURATION_MULTIPLIERS[unit]


def _format_duration(seconds: int) -> str:
    """Format a duration in seconds into a human-readable string."""
    if seconds <= 0:
        return "permanent"
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    if seconds < 86400:
        return f"{seconds // 3600}h"
    if seconds < 604800:
        return f"{seconds // 86400}d"
    return f"{seconds // 604800}w"


# ---------------------------------------------------------------------------
# Help texts — one entry per command
# ---------------------------------------------------------------------------

HELP_TEXTS: dict[str, dict[str, str]] = {
    "HELLO": {
        "usage": "HELLO <password> <email>",
        "short": "Register a new account",
        "detail": (
            "Creates a new services account using your current IRC nick as\n"
            "the username.  You will be automatically logged in after\n"
            "registration.\n"
            "\n"
            "Usage: HELLO <password> <email>\n"
            "  <password>  Your desired password.\n"
            "  <email>     A valid email address for account recovery."
        ),
    },
    "LOGIN": {
        "usage": "LOGIN <username> <password>",
        "short": "Authenticate to services",
        "detail": (
            "Log in to an existing services account.\n"
            "\n"
            "Usage: LOGIN <username> <password>\n"
            "  <username>  Your registered username.\n"
            "  <password>  Your password."
        ),
    },
    "LOGOUT": {
        "usage": "LOGOUT",
        "short": "End your current session",
        "detail": (
            "Logs you out of your current services session.\n"
            "\n"
            "Usage: LOGOUT"
        ),
    },
    "REGISTER": {
        "usage": "REGISTER <#channel> [description]",
        "short": "Register a channel",
        "detail": (
            "Register an IRC channel with services.  You become the\n"
            "channel owner (level 500) and the bot joins the channel.\n"
            "\n"
            "Usage: REGISTER <#channel> [description]\n"
            "  Requires: logged in."
        ),
    },
    "UNREGISTER": {
        "usage": "UNREGISTER <#channel>",
        "short": "Drop a channel registration",
        "detail": (
            "Unregisters a channel, removing all access entries and bans.\n"
            "The bot will part the channel.\n"
            "\n"
            "Usage: UNREGISTER <#channel>\n"
            "  Requires: logged in, channel owner (500) or admin."
        ),
    },
    "ADDUSER": {
        "usage": "ADDUSER <#channel> <username> <level>",
        "short": "Add a user to a channel's access list",
        "detail": (
            "Adds a user to the channel's access list at the given level.\n"
            "\n"
            "Usage: ADDUSER <#channel> <username> <level>\n"
            "  <level>  Must be 1-499.  Your level must be > granted level.\n"
            "  Requires: logged in, level >= 400 on channel."
        ),
    },
    "REMUSER": {
        "usage": "REMUSER <#channel> <username>",
        "short": "Remove a user from access list",
        "detail": (
            "Removes a user from the channel's access list.\n"
            "\n"
            "Usage: REMUSER <#channel> <username>\n"
            "  Requires: logged in, level >= 400, your level > target's."
        ),
    },
    "MODINFO": {
        "usage": "MODINFO <#channel> <ACCESS|AUTOMODE> <username> <value>",
        "short": "Modify a user's access entry",
        "detail": (
            "Modify a user's access level or automode setting.\n"
            "\n"
            "Usage:\n"
            "  MODINFO <#channel> ACCESS <username> <new_level>\n"
            "  MODINFO <#channel> AUTOMODE <username> <none|voice|op>\n"
            "\n"
            "  Requires: logged in, level >= 400, your level > target's\n"
            "  level AND > new level (for ACCESS changes)."
        ),
    },
    "ACCESS": {
        "usage": "ACCESS <#channel> [pattern]",
        "short": "Show channel access list",
        "detail": (
            "Displays the access list for a channel, optionally filtered\n"
            "by a username pattern (glob/fnmatch).\n"
            "\n"
            "Usage: ACCESS <#channel> [pattern]\n"
            "  Requires: logged in."
        ),
    },
    "OP": {
        "usage": "OP <#channel> [nick]",
        "short": "Op a user in channel",
        "detail": (
            "Gives channel operator status (+o) to a user.\n"
            "If no nick is specified, ops yourself.\n"
            "\n"
            "Usage: OP <#channel> [nick]\n"
            "  Requires: logged in, level >= 100."
        ),
    },
    "DEOP": {
        "usage": "DEOP <#channel> [nick]",
        "short": "Deop a user in channel",
        "detail": (
            "Removes channel operator status (-o) from a user.\n"
            "If no nick is specified, deops yourself.\n"
            "\n"
            "Usage: DEOP <#channel> [nick]\n"
            "  Requires: logged in, level >= 100."
        ),
    },
    "VOICE": {
        "usage": "VOICE <#channel> [nick]",
        "short": "Voice a user in channel",
        "detail": (
            "Gives voice (+v) to a user in the channel.\n"
            "If no nick is specified, voices yourself.\n"
            "\n"
            "Usage: VOICE <#channel> [nick]\n"
            "  Requires: logged in, level >= 25."
        ),
    },
    "DEVOICE": {
        "usage": "DEVOICE <#channel> [nick]",
        "short": "Devoice a user in channel",
        "detail": (
            "Removes voice (-v) from a user in the channel.\n"
            "If no nick is specified, devoices yourself.\n"
            "\n"
            "Usage: DEVOICE <#channel> [nick]\n"
            "  Requires: logged in, level >= 25."
        ),
    },
    "KICK": {
        "usage": "KICK <#channel> <nick> [reason]",
        "short": "Kick a user from channel",
        "detail": (
            "Kicks a user from the channel using SAKICK.\n"
            "\n"
            "Usage: KICK <#channel> <nick> [reason]\n"
            "  Requires: logged in, level >= 50."
        ),
    },
    "BAN": {
        "usage": "BAN <#channel> <nick|mask> [duration] [level] [reason]",
        "short": "Ban a user from channel",
        "detail": (
            "Sets a ban on a channel.  If a nick is given instead of a\n"
            "hostmask, it is resolved to *!*user@host.\n"
            "\n"
            "Usage: BAN <#channel> <nick|mask> [duration] [level] [reason]\n"
            "  <duration>  e.g. 5m, 1h, 30d, 0 for permanent.  Default: 1h.\n"
            "  <level>     Ban level, default 75.\n"
            "  <reason>    Reason for the ban.\n"
            "  Requires: logged in, level >= 75."
        ),
    },
    "UNBAN": {
        "usage": "UNBAN <#channel> <mask|id>",
        "short": "Remove a ban from channel",
        "detail": (
            "Removes a ban by hostmask or ban ID.\n"
            "\n"
            "Usage: UNBAN <#channel> <mask|id>\n"
            "  Requires: logged in, level >= 75."
        ),
    },
    "BANLIST": {
        "usage": "BANLIST <#channel>",
        "short": "Show channel ban list",
        "detail": (
            "Displays all active bans on a channel.\n"
            "\n"
            "Usage: BANLIST <#channel>\n"
            "  Requires: logged in, level >= 1."
        ),
    },
    "TOPIC": {
        "usage": "TOPIC <#channel> <new topic>",
        "short": "Set channel topic",
        "detail": (
            "Sets the topic of the specified channel.\n"
            "\n"
            "Usage: TOPIC <#channel> <new topic>\n"
            "  Requires: logged in, level >= 50."
        ),
    },
    "INVITE": {
        "usage": "INVITE <#channel>",
        "short": "Invite yourself to a channel",
        "detail": (
            "Sends an INVITE to you for the specified channel.\n"
            "\n"
            "Usage: INVITE <#channel>\n"
            "  Requires: logged in, level >= 1 on channel."
        ),
    },
    "SET": {
        "usage": "SET <#channel> <option> <value>",
        "short": "Modify channel settings",
        "detail": (
            "Modify settings for a registered channel.\n"
            "\n"
            "Usage: SET <#channel> <option> <value>\n"
            "  Options:\n"
            "    DESCRIPTION <text>  — set channel description   (400+)\n"
            "    URL <url>           — set channel URL            (400+)\n"
            "    TOPIC <text>        — set default topic          (400+)\n"
            "    AUTOTOPIC on|off    — auto-set topic on join     (400+)\n"
            "    MODELOCK <modes>    — lock channel modes         (400+)\n"
            "  Requires: logged in, level >= 400."
        ),
    },
    "INFO": {
        "usage": "INFO <#channel>",
        "short": "Show channel information",
        "detail": (
            "Displays public information about a registered channel.\n"
            "\n"
            "Usage: INFO <#channel>\n"
            "  No authentication required."
        ),
    },
    "CHANINFO": {
        "usage": "CHANINFO <#channel>",
        "short": "Detailed channel info (admin/manager)",
        "detail": (
            "Displays detailed channel information including settings.\n"
            "\n"
            "Usage: CHANINFO <#channel>\n"
            "  Requires: admin or level >= 400 on channel."
        ),
    },
    "STATUS": {
        "usage": "STATUS",
        "short": "Show bot status",
        "detail": (
            "Displays bot status: uptime, channels managed, users\n"
            "registered, and active sessions.\n"
            "\n"
            "Usage: STATUS"
        ),
    },
    "VERIFY": {
        "usage": "VERIFY <nick>",
        "short": "Check if a nick is logged in",
        "detail": (
            "Checks whether a given nick is currently logged in to services.\n"
            "\n"
            "Usage: VERIFY <nick>"
        ),
    },
    "SHOWCOMMANDS": {
        "usage": "SHOWCOMMANDS",
        "short": "List available commands",
        "detail": (
            "Lists all available commands with a brief description.\n"
            "\n"
            "Usage: SHOWCOMMANDS"
        ),
    },
    "HELP": {
        "usage": "HELP [command]",
        "short": "Show help for a command",
        "detail": (
            "Displays help for the specified command, or lists all\n"
            "commands if none is given.\n"
            "\n"
            "Usage: HELP [command]"
        ),
    },
    "ADMIN": {
        "usage": "ADMIN <username> <on|off>",
        "short": "Grant/revoke admin status",
        "detail": (
            "Grants or revokes administrator status for a user.\n"
            "\n"
            "Usage: ADMIN <username> <on|off>\n"
            "  Requires: logged in, admin."
        ),
    },
    "SUSPEND": {
        "usage": "SUSPEND <username>",
        "short": "Suspend a user account",
        "detail": (
            "Suspends a user account, preventing them from logging in.\n"
            "\n"
            "Usage: SUSPEND <username>\n"
            "  Requires: admin."
        ),
    },
    "UNSUSPEND": {
        "usage": "UNSUSPEND <username>",
        "short": "Unsuspend a user account",
        "detail": (
            "Restores a suspended user account.\n"
            "\n"
            "Usage: UNSUSPEND <username>\n"
            "  Requires: admin."
        ),
    },
    "SAY": {
        "usage": "SAY <#channel> <message>",
        "short": "Make bot speak in a channel",
        "detail": (
            "Makes the bot send a PRIVMSG to the specified channel.\n"
            "\n"
            "Usage: SAY <#channel> <message>\n"
            "  Requires: admin."
        ),
    },
    "BROADCAST": {
        "usage": "BROADCAST <message>",
        "short": "Notice all registered channels",
        "detail": (
            "Sends a NOTICE to all registered channels.\n"
            "\n"
            "Usage: BROADCAST <message>\n"
            "  Requires: admin."
        ),
    },
    "VHOST": {
        "usage": "VHOST <LIST|SET|CLEAR|ADD|DEL|OFF|ON>",
        "short": "Manage virtual hostnames",
        "detail": (
            "View and manage virtual hostnames (vhosts).\n"
            "\n"
            "User commands:\n"
            "  VHOST LIST             — List all available vhosts.\n"
            "  VHOST SET <vhost>      — Apply a vhost to yourself.\n"
            "  VHOST CLEAR            — Remove your current vhost.\n"
            "\n"
            "Admin commands:\n"
            "  VHOST ADD <pattern> [description]\n"
            "                         — Add a new available vhost.\n"
            "  VHOST DEL <pattern>    — Remove an available vhost.\n"
            "  VHOST ON <pattern>     — Re-enable a disabled vhost.\n"
            "  VHOST OFF <pattern>    — Disable a vhost (existing users keep it).\n"
            "  VHOST SETUSER <nick> <vhost>\n"
            "                         — Force-set a vhost on another user.\n"
            "\n"
            "Requires: logged in. Admin subcommands require admin."
        ),
    },
    "NICK": {
        "usage": "NICK <new_nick>",
        "short": "Change the bot's nickname",
        "detail": (
            "Changes the bot's IRC nickname.\n"
            "\n"
            "Usage: NICK <new_nick>\n"
            "  Requires: admin."
        ),
    },
    "AUTOVHOST": {
        "usage": "AUTOVHOST <ADD|DEL|LIST>",
        "short": "Manage automatic vhost assignments",
        "detail": (
            "Manage hostmask-based automatic vhost assignments.\n"
            "When a user joins whose hostmask matches a rule, the bot\n"
            "will automatically CHGHOST them.\n"
            "\n"
            "  AUTOVHOST ADD <hostmask> <vhost>\n"
            "      Add a rule. Hostmask is a glob pattern, e.g.\n"
            "      *!*@185.255.121.49 or ussybot!*@*\n"
            "  AUTOVHOST DEL <hostmask>\n"
            "      Remove a rule.\n"
            "  AUTOVHOST LIST\n"
            "      Show all rules.\n"
            "\n"
            "Requires: admin."
        ),
    },
}


# ---------------------------------------------------------------------------
# CommandHandler
# ---------------------------------------------------------------------------

class CommandHandler:
    """Dispatches and executes all ussynet channel-services commands.

    Manages user sessions, access checks, and channel operations.
    All responses are sent via NOTICE (Undernet X convention).
    """

    def __init__(self, irc: IRCClient, db: Database, config: dict) -> None:
        self.irc = irc
        self.db = db
        self.config = config

        # Configurable bot display name for user-facing messages
        self.bot_name = config.get("bot", {}).get("nick", "ussynet")

        # Session tracking: lowercase nick -> session dict
        self.sessions: dict[str, dict] = {}

        # Bot start time (for STATUS uptime)
        self.start_time: datetime = datetime.now(timezone.utc)

        # Admin hostmask patterns from config
        self._admin_hostmasks: list[str] = (
            config.get("services", {}).get("admin_hostmasks", [])
        )

        # Command dispatch table: command name (uppercase) -> async method
        self.commands: dict[str, Callable] = {
            # Authentication
            "HELLO":        self.cmd_hello,
            "LOGIN":        self.cmd_login,
            "LOGOUT":       self.cmd_logout,
            # Channel registration
            "REGISTER":     self.cmd_register,
            "UNREGISTER":   self.cmd_unregister,
            # Access management
            "ADDUSER":      self.cmd_adduser,
            "REMUSER":      self.cmd_remuser,
            "MODINFO":      self.cmd_modinfo,
            "ACCESS":       self.cmd_access,
            # Channel operations
            "OP":           self.cmd_op,
            "DEOP":         self.cmd_deop,
            "VOICE":        self.cmd_voice,
            "DEVOICE":      self.cmd_devoice,
            "KICK":         self.cmd_kick,
            "BAN":          self.cmd_ban,
            "UNBAN":        self.cmd_unban,
            "BANLIST":      self.cmd_banlist,
            "TOPIC":        self.cmd_topic,
            "INVITE":       self.cmd_invite,
            # Channel settings
            "SET":          self.cmd_set,
            # Information
            "INFO":         self.cmd_info,
            "CHANINFO":     self.cmd_chaninfo,
            "STATUS":       self.cmd_status,
            "VERIFY":       self.cmd_verify,
            "SHOWCOMMANDS": self.cmd_showcommands,
            "HELP":         self.cmd_help,
            # Admin
            "ADMIN":        self.cmd_admin,
            "SUSPEND":      self.cmd_suspend,
            "UNSUSPEND":    self.cmd_unsuspend,
            "SAY":          self.cmd_say,
            "BROADCAST":    self.cmd_broadcast,
            # Vhost management
            "VHOST":        self.cmd_vhost,
            # Bot management
            "NICK":         self.cmd_nick,
            # Auto-vhost management
            "AUTOVHOST":    self.cmd_autovhost,
        }

        # NOTE: IRC event handlers are registered by the bot orchestrator
        # (bot.py) — do NOT register them here to avoid duplicates.

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------

    def get_session(self, nick: str) -> Optional[dict]:
        """Return the login session for *nick*, or None if not logged in."""
        return self.sessions.get(nick.lower())

    def _create_session(self, nick: str, user_id: int, username: str,
                        hostmask: str) -> dict:
        """Create and store a new login session for *nick*."""
        session = {
            "user_id": user_id,
            "username": username,
            "hostmask": hostmask,
            "logged_in_at": datetime.now(timezone.utc),
        }
        self.sessions[nick.lower()] = session
        logger.info("Session created: %s (user_id=%d)", nick, user_id)
        return session

    def _remove_session(self, nick: str) -> Optional[dict]:
        """Remove and return the session for *nick*, or None."""
        return self.sessions.pop(nick.lower(), None)

    def _is_admin_hostmask(self, hostmask: str) -> bool:
        """Check whether *hostmask* matches any configured admin hostmask pattern."""
        for pattern in self._admin_hostmasks:
            if fnmatch.fnmatch(hostmask.lower(), pattern.lower()):
                return True
        return False

    def _get_user_level(self, user_id: int, channel_name: str) -> int:
        """Return a user's access level on *channel_name*, or 0 if none."""
        user = self.db.get_user_by_id(user_id)
        if user is None:
            return 0
        try:
            entry = self.db.get_access(channel_name, user["username"])
        except ValueError:
            return 0
        if entry is None:
            return 0
        return entry["level"]

    # ------------------------------------------------------------------
    # Response helpers
    # ------------------------------------------------------------------

    async def reply(self, nick: str, message: str) -> None:
        """Send a NOTICE to *nick*.  All command responses use NOTICE."""
        await self.irc.send_notice(nick, message)

    async def reply_lines(self, nick: str, lines: list[str]) -> None:
        """Send multiple NOTICE lines to *nick*."""
        for line in lines:
            await self.irc.send_notice(nick, line)

    # ------------------------------------------------------------------
    # Main dispatch
    # ------------------------------------------------------------------

    async def handle_message(self, nick: str, hostmask: str, target: str,
                             message: str) -> None:
        """Entry point for all PRIVMSGs.  Dispatches private messages as commands.

        Args:
            nick:     Sender's current nick.
            hostmask: Full nick!user@host prefix.
            target:   Message target (channel or bot nick).
            message:  The message body.
        """
        # Only process private messages (target is the bot's nick)
        if target.lower() != self.irc.nick.lower():
            return

        message = message.strip()
        if not message:
            return

        parts = message.split()
        command_name = parts[0].upper()
        args = parts[1:]

        handler = self.commands.get(command_name)
        if handler is None:
            await self.reply(
                nick,
                f"Unknown command \x02{command_name}\x02. "
                "Use SHOWCOMMANDS for a list of commands.",
            )
            return

        try:
            await handler(nick, hostmask, args)
        except Exception:
            logger.exception(
                "Error executing command %s from %s", command_name, nick,
            )
            await self.reply(nick, "An internal error occurred. Please try again later.")

    # ==================================================================
    # AUTHENTICATION COMMANDS
    # ==================================================================

    async def cmd_hello(self, nick: str, hostmask: str, args: list[str]) -> None:
        """HELLO <password> <email> — Register a new account."""
        if len(args) < 2:
            await self.reply(nick, f"Usage: HELLO <password> <email>")
            return

        password, email = args[0], args[1]

        # Username is the sender's current IRC nick
        username = nick

        try:
            user_id = self.db.create_user(username, password, email)
        except ValueError:
            await self.reply(nick, f"The username \x02{username}\x02 is already registered.")
            return

        # If the hostmask matches an admin pattern, auto-grant admin
        if self._is_admin_hostmask(hostmask):
            self.db.set_admin(user_id, True)
            logger.info("Auto-granted admin to %s (hostmask match)", username)

        # Auto-login after registration
        self.db.update_last_seen(user_id, hostmask)
        self._create_session(nick, user_id, username, hostmask)

        await self.reply(
            nick,
            f"Welcome to {self.bot_name}, \x02{username}\x02! You are now logged in.",
        )

    async def cmd_login(self, nick: str, hostmask: str, args: list[str]) -> None:
        """LOGIN <username> <password> — Authenticate to services."""
        if len(args) < 2:
            await self.reply(nick, "Usage: LOGIN <username> <password>")
            return

        username, password = args[0], args[1]

        user = self.db.authenticate(username, password)
        if user is None:
            await self.reply(nick, "Login failed. Invalid username or password.")
            return

        # Update last seen and create session
        self.db.update_last_seen(user["id"], hostmask)
        self._create_session(nick, user["id"], user["username"], hostmask)

        await self.reply(nick, f"Successfully logged in as \x02{user['username']}\x02.")

        # Auto-apply saved vhost if the user has one
        saved_vhost = self.db.get_user_vhost(user["id"])
        if saved_vhost and saved_vhost["is_active"]:
            await self.irc.chghost(nick, saved_vhost["pattern"])
            await self.reply(nick, f"Vhost \x02{saved_vhost['pattern']}\x02 applied.")

    async def cmd_logout(self, nick: str, hostmask: str, args: list[str]) -> None:
        """LOGOUT — End session."""
        session = self._remove_session(nick)
        if session is None:
            await self.reply(nick, "You are not logged in.")
            return

        await self.reply(nick, "Successfully logged out.")

    # ==================================================================
    # CHANNEL REGISTRATION
    # ==================================================================

    async def cmd_register(self, nick: str, hostmask: str, args: list[str]) -> None:
        """REGISTER <#channel> [description] — Register a channel."""
        session = self.get_session(nick)
        if session is None:
            await self.reply(nick, "You must be logged in to use this command.")
            return

        if len(args) < 1:
            await self.reply(nick, "Usage: REGISTER <#channel> [description]")
            return

        channel = args[0]
        if not channel.startswith("#"):
            await self.reply(nick, "Channel name must start with #.")
            return

        description = " ".join(args[1:]) if len(args) > 1 else ""

        try:
            self.db.register_channel(channel, session["user_id"], description)
        except ValueError:
            await self.reply(nick, f"Channel \x02{channel}\x02 is already registered.")
            return

        # Bot joins the channel and ops itself
        await self.irc.join(channel)
        await self.irc.samode(channel, f"+o {self.irc.nick}")

        await self.reply(
            nick,
            f"Channel \x02{channel}\x02 has been registered. "
            "You have been added as owner (500).",
        )

    async def cmd_unregister(self, nick: str, hostmask: str, args: list[str]) -> None:
        """UNREGISTER <#channel> — Drop a channel registration."""
        session = self.get_session(nick)
        if session is None:
            await self.reply(nick, "You must be logged in to use this command.")
            return

        if len(args) < 1:
            await self.reply(nick, "Usage: UNREGISTER <#channel>")
            return

        channel = args[0]
        if not channel.startswith("#"):
            await self.reply(nick, "Channel name must start with #.")
            return

        chan = self.db.get_channel(channel)
        if chan is None:
            await self.reply(nick, f"Channel \x02{channel}\x02 is not registered.")
            return

        # Check permissions: must be owner (500) or admin
        user_level = self._get_user_level(session["user_id"], channel)
        is_admin = self.db.is_admin(session["user_id"])

        if user_level < 500 and not is_admin:
            await self.reply(
                nick,
                "You must be the channel owner (500) or an admin to unregister.",
            )
            return

        # Part the channel and unregister
        await self.irc.part(channel, "Channel unregistered.")
        self.db.unregister_channel(channel)

        await self.reply(nick, f"Channel \x02{channel}\x02 has been unregistered.")

    # ==================================================================
    # ACCESS MANAGEMENT
    # ==================================================================

    async def cmd_adduser(self, nick: str, hostmask: str, args: list[str]) -> None:
        """ADDUSER <#channel> <username> <level> — Add a user to access list."""
        session = self.get_session(nick)
        if session is None:
            await self.reply(nick, "You must be logged in to use this command.")
            return

        if len(args) < 3:
            await self.reply(nick, "Usage: ADDUSER <#channel> <username> <level>")
            return

        channel, username = args[0], args[1]

        if not channel.startswith("#"):
            await self.reply(nick, "Channel name must start with #.")
            return

        # Validate level
        try:
            level = int(args[2])
        except ValueError:
            await self.reply(nick, "Level must be a number between 1 and 499.")
            return

        if level < 1 or level > 499:
            await self.reply(nick, "Level must be between 1 and 499.")
            return

        # Check channel exists
        chan = self.db.get_channel(channel)
        if chan is None:
            await self.reply(nick, f"Channel \x02{channel}\x02 is not registered.")
            return

        # Check requester's level
        my_level = self._get_user_level(session["user_id"], channel)
        if my_level < 400:
            await self.reply(nick, "You need level 400 or higher to add users.")
            return

        if my_level <= level:
            await self.reply(
                nick,
                f"Your level ({my_level}) must be higher than the level you're "
                f"granting ({level}).",
            )
            return

        # Check target user exists
        target_user = self.db.get_user(username)
        if target_user is None:
            await self.reply(nick, f"User \x02{username}\x02 does not exist.")
            return

        # Add access
        try:
            self.db.add_access(
                channel, username, level, session["user_id"],
            )
        except ValueError as e:
            await self.reply(nick, str(e))
            return

        await self.reply(
            nick,
            f"Added \x02{username}\x02 to \x02{channel}\x02 with access level {level}.",
        )

    async def cmd_remuser(self, nick: str, hostmask: str, args: list[str]) -> None:
        """REMUSER <#channel> <username> — Remove a user from access list."""
        session = self.get_session(nick)
        if session is None:
            await self.reply(nick, "You must be logged in to use this command.")
            return

        if len(args) < 2:
            await self.reply(nick, "Usage: REMUSER <#channel> <username>")
            return

        channel, username = args[0], args[1]

        if not channel.startswith("#"):
            await self.reply(nick, "Channel name must start with #.")
            return

        chan = self.db.get_channel(channel)
        if chan is None:
            await self.reply(nick, f"Channel \x02{channel}\x02 is not registered.")
            return

        # Check requester's level
        my_level = self._get_user_level(session["user_id"], channel)
        if my_level < 400:
            await self.reply(nick, "You need level 400 or higher to remove users.")
            return

        # Can't remove yourself (use UNREGISTER for owner)
        if username.lower() == session["username"].lower():
            await self.reply(nick, "You cannot remove yourself. Use UNREGISTER to drop ownership.")
            return

        # Check target's level
        try:
            target_access = self.db.get_access(channel, username)
        except ValueError as e:
            await self.reply(nick, str(e))
            return

        if target_access is None:
            await self.reply(
                nick,
                f"User \x02{username}\x02 does not have access on \x02{channel}\x02.",
            )
            return

        if my_level <= target_access["level"]:
            await self.reply(
                nick,
                f"Your level ({my_level}) must be higher than the target's "
                f"level ({target_access['level']}).",
            )
            return

        self.db.remove_access(channel, username)
        await self.reply(nick, f"Removed \x02{username}\x02 from \x02{channel}\x02.")

    async def cmd_modinfo(self, nick: str, hostmask: str, args: list[str]) -> None:
        """MODINFO <#channel> ACCESS|AUTOMODE <username> <value> — Modify access entry."""
        session = self.get_session(nick)
        if session is None:
            await self.reply(nick, "You must be logged in to use this command.")
            return

        if len(args) < 4:
            await self.reply(
                nick,
                "Usage: MODINFO <#channel> ACCESS <username> <new_level> | "
                "MODINFO <#channel> AUTOMODE <username> <none|voice|op>",
            )
            return

        channel = args[0]
        option = args[1].upper()
        username = args[2]
        value = args[3]

        if not channel.startswith("#"):
            await self.reply(nick, "Channel name must start with #.")
            return

        chan = self.db.get_channel(channel)
        if chan is None:
            await self.reply(nick, f"Channel \x02{channel}\x02 is not registered.")
            return

        # Check requester's level
        my_level = self._get_user_level(session["user_id"], channel)
        if my_level < 400:
            await self.reply(nick, "You need level 400 or higher to modify users.")
            return

        # Check target's current access
        try:
            target_access = self.db.get_access(channel, username)
        except ValueError as e:
            await self.reply(nick, str(e))
            return

        if target_access is None:
            await self.reply(
                nick,
                f"User \x02{username}\x02 does not have access on \x02{channel}\x02.",
            )
            return

        if my_level <= target_access["level"]:
            await self.reply(
                nick,
                f"Your level ({my_level}) must be higher than the target's "
                f"level ({target_access['level']}).",
            )
            return

        if option == "ACCESS":
            try:
                new_level = int(value)
            except ValueError:
                await self.reply(nick, "Level must be a number.")
                return

            if new_level < 1 or new_level > 499:
                await self.reply(nick, "Level must be between 1 and 499.")
                return

            if my_level <= new_level:
                await self.reply(
                    nick,
                    f"Your level ({my_level}) must be higher than the new level ({new_level}).",
                )
                return

            self.db.modify_access(channel, username, level=new_level)
            await self.reply(
                nick,
                f"Modified \x02{username}\x02 on \x02{channel}\x02: "
                f"access level changed to {new_level}.",
            )

        elif option == "AUTOMODE":
            value_lower = value.lower()
            if value_lower not in ("none", "voice", "op"):
                await self.reply(nick, "Automode must be one of: none, voice, op.")
                return

            self.db.modify_access(channel, username, automode=value_lower)
            await self.reply(
                nick,
                f"Modified \x02{username}\x02 on \x02{channel}\x02: "
                f"automode changed to {value_lower}.",
            )

        else:
            await self.reply(
                nick,
                "Unknown MODINFO option. Use ACCESS or AUTOMODE.",
            )

    async def cmd_access(self, nick: str, hostmask: str, args: list[str]) -> None:
        """ACCESS <#channel> [pattern] — Show access list."""
        session = self.get_session(nick)
        if session is None:
            await self.reply(nick, "You must be logged in to use this command.")
            return

        if len(args) < 1:
            await self.reply(nick, "Usage: ACCESS <#channel> [pattern]")
            return

        channel = args[0]
        pattern = args[1] if len(args) > 1 else None

        if not channel.startswith("#"):
            await self.reply(nick, "Channel name must start with #.")
            return

        chan = self.db.get_channel(channel)
        if chan is None:
            await self.reply(nick, f"Channel \x02{channel}\x02 is not registered.")
            return

        try:
            access_list = self.db.get_access_list(channel)
        except ValueError as e:
            await self.reply(nick, str(e))
            return

        # Filter by pattern if given
        if pattern:
            access_list = [
                entry for entry in access_list
                if fnmatch.fnmatch(entry["username"].lower(), pattern.lower())
            ]

        lines = [f"Access list for \x02{channel}\x02:"]

        if not access_list:
            lines.append("  (empty)")
        else:
            for entry in access_list:
                # Look up who added this entry
                added_by_name = "unknown"
                if entry.get("added_by"):
                    added_by_user = self.db.get_user_by_id(entry["added_by"])
                    if added_by_user:
                        added_by_name = added_by_user["username"]

                lines.append(
                    f"  {entry['level']:>3}  {entry['username']:<20}  "
                    f"{entry['automode']:<6}  (added by {added_by_name})"
                )

        lines.append("End of access list.")
        await self.reply_lines(nick, lines)

    # ==================================================================
    # CHANNEL OPERATIONS
    # ==================================================================

    async def cmd_op(self, nick: str, hostmask: str, args: list[str]) -> None:
        """OP <#channel> [nick] — Op a user."""
        session = self.get_session(nick)
        if session is None:
            await self.reply(nick, "You must be logged in to use this command.")
            return

        if len(args) < 1:
            await self.reply(nick, "Usage: OP <#channel> [nick]")
            return

        channel = args[0]
        target_nick = args[1] if len(args) > 1 else nick

        if not channel.startswith("#"):
            await self.reply(nick, "Channel name must start with #.")
            return

        chan = self.db.get_channel(channel)
        if chan is None:
            await self.reply(nick, f"Channel \x02{channel}\x02 is not registered.")
            return

        my_level = self._get_user_level(session["user_id"], channel)
        if my_level < 100:
            await self.reply(nick, "You need level 100 or higher to op users.")
            return

        await self.irc.samode(channel, f"+o {target_nick}")
        await self.reply(nick, f"Opped \x02{target_nick}\x02 on \x02{channel}\x02.")

    async def cmd_deop(self, nick: str, hostmask: str, args: list[str]) -> None:
        """DEOP <#channel> [nick] — Deop a user."""
        session = self.get_session(nick)
        if session is None:
            await self.reply(nick, "You must be logged in to use this command.")
            return

        if len(args) < 1:
            await self.reply(nick, "Usage: DEOP <#channel> [nick]")
            return

        channel = args[0]
        target_nick = args[1] if len(args) > 1 else nick

        if not channel.startswith("#"):
            await self.reply(nick, "Channel name must start with #.")
            return

        chan = self.db.get_channel(channel)
        if chan is None:
            await self.reply(nick, f"Channel \x02{channel}\x02 is not registered.")
            return

        my_level = self._get_user_level(session["user_id"], channel)
        if my_level < 100:
            await self.reply(nick, "You need level 100 or higher to deop users.")
            return

        await self.irc.samode(channel, f"-o {target_nick}")
        await self.reply(nick, f"Deopped \x02{target_nick}\x02 on \x02{channel}\x02.")

    async def cmd_voice(self, nick: str, hostmask: str, args: list[str]) -> None:
        """VOICE <#channel> [nick] — Voice a user."""
        session = self.get_session(nick)
        if session is None:
            await self.reply(nick, "You must be logged in to use this command.")
            return

        if len(args) < 1:
            await self.reply(nick, "Usage: VOICE <#channel> [nick]")
            return

        channel = args[0]
        target_nick = args[1] if len(args) > 1 else nick

        if not channel.startswith("#"):
            await self.reply(nick, "Channel name must start with #.")
            return

        chan = self.db.get_channel(channel)
        if chan is None:
            await self.reply(nick, f"Channel \x02{channel}\x02 is not registered.")
            return

        my_level = self._get_user_level(session["user_id"], channel)
        if my_level < 25:
            await self.reply(nick, "You need level 25 or higher to voice users.")
            return

        await self.irc.samode(channel, f"+v {target_nick}")
        await self.reply(nick, f"Voiced \x02{target_nick}\x02 on \x02{channel}\x02.")

    async def cmd_devoice(self, nick: str, hostmask: str, args: list[str]) -> None:
        """DEVOICE <#channel> [nick] — Devoice a user."""
        session = self.get_session(nick)
        if session is None:
            await self.reply(nick, "You must be logged in to use this command.")
            return

        if len(args) < 1:
            await self.reply(nick, "Usage: DEVOICE <#channel> [nick]")
            return

        channel = args[0]
        target_nick = args[1] if len(args) > 1 else nick

        if not channel.startswith("#"):
            await self.reply(nick, "Channel name must start with #.")
            return

        chan = self.db.get_channel(channel)
        if chan is None:
            await self.reply(nick, f"Channel \x02{channel}\x02 is not registered.")
            return

        my_level = self._get_user_level(session["user_id"], channel)
        if my_level < 25:
            await self.reply(nick, "You need level 25 or higher to devoice users.")
            return

        await self.irc.samode(channel, f"-v {target_nick}")
        await self.reply(nick, f"Devoiced \x02{target_nick}\x02 on \x02{channel}\x02.")

    async def cmd_kick(self, nick: str, hostmask: str, args: list[str]) -> None:
        """KICK <#channel> <nick> [reason] — Kick a user from channel."""
        session = self.get_session(nick)
        if session is None:
            await self.reply(nick, "You must be logged in to use this command.")
            return

        if len(args) < 2:
            await self.reply(nick, "Usage: KICK <#channel> <nick> [reason]")
            return

        channel = args[0]
        target_nick = args[1]
        reason = " ".join(args[2:]) if len(args) > 2 else "Requested"

        if not channel.startswith("#"):
            await self.reply(nick, "Channel name must start with #.")
            return

        chan = self.db.get_channel(channel)
        if chan is None:
            await self.reply(nick, f"Channel \x02{channel}\x02 is not registered.")
            return

        my_level = self._get_user_level(session["user_id"], channel)
        if my_level < 50:
            await self.reply(nick, "You need level 50 or higher to kick users.")
            return

        await self.irc.sakick(channel, target_nick, reason)
        # No reply — the kick speaks for itself

    async def cmd_ban(self, nick: str, hostmask: str, args: list[str]) -> None:
        """BAN <#channel> <nick|mask> [duration] [level] [reason] — Ban a user."""
        session = self.get_session(nick)
        if session is None:
            await self.reply(nick, "You must be logged in to use this command.")
            return

        if len(args) < 2:
            await self.reply(
                nick,
                "Usage: BAN <#channel> <nick|mask> [duration] [level] [reason]",
            )
            return

        channel = args[0]
        target = args[1]

        if not channel.startswith("#"):
            await self.reply(nick, "Channel name must start with #.")
            return

        chan = self.db.get_channel(channel)
        if chan is None:
            await self.reply(nick, f"Channel \x02{channel}\x02 is not registered.")
            return

        my_level = self._get_user_level(session["user_id"], channel)
        if my_level < 75:
            await self.reply(nick, "You need level 75 or higher to set bans.")
            return

        # Parse optional args: [duration] [level] [reason...]
        duration_seconds = 3600  # default: 1 hour
        ban_level = 75           # default
        reason_parts: list[str] = []
        remaining = args[2:]

        # Try to parse duration from first remaining arg
        if remaining:
            try:
                duration_seconds = parse_duration(remaining[0])
                remaining = remaining[1:]
            except ValueError:
                pass  # Not a duration, leave it for level/reason

        # Try to parse ban level from next remaining arg
        if remaining:
            try:
                candidate_level = int(remaining[0])
                if 1 <= candidate_level <= 500:
                    ban_level = candidate_level
                    remaining = remaining[1:]
            except ValueError:
                pass  # Not a level, treat as reason

        # Everything else is the reason
        reason = " ".join(remaining) if remaining else "No reason given"

        # Resolve target to a ban mask
        # If it looks like a mask (contains ! or @ or *), use it directly
        if "!" in target or "@" in target or "*" in target:
            mask = target
        else:
            # It's a nick — try to find their hostmask from sessions
            target_session = self.get_session(target)
            if target_session and target_session.get("hostmask"):
                # Parse hostmask: nick!user@host -> *!*user@host
                full_hm = target_session["hostmask"]
                if "!" in full_hm and "@" in full_hm:
                    _, user_host = full_hm.split("!", 1)
                    user_part, host_part = user_host.split("@", 1)
                    mask = f"*!*{user_part}@{host_part}"
                else:
                    mask = f"*!*@{target}*"
            else:
                # No session info, create a nick-based mask
                mask = f"{target}!*@*"

        # Add ban to database
        self.db.add_ban(
            channel, mask, reason, session["user_id"],
            duration=duration_seconds, level=ban_level,
        )

        # Set +b on channel via samode
        await self.irc.samode(channel, f"+b {mask}")

        # Also kick the target if it was a nick (not a mask pattern)
        if "!" not in target and "@" not in target and "*" not in target:
            await self.irc.sakick(channel, target, reason)

        duration_str = _format_duration(duration_seconds)
        await self.reply(
            nick,
            f"Banned \x02{mask}\x02 on \x02{channel}\x02 ({duration_str}).",
        )

    async def cmd_unban(self, nick: str, hostmask: str, args: list[str]) -> None:
        """UNBAN <#channel> <mask|id> — Remove a ban."""
        session = self.get_session(nick)
        if session is None:
            await self.reply(nick, "You must be logged in to use this command.")
            return

        if len(args) < 2:
            await self.reply(nick, "Usage: UNBAN <#channel> <mask|id>")
            return

        channel = args[0]
        target = args[1]

        if not channel.startswith("#"):
            await self.reply(nick, "Channel name must start with #.")
            return

        chan = self.db.get_channel(channel)
        if chan is None:
            await self.reply(nick, f"Channel \x02{channel}\x02 is not registered.")
            return

        my_level = self._get_user_level(session["user_id"], channel)
        if my_level < 75:
            await self.reply(nick, "You need level 75 or higher to remove bans.")
            return

        # Determine if target is a numeric ID or a mask
        mask_to_unset = None
        try:
            ban_id = int(target)
            # It's a ban ID — look up the mask for -b
            bans = self.db.get_bans(channel)
            found_ban = None
            for ban in bans:
                if ban["id"] == ban_id:
                    found_ban = ban
                    break
            if found_ban is None:
                await self.reply(nick, f"Ban ID {ban_id} not found on \x02{channel}\x02.")
                return
            mask_to_unset = found_ban["mask"]
            self.db.remove_ban(ban_id)
        except ValueError:
            # It's a mask
            mask_to_unset = target
            self.db.remove_ban_by_mask(channel, target)

        # Remove +b on channel via samode
        if mask_to_unset:
            await self.irc.samode(channel, f"-b {mask_to_unset}")

        await self.reply(
            nick,
            f"Unbanned \x02{mask_to_unset}\x02 on \x02{channel}\x02.",
        )

    async def cmd_banlist(self, nick: str, hostmask: str, args: list[str]) -> None:
        """BANLIST <#channel> — Show ban list."""
        session = self.get_session(nick)
        if session is None:
            await self.reply(nick, "You must be logged in to use this command.")
            return

        if len(args) < 1:
            await self.reply(nick, "Usage: BANLIST <#channel>")
            return

        channel = args[0]

        if not channel.startswith("#"):
            await self.reply(nick, "Channel name must start with #.")
            return

        chan = self.db.get_channel(channel)
        if chan is None:
            await self.reply(nick, f"Channel \x02{channel}\x02 is not registered.")
            return

        my_level = self._get_user_level(session["user_id"], channel)
        if my_level < 1:
            await self.reply(nick, "You need level 1 or higher to view the ban list.")
            return

        try:
            bans = self.db.get_bans(channel)
        except ValueError as e:
            await self.reply(nick, str(e))
            return

        lines = [f"Ban list for \x02{channel}\x02:"]
        lines.append(
            f"  {'ID':>4}  {'MASK':<30}  {'LVL':>3}  {'SET BY':<15}  "
            f"{'EXPIRES':<20}  REASON"
        )

        if not bans:
            lines.append("  (no bans)")
        else:
            for ban in bans:
                set_by = ban.get("set_by_username") or "unknown"
                expires = ban.get("expires_at") or "never"
                reason = ban.get("reason") or ""
                lines.append(
                    f"  {ban['id']:>4}  {ban['mask']:<30}  "
                    f"{ban['level']:>3}  {set_by:<15}  "
                    f"{str(expires):<20}  {reason}"
                )

        lines.append("End of ban list.")
        await self.reply_lines(nick, lines)

    async def cmd_topic(self, nick: str, hostmask: str, args: list[str]) -> None:
        """TOPIC <#channel> <new topic> — Set channel topic."""
        session = self.get_session(nick)
        if session is None:
            await self.reply(nick, "You must be logged in to use this command.")
            return

        if len(args) < 2:
            await self.reply(nick, "Usage: TOPIC <#channel> <new topic>")
            return

        channel = args[0]
        new_topic = " ".join(args[1:])

        if not channel.startswith("#"):
            await self.reply(nick, "Channel name must start with #.")
            return

        chan = self.db.get_channel(channel)
        if chan is None:
            await self.reply(nick, f"Channel \x02{channel}\x02 is not registered.")
            return

        my_level = self._get_user_level(session["user_id"], channel)
        if my_level < 50:
            await self.reply(nick, "You need level 50 or higher to set the topic.")
            return

        await self.irc.set_topic(channel, new_topic)
        await self.reply(nick, f"Topic for \x02{channel}\x02 changed.")

    async def cmd_invite(self, nick: str, hostmask: str, args: list[str]) -> None:
        """INVITE <#channel> — Invite self to a channel."""
        session = self.get_session(nick)
        if session is None:
            await self.reply(nick, "You must be logged in to use this command.")
            return

        if len(args) < 1:
            await self.reply(nick, "Usage: INVITE <#channel>")
            return

        channel = args[0]

        if not channel.startswith("#"):
            await self.reply(nick, "Channel name must start with #.")
            return

        chan = self.db.get_channel(channel)
        if chan is None:
            await self.reply(nick, f"Channel \x02{channel}\x02 is not registered.")
            return

        my_level = self._get_user_level(session["user_id"], channel)
        if my_level < 1:
            await self.reply(
                nick, "You need level 1 or higher on this channel to be invited."
            )
            return

        await self.irc.invite(nick, channel)
        # No reply — the invite is sent

    # ==================================================================
    # CHANNEL SET COMMAND
    # ==================================================================

    async def cmd_set(self, nick: str, hostmask: str, args: list[str]) -> None:
        """SET <#channel> <option> <value> — Modify channel settings."""
        session = self.get_session(nick)
        if session is None:
            await self.reply(nick, "You must be logged in to use this command.")
            return

        if len(args) < 3:
            await self.reply(nick, "Usage: SET <#channel> <option> <value>")
            return

        channel = args[0]
        option = args[1].upper()
        value = " ".join(args[2:])

        if not channel.startswith("#"):
            await self.reply(nick, "Channel name must start with #.")
            return

        chan = self.db.get_channel(channel)
        if chan is None:
            await self.reply(nick, f"Channel \x02{channel}\x02 is not registered.")
            return

        my_level = self._get_user_level(session["user_id"], channel)
        if my_level < 400:
            await self.reply(nick, "You need level 400 or higher to change channel settings.")
            return

        if option == "DESCRIPTION":
            self.db.update_channel(channel, description=value)
            await self.reply(
                nick, f"SET: DESCRIPTION for \x02{channel}\x02 set to \x02{value}\x02."
            )

        elif option == "URL":
            self.db.update_channel(channel, url=value)
            await self.reply(
                nick, f"SET: URL for \x02{channel}\x02 set to \x02{value}\x02."
            )

        elif option == "TOPIC":
            self.db.update_channel(channel, topic=value)
            await self.reply(
                nick, f"SET: TOPIC for \x02{channel}\x02 set to \x02{value}\x02."
            )

        elif option == "AUTOTOPIC":
            value_lower = value.lower()
            if value_lower not in ("on", "off"):
                await self.reply(nick, "AUTOTOPIC must be 'on' or 'off'.")
                return
            self.db.update_channel(channel, autotopic=(1 if value_lower == "on" else 0))
            await self.reply(
                nick, f"SET: AUTOTOPIC for \x02{channel}\x02 set to \x02{value_lower}\x02."
            )

        elif option == "MODELOCK":
            self.db.update_channel(channel, mode_lock=value)
            await self.reply(
                nick, f"SET: MODELOCK for \x02{channel}\x02 set to \x02{value}\x02."
            )
            # Apply the mode lock immediately
            await self.irc.samode(channel, value)

        else:
            await self.reply(
                nick,
                f"Unknown SET option \x02{option}\x02. "
                "Valid options: DESCRIPTION, URL, TOPIC, AUTOTOPIC, MODELOCK.",
            )

    # ==================================================================
    # INFORMATION COMMANDS
    # ==================================================================

    async def cmd_info(self, nick: str, hostmask: str, args: list[str]) -> None:
        """INFO <#channel> — Show channel information."""
        if len(args) < 1:
            await self.reply(nick, "Usage: INFO <#channel>")
            return

        channel = args[0]

        if not channel.startswith("#"):
            await self.reply(nick, "Channel name must start with #.")
            return

        chan = self.db.get_channel(channel)
        if chan is None:
            await self.reply(nick, f"Channel \x02{channel}\x02 is not registered.")
            return

        # Look up registrant
        registered_by = "unknown"
        if chan.get("registered_by"):
            reg_user = self.db.get_user_by_id(chan["registered_by"])
            if reg_user:
                registered_by = reg_user["username"]

        # Count access list entries
        try:
            access_list = self.db.get_access_list(channel)
            access_count = len(access_list)
        except ValueError:
            access_count = 0

        lines = [
            f"Information for \x02{channel}\x02:",
            f"  Registered by: {registered_by}",
            f"  Registered at: {chan.get('registered_at', 'unknown')}",
            f"  Description:   {chan.get('description', '') or '(none)'}",
            f"  URL:           {chan.get('url', '') or '(none)'}",
            f"  Mode lock:     {chan.get('mode_lock', '') or '(none)'}",
            f"  Access list:   {access_count} user(s)",
            "End of INFO.",
        ]
        await self.reply_lines(nick, lines)

    async def cmd_chaninfo(self, nick: str, hostmask: str, args: list[str]) -> None:
        """CHANINFO <#channel> — Detailed channel info (admin/manager)."""
        session = self.get_session(nick)
        if session is None:
            await self.reply(nick, "You must be logged in to use this command.")
            return

        if len(args) < 1:
            await self.reply(nick, "Usage: CHANINFO <#channel>")
            return

        channel = args[0]

        if not channel.startswith("#"):
            await self.reply(nick, "Channel name must start with #.")
            return

        chan = self.db.get_channel(channel)
        if chan is None:
            await self.reply(nick, f"Channel \x02{channel}\x02 is not registered.")
            return

        # Check permissions: admin or level >= 400
        my_level = self._get_user_level(session["user_id"], channel)
        is_admin = self.db.is_admin(session["user_id"])

        if my_level < 400 and not is_admin:
            await self.reply(
                nick, "You need level 400 or higher (or admin) to view detailed channel info."
            )
            return

        # Look up registrant
        registered_by = "unknown"
        if chan.get("registered_by"):
            reg_user = self.db.get_user_by_id(chan["registered_by"])
            if reg_user:
                registered_by = reg_user["username"]

        # Count access list entries
        try:
            access_list = self.db.get_access_list(channel)
            access_count = len(access_list)
        except ValueError:
            access_count = 0

        # Count bans
        try:
            bans = self.db.get_bans(channel)
            ban_count = len(bans)
        except ValueError:
            ban_count = 0

        autotopic_str = "on" if chan.get("autotopic") else "off"

        lines = [
            f"Detailed information for \x02{channel}\x02:",
            f"  Channel ID:    {chan.get('id', '?')}",
            f"  Registered by: {registered_by}",
            f"  Registered at: {chan.get('registered_at', 'unknown')}",
            f"  Description:   {chan.get('description', '') or '(none)'}",
            f"  URL:           {chan.get('url', '') or '(none)'}",
            f"  Default topic: {chan.get('topic', '') or '(none)'}",
            f"  Autotopic:     {autotopic_str}",
            f"  Mode lock:     {chan.get('mode_lock', '') or '(none)'}",
            f"  Flags:         {chan.get('flags', 0)}",
            f"  Access list:   {access_count} user(s)",
            f"  Active bans:   {ban_count}",
            "End of CHANINFO.",
        ]
        await self.reply_lines(nick, lines)

    async def cmd_status(self, nick: str, hostmask: str, args: list[str]) -> None:
        """STATUS — Show bot status."""
        now = datetime.now(timezone.utc)
        uptime = now - self.start_time
        uptime_str = _format_uptime(uptime.total_seconds())

        channels = self.db.get_registered_channels()
        channel_count = len(channels)

        # Count all users (approximate — we don't have a count method, so just note sessions)
        active_sessions = len(self.sessions)

        lines = [
            f"\x02{self.bot_name}\x02 Channel Services Status:",
            f"  Uptime:           {uptime_str}",
            f"  Channels managed: {channel_count}",
            f"  Active sessions:  {active_sessions}",
            f"  Connected to:     {self.irc.host}:{self.irc.port}",
            f"  Bot nick:         {self.irc.nick}",
        ]
        await self.reply_lines(nick, lines)

    async def cmd_verify(self, nick: str, hostmask: str, args: list[str]) -> None:
        """VERIFY <nick> — Check if a nick is logged in."""
        if len(args) < 1:
            await self.reply(nick, "Usage: VERIFY <nick>")
            return

        target_nick = args[0]
        session = self.get_session(target_nick)

        if session:
            await self.reply(
                nick,
                f"\x02{target_nick}\x02 is logged in as \x02{session['username']}\x02.",
            )
        else:
            await self.reply(nick, f"\x02{target_nick}\x02 is NOT logged in.")

    async def cmd_showcommands(self, nick: str, hostmask: str, args: list[str]) -> None:
        """SHOWCOMMANDS — List all available commands."""
        lines = ["\x02Available commands:\x02"]

        # Group commands by category
        categories = {
            "Authentication": ["HELLO", "LOGIN", "LOGOUT"],
            "Channel Registration": ["REGISTER", "UNREGISTER"],
            "Access Management": ["ADDUSER", "REMUSER", "MODINFO", "ACCESS"],
            "Channel Operations": [
                "OP", "DEOP", "VOICE", "DEVOICE", "KICK",
                "BAN", "UNBAN", "BANLIST", "TOPIC", "INVITE",
            ],
            "Channel Settings": ["SET"],
            "Information": ["INFO", "CHANINFO", "STATUS", "VERIFY", "SHOWCOMMANDS", "HELP"],
            "Vhost": ["VHOST"],
            "Admin": ["ADMIN", "SUSPEND", "UNSUSPEND", "SAY", "BROADCAST", "NICK", "AUTOVHOST"],
        }

        for category, cmds in categories.items():
            lines.append(f"  \x02{category}:\x02")
            for cmd in cmds:
                help_entry = HELP_TEXTS.get(cmd, {})
                short_desc = help_entry.get("short", "")
                lines.append(f"    {cmd:<15} {short_desc}")

        lines.append("Use HELP <command> for detailed help on a specific command.")
        await self.reply_lines(nick, lines)

    async def cmd_help(self, nick: str, hostmask: str, args: list[str]) -> None:
        """HELP [command] — Show help for a command."""
        if len(args) < 1:
            # No command specified — show command list
            await self.cmd_showcommands(nick, hostmask, args)
            return

        command_name = args[0].upper()
        help_entry = HELP_TEXTS.get(command_name)

        if help_entry is None:
            await self.reply(
                nick,
                f"No help available for \x02{command_name}\x02. "
                "Use SHOWCOMMANDS for a list of commands.",
            )
            return

        lines = [f"\x02Help for {command_name}:\x02"]
        # Split detail text into individual lines
        for line in help_entry["detail"].split("\n"):
            lines.append(f"  {line}")
        await self.reply_lines(nick, lines)

    # ==================================================================
    # ADMIN COMMANDS
    # ==================================================================

    async def cmd_admin(self, nick: str, hostmask: str, args: list[str]) -> None:
        """ADMIN <username> <on|off> — Grant/revoke admin status."""
        session = self.get_session(nick)
        if session is None:
            await self.reply(nick, "You must be logged in to use this command.")
            return

        if not self.db.is_admin(session["user_id"]):
            await self.reply(nick, "You must be an admin to use this command.")
            return

        if len(args) < 2:
            await self.reply(nick, "Usage: ADMIN <username> <on|off>")
            return

        username = args[0]
        toggle = args[1].lower()

        if toggle not in ("on", "off"):
            await self.reply(nick, "Usage: ADMIN <username> <on|off>")
            return

        target_user = self.db.get_user(username)
        if target_user is None:
            await self.reply(nick, f"User \x02{username}\x02 does not exist.")
            return

        if toggle == "on":
            self.db.set_admin(target_user["id"], True)
            await self.reply(nick, f"\x02{username}\x02 is now an admin.")
        else:
            self.db.set_admin(target_user["id"], False)
            await self.reply(nick, f"\x02{username}\x02 is no longer an admin.")

    async def cmd_suspend(self, nick: str, hostmask: str, args: list[str]) -> None:
        """SUSPEND <username> — Suspend a user account."""
        session = self.get_session(nick)
        if session is None:
            await self.reply(nick, "You must be logged in to use this command.")
            return

        if not self.db.is_admin(session["user_id"]):
            await self.reply(nick, "You must be an admin to use this command.")
            return

        if len(args) < 1:
            await self.reply(nick, "Usage: SUSPEND <username>")
            return

        username = args[0]
        target_user = self.db.get_user(username)
        if target_user is None:
            await self.reply(nick, f"User \x02{username}\x02 does not exist.")
            return

        self.db.set_suspended(target_user["id"], True)

        # Also force-logout the suspended user if they have a session
        for session_nick, sess in list(self.sessions.items()):
            if sess["username"].lower() == username.lower():
                self._remove_session(session_nick)

        await self.reply(nick, f"\x02{username}\x02 has been suspended.")

    async def cmd_unsuspend(self, nick: str, hostmask: str, args: list[str]) -> None:
        """UNSUSPEND <username> — Unsuspend a user account."""
        session = self.get_session(nick)
        if session is None:
            await self.reply(nick, "You must be logged in to use this command.")
            return

        if not self.db.is_admin(session["user_id"]):
            await self.reply(nick, "You must be an admin to use this command.")
            return

        if len(args) < 1:
            await self.reply(nick, "Usage: UNSUSPEND <username>")
            return

        username = args[0]
        target_user = self.db.get_user(username)
        if target_user is None:
            await self.reply(nick, f"User \x02{username}\x02 does not exist.")
            return

        self.db.set_suspended(target_user["id"], False)
        await self.reply(nick, f"\x02{username}\x02 has been unsuspended.")

    async def cmd_say(self, nick: str, hostmask: str, args: list[str]) -> None:
        """SAY <#channel> <message> — Make bot speak in a channel."""
        session = self.get_session(nick)
        if session is None:
            await self.reply(nick, "You must be logged in to use this command.")
            return

        if not self.db.is_admin(session["user_id"]):
            await self.reply(nick, "You must be an admin to use this command.")
            return

        if len(args) < 2:
            await self.reply(nick, "Usage: SAY <#channel> <message>")
            return

        channel = args[0]
        message = " ".join(args[1:])

        if not channel.startswith("#"):
            await self.reply(nick, "Channel name must start with #.")
            return

        await self.irc.send_privmsg(channel, message)

    async def cmd_broadcast(self, nick: str, hostmask: str, args: list[str]) -> None:
        """BROADCAST <message> — Send a notice to all registered channels."""
        session = self.get_session(nick)
        if session is None:
            await self.reply(nick, "You must be logged in to use this command.")
            return

        if not self.db.is_admin(session["user_id"]):
            await self.reply(nick, "You must be an admin to use this command.")
            return

        if len(args) < 1:
            await self.reply(nick, "Usage: BROADCAST <message>")
            return

        message = " ".join(args)
        channels = self.db.get_registered_channels()

        broadcast_msg = f"[\x02{self.bot_name}\x02] {message}"
        for chan in channels:
            await self.irc.send_notice(chan["name"], broadcast_msg)

        await self.reply(
            nick, f"Broadcast sent to {len(channels)} channel(s)."
        )

    # ==================================================================
    # VHOST COMMANDS
    # ==================================================================

    async def cmd_vhost(self, nick: str, hostmask: str, args: list[str]) -> None:
        """VHOST <subcommand> — Manage virtual hostnames."""
        session = self.get_session(nick)
        if session is None:
            await self.reply(nick, "You must be logged in to use this command.")
            return

        if len(args) < 1:
            await self.reply(nick, "Usage: VHOST <LIST|SET|CLEAR|ADD|DEL|OFF|ON|SETUSER>")
            return

        subcmd = args[0].upper()
        sub_args = args[1:]

        if subcmd == "LIST":
            await self._vhost_list(nick, session)
        elif subcmd == "SET":
            await self._vhost_set(nick, session, sub_args)
        elif subcmd == "CLEAR":
            await self._vhost_clear(nick, session)
        elif subcmd == "ADD":
            await self._vhost_add(nick, session, sub_args)
        elif subcmd == "DEL":
            await self._vhost_del(nick, session, sub_args)
        elif subcmd == "ON":
            await self._vhost_toggle(nick, session, sub_args, active=True)
        elif subcmd == "OFF":
            await self._vhost_toggle(nick, session, sub_args, active=False)
        elif subcmd == "SETUSER":
            await self._vhost_setuser(nick, session, sub_args)
        else:
            await self.reply(nick, f"Unknown VHOST subcommand: \x02{subcmd}\x02")
            await self.reply(nick, "Usage: VHOST <LIST|SET|CLEAR|ADD|DEL|OFF|ON|SETUSER>")

    async def _vhost_list(self, nick: str, session: dict) -> None:
        """VHOST LIST — Show available vhosts."""
        vhosts = self.db.list_vhosts(active_only=True)
        if not vhosts:
            await self.reply(nick, "No vhosts are currently available.")
            return

        # Also get the user's current vhost
        current = self.db.get_user_vhost(session["user_id"])
        current_pattern = current["pattern"] if current else None

        lines = [f"\x02Available vhosts ({len(vhosts)}):\x02"]
        for vh in vhosts:
            marker = " \x02[current]\x02" if current_pattern and vh["pattern"].lower() == current_pattern.lower() else ""
            desc = f" — {vh['description']}" if vh["description"] else ""
            lines.append(f"  {vh['pattern']}{desc}{marker}")

        lines.append("Use \x02VHOST SET <vhost>\x02 to apply one.")
        await self.reply_lines(nick, lines)

    async def _vhost_set(self, nick: str, session: dict, args: list[str]) -> None:
        """VHOST SET <vhost> — Apply a vhost to yourself."""
        if len(args) < 1:
            await self.reply(nick, "Usage: VHOST SET <vhost>")
            return

        pattern = args[0]
        vhost = self.db.get_vhost(pattern)
        if vhost is None or not vhost["is_active"]:
            await self.reply(nick, f"Vhost \x02{pattern}\x02 is not available. Use VHOST LIST to see options.")
            return

        # Save to database
        self.db.set_user_vhost(session["user_id"], vhost["id"])

        # Apply via CHGHOST on IRC
        await self.irc.chghost(nick, vhost["pattern"])
        await self.reply(nick, f"Your vhost has been set to \x02{vhost['pattern']}\x02.")

    async def _vhost_clear(self, nick: str, session: dict) -> None:
        """VHOST CLEAR — Remove your current vhost."""
        current = self.db.get_user_vhost(session["user_id"])
        if current is None:
            await self.reply(nick, "You don't have a vhost set.")
            return

        self.db.clear_user_vhost(session["user_id"])
        # Reset their host to their real connecting host by setting it to
        # their ident@host from the hostmask. There's no "undo CHGHOST"
        # in InspIRCd, so we just inform them.
        await self.reply(nick, "Your vhost has been cleared. Your real host will be visible after reconnecting.")

    async def _vhost_add(self, nick: str, session: dict, args: list[str]) -> None:
        """VHOST ADD <pattern> [description] — Add a new vhost (admin)."""
        if not self.db.is_admin(session["user_id"]):
            await self.reply(nick, "You must be an admin to add vhosts.")
            return

        if len(args) < 1:
            await self.reply(nick, "Usage: VHOST ADD <pattern> [description]")
            return

        pattern = args[0]
        description = " ".join(args[1:]) if len(args) > 1 else ""

        # Basic validation — vhosts should look like hostnames
        if " " in pattern or "!" in pattern or "@" in pattern:
            await self.reply(nick, "Invalid vhost pattern. Must be a valid hostname (no spaces, !, or @).")
            return

        try:
            vhost_id = self.db.add_vhost(pattern, description, session["user_id"])
            await self.reply(nick, f"Vhost \x02{pattern}\x02 added (id={vhost_id}).")
        except ValueError as exc:
            await self.reply(nick, str(exc))

    async def _vhost_del(self, nick: str, session: dict, args: list[str]) -> None:
        """VHOST DEL <pattern> — Remove a vhost (admin)."""
        if not self.db.is_admin(session["user_id"]):
            await self.reply(nick, "You must be an admin to remove vhosts.")
            return

        if len(args) < 1:
            await self.reply(nick, "Usage: VHOST DEL <pattern>")
            return

        pattern = args[0]
        try:
            self.db.remove_vhost(pattern)
            await self.reply(nick, f"Vhost \x02{pattern}\x02 has been removed.")
        except ValueError as exc:
            await self.reply(nick, str(exc))

    async def _vhost_toggle(self, nick: str, session: dict, args: list[str], active: bool) -> None:
        """VHOST ON/OFF <pattern> — Enable or disable a vhost (admin)."""
        if not self.db.is_admin(session["user_id"]):
            await self.reply(nick, "You must be an admin to manage vhosts.")
            return

        if len(args) < 1:
            await self.reply(nick, f"Usage: VHOST {'ON' if active else 'OFF'} <pattern>")
            return

        pattern = args[0]
        try:
            self.db.toggle_vhost(pattern, active)
            state = "enabled" if active else "disabled"
            await self.reply(nick, f"Vhost \x02{pattern}\x02 has been {state}.")
        except ValueError as exc:
            await self.reply(nick, str(exc))

    async def _vhost_setuser(self, nick: str, session: dict, args: list[str]) -> None:
        """VHOST SETUSER <nick> <vhost> — Force-set a vhost on another user (admin)."""
        if not self.db.is_admin(session["user_id"]):
            await self.reply(nick, "You must be an admin to use this command.")
            return

        if len(args) < 2:
            await self.reply(nick, "Usage: VHOST SETUSER <nick> <vhost>")
            return

        target_nick = args[0]
        pattern = args[1]

        # Look up the target's session
        target_session = self.get_session(target_nick)
        if target_session is None:
            await self.reply(nick, f"\x02{target_nick}\x02 is not logged in.")
            return

        vhost = self.db.get_vhost(pattern)
        if vhost is None:
            await self.reply(nick, f"Vhost \x02{pattern}\x02 does not exist. Add it first with VHOST ADD.")
            return

        # Save and apply
        self.db.set_user_vhost(target_session["user_id"], vhost["id"])
        await self.irc.chghost(target_nick, vhost["pattern"])
        await self.reply(nick, f"Vhost for \x02{target_nick}\x02 set to \x02{vhost['pattern']}\x02.")

    # ==================================================================
    # BOT MANAGEMENT COMMANDS
    # ==================================================================

    async def cmd_nick(self, nick: str, hostmask: str, args: list[str]) -> None:
        """NICK <new_nick> — Change the bot's nickname (admin only)."""
        session = self.get_session(nick)
        if session is None:
            await self.reply(nick, "You must be logged in to use this command.")
            return

        if not self.db.is_admin(session["user_id"]):
            await self.reply(nick, "You must be an admin to use this command.")
            return

        if len(args) < 1:
            await self.reply(nick, "Usage: NICK <new_nick>")
            return

        new_nick = args[0]

        # Basic nick validation (IRC nicks: letters, digits, special chars, no spaces)
        if not re.match(r'^[A-Za-z\[\]\\`_^{|}][A-Za-z0-9\[\]\\`_^{|}\-]*$', new_nick):
            await self.reply(nick, f"Invalid nickname: \x02{new_nick}\x02")
            return

        if len(new_nick) > 30:
            await self.reply(nick, "Nickname too long (max 30 characters).")
            return

        old_nick = self.irc.nick
        await self.irc.change_nick(new_nick)

        # Update the bot_name used in messages
        self.bot_name = new_nick
        await self.reply(nick, f"Bot nick changed from \x02{old_nick}\x02 to \x02{new_nick}\x02.")

    async def cmd_autovhost(self, nick: str, hostmask: str, args: list[str]) -> None:
        """AUTOVHOST <ADD|DEL|LIST> — Manage automatic vhost assignments (admin)."""
        session = self.get_session(nick)
        if session is None:
            await self.reply(nick, "You must be logged in to use this command.")
            return

        if not self.db.is_admin(session["user_id"]):
            await self.reply(nick, "You must be an admin to use this command.")
            return

        if len(args) < 1:
            await self.reply(nick, "Usage: AUTOVHOST <ADD|DEL|LIST>")
            return

        subcmd = args[0].upper()

        if subcmd == "ADD":
            if len(args) < 3:
                await self.reply(nick, "Usage: AUTOVHOST ADD <hostmask> <vhost>")
                return
            mask = args[1]
            vhost = args[2]
            try:
                av_id = self.db.add_auto_vhost(mask, vhost, session["user_id"])
                await self.reply(nick, f"Auto-vhost added (id={av_id}): \x02{mask}\x02 -> \x02{vhost}\x02")
            except ValueError as exc:
                await self.reply(nick, str(exc))

        elif subcmd == "DEL":
            if len(args) < 2:
                await self.reply(nick, "Usage: AUTOVHOST DEL <hostmask>")
                return
            mask = args[1]
            try:
                self.db.remove_auto_vhost(mask)
                await self.reply(nick, f"Auto-vhost for \x02{mask}\x02 removed.")
            except ValueError as exc:
                await self.reply(nick, str(exc))

        elif subcmd == "LIST":
            entries = self.db.list_auto_vhosts()
            if not entries:
                await self.reply(nick, "No auto-vhost rules configured.")
                return
            lines = [f"\x02Auto-vhost rules ({len(entries)}):\x02"]
            for e in entries:
                lines.append(f"  {e['hostmask']}  ->  {e['vhost']}")
            await self.reply_lines(nick, lines)

        else:
            await self.reply(nick, f"Unknown subcommand: \x02{subcmd}\x02. Use ADD, DEL, or LIST.")

    # ==================================================================
    # IRC EVENT HANDLERS
    # ==================================================================

    async def on_join(self, nick: str, hostmask: str, channel: str) -> None:
        """Handle user joins — apply automode, enforce bans, and auto-vhosts.

        Called by the IRC client's "join" event.
        """
        # Skip if it's the bot itself joining
        if nick.lower() == self.irc.nick.lower():
            return

        # Auto-vhost: check if the joining user's hostmask matches any rule
        auto_vhost = self.db.match_auto_vhost(hostmask)
        if auto_vhost:
            await self.irc.chghost(nick, auto_vhost)

        # Check if channel is registered
        chan = self.db.get_channel(channel)
        if chan is None:
            return

        # Check if the joining user matches any active ban
        try:
            matching_ban = self.db.get_matching_ban(channel, hostmask)
            if matching_ban:
                reason = matching_ban.get("reason", "Banned")
                await self.irc.samode(channel, f"+b {matching_ban['mask']}")
                await self.irc.sakick(channel, nick, f"Banned: {reason}")
                return
        except ValueError:
            pass  # Channel lookup issue, skip ban check

        # Check if the joining user is logged in
        session = self.get_session(nick)
        if session is None:
            return

        # Check their access level and automode
        try:
            access = self.db.get_access(channel, session["username"])
        except ValueError:
            return

        if access is None:
            return

        automode = access.get("automode", "none")
        level = access.get("level", 0)

        if automode == "op" and level >= 100:
            await self.irc.samode(channel, f"+o {nick}")
        elif automode == "voice" and level >= 25:
            await self.irc.samode(channel, f"+v {nick}")

        # Apply autotopic if enabled
        if chan.get("autotopic") and chan.get("topic"):
            # Only set topic when the first user joins (or we could do it
            # every time — but to avoid spam, skip for now; the topic is set
            # on bot join via on_connected)
            pass

    async def on_connected(self, server_name: str) -> None:
        """Handle successful connection — join all registered channels.

        Called by the IRC client's "connected" event.
        """
        logger.info("Connected to %s — joining registered channels", server_name)

        channels = self.db.get_registered_channels()
        for chan in channels:
            channel_name = chan["name"]
            await self.irc.join(channel_name)
            # Op ourselves via samode
            await self.irc.samode(channel_name, f"+o {self.irc.nick}")

            # Apply mode lock if set
            if chan.get("mode_lock"):
                await self.irc.samode(channel_name, chan["mode_lock"])

            # Set default topic if autotopic is enabled
            if chan.get("autotopic") and chan.get("topic"):
                await self.irc.set_topic(channel_name, chan["topic"])

    async def on_quit(self, nick: str, hostmask: str, reason: str) -> None:
        """Handle user quits — clear their session.

        Called by the IRC client's "quit" event.
        """
        removed = self._remove_session(nick)
        if removed:
            logger.debug("Cleared session for %s (quit: %s)", nick, reason)

    async def on_nick(self, old_nick: str, hostmask: str, new_nick: str) -> None:
        """Handle nick changes — move session from old nick to new nick.

        Called by the IRC client's "nick" event.
        """
        session = self._remove_session(old_nick)
        if session:
            # Update the hostmask in the session (the nick part changed)
            session["hostmask"] = hostmask
            self.sessions[new_nick.lower()] = session
            logger.debug(
                "Moved session from %s to %s (user: %s)",
                old_nick, new_nick, session["username"],
            )

    async def on_kick(self, nick: str, hostmask: str, channel: str,
                      kicked_nick: str, reason: str) -> None:
        """Handle kicks — rejoin if the bot was kicked.

        Called by the IRC client's "kick" event.
        """
        if kicked_nick.lower() == self.irc.nick.lower():
            logger.warning(
                "Bot was kicked from %s by %s (%s) — rejoining",
                channel, nick, reason,
            )
            # Rejoin and re-op
            await self.irc.join(channel)
            await self.irc.samode(channel, f"+o {self.irc.nick}")


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _format_uptime(total_seconds: float) -> str:
    """Format seconds into a human-readable uptime string."""
    total_seconds = int(total_seconds)
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)

    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")
    return " ".join(parts)
