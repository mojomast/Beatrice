#!/usr/bin/env python3
"""
ussynet - Channel Services Bot
Modeled after Undernet's X/CService bot.

Honoring the legacy of Dancer IRC bot (1998) — the bot whose HTTP
download feature (httpget) evolved into curl, now running on 30+ billion
devices including the Mars Ingenuity helicopter.

All configuration values can be overridden via environment variables
prefixed with USSYNET_. See apply_env_overrides() for the full mapping.

Usage:
    python3 -m services.bot [--config path/to/config.json] [--backup]

    --backup    Run as backup instance
    --config    Path to config.json (default: services/config.json)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.database import Database
from services.irc import IRCClient
from services.commands import CommandHandler


def setup_logging(level: str = "INFO") -> None:
    """Configure logging format and level for the ussynet bot."""
    log_format = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=log_format,
        datefmt=date_format,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def load_config(config_path: str) -> dict:
    """Load the JSON configuration file from disk.

    Args:
        config_path: Filesystem path to the config.json file.

    Returns:
        Parsed configuration dictionary.
    """
    path = Path(config_path)
    if not path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def apply_env_overrides(config: dict) -> dict:
    """Apply environment variable overrides to the loaded configuration.

    Any USSYNET_* environment variable that is set will take precedence
    over the corresponding value in the JSON config file.  This makes it
    easy to run the bot in containerised / CI environments without
    modifying the config on disk.

    Environment variable mapping:
        USSYNET_IRC_HOST           -> config["server"]["host"]
        USSYNET_IRC_PORT           -> config["server"]["port"]           (int)
        USSYNET_IRC_SSL            -> config["server"]["use_ssl"]        (bool)
        USSYNET_BOT_NICK           -> config["bot"]["nick"]
        USSYNET_BOT_IDENT          -> config["bot"]["ident"]
        USSYNET_BOT_REALNAME       -> config["bot"]["realname"]
        USSYNET_BOT_OPER_NAME      -> config["bot"]["oper_name"]
        USSYNET_BOT_OPER_PASSWORD  -> config["bot"]["oper_password"]
        USSYNET_BACKUP_NICK        -> config["backup"]["nick"]
        USSYNET_BACKUP_OPER_NAME   -> config["backup"]["oper_name"]
        USSYNET_BACKUP_OPER_PASSWORD -> config["backup"]["oper_password"]
        USSYNET_DB_PATH            -> config["database"]["path"]
        USSYNET_ADMIN_HOSTMASKS    -> config["services"]["admin_hostmasks"] (comma-separated)
        USSYNET_DEFAULT_CHANNEL    -> config["services"]["default_channel"]
        USSYNET_LOG_CHANNEL        -> config["services"]["log_channel"]
        USSYNET_NETWORK_NAME       -> config["network"]["name"]

    Args:
        config: The configuration dictionary loaded from JSON.

    Returns:
        The same dictionary, mutated in place, with env overrides applied.
    """
    # Ensure all top-level sections exist so env overrides don't KeyError
    config.setdefault("server", {})
    config.setdefault("bot", {})
    config.setdefault("backup", {})
    config.setdefault("database", {})
    config.setdefault("services", {})
    config.setdefault("network", {"name": "ussynet"})

    # --- simple string overrides ------------------------------------------------
    _str_mappings: list[tuple[str, str, str]] = [
        # (env_var,                    section,     key)
        ("USSYNET_IRC_HOST",           "server",    "host"),
        ("USSYNET_BOT_NICK",           "bot",       "nick"),
        ("USSYNET_BOT_IDENT",          "bot",       "ident"),
        ("USSYNET_BOT_REALNAME",       "bot",       "realname"),
        ("USSYNET_BOT_OPER_NAME",      "bot",       "oper_name"),
        ("USSYNET_BOT_OPER_PASSWORD",  "bot",       "oper_password"),
        ("USSYNET_BACKUP_NICK",        "backup",    "nick"),
        ("USSYNET_BACKUP_OPER_NAME",   "backup",    "oper_name"),
        ("USSYNET_BACKUP_OPER_PASSWORD", "backup",  "oper_password"),
        ("USSYNET_DB_PATH",            "database",  "path"),
        ("USSYNET_DEFAULT_CHANNEL",    "services",  "default_channel"),
        ("USSYNET_LOG_CHANNEL",        "services",  "log_channel"),
        ("USSYNET_NETWORK_NAME",       "network",   "name"),
    ]

    for env_var, section, key in _str_mappings:
        value = os.environ.get(env_var)
        if value is not None:
            config[section][key] = value

    # --- int override: port ------------------------------------------------------
    port_str = os.environ.get("USSYNET_IRC_PORT")
    if port_str is not None:
        config["server"]["port"] = int(port_str)

    # --- bool override: use_ssl --------------------------------------------------
    ssl_str = os.environ.get("USSYNET_IRC_SSL")
    if ssl_str is not None:
        config["server"]["use_ssl"] = ssl_str.lower() in ("true", "1")

    # --- list override: admin_hostmasks (comma-separated) ------------------------
    hostmasks_str = os.environ.get("USSYNET_ADMIN_HOSTMASKS")
    if hostmasks_str is not None:
        config["services"]["admin_hostmasks"] = [
            h.strip() for h in hostmasks_str.split(",") if h.strip()
        ]

    return config


class UssynetBot:
    """Main bot class for the ussynet Channel Services Bot.

    Manages the IRC connection, database, command handling, and
    periodic maintenance tasks such as ban expiry cleanup.
    """

    def __init__(self, config: dict, backup: bool = False):
        self.config = config
        self.backup = backup
        bot_conf = config["backup"] if backup else config["bot"]
        server_conf = config["server"]
        self.nick = bot_conf["nick"]

        # Derive the bot display name from the network config (default "ussynet")
        self.bot_name = config.get("network", {}).get("name", "ussynet")

        self.logger = logging.getLogger(f"ussynet.{self.nick}")

        db_path = config["database"]["path"]
        if not os.path.isabs(db_path):
            db_path = str(Path(__file__).resolve().parent / db_path)
        self.db = Database(db_path)

        self.irc = IRCClient(
            host=server_conf["host"],
            port=server_conf["port"],
            nick=bot_conf["nick"],
            ident=bot_conf["ident"],
            realname=bot_conf["realname"],
            oper_name=bot_conf["oper_name"],
            oper_password=bot_conf["oper_password"],
            use_ssl=server_conf.get("use_ssl", False),
        )

        self.cmd_handler = CommandHandler(self.irc, self.db, config)
        self._register_handlers()

        self.logger.info(
            "%s initialized as %s (%s mode)",
            self.bot_name, self.nick, "backup" if backup else "primary",
        )
        self._stopped = False

    def _register_handlers(self) -> None:
        """Wire up IRC event callbacks to the command handler."""
        self.irc.on("connected", self.cmd_handler.on_connected)
        self.irc.on("privmsg", self._on_privmsg)
        self.irc.on("join", self.cmd_handler.on_join)
        self.irc.on("quit", self.cmd_handler.on_quit)
        self.irc.on("nick", self.cmd_handler.on_nick)
        self.irc.on("kick", self.cmd_handler.on_kick)

    async def _on_privmsg(self, nick: str, hostmask: str, target: str, message: str) -> None:
        """Dispatch an incoming PRIVMSG to the command handler."""
        await self.cmd_handler.handle_message(nick, hostmask, target, message)

    async def start(self) -> None:
        """Connect to the database and IRC server, then enter the main loop."""
        self.logger.info("Starting %s...", self.nick)
        self.db.connect()
        self.logger.info("Database connected: %s", self.db.db_path)
        asyncio.create_task(self._ban_cleanup_loop())
        await self.irc.run()

    async def stop(self) -> None:
        """Gracefully shut down the bot, disconnecting from IRC and closing the DB."""
        if self._stopped:
            return
        self._stopped = True
        self.logger.info("Shutting down %s...", self.nick)
        await self.irc.disconnect(f"{self.nick} shutting down")
        self.db.close()
        self.logger.info("Shutdown complete.")

    async def _ban_cleanup_loop(self) -> None:
        """Periodically remove expired bans from the database (every 300 s)."""
        while True:
            try:
                await asyncio.sleep(300)
                count = self.db.cleanup_expired_bans()
                if count:
                    self.logger.info("Cleaned up %d expired bans", count)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in ban cleanup: %s", e)


async def main() -> None:
    """Entry point for the ussynet Channel Services Bot."""
    parser = argparse.ArgumentParser(
        description="ussynet Channel Services Bot",
        epilog="Honoring the legacy of Dancer IRC bot (1998)",
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "config.json"),
        help="Path to config.json",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Run as backup instance",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger("ussynet.main")

    logger.info("=" * 60)
    logger.info("ussynet Channel Services Bot v1.0.0")
    logger.info("Honoring the legacy of Dancer IRC bot (1998)")
    logger.info("=" * 60)

    config = load_config(args.config)
    apply_env_overrides(config)

    # Ensure the network section has a sensible default even if the
    # config file omits it entirely.
    config.setdefault("network", {"name": "ussynet"})

    bot = UssynetBot(config, backup=args.backup)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(bot.stop()))

    try:
        await bot.start()
    except KeyboardInterrupt:
        pass
    finally:
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(main())
