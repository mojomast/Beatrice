from __future__ import annotations

import asyncio
import logging
import os
import signal

from .app import BeatriceBot, RequestRoute, setup_logging
from .child_bots import CHILD_RESPONSE_MODE_AMBIENT
from .config import BotSettings


class ManagedChildBot(BeatriceBot):
    def __init__(self, settings: BotSettings) -> None:
        super().__init__(settings)
        self._child_response_mode = os.getenv("BOT_CHILD_RESPONSE_MODE", "addressed_only").strip().lower() or "addressed_only"

    async def run(self) -> None:
        logging.getLogger("beatrice").info("Starting managed child bot id=%s nick=%s", os.getenv("BOT_CHILD_ID", "unknown"), self.settings.irc_nick)
        await self.irc.run()

    async def _on_connected(self, _server_name: str) -> None:
        await super()._on_connected(_server_name)
        if self._child_response_mode == CHILD_RESPONSE_MODE_AMBIENT:
            for channel in self.settings.irc_channels:
                self._chat_channels.add(channel.lower())

    async def stop(self) -> None:
        if self._stopping:
            return
        self._stopping = True
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        await self.irc.disconnect(quit_message=f"{self.settings.irc_nick} signing off")
        await self.openrouter.aclose()

    def _known_bot_nicks(self) -> set[str]:
        """Read known bot nicks from env var set by parent bot."""
        raw = os.getenv("BOT_KNOWN_BOT_NICKS", "")
        nicks = {n.strip().lower() for n in raw.split(",") if n.strip()}
        if self.irc.nick:
            nicks.add(self.irc.nick.lower())
        return nicks

    def _classify_request(self, context, prompt: str, github_scope):
        return RequestRoute(model_route="chat", use_tools=False, reason="child_chat_only")

    def _allows_private_capabilities(self, context) -> bool:
        return False

    def _should_force_admin_public_tools(self, context, prompt: str) -> bool:
        return False

    def _tool_definitions(self, allowed_names=None):
        return []

    def _ambient_requires_invitation(self, context) -> bool:
        return self._child_response_mode != CHILD_RESPONSE_MODE_AMBIENT


def _child_settings() -> BotSettings:
    settings = BotSettings.from_env()
    system_prompt = os.getenv("BOT_CHILD_SYSTEM_PROMPT", "").strip()
    model = os.getenv("BOT_CHILD_MODEL", "").strip() or settings.child_default_model
    temperature = float(os.getenv("BOT_CHILD_TEMPERATURE", "0.7"))
    top_p = float(os.getenv("BOT_CHILD_TOP_P", "1.0"))
    max_tokens = int(os.getenv("BOT_CHILD_MAX_TOKENS", "180"))
    reply_interval_seconds = float(os.getenv("BOT_CHILD_REPLY_INTERVAL_SECONDS", "4"))
    defaults = settings.runtime_defaults
    defaults = defaults.__class__(
        system_prompt=system_prompt or defaults.system_prompt,
        model=model,
        models=defaults.models,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stream=False,
        reply_interval_seconds=reply_interval_seconds,
    )
    return BotSettings(
        openrouter_api_key=settings.openrouter_api_key,
        irc_server=settings.irc_server,
        irc_port=settings.irc_port,
        irc_nick=settings.irc_nick,
        irc_user=settings.irc_user,
        irc_realname=settings.irc_realname,
        irc_channels=settings.irc_channels,
        irc_password=settings.irc_password,
        irc_message_length=settings.irc_message_length,
        irc_max_line_bytes=settings.irc_max_line_bytes,
        command_prefix=settings.command_prefix,
        admin_password=settings.admin_password,
        admin_nicks=(),
        approval_timeout_seconds=settings.approval_timeout_seconds,
        openrouter_base_url=settings.openrouter_base_url,
        openrouter_http_referer=settings.openrouter_http_referer,
        openrouter_title=settings.openrouter_title,
        settings_file=settings.settings_file,
        secrets_file=settings.secrets_file,
        runtime_file=settings.runtime_file,
        memory_db_file=settings.memory_db_file,
        audit_log_file=settings.audit_log_file,
        child_bots_file=settings.child_bots_file,
        child_state_file=settings.child_state_file,
        child_data_dir=settings.child_data_dir,
        child_default_model=settings.child_default_model,
        history_turns=min(settings.history_turns, 4),
        runtime_defaults=defaults,
    )


async def main() -> None:
    setup_logging()
    settings = _child_settings()
    bot = ManagedChildBot(settings)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(bot.stop()))
    try:
        await bot.run()
    finally:
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(main())
