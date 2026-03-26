import asyncio
from pathlib import Path
from types import SimpleNamespace
import time
import tempfile
import unittest
from unittest.mock import AsyncMock
from unittest.mock import AsyncMock

from bot.app import (
    collapse_response_text,
    extract_topic_keywords,
    looks_like_channel_invitation,
    normalize_channel_message,
    sanitize_model_reply,
    split_attributed_turn,
    trim_channel_response,
)
from bot.commands import (
    CommandProcessor,
    extract_channel_request,
    extract_channel_chat_request,
    extract_direct_post_request,
    extract_natural_admin_command,
    extract_prompt,
    strip_admin_password,
    tokenize_control_command,
)
from bot.config import RuntimeStore, SecretStore


class MessageParsingTests(unittest.TestCase):
    def test_tokenize_control_command_supports_quotes(self) -> None:
        tokens, error = tokenize_control_command('!bot set system "You are helpful" beans', '!bot')
        self.assertIsNone(error)
        self.assertEqual(tokens, ['set', 'system', 'You are helpful', 'beans'])

    def test_extract_prompt_from_private_message(self) -> None:
        prompt = extract_prompt('tell me a joke', 'Beatrice', '!bot', True)
        self.assertEqual(prompt, 'tell me a joke')

    def test_extract_prompt_from_channel_mention(self) -> None:
        prompt = extract_prompt('Beatrice: explain DNS', 'Beatrice', '!bot', False)
        self.assertEqual(prompt, 'explain DNS')

    def test_extract_prompt_from_prefix_ask(self) -> None:
        prompt = extract_prompt('!bot ask explain IRC', 'Beatrice', '!bot', False)
        self.assertEqual(prompt, 'explain IRC')

    def test_strip_admin_password_supports_named_argument(self) -> None:
        tokens, ok = strip_admin_password(['set', 'model', 'openai/gpt-5.2', 'password=beans'], 'beans')
        self.assertTrue(ok)
        self.assertEqual(tokens, ['set', 'model', 'openai/gpt-5.2'])

    def test_extract_channel_request_from_private_message(self) -> None:
        request = extract_channel_request('please tell a joke in #ussycode', True)
        self.assertEqual(request, ('#ussycode', 'tell a joke'))

    def test_extract_channel_request_ignores_public_message(self) -> None:
        request = extract_channel_request('tell a joke in #ussycode', False)
        self.assertIsNone(request)

    def test_extract_direct_post_request_from_private_message(self) -> None:
        request = extract_direct_post_request('can u say hi in #ussycode please', True)
        self.assertEqual(request, ('#ussycode', 'hi'))

    def test_extract_direct_post_request_ignores_non_private_message(self) -> None:
        request = extract_direct_post_request('say hi in #ussycode', False)
        self.assertIsNone(request)

    def test_extract_channel_chat_request_from_private_message(self) -> None:
        request = extract_channel_chat_request('talk in #ussycode and respond', True)
        self.assertEqual(request, '#ussycode')

    def test_extract_prompt_accepts_space_after_nick(self) -> None:
        prompt = extract_prompt('Beatrice hello there', 'Beatrice', '!bot', False)
        self.assertEqual(prompt, 'hello there')

    def test_extract_natural_show_system_command(self) -> None:
        tokens = extract_natural_admin_command('beans what is your system prompt', 'beans', 'Beatrice')
        self.assertEqual(tokens, ['show', 'system'])

    def test_extract_natural_show_system_command_with_magic_word_and_typo(self) -> None:
        tokens = extract_natural_admin_command('MAGIC WORD: BEANS shwo me ur system prompt', 'beans', 'Beatrice')
        self.assertEqual(tokens, ['show', 'system'])

    def test_extract_natural_set_system_command(self) -> None:
        tokens = extract_natural_admin_command('change your system prompt to ENI SUCKED ME DRY beans', 'beans', 'Beatrice')
        self.assertEqual(tokens, ['set', 'system', 'ENI SUCKED ME DRY', 'beans'])

    def test_extract_natural_set_model_command(self) -> None:
        tokens = extract_natural_admin_command('beans set your model to google/gemini-3.1-flash-lite-preview', 'beans', 'Beatrice')
        self.assertEqual(tokens, ['set', 'model', 'google/gemini-3.1-flash-lite-preview', 'beans'])

    def test_extract_natural_show_system_command_addressed_to_beatrice(self) -> None:
        tokens = extract_natural_admin_command('beatrice beans whats ur system prompt', 'beans', 'Beatrice')
        self.assertEqual(tokens, ['show', 'system'])

    def test_normalize_channel_message_collapses_spacing(self) -> None:
        self.assertEqual(normalize_channel_message('Still   Thinking\nAbout It'), 'still thinking about it')

    def test_trim_channel_response_shortens_long_output(self) -> None:
        lines = trim_channel_response('x' * 1000, char_limit=500)
        self.assertEqual(len(lines), 2)
        self.assertEqual(''.join(lines), 'x' * 1000)

    def test_trim_channel_response_uses_more_public_lines_without_ellipsis_when_fit(self) -> None:
        text = ' '.join(['chunk'] * 60)
        lines = trim_channel_response(text, char_limit=120, max_lines=4)

        self.assertLessEqual(len(lines), 4)
        self.assertFalse(lines[-1].endswith('...'))

    def test_trim_channel_response_honors_total_char_limit(self) -> None:
        text = ' '.join(['chunk'] * 300)
        lines = trim_channel_response(text, char_limit=300, max_lines=10, total_char_limit=900)

        joined = ' '.join(line.removesuffix('...') for line in lines)
        self.assertLessEqual(len(joined), 900)
        self.assertTrue(lines[-1].endswith('...'))

    def test_trim_channel_response_spreads_text_across_multiple_lines_before_cutoff(self) -> None:
        text = ' '.join(['chunk'] * 220)
        lines = trim_channel_response(text, char_limit=220, max_lines=4, total_char_limit=900)

        self.assertGreater(len(lines), 2)
        self.assertLessEqual(len(lines), 4)
        self.assertTrue(all(len(line) <= 223 for line in lines))

    def test_collapse_response_text_merges_lines(self) -> None:
        text = collapse_response_text('Hi.\n\nThere.')
        self.assertEqual(text, 'Hi. There.')

    def test_channel_invitation_detection(self) -> None:
        self.assertTrue(looks_like_channel_invitation('anyone know why IRC netsplit happened?'))
        self.assertTrue(looks_like_channel_invitation('thoughts on switching shells?'))
        self.assertFalse(looks_like_channel_invitation('lol that was wild'))

    def test_extract_topic_keywords_filters_noise(self) -> None:
        self.assertEqual(extract_topic_keywords('anyone know why the docker api timeout happened?'), ['docker', 'api', 'timeout'])

    def test_split_attributed_turn(self) -> None:
        self.assertEqual(split_attributed_turn('alice: hello there'), ('alice', 'hello there'))
        self.assertEqual(split_attributed_turn('not an attributed turn'), (None, 'not an attributed turn'))

    def test_sanitize_model_reply_removes_repeated_prefixes(self) -> None:
        self.assertEqual(sanitize_model_reply('Beatrice: alice: Beatrice: hello', 'Beatrice', 'alice'), 'hello')


class CommandProcessorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = RuntimeStore()
        self.secrets = SecretStore(secrets_file='tests/secrets.test.json')
        self.api_key_updates: list[str | None] = []
        self.processor = CommandProcessor(
            self.store,
            self.secrets,
            '!bot',
            'beans',
            'Beatrice',
            self.api_key_updates.append,
            lambda scope: f'reset:{scope or "all"}',
            lambda scope: f'status:{scope or "all"}',
            lambda: 'runtime saved',
            lambda: 'approval list',
            lambda approval_id, actor, is_private: f'approved:{approval_id}:{actor}:{is_private}',
            lambda approval_id, actor, is_private: f'rejected:{approval_id}:{actor}:{is_private}',
            lambda tokens, actor, is_private: [f"child:{','.join(tokens)}:{actor}:{is_private}"],
        )

    def test_set_temperature_clamps_to_upper_bound(self) -> None:
        lines = self.processor.handle(['set', 'temperature', '9', 'beans'])
        self.assertEqual(lines, ['temperature set to 2.00'])
        self.assertEqual(self.store.current().temperature, 2.0)

    def test_set_requires_password(self) -> None:
        lines = self.processor.handle(['set', 'stream', 'on'])
        self.assertEqual(lines, ['admin password required'])

    def test_reset_restores_defaults(self) -> None:
        self.processor.handle(['set', 'model', 'anthropic/claude-sonnet-4', 'beans'])
        lines = self.processor.handle(['reset', 'beans'])
        self.assertTrue(lines[0].startswith('runtime config reset. model=deepseek/deepseek-v3.2'))
        self.assertEqual(self.store.current().model, 'deepseek/deepseek-v3.2')

    def test_set_openrouter_key_updates_secret_store(self) -> None:
        lines = self.processor.handle(['set', 'openrouter_key', 'sk-test', 'beans'])
        self.assertEqual(lines, ['OpenRouter API key updated'])
        self.assertEqual(self.secrets.openrouter_api_key, 'sk-test')
        self.assertEqual(self.api_key_updates, ['sk-test'])

    def test_clear_openrouter_key_clears_secret_store(self) -> None:
        self.processor.handle(['set', 'openrouter_key', 'sk-test', 'beans'])
        lines = self.processor.handle(['clear', 'openrouter_key', 'beans'])
        self.assertEqual(lines, ['OpenRouter API key cleared'])
        self.assertIsNone(self.secrets.openrouter_api_key)
        self.assertEqual(self.api_key_updates[-1], None)

    def test_context_status_defaults_to_all(self) -> None:
        lines = self.processor.handle(['context'])
        self.assertEqual(lines, ['status:all'])

    def test_context_reset_requires_password(self) -> None:
        lines = self.processor.handle(['context', 'reset'])
        self.assertEqual(lines, ['admin password required'])

    def test_context_reset_with_scope(self) -> None:
        lines = self.processor.handle(['context', 'reset', '#ussycode', 'beans'])
        self.assertEqual(lines, ['reset:#ussycode'])

    def test_set_reply_interval_seconds(self) -> None:
        lines = self.processor.handle(['set', 'reply_interval_seconds', '60', 'beans'])
        self.assertEqual(lines, ['reply_interval_seconds set to 60'])
        self.assertEqual(self.store.current().reply_interval_seconds, 60.0)

    def test_save_runtime_requires_password(self) -> None:
        lines = self.processor.handle(['save', 'runtime'])
        self.assertEqual(lines, ['admin password required'])

    def test_save_runtime_returns_persisted_message(self) -> None:
        lines = self.processor.handle(['save', 'runtime', 'beans'])
        self.assertEqual(lines, ['runtime saved'])

    def test_help_mentions_private_tools_and_approval_flow(self) -> None:
        lines = self.processor.handle(['help'])

        self.assertTrue(any('IRC awareness' in line for line in lines))
        self.assertTrue(any('safe web fetch' in line for line in lines))
        self.assertTrue(any('typed memories' in line for line in lines))
        self.assertTrue(any('approval IDs' in line for line in lines))
        self.assertTrue(any('show models' in line for line in lines))

    def test_approvals_lists_pending_items(self) -> None:
        lines = self.processor.handle(['approvals'])
        self.assertEqual(lines, ['approval list'])

    def test_child_command_is_forwarded(self) -> None:
        lines = self.processor.handle(['child', 'list'], actor='admin', is_private=True)
        self.assertEqual(lines, ['child:list:admin:True'])

    def test_show_models_returns_route_mapping(self) -> None:
        self.store.current().set_models({'research': 'google/gemini-2.5-flash-lite', 'code': 'qwen/qwen3-coder-30b-a3b-instruct'})

        lines = self.processor.handle(['show', 'models'])

        self.assertEqual(
            lines,
            ['models default=deepseek/deepseek-v3.2 chat=deepseek/deepseek-v3.2 research=google/gemini-2.5-flash-lite code=qwen/qwen3-coder-30b-a3b-instruct'],
        )

    def test_approve_requires_password(self) -> None:
        lines = self.processor.handle(['approve', 'abc123'], actor='alice', is_private=True)
        self.assertEqual(lines, ['admin password required'])

    def test_approve_passes_actor_and_privacy(self) -> None:
        lines = self.processor.handle(['approve', 'abc123', 'beans'], actor='alice', is_private=True)
        self.assertEqual(lines, ['approved:abc123:alice:True'])

    def test_reject_passes_actor_and_privacy(self) -> None:
        lines = self.processor.handle(['reject', 'abc123', 'beans'], actor='alice', is_private=True)
        self.assertEqual(lines, ['rejected:abc123:alice:True'])


class BotCooldownTests(unittest.IsolatedAsyncioTestCase):
    async def test_reply_interval_waits_before_second_response(self) -> None:
        from bot.app import BeatriceBot, MessageContext
        from bot.config import BotSettings

        settings = BotSettings(openrouter_api_key='sk-test')
        bot = BeatriceBot(settings)
        bot.store.current().set_reply_interval_seconds(60)

        sleeps: list[float] = []
        now = [0.0]

        async def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)
            now[0] += seconds

        def fake_monotonic() -> float:
            return now[0]

        bot.openrouter.complete = AsyncMock(return_value='first reply')
        bot.irc.send_privmsg = AsyncMock()

        original_sleep = asyncio.sleep
        original_monotonic = time.monotonic
        try:
            asyncio.sleep = fake_sleep
            time.monotonic = fake_monotonic

            context = MessageContext(nick='user', target='#ussycode', is_private=False)
            await bot._answer_prompt(context, 'hello')
            await bot._answer_prompt(context, 'hello again')
        finally:
            asyncio.sleep = original_sleep
            time.monotonic = original_monotonic

        self.assertGreaterEqual(len(sleeps), 1)
        self.assertEqual(sleeps[0], 60)


class ChannelParticipationTests(unittest.TestCase):
    def test_channel_reply_gate_requires_invitation_and_message_gap(self) -> None:
        from bot.app import BeatriceBot, MessageContext
        from bot.config import BotSettings

        bot = BeatriceBot(BotSettings(openrouter_api_key='sk-test'))
        context = MessageContext(nick='alice', target='#ussycode', is_private=False)
        bot._record_channel_activity('#ussycode', 'alice', 'docker api timeout in logs')
        bot._record_channel_activity('#ussycode', 'bob', 'api timeout still happening')
        bot._mark_channel_pause('#ussycode')

        for _ in range(5):
            bot._note_public_message('#ussycode')
        self.assertFalse(bot._assess_channel_reply(context, 'anyone know why docker api broke?').should_reply)

        bot._next_channel_response_times['#ussycode'] = 0.0
        bot._note_public_message('#ussycode')
        self.assertFalse(bot._assess_channel_reply(context, 'anyone know why docker api broke?').should_reply)
        self.assertFalse(bot._assess_channel_reply(context, 'lol same').should_reply)

    def test_ambient_replies_disabled_by_default(self) -> None:
        from bot.app import BeatriceBot, MessageContext
        from bot.config import BotSettings

        bot = BeatriceBot(BotSettings(openrouter_api_key='sk-test'))
        context = MessageContext(nick='alice', target='#ussycode', is_private=False)
        for _ in range(10):
            bot._note_public_message('#ussycode')
            bot._record_channel_activity('#ussycode', 'alice', 'anyone know why docker api broke?')

        assessment = bot._assess_channel_reply(context, 'anyone know why docker api broke?')
        self.assertFalse(assessment.should_reply)

    def test_build_messages_include_nick_attribution(self) -> None:
        from bot.app import BeatriceBot, MessageContext
        from bot.config import BotSettings

        bot = BeatriceBot(BotSettings(openrouter_api_key='sk-test'))
        context = MessageContext(nick='alice', target='#ussycode', is_private=False)
        messages = bot._build_messages(context, 'what changed?', 'system prompt')

        self.assertEqual(messages[-1]['content'], 'alice: what changed?')
        self.assertIn('live multi-user IRC channel', messages[0]['content'])
        self.assertTrue(any(message['content'].startswith('IRC environment:') for message in messages))

    def test_history_overflow_creates_summary(self) -> None:
        from bot.app import BeatriceBot, MessageContext
        from bot.config import BotSettings

        bot = BeatriceBot(BotSettings(openrouter_api_key='sk-test', history_turns=2))
        context = MessageContext(nick='alice', target='#ussycode', is_private=False)

        for index in range(6):
            bot._append_history(context.history_scope, 'user', f'alice: message {index}')

        summary = bot._history_summary_for(context.history_scope)
        recent = list(bot._history_for(context.history_scope))

        self.assertTrue(summary)
        self.assertIn('alice: message 0', summary)
        self.assertEqual(len(recent), 4)

    def test_build_messages_include_summary_when_present(self) -> None:
        from bot.app import BeatriceBot, MessageContext
        from bot.config import BotSettings

        bot = BeatriceBot(BotSettings(openrouter_api_key='sk-test', history_turns=2))
        context = MessageContext(nick='alice', target='#ussycode', is_private=False)

        for index in range(6):
            bot._append_history(context.history_scope, 'user', f'alice: message {index}')

        messages = bot._build_messages(context, 'latest question?', 'system prompt')
        self.assertTrue(any(message['content'].startswith('Earlier conversation summary:') for message in messages))

    def test_topic_snapshot_added_to_messages(self) -> None:
        from bot.app import BeatriceBot, MessageContext
        from bot.config import BotSettings

        bot = BeatriceBot(BotSettings(openrouter_api_key='sk-test'))
        context = MessageContext(nick='alice', target='#ussycode', is_private=False)
        bot._record_channel_activity('#ussycode', 'alice', 'docker api timeout on irc bridge')
        bot._record_channel_activity('#ussycode', 'bob', 'api timeout still happening in docker logs')

        messages = bot._build_messages(context, 'anyone know what changed?', 'system prompt')
        self.assertTrue(any(message['content'].startswith('Recent channel topics:') for message in messages))

    def test_channel_ask_caps_public_max_tokens(self) -> None:
        from bot.app import BeatriceBot, MessageContext
        from bot.config import BotSettings

        bot = BeatriceBot(BotSettings(openrouter_api_key='sk-test'))
        captured_runtime = None

        async def fake_complete(runtime, _prompt, messages=None):
            nonlocal captured_runtime
            captured_runtime = runtime
            return 'short answer'

        bot.openrouter.complete = fake_complete
        bot.irc.send_privmsg = AsyncMock()

        async def run_test() -> None:
            context = MessageContext(nick='alice', target='#ussycode', is_private=False)
            await bot._answer_prompt(context, 'hello there')

        asyncio.run(run_test())
        self.assertIsNotNone(captured_runtime)
        self.assertLessEqual(captured_runtime.max_tokens, 160)

    def test_response_lines_strip_bot_and_user_prefixes(self) -> None:
        from bot.app import BeatriceBot, MessageContext
        from bot.config import BotSettings

        bot = BeatriceBot(BotSettings(openrouter_api_key='sk-test'))
        context = MessageContext(nick='alice', target='#ussycode', is_private=False)

        lines = bot._response_lines('Beatrice: alice: Beatrice: hello there', context, None)
        self.assertEqual(lines, ['hello there'])

    def test_summary_fragment_includes_topic_prefix(self) -> None:
        from bot.app import BeatriceBot
        from bot.config import BotSettings

        bot = BeatriceBot(BotSettings(openrouter_api_key='sk-test'))
        fragment = bot._summary_fragment({'content': 'alice: docker api timeout happened again'})
        self.assertIn('[timeout]', fragment)


class AgentToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_child_command_requires_admin_private_message(self) -> None:
        from bot.app import BeatriceBot
        from bot.config import BotSettings

        bot = BeatriceBot(BotSettings(openrouter_api_key='sk-test', admin_nicks=('mojo',)))

        denied = bot.commands.handle(['child', 'list'], actor='alice', is_private=True)

        self.assertEqual(denied, ['child bot control denied: admin private message required'])

    async def test_child_create_start_enable_disable_remove_flow(self) -> None:
        from bot.app import BeatriceBot
        from bot.config import BotSettings

        with tempfile.TemporaryDirectory() as temp_dir:
            settings = BotSettings(
                openrouter_api_key='sk-test',
                admin_nicks=('mojo',),
                child_bots_file=str(Path(temp_dir) / 'children.json'),
                child_state_file=str(Path(temp_dir) / 'children-state.json'),
                child_data_dir=str(Path(temp_dir) / 'children'),
                audit_log_file=str(Path(temp_dir) / 'audit.jsonl'),
            )
            bot = BeatriceBot(settings)

            start_calls: list[str] = []
            stop_calls: list[str] = []

            async def fake_start(child_id: str):
                start_calls.append(child_id)

            async def fake_stop(child_id: str):
                stop_calls.append(child_id)

            bot.child_manager.start_child = fake_start
            bot.child_manager.stop_child = fake_stop

            created = bot.commands.handle(
                [
                    'child',
                    'create',
                    'id=helper',
                    'nick=HelperBot',
                    'channels=#ussycode',
                    'prompt=You are a concise helper bot.',
                    'response_mode=ambient',
                ],
                actor='mojo',
                is_private=True,
            )
            await asyncio.sleep(0)

            self.assertIn('child helper created', created[0])
            self.assertEqual(start_calls, ['helper'])
            listed = bot.commands.handle(['child', 'list'], actor='mojo', is_private=True)
            self.assertIn('child helper nick=HelperBot', listed[0])
            self.assertIn('mode=ambient', listed[0])

            disabled = bot.commands.handle(['child', 'disable', 'helper'], actor='mojo', is_private=True)
            enabled = bot.commands.handle(['child', 'enable', 'helper'], actor='mojo', is_private=True)
            self.assertEqual(disabled, ['child helper disabled'])
            self.assertEqual(enabled, ['child helper enabled'])

            started = bot.commands.handle(['child', 'start', 'helper'], actor='mojo', is_private=True)
            stopped = bot.commands.handle(['child', 'stop', 'helper'], actor='mojo', is_private=True)
            await asyncio.sleep(0)
            self.assertEqual(started, ['starting child helper'])
            self.assertEqual(stopped, ['stopping child helper'])
            self.assertEqual(start_calls, ['helper', 'helper'])
            self.assertEqual(stop_calls, ['helper'])

            removed = bot.commands.handle(['child', 'remove', 'helper'], actor='mojo', is_private=True)
            self.assertEqual(removed, ['child helper removed'])

    async def test_child_update_changes_prompt_and_model(self) -> None:
        from bot.app import BeatriceBot
        from bot.config import BotSettings

        with tempfile.TemporaryDirectory() as temp_dir:
            settings = BotSettings(
                openrouter_api_key='sk-test',
                admin_nicks=('mojo',),
                child_bots_file=str(Path(temp_dir) / 'children.json'),
                child_state_file=str(Path(temp_dir) / 'children-state.json'),
                child_data_dir=str(Path(temp_dir) / 'children'),
            )
            bot = BeatriceBot(settings)
            bot.child_manager.create_child(
                child_id='helper',
                nick='HelperBot',
                channels=('#ussycode',),
                system_prompt='You are calm.',
            )

            updated = bot.commands.handle(
                [
                    'child',
                    'update',
                    'id=helper',
                    'model=google/gemini-2.5-flash-lite',
                    'prompt=You are very terse.',
                    'response_mode=addressed_only',
                ],
                actor='mojo',
                is_private=True,
            )

            self.assertEqual(updated, ['child helper updated'])
            spec = bot.child_manager.get_spec('helper')
            assert spec is not None
            self.assertEqual(spec.model, 'google/gemini-2.5-flash-lite')
            self.assertEqual(spec.system_prompt, 'You are very terse.')
            self.assertEqual(spec.response_mode, 'addressed_only')

    async def test_child_manager_persists_registry(self) -> None:
        from bot.child_bots import ChildBotManager
        from bot.audit import AuditLogger
        from bot.config import BotSettings

        with tempfile.TemporaryDirectory() as temp_dir:
            settings = BotSettings(
                openrouter_api_key='sk-test',
                child_bots_file=str(Path(temp_dir) / 'children.json'),
                child_state_file=str(Path(temp_dir) / 'children-state.json'),
                child_data_dir=str(Path(temp_dir) / 'children'),
                audit_log_file=str(Path(temp_dir) / 'audit.jsonl'),
            )
            manager = ChildBotManager(settings, AuditLogger(settings.audit_log_file))
            manager.create_child(
                child_id='greeter',
                nick='GreeterBot',
                channels=('#ussycode',),
                system_prompt='You greet people.',
                response_mode='ambient',
            )

            reloaded = ChildBotManager(settings, AuditLogger(settings.audit_log_file))
            spec = reloaded.get_spec('greeter')
            assert spec is not None
            self.assertEqual(spec.nick, 'GreeterBot')
            self.assertEqual(spec.channels, ('#ussycode',))
            self.assertEqual(spec.response_mode, 'ambient')

    async def test_private_admin_child_management_request_uses_tools(self) -> None:
        from bot.app import BeatriceBot, MessageContext
        from bot.config import BotSettings

        bot = BeatriceBot(BotSettings(openrouter_api_key='sk-test', admin_nicks=('mojo',)))
        context = MessageContext(nick='mojo', target='Beatrice', is_private=True)

        route = bot._classify_request(context, 'please make 5 helper chatbots for #ussycode', None)
        selected = bot._select_tool_subset(context, 'please make 5 helper chatbots for #ussycode', None, False)

        self.assertEqual(route.reason, 'child_management')
        self.assertTrue(route.use_tools)
        self.assertEqual(selected, {'list_child_bots', 'request_child_bot_changes'})

    async def test_request_child_bot_changes_queues_approval_with_variation(self) -> None:
        from bot.app import BeatriceBot, MessageContext
        from bot.config import BotSettings

        bot = BeatriceBot(BotSettings(openrouter_api_key='sk-test', admin_nicks=('mojo',)))
        context = MessageContext(nick='mojo', target='Beatrice', is_private=True)

        queued = await bot._execute_tool_call(
            SimpleNamespace(
                name='request_child_bot_changes',
                arguments={
                    'operations': [
                        {
                            'action': 'create',
                            'count': 5,
                            'id_prefix': 'helper',
                            'nick_prefix': 'HelperBot',
                            'channels': ['#ussycode'],
                            'purpose': 'Helpful helper chatbots for channel banter and quick guidance.',
                            'persona': 'same purpose, slightly different vibes',
                            'response_mode': 'ambient',
                            'start_after_create': False,
                        }
                    ]
                },
            ),
            context,
        )

        self.assertTrue(queued['approval_required'])
        pending = next(iter(bot._pending_approvals.values()))
        operations = pending.arguments['operations']
        self.assertEqual(len(operations), 5)
        prompts = {operation['system_prompt'] for operation in operations}
        self.assertEqual(len(prompts), 5)
        self.assertTrue(all(operation['response_mode'] == 'ambient' for operation in operations))
        self.assertIn('create=5', pending.summary)

    async def test_markup_tool_response_retries_instead_of_leaking(self) -> None:
        from bot.app import BeatriceBot, MessageContext
        from bot.config import BotSettings
        from bot.openrouter import ChatResponse

        bot = BeatriceBot(BotSettings(openrouter_api_key='sk-test', admin_nicks=('mojo',)))
        context = MessageContext(nick='mojo', target='#ussycode', is_private=False)
        messages = [{'role': 'system', 'content': 'hi'}, {'role': 'user', 'content': 'make 5 bots'}]
        first = ChatResponse(
            content='<function_calls><invoke name="request_child_bot_changes"><parameter name="operations">[]</parameter></invoke></function_calls>',
            tool_calls=(),
            assistant_message={'role': 'assistant', 'content': '<function_calls><invoke name="request_child_bot_changes"><parameter name="operations">[]</parameter></invoke></function_calls>'},
        )
        second = ChatResponse(
            content='done',
            tool_calls=(),
            assistant_message={'role': 'assistant', 'content': 'done'},
        )
        bot.openrouter.chat = AsyncMock(side_effect=[first, second])

        result = await bot._run_private_agent_loop(context, 'please spin up 5 random bots', bot.store.current(), messages, None, max_rounds=2)

        self.assertEqual(result, 'done')
        self.assertEqual(bot.openrouter.chat.await_count, 2)

    async def test_approval_applies_five_varied_child_bots(self) -> None:
        from bot.app import BeatriceBot, MessageContext
        from bot.config import BotSettings

        with tempfile.TemporaryDirectory() as temp_dir:
            settings = BotSettings(
                openrouter_api_key='sk-test',
                admin_nicks=('mojo',),
                child_bots_file=str(Path(temp_dir) / 'children.json'),
                child_state_file=str(Path(temp_dir) / 'children-state.json'),
                child_data_dir=str(Path(temp_dir) / 'children'),
                audit_log_file=str(Path(temp_dir) / 'audit.jsonl'),
            )
            bot = BeatriceBot(settings)
            context = MessageContext(nick='mojo', target='Beatrice', is_private=True)
            start_calls: list[str] = []

            async def fake_start(child_id: str):
                start_calls.append(child_id)

            bot.child_manager.start_child = fake_start

            queued = await bot._execute_tool_call(
                SimpleNamespace(
                    name='request_child_bot_changes',
                    arguments={
                        'operations': [
                            {
                                'action': 'create',
                                'count': 5,
                                'id_prefix': 'greeter',
                                'nick_prefix': 'Greeter',
                                'channels': ['#ussycode'],
                                'purpose': 'Greeter bots for welcoming people and light conversation.',
                                'response_mode': 'ambient',
                                'start_after_create': True,
                            }
                        ]
                    },
                ),
                context,
            )

            approved = bot.commands.handle(['approve', queued['approval_id'], 'beans'], actor='mojo', is_private=True)
            await asyncio.sleep(0)

            self.assertIn('approved', approved[0])
            specs = bot.child_manager.list_specs()
            self.assertEqual(len(specs), 5)
            self.assertEqual(len({spec.system_prompt for spec in specs}), 5)
            self.assertTrue(all(spec.response_mode == 'ambient' for spec in specs))
            self.assertEqual(len(start_calls), 5)

    async def test_managed_child_bot_ambient_mode_enables_chat_channels(self) -> None:
        import os

        from bot.child_runner import ManagedChildBot
        from bot.config import BotSettings

        settings = BotSettings(openrouter_api_key='sk-test', irc_channels=('#ussycode',))
        previous = os.environ.get('BOT_CHILD_RESPONSE_MODE')
        os.environ['BOT_CHILD_RESPONSE_MODE'] = 'ambient'
        try:
            bot = ManagedChildBot(settings)
            await bot._on_connected('irc.example.net')
        finally:
            if previous is None:
                os.environ.pop('BOT_CHILD_RESPONSE_MODE', None)
            else:
                os.environ['BOT_CHILD_RESPONSE_MODE'] = previous

        self.assertIn('#ussycode', bot._chat_channels)

    async def test_managed_child_bot_addressed_mode_requires_invitation(self) -> None:
        import os

        from bot.child_runner import ManagedChildBot
        from bot.app import MessageContext
        from bot.config import BotSettings

        settings = BotSettings(openrouter_api_key='sk-test', irc_channels=('#ussycode',))
        previous = os.environ.get('BOT_CHILD_RESPONSE_MODE')
        os.environ['BOT_CHILD_RESPONSE_MODE'] = 'addressed_only'
        try:
            bot = ManagedChildBot(settings)
        finally:
            if previous is None:
                os.environ.pop('BOT_CHILD_RESPONSE_MODE', None)
            else:
                os.environ['BOT_CHILD_RESPONSE_MODE'] = previous
        bot._chat_channels.add('#ussycode')
        bot._channel_human_messages_since_reply['#ussycode'] = 10
        context = MessageContext(nick='alice', target='#ussycode', is_private=False)

        assessment = bot._assess_channel_reply(context, 'docker timeout still happening')

        self.assertFalse(assessment.should_reply)


class OpenRouterToolParsingTests(unittest.IsolatedAsyncioTestCase):
    def test_extract_markup_tool_calls_parses_function_markup(self) -> None:
        from bot.openrouter import OpenRouterClient

        message = {
            'role': 'assistant',
            'content': '<function_calls><invoke name="request_child_bot_changes"><parameter name="operations">[{"action":"create","count":5}]</parameter></invoke></function_calls>',
        }

        tool_calls = OpenRouterClient._extract_tool_calls(message)

        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].name, 'request_child_bot_changes')
        self.assertEqual(tool_calls[0].arguments['operations'][0]['action'], 'create')
    async def test_private_agent_tools_include_approval_based_runtime_mutation(self) -> None:
        from bot.app import BeatriceBot, MessageContext
        from bot.config import BotSettings

        bot = BeatriceBot(BotSettings(openrouter_api_key='sk-test'))
        context = MessageContext(nick='alice', target='Beatrice', is_private=True)

        prompt, has_password = bot._sanitize_prompt_for_model('please change your temperature to 0.4 beans', context)

        self.assertTrue(has_password)
        self.assertNotIn('beans', prompt.lower())
        names = [tool.name for tool in bot._tool_definitions()]
        self.assertIn('set_runtime_config', names)
        self.assertIn('persist_runtime_config', names)

    async def test_admin_public_messages_use_private_capability_path(self) -> None:
        from bot.app import BeatriceBot, MessageContext
        from bot.config import BotSettings

        bot = BeatriceBot(BotSettings(openrouter_api_key='sk-test', admin_nicks=('mojo',)))
        public_admin = MessageContext(nick='mojo', target='#ussycode', is_private=False)
        public_non_admin = MessageContext(nick='alice', target='#ussycode', is_private=False)

        self.assertTrue(bot._allows_private_capabilities(public_admin))
        self.assertFalse(bot._allows_private_capabilities(public_non_admin))

    async def test_admin_public_actor_can_approve(self) -> None:
        from bot.app import BeatriceBot, MessageContext
        from bot.config import BotSettings

        bot = BeatriceBot(BotSettings(openrouter_api_key='sk-test', admin_nicks=('mojo',)))
        context = MessageContext(nick='alice', target='Beatrice', is_private=True)
        queued = await bot._execute_tool_call(
            SimpleNamespace(name='set_runtime_config', arguments={'temperature': 0.4}),
            context,
        )

        approved = bot.commands.handle(['approve', queued['approval_id'], 'beans'], actor='mojo', is_private=False)

        self.assertIn('approved', approved[0])
        self.assertEqual(bot.store.current().temperature, 0.4)

    def test_extract_github_scope_parses_owner_and_repo(self) -> None:
        from bot.app import BeatriceBot
        from bot.config import BotSettings

        bot = BeatriceBot(BotSettings(openrouter_api_key='sk-test', admin_nicks=('mojo',)))

        owner_scope = bot._extract_github_scope('search github/mojomast for ussyverse')
        repo_scope = bot._extract_github_scope('check github/mojomast/ussynet please')

        self.assertIsNotNone(owner_scope)
        self.assertEqual(owner_scope.owner, 'mojomast')
        self.assertIsNone(owner_scope.repo)
        self.assertIsNotNone(repo_scope)
        self.assertEqual(repo_scope.owner, 'mojomast')
        self.assertEqual(repo_scope.repo, 'ussynet')

    async def test_github_scope_blocks_repo_read_when_repo_not_explicit(self) -> None:
        from bot.app import BeatriceBot, MessageContext
        from bot.config import BotSettings

        bot = BeatriceBot(BotSettings(openrouter_api_key='sk-test', admin_nicks=('mojo',)))
        context = MessageContext(nick='mojo', target='#ussycode', is_private=False)
        scope = bot._extract_github_scope('search github/mojomast for ussyverse')

        blocked = await bot._execute_tool_call(
            SimpleNamespace(name='github_read_repository_readme', arguments={'owner': 'mojomast', 'repo': 'ussycode'}),
            context,
            scope,
        )

        self.assertFalse(blocked['ok'])
        self.assertIn('repo must be explicit', blocked['error'])

    async def test_github_list_owner_repositories_respects_owner_scope(self) -> None:
        from bot.app import BeatriceBot, MessageContext
        from bot.config import BotSettings

        bot = BeatriceBot(BotSettings(openrouter_api_key='sk-test', admin_nicks=('mojo',)))
        context = MessageContext(nick='mojo', target='#ussycode', is_private=False)
        scope = bot._extract_github_scope('projects from github/mojomast')

        async def fake_list(owner: str, limit: int = 8):
            return {'owner': owner, 'repositories': [{'full_name': 'mojomast/ussynet'}]}

        bot.github.list_owner_repositories = fake_list

        result = await bot._execute_tool_call(
            SimpleNamespace(name='github_list_owner_repositories', arguments={'owner': 'mojomast', 'limit': 5}),
            context,
            scope,
        )

        self.assertTrue(result['ok'])
        self.assertEqual(result['result']['repositories'][0]['full_name'], 'mojomast/ussynet')

    def test_requires_web_lookup_for_current_events(self) -> None:
        from bot.app import BeatriceBot
        from bot.config import BotSettings

        bot = BeatriceBot(BotSettings(openrouter_api_key='sk-test', admin_nicks=('mojo',)))

        self.assertTrue(bot._requires_web_lookup('can you look for current events', None))
        self.assertTrue(bot._requires_web_lookup('please websearch this for me', None))
        self.assertFalse(bot._requires_web_lookup('tell me about github/mojomast/ussynet', bot._extract_github_scope('github/mojomast/ussynet')))

    def test_extract_domain_hint_finds_explicit_site(self) -> None:
        from bot.app import BeatriceBot
        from bot.config import BotSettings

        bot = BeatriceBot(BotSettings(openrouter_api_key='sk-test', admin_nicks=('mojo',)))

        self.assertEqual(bot._extract_domain_hint('its ussyverse from ussy.host'), 'ussy.host')
        self.assertIsNone(bot._extract_domain_hint('check github.com/mojomast/ussynet'))

    def test_prefer_direct_web_fetch_url_for_github_trending(self) -> None:
        from bot.app import BeatriceBot
        from bot.config import BotSettings

        bot = BeatriceBot(BotSettings(openrouter_api_key='sk-test', admin_nicks=('mojo',)))

        self.assertEqual(bot._prefer_direct_web_fetch_url('tell me whats hot on github today'), 'https://github.com/trending')
        self.assertEqual(bot._prefer_direct_web_fetch_url('its ussyverse from ussy.host'), 'https://ussy.host')

    def test_forced_first_tool_choice_prefers_direct_fetch_or_search(self) -> None:
        from bot.app import BeatriceBot
        from bot.config import BotSettings

        bot = BeatriceBot(BotSettings(openrouter_api_key='sk-test', admin_nicks=('mojo',)))

        self.assertEqual(
            bot._forced_first_tool_choice(True, 'https://github.com/trending', None, None),
            {'type': 'function', 'function': {'name': 'web_fetch'}},
        )
        self.assertEqual(
            bot._forced_first_tool_choice(True, None, None, None),
            {'type': 'function', 'function': {'name': 'web_search'}},
        )

    def test_forced_first_tool_choice_prefers_child_change_tool(self) -> None:
        from bot.app import BeatriceBot
        from bot.config import BotSettings

        bot = BeatriceBot(BotSettings(openrouter_api_key='sk-test', admin_nicks=('mojo',)))

        self.assertEqual(
            bot._forced_first_tool_choice(
                False,
                None,
                None,
                'please spin up 5 random bots with very different personalities',
                frozenset({'list_child_bots', 'request_child_bot_changes'}),
            ),
            {'type': 'function', 'function': {'name': 'request_child_bot_changes'}},
        )

    def test_classify_request_routes_chat_research_and_code(self) -> None:
        from bot.app import BeatriceBot, MessageContext
        from bot.config import BotSettings

        bot = BeatriceBot(BotSettings(openrouter_api_key='sk-test', admin_nicks=('mojo',)))

        public_context = MessageContext(nick='alice', target='#ussycode', is_private=False)
        private_context = MessageContext(nick='alice', target='Beatrice', is_private=True)
        admin_public = MessageContext(nick='mojo', target='#ussycode', is_private=False)

        self.assertEqual(bot._classify_request(public_context, 'hello there', None).model_route, 'chat')
        self.assertEqual(bot._classify_request(private_context, 'hello there', None).model_route, 'chat')
        self.assertEqual(bot._classify_request(admin_public, 'please research current events', None).model_route, 'research')
        self.assertTrue(bot._classify_request(admin_public, 'please research current events', None).use_tools)
        self.assertEqual(bot._classify_request(public_context, 'Traceback in bot/app.py line 10', None).model_route, 'code')
        github_scope = bot._extract_github_scope('check github/mojomast/ussynet')
        self.assertEqual(bot._classify_request(admin_public, 'check github/mojomast/ussynet', github_scope).model_route, 'code')
        self.assertTrue(bot._classify_request(admin_public, 'check github/mojomast/ussynet', github_scope).use_tools)

    def test_force_admin_public_tools_for_research_verbs(self) -> None:
        from bot.app import BeatriceBot, MessageContext
        from bot.config import BotSettings

        bot = BeatriceBot(BotSettings(openrouter_api_key='sk-test', admin_nicks=('mojo',)))
        admin_public = MessageContext(nick='mojo', target='#ussycode', is_private=False)
        public_user = MessageContext(nick='alice', target='#ussycode', is_private=False)

        self.assertTrue(bot._should_force_admin_public_tools(admin_public, 'research the ussyverse for me'))
        self.assertTrue(bot._should_force_admin_public_tools(admin_public, 'find out about kyle durepos'))
        self.assertFalse(bot._should_force_admin_public_tools(public_user, 'research the ussyverse for me'))

    def test_select_tool_subset_prefers_minimal_web_research_tools(self) -> None:
        from bot.app import BeatriceBot, MessageContext, WEB_RESEARCH_TOOLS
        from bot.config import BotSettings

        bot = BeatriceBot(BotSettings(openrouter_api_key='sk-test', admin_nicks=('mojo',)))
        admin_public = MessageContext(nick='mojo', target='#ussycode', is_private=False)

        selected = bot._select_tool_subset(admin_public, 'research current events', None, True)

        self.assertEqual(selected, WEB_RESEARCH_TOOLS)

    def test_tool_signature_normalizes_duplicate_web_calls(self) -> None:
        from bot.app import BeatriceBot
        from bot.config import BotSettings
        from bot.openrouter import ToolCall

        bot = BeatriceBot(BotSettings(openrouter_api_key='sk-test', admin_nicks=('mojo',)))
        call_a = ToolCall(id='1', name='web_search', arguments={'query': ' current   events ', 'limit': 3})
        call_b = ToolCall(id='2', name='web_search', arguments={'limit': 3, 'query': 'current events'})

        self.assertEqual(bot._tool_signature(call_a), bot._tool_signature(call_b))

    async def test_execute_tool_call_lists_repository_directory(self) -> None:
        from bot.app import BeatriceBot, MessageContext
        from bot.config import BotSettings

        bot = BeatriceBot(BotSettings(openrouter_api_key='sk-test', admin_nicks=('mojo',)))
        context = MessageContext(nick='mojo', target='#ussycode', is_private=False)
        scope = bot._extract_github_scope('check github/mojomast/ussynet')

        async def fake_list_directory(owner: str, repo: str, path=None, ref=None):
            return {'owner': owner, 'repo': repo, 'path': path or '', 'ref': ref, 'entries': [{'name': 'bot', 'type': 'dir'}]}

        bot.github.list_repository_directory = fake_list_directory

        result = await bot._execute_tool_call(
            SimpleNamespace(name='github_list_repository_directory', arguments={'owner': 'mojomast', 'repo': 'ussynet', 'path': ''}),
            context,
            scope,
        )

        self.assertTrue(result['ok'])
        self.assertEqual(result['result']['entries'][0]['name'], 'bot')

    async def test_research_timeout_retries_with_trimmed_tools(self) -> None:
        from bot.app import BeatriceBot, MessageContext
        from bot.config import BotSettings
        from bot.openrouter import OpenRouterTimeout

        bot = BeatriceBot(BotSettings(openrouter_api_key='sk-test', admin_nicks=('mojo',)))
        context = MessageContext(nick='mojo', target='#ussycode', is_private=False)

        first = AsyncMock(side_effect=OpenRouterTimeout('timeout'))
        second = AsyncMock(return_value='retried answer')
        bot._run_private_agent_loop = AsyncMock(side_effect=[OpenRouterTimeout('timeout'), 'retried answer'])

        await bot._answer_prompt_locked(context, 'research kyle durepos for me please', None, '#ussycode', 'research kyle durepos for me please')

        self.assertEqual(bot._run_private_agent_loop.await_count, 2)
        retry_call = bot._run_private_agent_loop.await_args_list[1]
        self.assertEqual(retry_call.kwargs['max_rounds'], 2)
        tool_names = {tool.name for tool in retry_call.kwargs['tools_override']}
        self.assertEqual(tool_names, {'web_search', 'web_fetch'})

    async def test_execute_tool_call_runs_web_search(self) -> None:
        from bot.app import BeatriceBot, MessageContext
        from bot.config import BotSettings

        bot = BeatriceBot(BotSettings(openrouter_api_key='sk-test', admin_nicks=('mojo',)))
        context = MessageContext(nick='mojo', target='#ussycode', is_private=False)

        async def fake_search(query: str, limit: int = 5):
            return {'query': query, 'results': [{'title': 'Example', 'url': 'https://example.com'}]}

        bot.web.search_tool_result = fake_search

        result = await bot._execute_tool_call(
            SimpleNamespace(name='web_search', arguments={'query': 'current events', 'limit': 3}),
            context,
        )

        self.assertTrue(result['ok'])
        self.assertEqual(result['result']['results'][0]['title'], 'Example')

    async def test_execute_tool_call_derives_missing_web_search_query_from_history(self) -> None:
        from bot.app import BeatriceBot, MessageContext
        from bot.config import BotSettings

        bot = BeatriceBot(BotSettings(openrouter_api_key='sk-test', admin_nicks=('mojo',)))
        context = MessageContext(nick='mojo', target='#ussycode', is_private=False)
        bot._append_history(context.history_scope, 'user', bot._format_user_turn(context, 'research the ussyverse'))

        async def fake_search(query: str, limit: int = 5):
            return {'query': query, 'results': [{'title': 'Example', 'url': 'https://example.com'}]}

        bot.web.search_tool_result = fake_search

        result = await bot._execute_tool_call(
            SimpleNamespace(name='web_search', arguments={'query': '', 'limit': 3}),
            context,
        )

        self.assertTrue(result['ok'])
        self.assertEqual(result['result']['query'], 'research the ussyverse')

    async def test_tool_budget_blocks_second_github_discovery_call(self) -> None:
        from bot.app import BeatriceBot
        from bot.config import BotSettings
        from bot.openrouter import ToolCall

        bot = BeatriceBot(BotSettings(openrouter_api_key='sk-test', admin_nicks=('mojo',)))
        budget = bot.__class__.__dict__['_reserve_tool_budget']
        state_cls = __import__('bot.app', fromlist=['ToolBudgetState']).ToolBudgetState
        state = state_cls(total_limit=8, category_limits={'github_discovery': 1, 'other': 2})

        first = budget(bot, state, ToolCall(id='1', name='github_search_owner_repositories', arguments={'owner': 'mojomast', 'query': 'ussy'}))
        second = budget(bot, state, ToolCall(id='2', name='github_list_owner_repositories', arguments={'owner': 'mojomast'}))

        self.assertIsNone(first)
        self.assertIsNotNone(second)
        assert second is not None
        self.assertEqual(second['error_type'], 'tool_budget_exceeded')
        self.assertEqual(second['category'], 'github_discovery')

    def test_summarize_tool_arguments_redacts_large_or_sensitive_fields(self) -> None:
        from bot.app import BeatriceBot
        from bot.config import BotSettings

        bot = BeatriceBot(BotSettings(openrouter_api_key='sk-test', admin_nicks=('mojo',)))

        summary = bot._summarize_tool_arguments(
            'set_runtime_config',
            {
                'system_prompt': 'x' * 50,
                'temperature': 0.4,
                'api_key': 'secret',
                'url': 'https://api.github.com/search/repositories?q=ussy#frag',
            },
        )

        self.assertEqual(summary['system_prompt_len'], 50)
        self.assertEqual(summary['temperature'], 0.4)
        self.assertEqual(summary['api_key'], '<redacted>')
        self.assertEqual(summary['url'], 'https://api.github.com/search/repositories')

    async def test_private_agent_tools_expose_environment_web_memory_and_whois(self) -> None:
        from bot.app import BeatriceBot, MessageContext
        from bot.config import BotSettings

        bot = BeatriceBot(BotSettings(openrouter_api_key='sk-test'))
        context = MessageContext(nick='alice', target='Beatrice', is_private=True)
        bot.irc.server_name = 'irc.example.net'
        await bot.irc._process_line(':Beatrice!bot@example JOIN :#ussycode')
        await bot.irc._process_line(':irc.example.net 353 Beatrice = #ussycode :@Beatrice alice bob')
        await bot.irc._process_line(':irc.example.net 366 Beatrice #ussycode :End of /NAMES list')
        await bot.irc._process_line(':irc.example.net 332 Beatrice #ussycode :Deploy window')
        await bot.irc._process_line(':alice!user@example NICK :alice_')

        environment = await bot._execute_tool_call(SimpleNamespace(name='get_environment_info', arguments={}), context)
        self.assertTrue(environment['ok'])
        self.assertEqual(environment['server'], 'irc.example.net')
        self.assertEqual(environment['joined_channels'], ['#ussycode'])
        self.assertEqual(environment['channel_topics'], {'#ussycode': 'Deploy window'})
        self.assertEqual(environment['channel_users']['#ussycode'], ['alice_', 'Beatrice', 'bob'])
        self.assertEqual(environment['recent_nick_changes'][0]['old_nick'], 'alice')
        self.assertEqual(environment['recent_nick_changes'][0]['new_nick'], 'alice_')

        names = [tool.name for tool in bot._tool_definitions()]
        self.assertIn('web_fetch', names)
        self.assertIn('remember_memory', names)
        self.assertIn('search_memories', names)
        self.assertIn('irc_whois', names)
        self.assertIn('get_subject_profile', names)
        self.assertIn('update_subject_profile', names)

    async def test_private_profile_prompt_uses_auto_profile_updates(self) -> None:
        from pathlib import Path
        import tempfile

        from bot.app import BeatriceBot, MessageContext
        from bot.config import BotSettings

        with tempfile.TemporaryDirectory() as temp_dir:
            settings = BotSettings(
                openrouter_api_key='sk-test',
                memory_db_file=str(Path(temp_dir) / 'memory.sqlite3'),
            )
            bot = BeatriceBot(settings)
            await bot.memory.initialize()
            context = MessageContext(nick='alice', target='Beatrice', is_private=True)

            await bot._auto_update_profile_from_message(context.history_scope, 'alice', "I'm a systems engineer. I prefer concise answers.")
            prompt = bot._private_profile_prompt(context)

            self.assertIsNotNone(prompt)
            assert prompt is not None
            self.assertIn('systems engineer', prompt)
            self.assertIn('concise answers', prompt)

    def test_settings_default_prompt_is_operator_friendly(self) -> None:
        from bot.config import BotSettings

        settings = BotSettings.from_env()

        self.assertIn('Treat Mojo as an authorized operator', settings.runtime_defaults.system_prompt)
        self.assertIn('Never drift into romance', settings.runtime_defaults.system_prompt)

    async def test_channel_prompt_context_includes_member_profiles_and_topics(self) -> None:
        from pathlib import Path
        import tempfile

        from bot.app import BeatriceBot, MessageContext
        from bot.config import BotSettings

        with tempfile.TemporaryDirectory() as temp_dir:
            settings = BotSettings(
                openrouter_api_key='sk-test',
                memory_db_file=str(Path(temp_dir) / 'memory.sqlite3'),
            )
            bot = BeatriceBot(settings)
            await bot.memory.initialize()
            context = MessageContext(nick='alice', target='#ussycode', is_private=False)

            await bot.irc._process_line(':Beatrice!bot@example JOIN :#ussycode')
            await bot.irc._process_line(':irc.example.net 353 Beatrice = #ussycode :@Beatrice alice bob')
            await bot.irc._process_line(':irc.example.net 366 Beatrice #ussycode :End of /NAMES list')
            await bot.irc._process_line(':irc.example.net 332 Beatrice #ussycode :Deploy window')
            bot._record_channel_activity('#ussycode', 'alice', 'docker timeout on bridge')
            await bot._auto_update_profile_from_message('#ussycode', 'alice', 'I use linux and I prefer concise answers')

            snippets = bot._channel_prompt_context(context)

            self.assertEqual(len(snippets), 2)
            self.assertIn('Deploy window', snippets[0])
            self.assertIn('alice', snippets[1])
            self.assertIn('concise answers', snippets[1])

    async def test_execute_tool_call_can_store_and_search_memory(self) -> None:
        from pathlib import Path
        import tempfile

        from bot.app import BeatriceBot, MessageContext
        from bot.config import BotSettings

        with tempfile.TemporaryDirectory() as temp_dir:
            settings = BotSettings(
                openrouter_api_key='sk-test',
                memory_db_file=str(Path(temp_dir) / 'memory.sqlite3'),
            )
            bot = BeatriceBot(settings)
            await bot.memory.initialize()
            context = MessageContext(nick='alice', target='Beatrice', is_private=True)

            remember = await bot._execute_tool_call(
                SimpleNamespace(name='remember_memory', arguments={'content': 'bob likes rust'}),
                context,
            )
            search = await bot._execute_tool_call(
                SimpleNamespace(name='search_memories', arguments={'query': 'rust', 'limit': 3}),
                context,
            )

            self.assertTrue(remember['ok'])
            self.assertTrue(search['ok'])
            self.assertEqual(search['count'], 1)
            self.assertIn('bob likes rust', search['memories'][0]['content'])

    async def test_execute_tool_call_can_store_fact_and_profile(self) -> None:
        from pathlib import Path
        import tempfile

        from bot.app import BeatriceBot, MessageContext
        from bot.config import BotSettings

        with tempfile.TemporaryDirectory() as temp_dir:
            settings = BotSettings(
                openrouter_api_key='sk-test',
                memory_db_file=str(Path(temp_dir) / 'memory.sqlite3'),
            )
            bot = BeatriceBot(settings)
            await bot.memory.initialize()
            context = MessageContext(nick='alice', target='Beatrice', is_private=True)

            remembered = await bot._execute_tool_call(
                SimpleNamespace(
                    name='remember_memory',
                    arguments={'content': 'bob likes rust', 'kind': 'fact', 'subject': 'bob'},
                ),
                context,
            )
            profile = await bot._execute_tool_call(
                SimpleNamespace(
                    name='update_subject_profile',
                    arguments={'subject': 'bob', 'profile': 'bob likes rust and concise answers'},
                ),
                context,
            )
            profile_lookup = await bot._execute_tool_call(
                SimpleNamespace(name='get_subject_profile', arguments={'subject': 'bob'}),
                context,
            )
            search = await bot._execute_tool_call(
                SimpleNamespace(name='search_memories', arguments={'subject': 'bob', 'kind': 'fact'}),
                context,
            )

            self.assertTrue(remembered['ok'])
            self.assertEqual(remembered['memory']['kind'], 'fact')
            self.assertEqual(remembered['memory']['subject'], 'bob')
            self.assertTrue(profile['ok'])
            self.assertTrue(profile_lookup['ok'])
            self.assertEqual(profile_lookup['profile'], 'bob likes rust and concise answers')
            self.assertEqual(search['count'], 1)
            self.assertEqual(search['memories'][0]['subject'], 'bob')

    async def test_execute_tool_call_queues_runtime_update_for_admin_approval(self) -> None:
        from bot.app import BeatriceBot, MessageContext
        from bot.config import BotSettings

        bot = BeatriceBot(BotSettings(openrouter_api_key='sk-test'))
        context = MessageContext(nick='alice', target='Beatrice', is_private=True)

        queued = await bot._execute_tool_call(
            SimpleNamespace(name='set_runtime_config', arguments={'temperature': 0.4}),
            context,
        )

        self.assertFalse(queued['ok'])
        self.assertTrue(queued['approval_required'])
        self.assertIn('approval_id', queued)
        self.assertEqual(bot.store.current().temperature, 0.7)

    async def test_execute_tool_call_queues_persist_runtime_for_admin_approval(self) -> None:
        from bot.app import BeatriceBot, MessageContext
        from bot.config import BotSettings

        bot = BeatriceBot(BotSettings(openrouter_api_key='sk-test'))
        context = MessageContext(nick='alice', target='Beatrice', is_private=True)

        queued = await bot._execute_tool_call(
            SimpleNamespace(name='persist_runtime_config', arguments={}),
            context,
        )

        self.assertFalse(queued['ok'])
        self.assertTrue(queued['approval_required'])

    async def test_approval_commands_require_admin_identity(self) -> None:
        from bot.app import BeatriceBot, MessageContext
        from bot.config import BotSettings

        settings = BotSettings(openrouter_api_key='sk-test', admin_nicks=('admin',))
        bot = BeatriceBot(settings)
        context = MessageContext(nick='alice', target='Beatrice', is_private=True)

        queued = await bot._execute_tool_call(
            SimpleNamespace(name='set_runtime_config', arguments={'temperature': 0.4}),
            context,
        )
        approval_id = queued['approval_id']

        denied_user = bot.commands.handle(['approve', approval_id, 'beans'], actor='mallory', is_private=True)

        self.assertEqual(denied_user, ['approval denied: admin private message required'])

    async def test_admin_can_approve_runtime_change_and_persist_it(self) -> None:
        from pathlib import Path
        import tempfile

        from bot.app import BeatriceBot, MessageContext
        from bot.config import BotSettings

        with tempfile.TemporaryDirectory() as temp_dir:
            runtime_path = Path(temp_dir) / 'runtime.json'
            memory_path = Path(temp_dir) / 'memory.sqlite3'
            settings = BotSettings(
                openrouter_api_key='sk-test',
                runtime_file=str(runtime_path),
                memory_db_file=str(memory_path),
                admin_nicks=('admin',),
            )
            bot = BeatriceBot(settings)
            context = MessageContext(nick='alice', target='Beatrice', is_private=True)

            queued_change = await bot._execute_tool_call(
                SimpleNamespace(name='set_runtime_config', arguments={'temperature': 0.4, 'stream': True}),
                context,
            )
            queued_persist = await bot._execute_tool_call(
                SimpleNamespace(name='persist_runtime_config', arguments={}),
                context,
            )

            approval_change = bot.commands.handle(['approve', queued_change['approval_id'], 'beans'], actor='admin', is_private=True)
            approval_persist = bot.commands.handle(['approve', queued_persist['approval_id'], 'beans'], actor='admin', is_private=True)

            self.assertIn('approved', approval_change[0])
            self.assertEqual(bot.store.current().temperature, 0.4)
            self.assertTrue(bot.store.current().stream)
            self.assertIn('runtime config saved', approval_persist[0])
            self.assertTrue(runtime_path.exists())

    async def test_admin_can_reject_pending_action(self) -> None:
        from bot.app import BeatriceBot, MessageContext
        from bot.config import BotSettings

        settings = BotSettings(openrouter_api_key='sk-test', admin_nicks=('admin',))
        bot = BeatriceBot(settings)
        context = MessageContext(nick='alice', target='Beatrice', is_private=True)

        queued = await bot._execute_tool_call(
            SimpleNamespace(name='set_runtime_config', arguments={'temperature': 0.4}),
            context,
        )
        approval_id = queued['approval_id']
        rejected = bot.commands.handle(['reject', approval_id, 'beans'], actor='admin', is_private=True)

        self.assertIn(f'rejected approval {approval_id}', rejected[0])
        self.assertEqual(bot.store.current().temperature, 0.7)

    async def test_approvals_command_lists_pending_actions(self) -> None:
        from bot.app import BeatriceBot, MessageContext
        from bot.config import BotSettings

        bot = BeatriceBot(BotSettings(openrouter_api_key='sk-test'))
        context = MessageContext(nick='alice', target='Beatrice', is_private=True)
        queued = await bot._execute_tool_call(
            SimpleNamespace(name='set_runtime_config', arguments={'temperature': 0.4}),
            context,
        )

        listed = bot.commands.handle(['approvals'])

        self.assertIn(queued['approval_id'], listed[0])

    async def test_approval_flow_writes_audit_log(self) -> None:
        from pathlib import Path
        import tempfile

        from bot.app import BeatriceBot, MessageContext
        from bot.config import BotSettings

        with tempfile.TemporaryDirectory() as temp_dir:
            runtime_path = Path(temp_dir) / 'runtime.json'
            memory_path = Path(temp_dir) / 'memory.sqlite3'
            audit_path = Path(temp_dir) / 'audit.jsonl'
            settings = BotSettings(
                openrouter_api_key='sk-test',
                runtime_file=str(runtime_path),
                memory_db_file=str(memory_path),
                audit_log_file=str(audit_path),
                admin_nicks=('admin',),
            )
            bot = BeatriceBot(settings)
            context = MessageContext(nick='alice', target='Beatrice', is_private=True)

            queued = await bot._execute_tool_call(
                SimpleNamespace(name='set_runtime_config', arguments={'temperature': 0.4}),
                context,
            )
            bot.commands.handle(['approve', queued['approval_id'], 'beans'], actor='admin', is_private=True)

            contents = audit_path.read_text(encoding='utf-8')

            self.assertIn('approval_requested', contents)
            self.assertIn('approval_granted', contents)
            self.assertIn('dangerous_action_result', contents)
