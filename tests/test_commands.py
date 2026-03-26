import asyncio
from types import SimpleNamespace
import time
import unittest
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

    def test_approvals_lists_pending_items(self) -> None:
        lines = self.processor.handle(['approvals'])
        self.assertEqual(lines, ['approval list'])

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

    async def test_approval_commands_require_private_admin_actor(self) -> None:
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

        denied_public = bot.commands.handle(['approve', approval_id, 'beans'], actor='admin', is_private=False)
        denied_user = bot.commands.handle(['approve', approval_id, 'beans'], actor='mallory', is_private=True)

        self.assertEqual(denied_public, ['approval denied: admin private message required'])
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
