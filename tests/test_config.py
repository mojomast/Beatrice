import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bot.config import BotSettings, RuntimeDefaults, RuntimeStore, SecretStore, default_irc_user, parse_channels, parse_csv_values


class ConfigTests(unittest.TestCase):
    def tearDown(self) -> None:
        test_secrets = Path('tests/secrets.test.json')
        if test_secrets.exists():
            test_secrets.unlink()

    def test_parse_channels_splits_commas(self) -> None:
        self.assertEqual(parse_channels('#a, #b ,#c'), ('#a', '#b', '#c'))

    def test_parse_csv_values_splits_commas(self) -> None:
        self.assertEqual(parse_csv_values('alice, bob ,carol'), ('alice', 'bob', 'carol'))

    def test_default_irc_user_sanitizes_nick(self) -> None:
        self.assertEqual(default_irc_user('Bea!trice'), 'beatrice')

    def test_runtime_store_snapshot_is_isolated(self) -> None:
        store = RuntimeStore(RuntimeDefaults(model='deepseek/deepseek-v3.2'))
        snapshot = store.snapshot()
        snapshot.model = 'other/model'
        self.assertEqual(store.current().model, 'deepseek/deepseek-v3.2')

    def test_settings_load_defaults_for_beatrice(self) -> None:
        env = {
            'IRC_CHANNEL': '#ussycode,#llm',
        }
        with patch.dict(os.environ, env, clear=True):
            settings = BotSettings.from_env()

        self.assertIsNone(settings.openrouter_api_key)
        self.assertEqual(settings.irc_server, 'irc.ussyco.de')
        self.assertEqual(settings.irc_nick, 'Beatrice')
        self.assertEqual(settings.irc_user, 'beatrice')
        self.assertEqual(settings.irc_channels, ('#ussycode', '#llm'))
        self.assertEqual(settings.runtime_defaults.max_tokens, 700)
        self.assertEqual(settings.runtime_defaults.reply_interval_seconds, 8.0)
        self.assertEqual(settings.history_turns, 12)

    def test_settings_file_provides_editable_defaults(self) -> None:
        payload = '''{
          "irc": {
            "server": "irc.example.net",
            "channels": ["#ussycode"]
          },
          "bot": {
            "command_prefix": "!bea"
          },
          "defaults": {
            "model": "deepseek/deepseek-v3.2",
            "temperature": 1.7,
            "max_tokens": 900
          }
        }'''

        with tempfile.TemporaryDirectory() as temp_dir:
            settings_path = os.path.join(temp_dir, 'settings.json')
            with open(settings_path, 'w', encoding='utf-8') as handle:
                handle.write(payload)

            env = {
                'BOT_SETTINGS_FILE': settings_path,
            }
            with patch.dict(os.environ, env, clear=True):
                settings = BotSettings.from_env()

        self.assertEqual(settings.irc_server, 'irc.example.net')
        self.assertEqual(settings.irc_channels, ('#ussycode',))
        self.assertEqual(settings.command_prefix, '!bea')
        self.assertEqual(settings.runtime_defaults.model, 'deepseek/deepseek-v3.2')
        self.assertEqual(settings.runtime_defaults.temperature, 1.7)
        self.assertEqual(settings.runtime_defaults.max_tokens, 900)
        self.assertEqual(settings.runtime_defaults.reply_interval_seconds, 8.0)

    def test_settings_file_can_expand_history_turns(self) -> None:
        payload = '''{
          "bot": {
            "history_turns": 12
          }
        }'''

        with tempfile.TemporaryDirectory() as temp_dir:
            settings_path = os.path.join(temp_dir, 'settings.json')
            with open(settings_path, 'w', encoding='utf-8') as handle:
                handle.write(payload)

            env = {
                'BOT_SETTINGS_FILE': settings_path,
            }
            with patch.dict(os.environ, env, clear=True):
                settings = BotSettings.from_env()

        self.assertEqual(settings.history_turns, 12)

    def test_secret_store_persists_openrouter_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            secrets_path = os.path.join(temp_dir, 'secrets.json')
            secrets = SecretStore.from_file(secrets_path)
            secrets.set_openrouter_api_key('sk-test')

            loaded = SecretStore.from_file(secrets_path)

        self.assertEqual(loaded.openrouter_api_key, 'sk-test')

    def test_secret_store_clears_persisted_openrouter_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            secrets_path = os.path.join(temp_dir, 'secrets.json')
            secrets = SecretStore.from_file(secrets_path)
            secrets.set_openrouter_api_key('sk-test')
            secrets.clear_openrouter_api_key()

            loaded = SecretStore.from_file(secrets_path)

        self.assertIsNone(loaded.openrouter_api_key)

    def test_runtime_store_supports_overrides_and_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime_path = os.path.join(temp_dir, 'runtime.json')
            store = RuntimeStore(RuntimeDefaults(model='deepseek/deepseek-v3.2'), {'model': 'openai/gpt-5'})

            self.assertEqual(store.current().model, 'openai/gpt-5')
            store.apply_updates({'temperature': 1.5, 'stream': True})
            store.persist(runtime_path)

            with open(runtime_path, encoding='utf-8') as handle:
                payload = handle.read()

        self.assertIn('"model": "openai/gpt-5"', payload)
        self.assertIn('"temperature": 1.5', payload)
        self.assertIn('"stream": true', payload)

    def test_settings_load_runtime_and_memory_paths(self) -> None:
        env = {
            'BOT_RUNTIME_FILE': 'state/runtime.json',
            'BOT_MEMORY_DB_FILE': 'state/memory.sqlite3',
            'BOT_AUDIT_LOG_FILE': 'state/audit.jsonl',
            'BOT_ADMIN_NICKS': 'alice,bob',
            'BOT_APPROVAL_TIMEOUT_SECONDS': '1200',
        }
        with patch.dict(os.environ, env, clear=True):
            settings = BotSettings.from_env()

        self.assertEqual(settings.runtime_file, 'state/runtime.json')
        self.assertEqual(settings.memory_db_file, 'state/memory.sqlite3')
        self.assertEqual(settings.audit_log_file, 'state/audit.jsonl')
        self.assertEqual(settings.admin_nicks, ('alice', 'bob'))
        self.assertEqual(settings.approval_timeout_seconds, 1200.0)
