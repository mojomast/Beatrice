import asyncio
import unittest
from unittest.mock import AsyncMock

from bot.irc import IRCClient, NickChange


class IRCClientTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.client = IRCClient(
            host="irc.example.net",
            port=6667,
            nick="Beatrice",
            user="beatrice",
            realname="Beatrice Bot",
        )

    async def test_welcome_tracks_server_name_and_emits_connected(self) -> None:
        connected: list[str] = []

        async def on_connected(server_name: str) -> None:
            connected.append(server_name)

        self.client.on("connected", on_connected)

        await self.client._process_line(":irc.example.net 001 Beatrice :Welcome to the network")

        self.assertEqual(self.client.nick, "Beatrice")
        self.assertEqual(self.client.server_name, "irc.example.net")
        self.assertEqual(connected, ["irc.example.net"])

    async def test_whois_collects_structured_response(self) -> None:
        self.client.connected = True
        self.client.send_raw = AsyncMock()

        task = asyncio.create_task(self.client.whois("alice"))
        await asyncio.sleep(0)

        self.client.send_raw.assert_awaited_once_with("WHOIS alice")

        await self.client._process_line(":irc.example.net 311 Beatrice alice user host * :Alice Example")
        await self.client._process_line(":irc.example.net 312 Beatrice alice irc.example.net :Example IRC")
        await self.client._process_line(":irc.example.net 313 Beatrice alice :is an IRC operator")
        await self.client._process_line(":irc.example.net 317 Beatrice alice 42 1700000000 :seconds idle, signon time")
        await self.client._process_line(":irc.example.net 319 Beatrice alice :@#ops #general")
        await self.client._process_line(":irc.example.net 318 Beatrice alice :End of /WHOIS list")

        result = await task

        self.assertEqual(result.status, "ok")
        self.assertIsNotNone(result.info)
        assert result.info is not None
        self.assertEqual(result.nick, "alice")
        self.assertEqual(result.info.nick, "alice")
        self.assertEqual(result.info.user, "user")
        self.assertEqual(result.info.host, "host")
        self.assertEqual(result.info.realname, "Alice Example")
        self.assertEqual(result.info.server, "irc.example.net")
        self.assertEqual(result.info.server_info, "Example IRC")
        self.assertEqual(result.info.channels, ("@#ops", "#general"))
        self.assertEqual(result.info.idle_seconds, 42)
        self.assertEqual(result.info.signon_time, 1700000000)
        self.assertTrue(result.info.is_operator)

    async def test_whois_returns_not_found(self) -> None:
        self.client.connected = True
        self.client.send_raw = AsyncMock()

        task = asyncio.create_task(self.client.whois("missing"))
        await asyncio.sleep(0)
        await self.client._process_line(":irc.example.net 401 Beatrice missing :No such nick/channel")

        result = await task

        self.assertEqual(result.status, "not_found")
        self.assertEqual(result.nick, "missing")
        self.assertEqual(result.error, "No such nick/channel")
        self.assertIsNone(result.info)

    async def test_whois_returns_error_when_not_connected(self) -> None:
        result = await self.client.whois("alice")

        self.assertEqual(result.status, "error")
        self.assertEqual(result.nick, "alice")
        self.assertEqual(result.error, "not connected")

    async def test_join_names_topic_nick_part_and_quit_update_environment_state(self) -> None:
        events: list[tuple] = []

        self.client.on("join", lambda nick, prefix, channel: events.append(("join", nick, channel)))
        self.client.on("names", lambda channel, names: events.append(("names", channel, names)))
        self.client.on("topic", lambda nick, prefix, channel, topic: events.append(("topic", nick, channel, topic)))
        self.client.on("nick", lambda old_nick, prefix, new_nick: events.append(("nick", old_nick, new_nick)))
        self.client.on("part", lambda nick, prefix, channel, reason: events.append(("part", nick, channel, reason)))
        self.client.on("quit", lambda nick, prefix, reason: events.append(("quit", nick, reason)))

        await self.client._process_line(":Beatrice!bot@example JOIN :#ussycode")
        await self.client._process_line(":irc.example.net 353 Beatrice = #ussycode :@Beatrice alice +bob")
        await self.client._process_line(":irc.example.net 366 Beatrice #ussycode :End of /NAMES list")
        await self.client._process_line(":irc.example.net 332 Beatrice #ussycode :Ship the fix")
        await self.client._process_line(":alice!user@example NICK :alice_")
        await self.client._process_line(":bob!user@example PART #ussycode :later")
        await self.client._process_line(":alice_!user@example QUIT :bye")

        self.assertEqual(self.client.joined_channels(), ("#ussycode",))
        self.assertEqual(self.client.channel_topic("#ussycode"), "Ship the fix")
        self.assertEqual(self.client.channel_users("#ussycode"), ("Beatrice",))

        changes = self.client.recent_nick_changes()
        self.assertEqual(len(changes), 1)
        self.assertIsInstance(changes[0], NickChange)
        self.assertEqual(changes[0].old_nick, "alice")
        self.assertEqual(changes[0].new_nick, "alice_")

        state = self.client.environment_state()
        self.assertEqual(state["joined_channels"], ["#ussycode"])
        self.assertEqual(state["channels"][0]["name"], "#ussycode")
        self.assertTrue(state["channels"][0]["joined"])
        self.assertEqual(state["channels"][0]["topic"], "Ship the fix")
        self.assertEqual(state["channels"][0]["users"], ["Beatrice"])

        self.assertEqual(
            events,
            [
                ("join", "Beatrice", "#ussycode"),
                ("names", "#ussycode", ("alice", "Beatrice", "bob")),
                ("topic", "irc.example.net", "#ussycode", "Ship the fix"),
                ("nick", "alice", "alice_"),
                ("part", "bob", "#ussycode", "later"),
                ("quit", "alice_", "bye"),
            ],
        )

    async def test_names_without_self_does_not_mark_joined(self) -> None:
        await self.client._process_line(":irc.example.net 353 Beatrice = #other :alice bob")
        await self.client._process_line(":irc.example.net 366 Beatrice #other :End of /NAMES list")

        self.assertEqual(self.client.joined_channels(), ())
        self.assertEqual(self.client.known_channels(), ("#other",))
        self.assertEqual(self.client.channel_users("#other"), ("alice", "bob"))

    async def test_self_nick_change_updates_client_nick_and_membership(self) -> None:
        await self.client._process_line(":Beatrice!bot@example JOIN :#ussycode")
        await self.client._process_line(":irc.example.net 353 Beatrice = #ussycode :@Beatrice alice")
        await self.client._process_line(":irc.example.net 366 Beatrice #ussycode :End of /NAMES list")
        await self.client._process_line(":Beatrice!bot@example NICK :Bea")

        self.assertEqual(self.client.nick, "Bea")
        self.assertEqual(self.client.channel_users("#ussycode"), ("alice", "Bea"))

    async def test_parting_self_drops_channel_state(self) -> None:
        await self.client._process_line(":Beatrice!bot@example JOIN :#ussycode")
        await self.client._process_line(":irc.example.net 332 Beatrice #ussycode :Ship the fix")
        await self.client._process_line(":Beatrice!bot@example PART #ussycode :bye")

        self.assertEqual(self.client.joined_channels(), ())
        self.assertEqual(self.client.known_channels(), ())
        self.assertIsNone(self.client.channel_topic("#ussycode"))
