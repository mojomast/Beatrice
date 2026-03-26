import unittest

from bot.profile_tools import (
    IRCActivity,
    build_channel_prompt_context,
    build_user_profile_fragment,
    extract_profile_facts,
    extract_topic_keywords,
    format_channel_member_prompt,
    format_channel_topic_prompt,
)


class ProfileToolsTests(unittest.TestCase):
    def test_extract_profile_facts_keeps_simple_explicit_facts(self) -> None:
        facts = extract_profile_facts(
            "I'm a systems engineer. I use neovim and tmux. I prefer concise answers.",
            "alice",
        )

        self.assertEqual(
            facts,
            [
                "alice is systems engineer",
                "alice uses neovim and tmux",
                "alice prefers concise answers",
            ],
        )

    def test_extract_profile_facts_supports_pronouns(self) -> None:
        facts = extract_profile_facts("my pronouns are she/her and I use linux", "zoe")

        self.assertEqual(facts[0], "zoe's pronouns are she/her")
        self.assertIn("zoe uses linux", facts)

    def test_extract_profile_facts_stays_conservative_for_questions_and_negation(self) -> None:
        self.assertEqual(extract_profile_facts("do I like rust?", "bob"), [])
        self.assertEqual(extract_profile_facts("I don't like rust right now", "bob"), [])

    def test_build_user_profile_fragment_combines_memory_and_activity(self) -> None:
        fragment = build_user_profile_fragment(
            "alice",
            remembered_profile="alice prefers concise answers",
            remembered_facts=["alice uses linux"],
            recent_activity=[
                IRCActivity("alice", "I work on the IRC bridge"),
                IRCActivity("alice", "docker timeout still happening on the bridge"),
                IRCActivity("bob", "alice is great"),
            ],
        )

        self.assertEqual(
            fragment,
            "alice: prefers concise answers; uses linux; works on IRC bridge; recent topics: docker, timeout, still",
        )

    def test_format_channel_member_prompt_prioritizes_active_members_and_profiles(self) -> None:
        snippet = format_channel_member_prompt(
            ["Beatrice", "alice", "bob", "zoe"],
            profiles={"alice": "alice prefers concise answers", "bob": "bob uses rust and nix"},
            active_nicks=["bob", "alice", "bob"],
            max_members=3,
        )

        self.assertEqual(
            snippet,
            "Channel members: bob (uses rust and nix), alice (prefers concise answers), Beatrice, +1 more. Recently active: bob, alice.",
        )

    def test_format_channel_topic_prompt_and_context_include_topic_and_keywords(self) -> None:
        topic = format_channel_topic_prompt(
            "#ussycode",
            topic="Deploy window and bridge cleanup",
            recent_topic_keywords=["docker", "bridge", "timeout", "docker"],
        )
        context = build_channel_prompt_context(
            "#ussycode",
            members=["alice", "bob"],
            member_profiles={"alice": "alice likes incident writeups"},
            active_nicks=["alice"],
            topic="Deploy window and bridge cleanup",
            recent_topic_keywords=["docker", "bridge", "timeout"],
        )

        self.assertEqual(
            topic,
            "Channel topic for #ussycode: Deploy window and bridge cleanup. Recent channel topics: docker, bridge, timeout.",
        )
        self.assertEqual(len(context), 2)
        self.assertEqual(context[0], topic)
        self.assertIn("Channel members: alice (likes incident writeups), bob.", context[1])

    def test_extract_topic_keywords_filters_noise_words(self) -> None:
        self.assertEqual(
            extract_topic_keywords("anyone know why the docker api timeout happened again?"),
            ["docker", "api", "timeout", "happened"],
        )


if __name__ == "__main__":
    unittest.main()
