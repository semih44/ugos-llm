#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the bridge's bearer-token auth and its bind guard.

The bridge is the authentication boundary in front of a gateway that has
none, so these two functions are the whole security story: get them wrong
and an unauthenticated LLM endpoint sits on the LAN.

Run: python3 -m unittest discover -s tests -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_units import px  # noqa: E402  (reuse the loaded proxy module)


KEY = "s3cret-key"


class TestCheckAuth(unittest.TestCase):
    def test_no_key_configured_allows_everything(self):
        # backwards compatibility: existing deployments (docker-only bind,
        # no key) must keep working untouched
        for header in (None, "", "Bearer whatever", "garbage"):
            with self.subTest(header=header):
                self.assertTrue(px.check_auth(header, ""))

    def test_correct_bearer_token_passes(self):
        self.assertTrue(px.check_auth(f"Bearer {KEY}", KEY))

    def test_scheme_is_case_insensitive(self):
        for scheme in ("Bearer", "bearer", "BEARER", "BeArEr"):
            with self.subTest(scheme=scheme):
                self.assertTrue(px.check_auth(f"{scheme} {KEY}", KEY))

    def test_missing_header_is_rejected(self):
        self.assertFalse(px.check_auth(None, KEY))
        self.assertFalse(px.check_auth("", KEY))

    def test_wrong_key_is_rejected(self):
        self.assertFalse(px.check_auth("Bearer wrong", KEY))

    def test_near_miss_keys_are_rejected(self):
        for bad in (KEY[:-1], KEY + "x", KEY.upper(), " " + KEY, KEY + " "):
            with self.subTest(bad=bad):
                self.assertFalse(px.check_auth(f"Bearer {bad}", KEY))

    def test_bare_key_without_scheme_is_rejected(self):
        # be strict: a client that forgets the scheme should get a clear 401
        # rather than silently working against half the ecosystem's habits
        self.assertFalse(px.check_auth(KEY, KEY))

    def test_other_schemes_are_rejected(self):
        self.assertFalse(px.check_auth(f"Basic {KEY}", KEY))
        self.assertFalse(px.check_auth(f"Token {KEY}", KEY))

    def test_empty_bearer_is_rejected(self):
        self.assertFalse(px.check_auth("Bearer ", KEY))
        self.assertFalse(px.check_auth("Bearer", KEY))

    def test_comparison_is_constant_time(self):
        # guard against a future refactor to plain == , which leaks the key
        # length and prefix through timing
        import inspect
        src = inspect.getsource(px.check_auth)
        self.assertIn("compare_digest", src)

    def test_non_ascii_key_does_not_crash(self):
        self.assertTrue(px.check_auth("Bearer schlüssel", "schlüssel"))
        self.assertFalse(px.check_auth("Bearer schlussel", "schlüssel"))


class TestThinkingSuppression(unittest.TestCase):
    """Emulated tool requests are budget-capped JSON extractions — a server
    whose default is enable_thinking:true would burn that budget on
    reasoning. The bridge therefore pins thinking off for exactly the
    requests it rewrites, and only those."""

    base = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    TOOL = {"type": "function", "function": {
        "name": "alpha", "description": "A",
        "parameters": {"type": "object",
                       "properties": {"x": {"type": "string"}}}}}

    def _rw(self, payload):
        import json as j
        body, _, plan = px.rewrite(j.dumps(payload).encode(),
                                   "/v1/chat/completions")
        return j.loads(body), plan

    def test_emulated_request_gets_thinking_off(self):
        data, plan = self._rw(dict(self.base, tools=[self.TOOL],
                                   tool_choice="required"))
        self.assertIsNotNone(plan)
        self.assertEqual(data["chat_template_kwargs"],
                         {"enable_thinking": False})

    def test_client_supplied_kwargs_win(self):
        data, _ = self._rw(dict(self.base, tools=[self.TOOL],
                                tool_choice="required",
                                chat_template_kwargs={"enable_thinking": True}))
        self.assertEqual(data["chat_template_kwargs"],
                         {"enable_thinking": True})

    def test_passthrough_requests_are_untouched(self):
        data, plan = self._rw(dict(self.base))
        self.assertIsNone(plan)
        self.assertNotIn("chat_template_kwargs", data)


class TestThinkingDefault(unittest.TestCase):
    """The gateway swallows server-side --chat-template-kwargs (verified:
    argv intact, /props kwargs empty, thinking stays off), while
    request-level kwargs demonstrably win. So the bridge owns the default,
    per instance, via THINKING_DEFAULT."""

    base = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    TOOL = {"type": "function", "function": {
        "name": "alpha", "description": "A",
        "parameters": {"type": "object",
                       "properties": {"x": {"type": "string"}}}}}

    def setUp(self):
        self._orig = px.THINKING_DEFAULT

    def tearDown(self):
        px.THINKING_DEFAULT = self._orig

    def _rw(self, payload):
        import json as j
        body, _, plan = px.rewrite(j.dumps(payload).encode(),
                                   "/v1/chat/completions")
        return j.loads(body), plan

    def test_on_injects_thinking_into_plain_chat(self):
        px.THINKING_DEFAULT = "on"
        data, plan = self._rw(dict(self.base))
        self.assertIsNone(plan)
        self.assertEqual(data["chat_template_kwargs"],
                         {"enable_thinking": True})

    def test_off_injects_no_thinking(self):
        px.THINKING_DEFAULT = "off"
        data, _ = self._rw(dict(self.base))
        self.assertEqual(data["chat_template_kwargs"],
                         {"enable_thinking": False})

    def test_unset_leaves_requests_alone(self):
        px.THINKING_DEFAULT = ""
        data, _ = self._rw(dict(self.base))
        self.assertNotIn("chat_template_kwargs", data)

    def test_client_kwargs_always_win(self):
        px.THINKING_DEFAULT = "on"
        data, _ = self._rw(dict(self.base,
                                chat_template_kwargs={"enable_thinking": False}))
        self.assertEqual(data["chat_template_kwargs"],
                         {"enable_thinking": False})

    def test_emulated_tool_requests_stay_sober_even_with_default_on(self):
        # the whole point: Paperless-style JSON extraction must never burn
        # its token budget on reasoning, whatever the instance default is
        px.THINKING_DEFAULT = "on"
        data, plan = self._rw(dict(self.base, tools=[self.TOOL],
                                   tool_choice="required"))
        self.assertIsNotNone(plan)
        self.assertEqual(data["chat_template_kwargs"],
                         {"enable_thinking": False})

    def test_reasoning_effort_low_beats_default_on(self):
        # VS Code's thinking-effort picker sends reasoning_effort — the
        # per-question choice must override the instance default
        px.THINKING_DEFAULT = "on"
        data, _ = self._rw(dict(self.base, reasoning_effort="low"))
        self.assertEqual(data["chat_template_kwargs"],
                         {"enable_thinking": False})

    def test_reasoning_effort_high_beats_default_off(self):
        px.THINKING_DEFAULT = "off"
        data, _ = self._rw(dict(self.base, reasoning_effort="high"))
        self.assertEqual(data["chat_template_kwargs"],
                         {"enable_thinking": True})

    def test_all_documented_effort_levels_map(self):
        px.THINKING_DEFAULT = ""
        for eff, expect in (("none", False), ("minimal", False),
                            ("low", False), ("medium", True),
                            ("high", True), ("xhigh", True), ("max", True)):
            with self.subTest(effort=eff):
                data, _ = self._rw(dict(self.base, reasoning_effort=eff))
                self.assertEqual(
                    data["chat_template_kwargs"]["enable_thinking"], expect)

    def test_unknown_effort_falls_back_to_default(self):
        px.THINKING_DEFAULT = "on"
        data, _ = self._rw(dict(self.base, reasoning_effort="turbo"))
        self.assertEqual(data["chat_template_kwargs"],
                         {"enable_thinking": True})

    def test_client_kwargs_beat_reasoning_effort(self):
        px.THINKING_DEFAULT = ""
        data, _ = self._rw(dict(
            self.base, reasoning_effort="high",
            chat_template_kwargs={"enable_thinking": False}))
        self.assertEqual(data["chat_template_kwargs"],
                         {"enable_thinking": False})

    def test_effort_cannot_unpin_tool_emulation(self):
        px.THINKING_DEFAULT = "on"
        data, plan = self._rw(dict(self.base, tools=[self.TOOL],
                                   tool_choice="required",
                                   reasoning_effort="high"))
        self.assertIsNotNone(plan)
        self.assertEqual(data["chat_template_kwargs"],
                         {"enable_thinking": False})


class TestAuthShape(unittest.TestCase):
    """401 diagnostics must describe the attempt without leaking the key."""

    def test_missing_header(self):
        self.assertEqual(px.auth_shape({}), "no-auth-header")

    def test_bearer_shape(self):
        s = px.auth_shape({"Authorization": "Bearer abcdef"})
        self.assertEqual(s, "Authorization=Bearer,len=6")

    def test_bare_token_shape(self):
        s = px.auth_shape({"Authorization": "abcdefgh"})
        self.assertEqual(s, "Authorization=bare,len=8")

    def test_api_key_header_shape(self):
        s = px.auth_shape({"api-key": "xyz"})
        self.assertEqual(s, "api-key,len=3")

    def test_never_contains_the_secret(self):
        secret = "SuperGeheim123"
        for h in ({"Authorization": f"Bearer {secret}"},
                  {"Authorization": secret}, {"api-key": secret}):
            self.assertNotIn(secret, px.auth_shape(h))


class TestBindGuard(unittest.TestCase):
    """Refusing to start is the only reliable way to stop someone putting an
    unauthenticated LLM endpoint on their LAN."""

    def test_loopback_without_key_is_allowed(self):
        for host in ("127.0.0.1", "127.0.0.5", "::1", "localhost"):
            with self.subTest(host=host):
                px.guard_bind(host, "", False)      # must not raise

    def test_docker_bridge_without_key_is_allowed(self):
        # the shipped default, and what the Paperless deployment uses
        for host in ("172.17.0.1", "172.18.0.1", "172.31.255.1"):
            with self.subTest(host=host):
                px.guard_bind(host, "", False)

    def test_wildcard_without_key_is_refused(self):
        for host in ("0.0.0.0", "::", ""):
            with self.subTest(host=host):
                with self.assertRaises(SystemExit):
                    px.guard_bind(host, "", False)

    def test_lan_address_without_key_is_refused(self):
        for host in ("192.168.1.221", "10.0.0.5", "fd00::1"):
            with self.subTest(host=host):
                with self.assertRaises(SystemExit):
                    px.guard_bind(host, "", False)

    def test_any_address_with_key_is_allowed(self):
        for host in ("0.0.0.0", "192.168.1.221", "::"):
            with self.subTest(host=host):
                px.guard_bind(host, KEY, False)

    def test_escape_hatch_allows_unauthenticated_exposure(self):
        px.guard_bind("0.0.0.0", "", True)

    def test_refusal_message_names_the_fix(self):
        with self.assertRaises(SystemExit) as cm:
            px.guard_bind("192.168.1.221", "", False)
        msg = str(cm.exception)
        self.assertIn("API_KEY", msg)


if __name__ == "__main__":
    unittest.main()
