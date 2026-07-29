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
