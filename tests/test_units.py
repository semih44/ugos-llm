#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the risky pure functions: name validation, GGUF parsing,
architecture rules, split detection, and the proxy's rewrite logic.

Run: python3 -m unittest discover -s tests -v
"""
import importlib.util
import io
import json
import os
import shutil
import struct
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ug = _load("ugos-llm.py", "ugos_llm")
px = _load("openai-bridge/proxy.py", "bridge_proxy")


def gguf_bytes(arch="qwen3_5", name="test", extra_kv=()):
    """Build a minimal valid GGUF header."""
    def s(v):
        b = v.encode()
        return struct.pack("<Q", len(b)) + b
    kv = [(b"general.architecture", 8, s(arch)),
          (b"general.name", 8, s(name))]
    kv += list(extra_kv)
    out = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0)
    out += struct.pack("<Q", len(kv))
    for key, typ, payload in kv:
        out += struct.pack("<Q", len(key)) + key + struct.pack("<I", typ)
        out += payload
    return out


class TestNameValidation(unittest.TestCase):
    def test_accepts_normal_names(self):
        for n in ("Qwen3.5-9B", "llama_3", "a", "M" * 64):
            self.assertEqual(ug.validate_name(n), n)

    def test_rejects_dangerous_names(self):
        for bad in ("../etc", "a/b", "a;rm -rf /", "a b", "a\nb", "",
                    ".", "..", "-lead", "M" * 65, "a'b", "a$(x)"):
            with self.subTest(bad=bad), self.assertRaises(SystemExit):
                ug.validate_name(bad)


class TestGGUF(unittest.TestCase):
    def test_parses_architecture(self):
        meta = ug.parse_gguf_meta(io.BytesIO(gguf_bytes("gemma3", "g3")))
        self.assertEqual(meta["general.architecture"], "gemma3")
        self.assertEqual(meta["general.name"], "g3")

    def test_rejects_non_gguf(self):
        with self.assertRaises(ug.GGUFError):
            ug.parse_gguf_meta(io.BytesIO(b"NOPE" + b"\x00" * 64))

    def test_rejects_truncated(self):
        with self.assertRaises(ug.GGUFError):
            ug.parse_gguf_meta(io.BytesIO(gguf_bytes()[:12]))

    def test_rejects_absurd_string_length(self):
        evil = (b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0)
                + struct.pack("<Q", 1) + struct.pack("<Q", 1 << 60))
        with self.assertRaises(ug.GGUFError):
            ug.parse_gguf_meta(io.BytesIO(evil))

    def test_rejects_unknown_type(self):
        bad = (b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0)
               + struct.pack("<Q", 1)
               + struct.pack("<Q", 3) + b"key" + struct.pack("<I", 99))
        with self.assertRaises(ug.GGUFError):
            ug.parse_gguf_meta(io.BytesIO(bad))


class TestArchRules(unittest.TestCase):
    def test_moe_is_broken(self):
        for a in ("qwen35moe", "qwen3moe", "Qwen3MoE", "glm4moe"):
            self.assertEqual(ug.classify_arch(a)[0], "BROKEN")

    def test_missing_and_tested(self):
        self.assertEqual(ug.classify_arch("gemma4")[0], "UNSUPPORTED")
        self.assertEqual(ug.classify_arch("qwen3_5")[0], "TESTED")
        self.assertEqual(ug.classify_arch("llama")[0], "EXPECTED")
        self.assertEqual(ug.classify_arch("something-new")[0], "UNKNOWN")


class TestSplitDetection(unittest.TestCase):
    def test_detects_shards(self):
        files = [("m-00001-of-00003.gguf", 1), ("m-00002-of-00003.gguf", 1),
                 ("m-00003-of-00003.gguf", 1), ("other.gguf", 1)]
        self.assertEqual(len(ug.split_siblings(files, "m-00001-of-00003.gguf")), 3)

    def test_single_file_is_not_split(self):
        self.assertEqual(ug.split_siblings([("solo.gguf", 1)], "solo.gguf"), [])


class TestCollapseScore(unittest.TestCase):
    def test_flags_repetition(self):
        self.assertLess(ug.collapse_score("- " * 200), 0.5)
        self.assertLess(ug.collapse_score("1. " * 100), 0.5)

    def test_accepts_prose(self):
        text = ("Die Lagerhalle wurde im Jahr 1987 errichtet und seitdem "
                "mehrfach umgebaut, zuletzt mit neuer Heizung.")
        self.assertGreater(ug.collapse_score(text), 0.5)


TOOL_A = {"type": "function", "function": {
    "name": "alpha", "description": "A",
    "parameters": {"type": "object", "properties": {"x": {"type": "string"}},
                   "strict": True, "additionalProperties": False}}}
TOOL_B = {"type": "function", "function": {
    "name": "beta", "description": "B",
    "parameters": {"type": "object", "properties": {"y": {"type": "string"}}}}}


def rw(payload):
    body, note, plan = px.rewrite(json.dumps(payload).encode(),
                                  "/v1/chat/completions")
    return json.loads(body), note, plan


class TestProxyRewrite(unittest.TestCase):
    base = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}

    def test_no_tools_passthrough(self):
        _, _, plan = rw(dict(self.base))
        self.assertIsNone(plan)

    def test_tool_choice_auto_and_none_passthrough(self):
        for choice in ("auto", "none"):
            with self.subTest(choice=choice):
                _, _, plan = rw(dict(self.base, tools=[TOOL_A, TOOL_B],
                                     tool_choice=choice))
                self.assertIsNone(plan, "auto/none must not be emulated")

    def test_required_single_tool(self):
        data, _, plan = rw(dict(self.base, tools=[TOOL_A],
                                tool_choice="required"))
        self.assertEqual(plan[0], "single")
        self.assertNotIn("tools", data)
        self.assertIn("alpha", json.dumps(data["messages"]) + "alpha")

    def test_required_multi_tool_offers_all(self):
        data, note, plan = rw(dict(self.base, tools=[TOOL_A, TOOL_B],
                                   tool_choice="required"))
        self.assertEqual(plan[0], "multi")
        self.assertIn("alpha", data["messages"][-1]["content"])
        self.assertIn("beta", data["messages"][-1]["content"])
        self.assertIn("2-tools", note)

    def test_named_tool_choice_selects_that_tool(self):
        data, _, plan = rw(dict(
            self.base, tools=[TOOL_A, TOOL_B],
            tool_choice={"type": "function", "function": {"name": "beta"}}))
        self.assertEqual(plan[0], "single")
        self.assertEqual(plan[1]["function"]["name"], "beta")

    def test_max_tokens_is_a_hard_cap(self):
        data, _, _ = rw(dict(self.base, tools=[TOOL_A],
                             tool_choice="required", max_tokens=999999))
        self.assertEqual(data["max_tokens"], px.MAX_TOKENS)

    def test_max_tokens_smaller_value_kept(self):
        data, _, _ = rw(dict(self.base, tools=[TOOL_A],
                             tool_choice="required", max_tokens=17))
        self.assertEqual(data["max_tokens"], 17)

    def test_streaming_disabled_for_emulation(self):
        data, note, _ = rw(dict(self.base, tools=[TOOL_A],
                                tool_choice="required", stream=True))
        self.assertNotIn("stream", data)
        self.assertIn("stream-disabled", note)

    def test_openai_only_keywords_scrubbed(self):
        data, _, _ = rw(dict(self.base, tools=[TOOL_A],
                             tool_choice="required"))
        blob = data["messages"][-1]["content"]
        # "strict" is an OpenAI API extension, not JSON Schema — drop it.
        # "additionalProperties" IS legitimate JSON Schema — keep it.
        self.assertNotIn('"strict"', blob)
        self.assertIn("additionalProperties", blob)


class TestProxyWrap(unittest.TestCase):
    def _resp(self, content):
        return json.dumps({"choices": [{"index": 0, "finish_reason": "stop",
                                        "message": {"role": "assistant",
                                                    "content": content}}]}).encode()

    def test_single_wraps_json(self):
        out = json.loads(px.wrap_as_tool_call(
            self._resp('{"x": "1"}'), ("single", TOOL_A)))
        call = out["choices"][0]["message"]["tool_calls"][0]
        self.assertEqual(call["function"]["name"], "alpha")
        self.assertEqual(json.loads(call["function"]["arguments"]), {"x": "1"})
        self.assertEqual(out["choices"][0]["finish_reason"], "tool_calls")

    def test_handles_code_fences(self):
        out = json.loads(px.wrap_as_tool_call(
            self._resp('```json\n{"x": "2"}\n```'), ("single", TOOL_A)))
        args = out["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        self.assertEqual(json.loads(args), {"x": "2"})

    def test_multi_maps_selected_tool(self):
        out = json.loads(px.wrap_as_tool_call(
            self._resp('{"tool_name":"beta","arguments":{"y":"z"}}'),
            ("multi", [TOOL_A, TOOL_B])))
        call = out["choices"][0]["message"]["tool_calls"][0]
        self.assertEqual(call["function"]["name"], "beta")

    def test_multi_rejects_hallucinated_tool(self):
        with self.assertRaises(ValueError):
            px.wrap_as_tool_call(
                self._resp('{"tool_name":"ghost","arguments":{}}'),
                ("multi", [TOOL_A, TOOL_B]))

    def test_non_json_raises(self):
        with self.assertRaises(ValueError):
            px.wrap_as_tool_call(self._resp("I am prose."), ("single", TOOL_A))



class TestProxyHardening(unittest.TestCase):
    """Malformed input must produce clean errors, never crashes or
    silent data loss."""
    base = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}

    def test_null_tool_entry_is_bad_request(self):
        with self.assertRaises(px.BadRequest):
            px.rewrite(json.dumps(dict(self.base, tools=[None],
                                       tool_choice="required")).encode(),
                       "/v1/chat/completions")

    def test_tools_not_a_list_is_bad_request(self):
        with self.assertRaises(px.BadRequest):
            px.rewrite(json.dumps(dict(self.base, tools="x",
                                       tool_choice="required")).encode(),
                       "/v1/chat/completions")

    def test_non_integer_max_tokens_is_bad_request(self):
        with self.assertRaises(px.BadRequest):
            px.rewrite(json.dumps(dict(self.base, tools=[TOOL_A],
                                       tool_choice="required",
                                       max_tokens="lots")).encode(),
                       "/v1/chat/completions")

    def test_messages_not_a_list_is_bad_request(self):
        with self.assertRaises(px.BadRequest):
            px.rewrite(json.dumps({"model": "m", "messages": "hi",
                                   "tools": [TOOL_A],
                                   "tool_choice": "required"}).encode(),
                       "/v1/chat/completions")

    def test_scrub_keeps_property_named_strict(self):
        schema = {"type": "object",
                  "properties": {"strict": {"type": "boolean"},
                                 "additionalProperties": {"type": "string"}},
                  "strict": True}
        out = px.scrub_schema(json.loads(json.dumps(schema)))
        self.assertNotIn("strict", out)               # keyword removed
        self.assertIn("strict", out["properties"])    # property kept
        self.assertIn("additionalProperties", out["properties"])

    def test_wrap_rejects_missing_required_args(self):
        resp = json.dumps({"choices": [{"index": 0, "finish_reason": "stop",
                                        "message": {"role": "assistant",
                                                    "content": '{"y": "1"}'}}]}
                          ).encode()
        tool = {"type": "function", "function": {
            "name": "alpha",
            "parameters": {"type": "object",
                           "properties": {"x": {"type": "string"}},
                           "required": ["x"]}}}
        with self.assertRaises(ValueError):
            px.wrap_as_tool_call(resp, ("single", tool))


class TestCLIValidation(unittest.TestCase):
    def test_ctx_bounds_rejected(self):
        for ctx in (-1, 0, 100, 999999):
            args = ug.build_parser().parse_args(
                ["install", "o/r", "--ctx", str(ctx)])
            with self.subTest(ctx=ctx), self.assertRaises(SystemExit):
                ug.cmd_install(args)

    def test_empty_quant_rejected(self):
        args = ug.build_parser().parse_args(
            ["install", "o/r", "--quant", ""])
        with self.assertRaises(SystemExit):
            ug.cmd_install(args)


class TestAtomicInstall(unittest.TestCase):
    """The install must never leave a half-populated model directory active."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.models = os.path.join(self.tmp, "models")
        os.makedirs(self.models)
        self._orig_dir = ug.MODELS_DIR
        self._orig_root = ug.is_root
        ug.MODELS_DIR = self.models
        ug.is_root = lambda: True          # exercise the native path
        self._staging = ug._staging_dir
        ug._staging_dir = lambda: self.tmp
        self.src = os.path.join(self.tmp, "src.gguf")
        with open(self.src, "wb") as f:
            f.write(b"GGUF" + b"\x00" * 32)

    def tearDown(self):
        ug.MODELS_DIR = self._orig_dir
        ug.is_root = self._orig_root
        ug._staging_dir = self._staging
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_installs_and_replaces(self):
        ug.priv_install_files("M", self.src, None, '{"a":1}')
        self.assertTrue(os.path.isfile(os.path.join(self.models, "M/M.gguf")))
        # replacing must remove stale files (e.g. an old projector)
        open(os.path.join(self.models, "M", "mmproj.gguf"), "w").close()
        ug.priv_install_files("M", self.src, None, '{"a":2}')
        self.assertFalse(os.path.exists(
            os.path.join(self.models, "M", "mmproj.gguf")))
        with open(os.path.join(self.models, "M", "model_config.json")) as f:
            self.assertIn('"a":2', f.read())

    def test_failure_keeps_previous_installation(self):
        ug.priv_install_files("M", self.src, None, '{"v":"old"}')
        boom = os.path.join(self.tmp, "gone.gguf")   # missing source
        with self.assertRaises(Exception):
            ug.priv_install_files("M", boom, None, '{"v":"new"}')
        with open(os.path.join(self.models, "M", "model_config.json")) as f:
            self.assertIn("old", f.read())          # rollback intact
        leftovers = [d for d in os.listdir(self.models) if d.startswith(".")]
        self.assertEqual(leftovers, [], "no temp dirs left behind")


class TestProxyStrictness(unittest.TestCase):
    base = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}

    def test_unknown_named_tool_is_rejected(self):
        with self.assertRaises(px.BadRequest):
            rw(dict(self.base, tools=[TOOL_A],
                    tool_choice={"type": "function",
                                 "function": {"name": "ghost"}}))

    def test_max_tokens_must_be_positive_int(self):
        for bad in ("100", 1.5, True, 0, -5):
            with self.subTest(bad=bad), self.assertRaises(px.BadRequest):
                rw(dict(self.base, tools=[TOOL_A], tool_choice="required",
                        max_tokens=bad))


class TestSwapFailures(unittest.TestCase):
    """Every step of the directory swap must be survivable."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.models = os.path.join(self.tmp, "models")
        os.makedirs(self.models)
        self._dir, self._root, self._rename = (ug.MODELS_DIR, ug.is_root,
                                               ug._priv_rename)
        ug.MODELS_DIR = self.models
        ug.is_root = lambda: True
        self._staging = ug._staging_dir
        ug._staging_dir = lambda: self.tmp
        self.src = os.path.join(self.tmp, "src.gguf")
        with open(self.src, "wb") as f:
            f.write(b"GGUF" + b"\x00" * 32)
        ug.priv_install_files("M", self.src, None, '{"v":"old"}')

    def tearDown(self):
        ug.MODELS_DIR, ug.is_root, ug._priv_rename = (self._dir, self._root,
                                                      self._rename)
        ug._staging_dir = self._staging
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _config(self):
        with open(os.path.join(self.models, "M", "model_config.json")) as f:
            return f.read()

    def test_failure_activating_restores_old(self):
        real = ug._priv_rename

        def flaky(src, dst):               # break only the activation step
            if src == ".M.new":
                raise OSError("simulated swap failure")
            return real(src, dst)

        ug._priv_rename = flaky
        with self.assertRaises(ug.PrivError):
            ug.priv_install_files("M", self.src, None, '{"v":"new"}')
        ug._priv_rename = real
        self.assertIn("old", self._config(), "previous install must be back")
        self.assertEqual([d for d in os.listdir(self.models)
                          if d.startswith(".")], [])

    def test_failure_parking_old_keeps_everything(self):
        real = ug._priv_rename

        def flaky(src, dst):               # break the "park the old one" step
            if dst == ".M.old":
                raise OSError("simulated park failure")
            return real(src, dst)

        ug._priv_rename = flaky
        with self.assertRaises(ug.PrivError):
            ug.priv_install_files("M", self.src, None, '{"v":"new"}')
        ug._priv_rename = real
        self.assertIn("old", self._config())

    def test_recovers_crash_between_renames(self):
        # simulate: M was parked as .M.old, then the process died
        os.rename(os.path.join(self.models, "M"),
                  os.path.join(self.models, ".M.old"))
        os.makedirs(os.path.join(self.models, ".M.new"))
        ug.priv_install_files("M", self.src, None, '{"v":"recovered"}')
        self.assertIn("recovered", self._config())
        self.assertEqual([d for d in os.listdir(self.models)
                          if d.startswith(".")], [])

    def test_concurrent_install_is_refused(self):
        import fcntl
        lock_path = os.path.join(ug._staging_dir(), ".M.install.lock")
        holder = open(lock_path, "w")
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with self.assertRaises(SystemExit):
                ug.priv_install_files("M", self.src, None, '{"v":"x"}')
        finally:
            fcntl.flock(holder, fcntl.LOCK_UN)
            holder.close()
        self.assertIn("old", self._config())   # untouched

    def test_recovers_after_crash_that_left_a_lock_file(self):
        """A real crash leaves the lock file behind. The kernel drops the
        lock itself, so the next run must recover instead of refusing."""
        os.rename(os.path.join(self.models, "M"),
                  os.path.join(self.models, ".M.old"))
        os.makedirs(os.path.join(self.models, ".M.new"))
        open(os.path.join(ug._staging_dir(), ".M.install.lock"), "w").close()
        ug.priv_install_files("M", self.src, None, '{"v":"recovered"}')
        self.assertIn("recovered", self._config())
        self.assertEqual([d for d in os.listdir(self.models)
                          if d.startswith(".")], [])

    def test_activation_succeeds_even_if_old_cleanup_fails(self):
        real = ug._priv_rmtree

        def flaky(dirname):
            if dirname == ".M.old":
                raise ug.PrivError("simulated cleanup failure")
            return real(dirname)

        ug._priv_rmtree = flaky
        try:
            ug.priv_install_files("M", self.src, None, '{"v":"new"}')
        finally:
            ug._priv_rmtree = real
        self.assertIn("new", self._config(), "install must count as done")

    def test_data_files_are_not_executable(self):
        mode = os.stat(os.path.join(self.models, "M", "M.gguf")).st_mode
        self.assertEqual(mode & 0o777, 0o644)


class TestDockerPath(unittest.TestCase):
    """The non-root path builds argv command lists — verify them without
    actually running docker."""

    def setUp(self):
        self._root, self._docker = ug.is_root, ug._docker
        ug.is_root = lambda: False
        self.calls = []
        # _priv_populate writes a temp config next to the source file
        self.tmp = tempfile.mkdtemp()
        self.src = os.path.join(self.tmp, "model.gguf")
        with open(self.src, "wb") as f:
            f.write(b"GGUF")

    def tearDown(self):
        ug.is_root, ug._docker = self._root, self._docker
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _fake(self, fail_on=None):
        class R:
            def __init__(self, rc):
                self.returncode, self.stderr, self.stdout = rc, "boom", ""

        def fake(argv, mounts, image=None, stdin=None):
            self.calls.append(argv)
            return R(1 if (fail_on and argv[0] == fail_on) else 0)
        return fake

    def test_no_shell_is_ever_used(self):
        ug._docker = self._fake()
        ug._priv_populate(".M.new", "M", self.src, None, "{}")
        for argv in self.calls:
            self.assertNotIn(argv[0], ("sh", "bash", "-c"),
                             f"privileged step must not use a shell: {argv}")

    def test_failed_copy_raises(self):
        ug._docker = self._fake(fail_on="cp")
        with self.assertRaises(ug.PrivError):
            ug._priv_populate(".M.new", "M", self.src, None, "{}")

    def test_failed_rename_raises(self):
        ug._docker = self._fake(fail_on="mv")
        with self.assertRaises(ug.PrivError):
            ug._priv_rename(".M.new", "M")

    def test_failed_cleanup_raises(self):
        ug._docker = self._fake(fail_on="rm")
        with self.assertRaises(ug.PrivError):
            ug._priv_rmtree(".M.old")


if __name__ == "__main__":
    unittest.main()
