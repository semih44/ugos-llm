#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for runtime profiles: companion files, the runtime marker,
the dispatcher script, and runtime-aware config linting.

Run: python3 -m unittest discover -s tests -v
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_units import ug  # noqa: E402  (reuse the loaded module)


class TestArchRulesPerRuntime(unittest.TestCase):
    def test_vendor_default_is_unchanged(self):
        self.assertEqual(ug.classify_arch("qwen35moe")[0], "BROKEN")
        self.assertEqual(ug.classify_arch("gemma4")[0], "UNSUPPORTED")

    def test_upstream_supports_gemma4_and_moe(self):
        rt = "upstream-b10143"
        self.assertEqual(ug.classify_arch("gemma4", runtime=rt)[0], "TESTED")
        self.assertEqual(ug.classify_arch("gemma4moe", runtime=rt)[0], "TESTED")
        self.assertEqual(ug.classify_arch("qwen35moe", runtime=rt)[0], "TESTED")

    def test_upstream_moe_is_flagged_slow_not_broken(self):
        verdict, why = ug.classify_arch("somethingmoe", runtime="upstream-b10143")
        self.assertNotEqual(verdict, "BROKEN")
        self.assertIn("slow", why.lower())

    def test_upstream_unknown_stays_unknown(self):
        self.assertEqual(
            ug.classify_arch("weird-arch", runtime="upstream-b10143")[0],
            "UNKNOWN")


class TestExtraArgs(unittest.TestCase):
    def test_vendor_args_keep_ub_512_and_underscore_mmap(self):
        ea = ug.build_extra_args(16384)
        self.assertIn("512", ea)
        self.assertIn("--no_mmap", ea)      # UGREEN's b8413 spelling
        self.assertNotIn("--jinja", ea)
        self.assertEqual(ea[ea.index("-c") + 1], "16384")

    def test_upstream_args_use_4096_and_dash_mmap(self):
        ea = ug.build_extra_args(8192, runtime="upstream-b10143")
        self.assertIn("4096", ea)
        self.assertNotIn("512", ea)
        self.assertIn("--no-mmap", ea)      # upstream parser rejects --no_mmap
        self.assertNotIn("--no_mmap", ea)
        self.assertIn("--jinja", ea)
        self.assertEqual(ea[ea.index("-c") + 1], "8192")

    def test_draft_args_reference_installed_draft_absolutely(self):
        ea = ug.build_extra_args(8192, runtime="upstream-b10143",
                                 draft_name="Gemma4-26B")
        i = ea.index("--spec-draft-model")
        self.assertEqual(ea[i + 1],
                         os.path.join(ug.MODELS_DIR, "Gemma4-26B", "draft.gguf"))
        self.assertIn("draft-mtp", ea)
        self.assertIn("--spec-draft-n-max", ea)

    def test_draft_without_upstream_runtime_is_refused(self):
        with self.assertRaises(SystemExit):
            ug.build_extra_args(8192, draft_name="M")


class TestCompanionFiles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.models = os.path.join(self.tmp, "models")
        os.makedirs(self.models)
        self._dir, self._root = ug.MODELS_DIR, ug.is_root
        ug.MODELS_DIR = self.models
        ug.is_root = lambda: True
        self._staging = ug._staging_dir
        ug._staging_dir = lambda: self.tmp
        self.src = os.path.join(self.tmp, "src.gguf")
        self.draft = os.path.join(self.tmp, "mtp.gguf")
        for p in (self.src, self.draft):
            with open(p, "wb") as f:
                f.write(b"GGUF" + b"\x00" * 32)

    def tearDown(self):
        ug.MODELS_DIR, ug.is_root = self._dir, self._root
        ug._staging_dir = self._staging
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_extra_files_and_marker_are_installed(self):
        ug.priv_install_files("M", self.src, None, "{}",
                              extra_files=[(self.draft, "draft.gguf")],
                              runtime="upstream-b10143")
        base = os.path.join(self.models, "M")
        self.assertTrue(os.path.isfile(os.path.join(base, "draft.gguf")))
        marker = os.path.join(base, ug.RUNTIME_MARKER)
        with open(marker) as f:
            self.assertEqual(f.read().strip(), "upstream-b10143")

    def test_no_marker_without_runtime(self):
        ug.priv_install_files("M", self.src, None, "{}")
        self.assertFalse(os.path.exists(
            os.path.join(self.models, "M", ug.RUNTIME_MARKER)))

    def test_reinstall_without_runtime_drops_stale_marker(self):
        ug.priv_install_files("M", self.src, None, "{}",
                              runtime="upstream-b10143")
        ug.priv_install_files("M", self.src, None, "{}")
        self.assertFalse(os.path.exists(
            os.path.join(self.models, "M", ug.RUNTIME_MARKER)))

    def test_companions_are_not_executable(self):
        ug.priv_install_files("M", self.src, None, "{}",
                              extra_files=[(self.draft, "draft.gguf")],
                              runtime="upstream-b10143")
        base = os.path.join(self.models, "M")
        for fn in ("draft.gguf", ug.RUNTIME_MARKER):
            mode = os.stat(os.path.join(base, fn)).st_mode
            self.assertEqual(mode & 0o777, 0o644, fn)


class TestCompanionDockerPath(unittest.TestCase):
    """Non-root: companion files and marker must be copied via argv lists,
    never through a shell."""

    def setUp(self):
        self._root, self._docker = ug.is_root, ug._docker
        ug.is_root = lambda: False
        self.calls = []
        self.tmp = tempfile.mkdtemp()
        self.src = os.path.join(self.tmp, "model.gguf")
        self.draft = os.path.join(self.tmp, "mtp.gguf")
        for p in (self.src, self.draft):
            with open(p, "wb") as f:
                f.write(b"GGUF")

        class R:
            returncode, stderr, stdout = 0, "", ""

        ug._docker = lambda argv, mounts, image=None, stdin=None: (
            self.calls.append(argv) or R())

    def tearDown(self):
        ug.is_root, ug._docker = self._root, self._docker
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_extra_files_copied_without_shell(self):
        ug._priv_populate(".M.new", "M", self.src, None, "{}",
                          extra_files=[(self.draft, "draft.gguf")],
                          runtime="upstream-b10143")
        flat = [a for argv in self.calls for a in argv]
        self.assertIn("/m/.M.new/draft.gguf", flat)
        self.assertIn(f"/m/.M.new/{ug.RUNTIME_MARKER}", flat)
        for argv in self.calls:
            self.assertNotIn(argv[0], ("sh", "bash", "-c"))


class TestDispatcherScript(unittest.TestCase):
    """The dispatcher runs as root in the gateway's spawn path — its routing
    logic is exercised for real (bash), not just string-matched."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.bundle = os.path.join(self.tmp, "bundle")
        self.runtimes = os.path.join(self.tmp, "runtimes")
        self.models = os.path.join(self.tmp, "models")
        rt = os.path.join(self.runtimes, "upstream-b10143")
        for d in (self.bundle, rt, os.path.join(self.models, "M")):
            os.makedirs(d)

        def stub(path, tag):
            with open(path, "w") as f:
                f.write(f'#!/bin/bash\necho {tag} "$@"\n')
            os.chmod(path, 0o755)

        stub(os.path.join(self.bundle, "llama-server.ugreen-orig"), "VENDOR")
        stub(os.path.join(rt, "llama-server-wrapper"), "UPSTREAM")

        self.disp = os.path.join(self.bundle, "llama-server")
        with open(self.disp, "w") as f:
            f.write(ug.dispatcher_script(self.runtimes))
        os.chmod(self.disp, 0o755)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, *argv):
        return subprocess.run(["bash", self.disp, *argv],
                              capture_output=True, text=True)

    def test_script_is_tagged_for_detection(self):
        with open(self.disp) as f:
            self.assertIn(ug.DISPATCHER_TAG, f.read())

    def test_model_without_marker_goes_to_vendor(self):
        model = os.path.join(self.models, "M", "M.gguf")
        r = self._run("-m", model, "-c", "4096")
        self.assertTrue(r.stdout.startswith("VENDOR"), r.stdout + r.stderr)
        self.assertIn("-c 4096", r.stdout)

    def test_model_with_marker_goes_to_upstream(self):
        with open(os.path.join(self.models, "M", ug.RUNTIME_MARKER), "w") as f:
            f.write("upstream-b10143\n")
        r = self._run("-m", os.path.join(self.models, "M", "M.gguf"))
        self.assertTrue(r.stdout.startswith("UPSTREAM"), r.stdout + r.stderr)

    def test_missing_runtime_fails_fast_not_vendor(self):
        """A marker pointing at an absent runtime must not silently fall back
        to the vendor build (which would choke on upstream-only flags)."""
        with open(os.path.join(self.models, "M", ug.RUNTIME_MARKER), "w") as f:
            f.write("upstream-b99999\n")
        r = self._run("-m", os.path.join(self.models, "M", "M.gguf"))
        self.assertNotEqual(r.returncode, 0)
        self.assertNotIn("VENDOR", r.stdout)
        self.assertIn("upstream-b99999", r.stderr)

    def test_no_model_arg_goes_to_vendor(self):
        r = self._run("--help")
        self.assertTrue(r.stdout.startswith("VENDOR"), r.stdout + r.stderr)


class TestRuntimeWrapper(unittest.TestCase):
    """The wrapper encodes hard-won driver findings — pin them."""

    def test_uses_vendor_opencl_userspace(self):
        w = ug.runtime_wrapper_script()
        self.assertIn(f"{ug.VENDOR_BUNDLE_DIR}/libigdrcl.so", w)
        self.assertIn("opencl:gpu", w)

    def test_vendor_dir_comes_after_runtime_dir(self):
        w = ug.runtime_wrapper_script()
        self.assertIn(f'"$DIR:{ug.VENDOR_BUNDLE_DIR}', w)

    def test_no_backend_path_and_no_level_zero(self):
        w = ug.runtime_wrapper_script()
        self.assertNotIn("GGML_BACKEND_PATH", w)   # llama.cpp treats it as a file
        # host L0 too old, glibc-compatible L0 segfaults — OpenCL is the way
        self.assertNotIn("ZE_ENABLE_ALT_DRIVERS", w)


class TestDispatcherState(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._bundle = ug.VENDOR_BUNDLE_DIR
        ug.VENDOR_BUNDLE_DIR = self.tmp

    def tearDown(self):
        ug.VENDOR_BUNDLE_DIR = self._bundle
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_missing(self):
        self.assertEqual(ug.dispatcher_state()[0], "missing")

    def test_vendor(self):
        with open(os.path.join(self.tmp, "llama-server"), "w") as f:
            f.write("#!/bin/bash\nexec ./.llama-server \"$@\"\n")
        self.assertEqual(ug.dispatcher_state()[0], "vendor")

    def test_ours(self):
        with open(os.path.join(self.tmp, "llama-server"), "w") as f:
            f.write(ug.dispatcher_script("/x"))
        self.assertEqual(ug.dispatcher_state()[0], "ours")


class TestRuntimeAwareLint(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _lint(self, extra_args, runtime=None, with_draft_file=False):
        base = os.path.join(self.tmp, "M")
        os.makedirs(base, exist_ok=True)
        if with_draft_file:
            open(os.path.join(base, "draft.gguf"), "w").close()
        cfg = {"extra_args": extra_args}
        return ug.lint_model_config("M", cfg, runtime, model_dir=base)

    def test_vendor_4096_is_flagged(self):
        warns = self._lint(["-ub", "4096", "-b", "4096"])
        self.assertTrue(any("4096" in w for w in warns))

    def test_upstream_4096_is_fine(self):
        warns = self._lint(["-ub", "4096", "-b", "4096"],
                           runtime="upstream-b10143")
        self.assertFalse(any("4096" in w for w in warns))

    def test_upstream_underscore_mmap_is_flagged(self):
        warns = self._lint(["--no_mmap"], runtime="upstream-b10143")
        self.assertTrue(any("no_mmap" in w for w in warns))

    def test_vendor_underscore_mmap_is_fine(self):
        self.assertEqual(self._lint(["--no_mmap"]), [])

    def test_missing_draft_file_is_flagged(self):
        warns = self._lint(["--spec-draft-model", "/nope/draft.gguf"],
                           runtime="upstream-b10143")
        self.assertTrue(any("draft" in w.lower() for w in warns))

    def test_present_draft_file_is_fine(self):
        base = os.path.join(self.tmp, "M")
        os.makedirs(base, exist_ok=True)
        warns = self._lint(["--spec-draft-model",
                            os.path.join(base, "draft.gguf")],
                           runtime="upstream-b10143", with_draft_file=True)
        self.assertFalse(any("draft" in w.lower() for w in warns))

    def test_marker_without_dispatcher_is_flagged(self):
        warns = ug.lint_runtime_setup({"M": "upstream-b10143"},
                                      dispatcher="vendor",
                                      runtimes_present=["upstream-b10143"])
        self.assertTrue(any("dispatcher" in w.lower() for w in warns))

    def test_marker_without_runtime_dir_is_flagged(self):
        warns = ug.lint_runtime_setup({"M": "upstream-b10143"},
                                      dispatcher="ours",
                                      runtimes_present=[])
        self.assertTrue(any("upstream-b10143" in w for w in warns))

    def test_clean_setup_has_no_warnings(self):
        warns = ug.lint_runtime_setup({"M": "upstream-b10143"},
                                      dispatcher="ours",
                                      runtimes_present=["upstream-b10143"])
        self.assertEqual(warns, [])


if __name__ == "__main__":
    unittest.main()
