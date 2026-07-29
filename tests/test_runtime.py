#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for runtime profiles: companion files, the runtime marker,
the dispatcher script, and runtime-aware config linting.

Run: python3 -m unittest discover -s tests -v
"""
import json
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

    def test_upstream_keeps_moe_speed_caveat_even_when_tested(self):
        # qwen35moe is verified AND slow — the verdict must not swallow the
        # warning that makes it actionable
        _, why = ug.classify_arch("qwen35moe", runtime="upstream-b10143")
        self.assertIn("slow", why.lower())
        self.assertIn("--draft", why)

    def test_upstream_inherits_vendor_architectures_as_expected(self):
        # a newer llama.cpp is an architectural superset of b8413, so nothing
        # the vendor build loads may come back as UNKNOWN upstream
        rt = "upstream-b10143"
        for arch in ("llama", "gemma2", "gemma3", "mistral", "qwen2",
                     "qwen3", "phi3"):
            with self.subTest(arch=arch):
                verdict, why = ug.classify_arch(arch, runtime=rt)
                self.assertEqual(verdict, "EXPECTED", why)

    def test_qwen35_dense_is_tested_upstream(self):
        # re-verified end-to-end on b10143 (29 Jul 2026): chat, long-prompt,
        # native tool calls and vision all passed through the gateway
        self.assertEqual(
            ug.classify_arch("qwen3_5", runtime="upstream-b10143")[0],
            "TESTED")

    def test_upstream_moe_is_flagged_slow_not_broken(self):
        verdict, why = ug.classify_arch("somethingmoe", runtime="upstream-b10143")
        self.assertNotEqual(verdict, "BROKEN")
        self.assertIn("slow", why.lower())

    def test_upstream_unknown_stays_unknown(self):
        self.assertEqual(
            ug.classify_arch("weird-arch", runtime="upstream-b10143")[0],
            "UNKNOWN")

    def test_unverified_upstream_runtime_is_not_tested(self):
        # a runtime is not capable just because its name says "upstream"
        verdict, why = ug.classify_arch("gemma4", runtime="upstream-old")
        self.assertEqual(verdict, "EXPECTED")
        self.assertIn("unverified", why)


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

    def test_system_icds_are_blocked(self):
        # /etc/OpenCL/vendors would add UGOS' old driver as a second device
        # and trip ggml's crashing multi-GPU peer-access path
        w = ug.runtime_wrapper_script()
        self.assertIn('OCL_ICD_VENDORS="$DIR/.no-system-icds"', w)


class TestSidecarFilter(unittest.TestCase):
    def test_sidecars_and_projectors_are_not_main_models(self):
        files = [("MTP/mtp-gemma-4-26B-A4B-it-Q4_0.gguf", 1),
                 ("mtp-gemma-4-26B-A4B-it.gguf", 1),
                 ("eagle-head.gguf", 2),
                 ("draft.gguf", 2),
                 ("mmproj-F16.gguf", 5),
                 ("gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf", 100),
                 ("README.md", 1)]
        mains = ug.main_model_ggufs(files)
        self.assertEqual([n for n, _ in mains],
                         ["gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"])

    def test_models_with_sidecar_like_substrings_survive(self):
        # "draft"/"mtp" must only match as a name prefix, not mid-word
        files = [("Landrafting-7B-Q4_K_M.gguf", 5)]
        self.assertEqual(len(ug.main_model_ggufs(files)), 1)

    def test_versioned_sidecars_are_filtered(self):
        # llama.cpp ships eagle3- (and future eagleN-) heads
        files = [("eagle3-model-Q4_0.gguf", 1), ("eagle4_head.gguf", 1),
                 ("main-Q4_K_M.gguf", 100)]
        self.assertEqual([n for n, _ in ug.main_model_ggufs(files)],
                         ["main-Q4_K_M.gguf"])


class TestThinkingFlag(unittest.TestCase):
    def test_off_sets_template_kwargs(self):
        ea = ug.build_extra_args(8192, runtime="upstream-b10143",
                                 thinking="off")
        i = ea.index("--chat-template-kwargs")
        self.assertEqual(json.loads(ea[i + 1]), {"enable_thinking": False})

    def test_on_sets_template_kwargs(self):
        ea = ug.build_extra_args(8192, runtime="upstream-b10143",
                                 thinking="on")
        i = ea.index("--chat-template-kwargs")
        self.assertEqual(json.loads(ea[i + 1]), {"enable_thinking": True})

    def test_auto_adds_nothing(self):
        ea = ug.build_extra_args(8192, runtime="upstream-b10143",
                                 thinking=None)
        self.assertNotIn("--chat-template-kwargs", ea)

    def test_thinking_without_upstream_is_refused(self):
        with self.assertRaises(SystemExit):
            ug.build_extra_args(8192, thinking="off")


class TestRuntimeDeployAtomic(unittest.TestCase):
    """The working runtime must never be deleted before its replacement
    is in place — same guarantees as the model installer."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._runtimes, self._root = ug.RUNTIMES_DIR, ug.is_root
        self._staging, self._rename = ug._staging_dir, ug._rt_rename
        ug.RUNTIMES_DIR = os.path.join(self.tmp, "runtimes")
        ug.is_root = lambda: True
        ug._staging_dir = lambda: self.tmp
        self.src = os.path.join(self.tmp, "src")
        os.makedirs(self.src)
        for fn, content in (("llama-server", "#!/bin/sh\necho v1\n"),
                            ("BUILD_INFO", "test build v1\n")):
            with open(os.path.join(self.src, fn), "w") as f:
                f.write(content)

    def tearDown(self):
        ug.RUNTIMES_DIR, ug.is_root = self._runtimes, self._root
        ug._staging_dir, ug._rt_rename = self._staging, self._rename
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _target(self):
        return os.path.join(ug.RUNTIMES_DIR, "upstream-b1")

    def _build_info(self):
        with open(os.path.join(self._target(), "BUILD_INFO")) as f:
            return f.read()

    def test_deploy_generates_wrapper(self):
        ug.runtime_deploy(self.src, "upstream-b1")
        wp = os.path.join(self._target(), "llama-server-wrapper")
        self.assertTrue(os.access(wp, os.X_OK))

    def test_force_replace_leaves_no_leftovers(self):
        ug.runtime_deploy(self.src, "upstream-b1")
        with open(os.path.join(self.src, "BUILD_INFO"), "w") as f:
            f.write("test build v2\n")
        ug.runtime_deploy(self.src, "upstream-b1", force=True)
        self.assertIn("v2", self._build_info())
        self.assertEqual([d for d in os.listdir(ug.RUNTIMES_DIR)
                          if d.startswith(".")], [])

    def test_failed_activation_restores_previous_runtime(self):
        ug.runtime_deploy(self.src, "upstream-b1")
        real = ug._rt_rename

        def flaky(src, dst):
            if src.endswith(".new") and dst == self._target():
                raise OSError("simulated activation failure")
            return real(src, dst)

        ug._rt_rename = flaky
        with self.assertRaises(SystemExit):
            ug.runtime_deploy(self.src, "upstream-b1", force=True)
        ug._rt_rename = real
        self.assertTrue(os.path.isdir(self._target()),
                        "previous runtime must still be active")
        self.assertIn("v1", self._build_info())

    def test_recovers_interrupted_swap(self):
        ug.runtime_deploy(self.src, "upstream-b1")
        # simulate: parked as .old, then the process died mid-swap
        os.rename(self._target(),
                  os.path.join(ug.RUNTIMES_DIR, ".upstream-b1.old"))
        ug.runtime_deploy(self.src, "upstream-b1", force=True)
        self.assertTrue(os.path.isdir(self._target()))
        self.assertEqual([d for d in os.listdir(ug.RUNTIMES_DIR)
                          if d.startswith(".")], [])


class TestRuntimeDeployDockerPath(unittest.TestCase):
    """Non-root deploy must follow the same park-before-replace order —
    the live runtime is never rm'd before its successor is in place."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._runtimes, self._root = ug.RUNTIMES_DIR, ug.is_root
        self._staging, self._docker = ug._staging_dir, ug._docker
        ug.RUNTIMES_DIR = os.path.join(self.tmp, "runtimes")
        ug.is_root = lambda: False
        ug._staging_dir = lambda: self.tmp
        self.calls = []

        class R:
            returncode, stderr, stdout = 0, "", ""

        ug._docker = lambda argv, mounts, image=None, stdin=None: (
            self.calls.append(argv) or R())
        # existing live runtime, visible to the host-side exists() checks
        os.makedirs(os.path.join(ug.RUNTIMES_DIR, "upstream-b1"))
        self.src = os.path.join(self.tmp, "src")
        os.makedirs(self.src)
        for fn in ("llama-server", "BUILD_INFO"):
            open(os.path.join(self.src, fn), "w").close()

    def tearDown(self):
        ug.RUNTIMES_DIR, ug.is_root = self._runtimes, self._root
        ug._staging_dir, ug._docker = self._staging, self._docker
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_live_runtime_is_parked_not_deleted(self):
        ug.runtime_deploy(self.src, "upstream-b1", force=True)
        live = "/rt/runtimes/upstream-b1"
        park = [i for i, argv in enumerate(self.calls)
                if argv[:2] == ["mv", live]]
        activate = [i for i, argv in enumerate(self.calls)
                    if argv[0] == "mv" and argv[2] == live]
        self.assertTrue(park and activate and park[0] < activate[0],
                        f"park must precede activation: {self.calls}")
        for argv in self.calls[:activate[0]]:
            self.assertNotEqual(argv[:3], ["rm", "-rf", live],
                                "live runtime must never be rm'd")

    def test_staging_file_is_per_runtime(self):
        ug.runtime_deploy(self.src, "upstream-b1", force=True)
        cp = [argv for argv in self.calls
              if argv[0] == "cp" and "runtime-wrapper" in argv[1]]
        self.assertTrue(cp and "upstream-b1" in cp[0][1],
                        f"staging file must carry the runtime name: {cp}")


class TestEnableBackupLifecycle(unittest.TestCase):
    """A firmware update replaces the dispatcher with a NEW vendor wrapper;
    re-enabling must back up the new one, not resurrect the old."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._bundle, self._root = ug.VENDOR_BUNDLE_DIR, ug.is_root
        self._staging = ug._staging_dir
        ug.VENDOR_BUNDLE_DIR = self.tmp
        ug.is_root = lambda: True
        ug._staging_dir = lambda: self.tmp
        self.wrapper = os.path.join(self.tmp, "llama-server")
        self.backup = os.path.join(self.tmp, "llama-server.ugreen-orig")
        with open(self.wrapper, "w") as f:
            f.write("VENDOR-V1\n")

    def tearDown(self):
        ug.VENDOR_BUNDLE_DIR, ug.is_root = self._bundle, self._root
        ug._staging_dir = self._staging
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _read(self, path):
        with open(path) as f:
            return f.read()

    def test_enable_backs_up_and_installs_dispatcher(self):
        ug.runtime_enable()
        self.assertEqual(self._read(self.backup), "VENDOR-V1\n")
        self.assertIn(ug.DISPATCHER_TAG, self._read(self.wrapper))

    def test_firmware_update_refreshes_stale_backup(self):
        ug.runtime_enable()
        with open(self.wrapper, "w") as f:      # firmware update: new vendor
            f.write("VENDOR-V2\n")
        ug.runtime_enable()
        self.assertEqual(self._read(self.backup), "VENDOR-V2\n")
        self.assertEqual(self._read(self.backup + ".prev"), "VENDOR-V1\n")
        self.assertIn(ug.DISPATCHER_TAG, self._read(self.wrapper))

    def test_disable_restores_current_vendor_wrapper(self):
        ug.runtime_enable()
        with open(self.wrapper, "w") as f:
            f.write("VENDOR-V2\n")
        ug.runtime_enable()
        ug.runtime_disable()
        self.assertEqual(self._read(self.wrapper), "VENDOR-V2\n")


class TestHeaderProbeRetry(unittest.TestCase):
    """qat GGUFs carry >4 MB of metadata — the probe must widen its byte
    range on a truncated header instead of dying."""

    def setUp(self):
        self._req = ug.hf_request
        from test_units import gguf_bytes
        self.full = gguf_bytes("gemma4moe", "g")
        self.ranges = []

    def tearDown(self):
        ug.hf_request = self._req

    def _fake(self, truncate_below_mb):
        import contextlib
        import io as _io

        def fake(url, byte_range=None, timeout=60):
            self.ranges.append(byte_range)
            end = int(byte_range.split("-")[1])
            data = self.full if end >= truncate_below_mb * 1024 * 1024 - 1 \
                else self.full[:8]
            return contextlib.closing(_io.BytesIO(data))
        return fake

    def test_retries_with_larger_range(self):
        ug.hf_request = self._fake(truncate_below_mb=16)
        meta = ug.hf_probe_arch("o/r", "big.gguf")
        self.assertEqual(meta["general.architecture"], "gemma4moe")
        self.assertGreater(len(self.ranges), 1)

    def test_single_fetch_when_header_fits(self):
        ug.hf_request = self._fake(truncate_below_mb=4)
        ug.hf_probe_arch("o/r", "small.gguf")
        self.assertEqual(len(self.ranges), 1)

    def test_dies_when_even_the_largest_range_truncates(self):
        ug.hf_request = self._fake(truncate_below_mb=1024)
        with self.assertRaises(SystemExit):
            ug.hf_probe_arch("o/r", "absurd.gguf")


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
