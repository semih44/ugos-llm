#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ugos-llm — run your own LLMs on UGREEN NASync AI NAS devices.

Single-file CLI, Python 3 stdlib only. Run it ON the NAS (via SSH).

  check    pre-flight a Hugging Face GGUF repo (no download)
  install  download + configure + register a model
  list     installed models, catalog status, loaded state
  test     acceptance suite: chat / long-prompt stability / tools / vision
  doctor   health checks + lint for known pitfalls
  ui       add/remove the Model Manager UI card (catalog DB, with backup)
  runtime  manage alternative llama.cpp runtimes (dispatcher, deploy, status)
  remove   delete an installed model

Privileges: check/list/test/doctor run as any user. install/remove/ui write to
root-owned paths: run them with sudo, or as a user in the `docker` group (file
operations are then routed through a throwaway container). No shell is ever
used for privileged operations, and model names are strictly validated.

Tested on: UGOS Pro, Model Manager 1.17.0.0055, infer_gateway 1.0.0.0004,
llama.cpp b8413 (SYCL). See docs/known-bugs.md before trusting anything.

License: MIT.
"""

import argparse
import contextlib
import fcntl
import io
import json
import os
import re
import shutil
import sqlite3
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request

# --------------------------------------------------------------------------
# Constants (verified on UGOS Pro / iDX6011, July 2026)
# --------------------------------------------------------------------------

MODELS_DIR = "/opt/ugreen/ai/models"
GATEWAY = "http://127.0.0.1:62891"
AICONSOLE_DB = "/volume1/@appstore/com.ugreen.aiconsole/db/aiconsole.db"

# Versioned images for privileged helpers. Version tags are not immutable
# (only digests are) — if you want bit-for-bit reproducibility, override
# these with digest references via the environment.
IMG_BUSYBOX = os.environ.get("UGOS_LLM_IMG_BUSYBOX",
                             "docker.io/library/alpine:3.20")
IMG_PYTHON = os.environ.get("UGOS_LLM_IMG_PYTHON",
                            "docker.io/library/python:3.12-alpine")

# Catalog rows created by this tool are marked with this release_id so that
# `remove`/`ui remove` can never delete a vendor row.
OUR_RELEASE_ID = 9000

# The single most important deviation from UGREEN's own defaults:
# their `-ub 4096 -b 4096` silently corrupts long-prompt inference on the
# shipped SYCL build (llama.cpp b8413). 512 is correct AND ~2x faster.
SAFE_EXTRA_ARGS = ["-ngl", "999", "-t", "10", "-e", "-lv", "3",
                   "--no-warmup", "--no_mmap", "-fa", "off", "-np", "1",
                   "-ub", "512", "-b", "512"]

# Upstream llama.cpp runtimes (installed via `runtime deploy`) fix both the
# ub-4096 numerical bug and MoE support, so their default config differs.
# Mind the spelling: UGREEN's patched b8413 accepts `--no_mmap`, the upstream
# parser only `--no-mmap` — the wrong variant kills the server at spawn.
UPSTREAM_EXTRA_ARGS = ["-ngl", "999", "-t", "10", "-e", "-lv", "3",
                       "--no-warmup", "--no-mmap", "-fa", "off", "-np", "1",
                       "-ub", "4096", "-b", "4096", "--jinja"]

# Where alternative runtimes live and how models opt in. The dispatcher
# replaces UGREEN's 258-byte llama-server wrapper; the original is kept
# next to it as llama-server.ugreen-orig and remains the default path.
RUNTIMES_DIR = "/volume1/@aiconsole/llamacppSycl/ugos-llm-runtimes"
VENDOR_BUNDLE_DIR = "/volume1/@aiconsole/llamacppSycl/llama-sycl/llama-sycl"
RUNTIME_MARKER = ".ugos-llm-runtime"
DISPATCHER_TAG = "# ugos-llm dispatcher"

ARCH_TESTED_OK = {"qwen3_5", "qwen35", "qwen3"}
ARCH_EXPECTED_OK = {"llama", "qwen2", "qwen2vl", "gemma", "gemma2",
                    "gemma3", "gemma3n", "mistral", "phi3"}
ARCH_BROKEN_MOE = re.compile(r"moe", re.IGNORECASE)
ARCH_KNOWN_MISSING = {"gemma4", "qwen3_6", "qwen36", "glm4moe"}

# Verified in the b10143 container on an iDX6011 (July 2026), see
# docs/known-bugs.md section 6.
UPSTREAM_ARCH_TESTED = {"qwen3_5", "qwen35", "qwen3", "qwen35moe",
                        "gemma4", "gemma4moe", "gemma4_assistant"}

STATUS = {1: "not installed", 3: "disabled", 8: "active"}

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
SPLIT_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)
MAX_GGUF_STRING = 8 << 20        # sanity bound while parsing headers


def log(msg=""):
    print(msg, flush=True)


def die(msg, code=1):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def validate_name(name):
    """Model directory names end up in filesystem paths and SQL. Be strict."""
    if not name or not NAME_RE.match(name) or name in (".", ".."):
        die(f"invalid model name {name!r}: allowed are letters, digits, "
            "dot, dash and underscore (max 64 chars, must start "
            "alphanumeric).")
    return name


# --------------------------------------------------------------------------
# Privileged operations — no shell, ever. Root does it natively, non-root
# routes through a pinned container with explicit argv.
# --------------------------------------------------------------------------

def is_root():
    return os.geteuid() == 0


def have_docker():
    return shutil.which("docker") is not None and subprocess.run(
        ["docker", "ps", "-q"], capture_output=True).returncode == 0


def _need_privileges():
    if not is_root() and not have_docker():
        die("This action needs root. Re-run with sudo, or add your user to "
            "the docker group (`sudo usermod -aG docker $USER`, then log in "
            "again).")


def _docker(argv, mounts, image=IMG_BUSYBOX, stdin=None):
    cmd = ["docker", "run", "--rm", "-i"]
    for host, cont in mounts.items():
        cmd += ["-v", f"{host}:{cont}"]
    cmd += [image] + argv
    return subprocess.run(cmd, input=stdin, capture_output=True, text=True)


class PrivError(RuntimeError):
    """A privileged step failed — caller decides how to recover."""


def _priv_rmtree(dirname):
    """Delete <MODELS_DIR>/<dirname>. Missing is fine, anything else is an
    error: silently leaving multi-GB leftovers behind would break both the
    disk-space and the rollback guarantees."""
    path = os.path.join(MODELS_DIR, dirname)
    if is_root():
        if not os.path.exists(path):
            return
        try:
            shutil.rmtree(path)
        except OSError as e:
            raise PrivError(f"could not remove {path}: {e}") from e
        return
    r = _docker(["rm", "-rf", f"/m/{dirname}"], {MODELS_DIR: "/m"})
    if r.returncode != 0:
        raise PrivError(f"could not remove {path}: "
                        f"{r.stderr.strip()[:200]}")


def _priv_rename(src_name, dst_name):
    """Rename inside MODELS_DIR (same filesystem => atomic)."""
    if is_root():
        try:
            os.rename(os.path.join(MODELS_DIR, src_name),
                      os.path.join(MODELS_DIR, dst_name))
        except OSError as e:
            raise PrivError(f"rename {src_name} -> {dst_name} failed: "
                            f"{e}") from e
        return
    r = _docker(["mv", f"/m/{src_name}", f"/m/{dst_name}"],
                {MODELS_DIR: "/m"})
    if r.returncode != 0:
        raise PrivError(f"rename {src_name} -> {dst_name} failed: "
                        f"{r.stderr.strip()[:200]}")


@contextlib.contextmanager
def _install_lock(name):
    """Advisory lock so two installs of the same model cannot interleave.

    Deliberately an flock, not a lock directory: the kernel releases it when
    the process dies (crash, power loss, SIGKILL), so a stale lock can never
    block the crash recovery on the next run. The lock file itself is
    allowed to persist; only the held lock matters. It lives in the staging
    directory because that is writable without privileges.
    """
    path = os.path.join(_staging_dir(), f".{name}.install.lock")
    fh = open(path, "w")
    try:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            die(f"another install of {name!r} is already running "
                f"(lock held on {path}). Wait for it to finish and retry.")
        yield
    finally:
        try:
            fcntl.flock(fh, fcntl.LOCK_UN)
        except OSError:
            pass          # never mask the real outcome
        fh.close()


def _recover_interrupted(name, work, old):
    """Repair state left behind by a crash during a previous swap.

    Cases (M = active dir, .M.new = build dir, .M.old = rollback dir):
      M missing, .M.old present  -> the swap died mid-way: restore .M.old
      M present, .M.old present  -> the swap succeeded: drop .M.old
      .M.new present             -> an aborted build: drop it
    """
    have = os.path.exists(os.path.join(MODELS_DIR, name))
    have_old = os.path.exists(os.path.join(MODELS_DIR, old))
    if not have and have_old:
        log(f"recovering interrupted install: restoring {old} -> {name}")
        _priv_rename(old, name)
    elif have and have_old:
        log(f"cleaning up leftover rollback directory {old}")
        _priv_rmtree(old)
    _priv_rmtree(work)


def _check_destname(destname):
    """Companion destinations are our own strings, but stay defensive."""
    if ("/" in destname or destname in (".", "..")
            or destname.startswith("..")):
        raise PrivError(f"unsafe companion file name {destname!r}")
    return destname


def _priv_populate(work_name, name, model_src, mmproj_src, config_json,
                   extra_files=None, runtime=None):
    """Fill <MODELS_DIR>/<work_name>/ with model, projector, config, any
    companion files (e.g. an MTP draft head) and the runtime marker.
    Raises PrivError on any failure (no partial state is ever activated)."""
    extra_files = [(src, _check_destname(dst))
                   for src, dst in (extra_files or [])]
    work_dir = os.path.join(MODELS_DIR, work_name)
    if is_root():
        os.makedirs(work_dir, exist_ok=True)
        shutil.copyfile(model_src, os.path.join(work_dir, f"{name}.gguf"))
        if mmproj_src:
            shutil.copyfile(mmproj_src, os.path.join(work_dir, "mmproj.gguf"))
        for src, dst in extra_files:
            shutil.copyfile(src, os.path.join(work_dir, dst))
        with open(os.path.join(work_dir, "model_config.json"), "w") as f:
            f.write(config_json)
        if runtime:
            with open(os.path.join(work_dir, RUNTIME_MARKER), "w") as f:
                f.write(runtime + "\n")
        os.chmod(work_dir, 0o755)
        for fn in os.listdir(work_dir):
            p = os.path.join(work_dir, fn)
            if os.path.isfile(p):
                os.chmod(p, 0o644)     # data files need no execute bit
        return

    stage = os.path.dirname(os.path.abspath(model_src))
    cfg_path = os.path.join(stage, f".{work_name}.model_config.json")
    with open(cfg_path, "w") as f:
        f.write(config_json)
    marker_path = None
    if runtime:
        marker_path = os.path.join(stage, f".{work_name}.runtime-marker")
        with open(marker_path, "w") as f:
            f.write(runtime + "\n")
    mounts = {MODELS_DIR: "/m", stage: "/s"}
    steps = [["mkdir", "-p", f"/m/{work_name}"],
             ["cp", f"/s/{os.path.basename(model_src)}",
              f"/m/{work_name}/{name}.gguf"]]
    if mmproj_src:
        steps.append(["cp", f"/s/{os.path.basename(mmproj_src)}",
                      f"/m/{work_name}/mmproj.gguf"])
    files = [f"/m/{work_name}/{name}.gguf",
             f"/m/{work_name}/model_config.json"]
    if mmproj_src:
        files.append(f"/m/{work_name}/mmproj.gguf")
    for src, dst in extra_files:
        steps.append(["cp", f"/s/{os.path.basename(src)}",
                      f"/m/{work_name}/{dst}"])
        files.append(f"/m/{work_name}/{dst}")
    if marker_path:
        steps.append(["cp", f"/s/{os.path.basename(marker_path)}",
                      f"/m/{work_name}/{RUNTIME_MARKER}"])
        files.append(f"/m/{work_name}/{RUNTIME_MARKER}")
    steps += [["cp", f"/s/{os.path.basename(cfg_path)}",
               f"/m/{work_name}/model_config.json"],
              ["chmod", "755", f"/m/{work_name}"],
              ["chmod", "644"] + files]
    try:
        for argv in steps:
            r = _docker(argv, mounts)
            if r.returncode != 0:
                raise PrivError(f"step {argv[0]} failed: "
                                f"{r.stderr.strip()[:200]}")
    finally:
        for p in (cfg_path, marker_path):
            if p and os.path.exists(p):
                os.unlink(p)


def priv_install_files(name, model_src, mmproj_src, config_json,
                       extra_files=None, runtime=None):
    """Install atomically: build in a temporary sibling directory, then swap.

    A previously working installation is only removed after the replacement
    is complete, and is restored if the swap fails. The gateway therefore
    never sees a half-populated model directory.
    """
    _need_privileges()
    validate_name(name)
    if runtime:
        validate_name(runtime)
    work = f".{name}.new"
    old = f".{name}.old"

    with _install_lock(name):
        _recover_interrupted(name, work, old)
        try:
            _priv_populate(work, name, model_src, mmproj_src, config_json,
                           extra_files=extra_files, runtime=runtime)
        except Exception as e:      # incl. OSError from the native path
            _priv_rmtree(work)
            raise PrivError(str(e)) from e

        existing = os.path.exists(os.path.join(MODELS_DIR, name))
        if existing:
            try:
                _priv_rename(name, old)    # keep the old one as rollback
            except Exception as e:
                _priv_rmtree(work)         # nothing was activated yet
                raise PrivError(
                    f"could not park the existing installation ({e}); it "
                    "remains active and untouched.") from e
        try:
            _priv_rename(work, name)
        except Exception as e:
            if existing:                   # put the working version back
                try:
                    _priv_rename(old, name)
                except Exception as e2:
                    raise PrivError(
                        f"activation failed ({e}) AND rollback failed ({e2}). "
                        f"Your previous installation is intact at "
                        f"{os.path.join(MODELS_DIR, old)} — rename it back to "
                        f"{name} manually.") from e
            _priv_rmtree(work)
            raise PrivError(str(e)) from e

        # From here on the new model IS active. A failure while dropping the
        # rollback copy must not be reported as a failed install — it only
        # wastes disk until the next run cleans it up.
        try:
            _priv_rmtree(old)
        except PrivError as e:
            log(f"WARNING: the new model is active, but the previous "
                f"installation could not be removed ({e}). It occupies disk "
                f"at {os.path.join(MODELS_DIR, old)} and will be cleaned up "
                "on the next install of this model.")


def priv_remove_dir(name):
    _need_privileges()
    validate_name(name)
    if is_root():
        shutil.rmtree(os.path.join(MODELS_DIR, name))
        return
    r = _docker(["rm", "-rf", f"/m/{name}"], {MODELS_DIR: "/m"})
    if r.returncode != 0:
        die(f"remove failed: {r.stderr.strip()[:300]}")


def db_read(sql, params=()):
    """Read-only catalog access — works for any user."""
    con = sqlite3.connect(f"file:{AICONSOLE_DB}?mode=ro", uri=True, timeout=10)
    try:
        return con.execute(sql, params).fetchall()
    finally:
        con.close()


def db_write(sql, params=()):
    """Parameterised catalog write. Never interpolates values into SQL."""
    _need_privileges()
    if is_root():
        con = sqlite3.connect(AICONSOLE_DB, timeout=10)
        try:
            con.execute(sql, params)
            con.commit()
        finally:
            con.close()
        return
    payload = json.dumps({"sql": sql, "params": list(params)})
    code = (
        "import json,sqlite3,sys;"
        "d=json.load(sys.stdin);"
        "c=sqlite3.connect('/db/aiconsole.db',timeout=10);"
        "c.execute(d['sql'],d['params']);c.commit();c.close()"
    )
    r = _docker(["python3", "-c", code],
                {os.path.dirname(AICONSOLE_DB): "/db"},
                image=IMG_PYTHON, stdin=payload)
    if r.returncode != 0:
        die(f"catalog write failed: {r.stderr.strip()[:300]}")


def db_backup():
    """Consistent backup via SQLite's backup API (copyfile misses the WAL)."""
    # millisecond suffix: two catalog changes in the same second must not
    # overwrite each other's backup
    stamp = f"{time.strftime('%Y%m%d-%H%M%S')}-{int(time.time() * 1000) % 1000:03d}"
    dest = os.path.join(_staging_dir(), f"aiconsole.db.backup-{stamp}")
    src = sqlite3.connect(f"file:{AICONSOLE_DB}?mode=ro", uri=True, timeout=10)
    try:
        out = sqlite3.connect(dest)
        try:
            src.backup(out)
        finally:
            out.close()
    finally:
        src.close()
    return dest


# --------------------------------------------------------------------------
# GGUF header parsing (defensive: the file is remote and untrusted)
# --------------------------------------------------------------------------

_GGUF_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
               10: 8, 11: 8, 12: 8}


class GGUFError(ValueError):
    pass


def parse_gguf_meta(stream, want=("general.architecture", "general.name")):
    """Parse selected string keys from a GGUF header. Raises GGUFError."""
    def rd(n):
        b = stream.read(n)
        if len(b) != n:
            raise GGUFError("truncated header (need a larger byte range?)")
        return b

    def rstr():
        (n,) = struct.unpack("<Q", rd(8))
        if n > MAX_GGUF_STRING:
            raise GGUFError(f"implausible string length {n}")
        return rd(n).decode("utf-8", "replace")

    if rd(4) != b"GGUF":
        raise GGUFError("not a GGUF file (bad magic)")
    rd(4 + 8)                       # version, tensor count
    (kv,) = struct.unpack("<Q", rd(8))
    if kv > 100000:
        raise GGUFError(f"implausible metadata count {kv}")
    out = {}
    for _ in range(kv):
        key = rstr()
        (t,) = struct.unpack("<I", rd(4))
        if t == 8:
            v = rstr()
            if key in want:
                out[key] = v
        elif t == 9:
            (et,) = struct.unpack("<I", rd(4))
            (cnt,) = struct.unpack("<Q", rd(8))
            if et == 8:
                for _ in range(cnt):
                    rstr()
            elif et in _GGUF_SIZES:
                rd(cnt * _GGUF_SIZES[et])
            else:
                raise GGUFError(f"unsupported array element type {et}")
        elif t in _GGUF_SIZES:
            rd(_GGUF_SIZES[t])
        else:
            raise GGUFError(f"unsupported metadata type {t}")
        if len(out) == len(want):
            break
    return out


# --------------------------------------------------------------------------
# Hugging Face
# --------------------------------------------------------------------------

def hf_request(url, byte_range=None, timeout=60):
    req = urllib.request.Request(url)
    tok = os.environ.get("HF_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    if byte_range:
        req.add_header("Range", f"bytes={byte_range}")
    return urllib.request.urlopen(req, timeout=timeout)


def hf_repo_files(repo):
    url = f"https://huggingface.co/api/models/{repo}?blobs=true"
    try:
        with hf_request(url) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            die(f"{repo} is gated on Hugging Face (HTTP {e.code}). Accept the "
                "license on the model page, create a token and export "
                "HF_TOKEN=<token>.")
        if e.code == 404:
            die(f"repo {repo} not found on Hugging Face.")
        die(f"Hugging Face API returned HTTP {e.code} for {repo}.")
    except urllib.error.URLError as e:
        die(f"cannot reach huggingface.co ({e.reason}). Check the NAS's "
            "internet connection/DNS.")
    except json.JSONDecodeError:
        die("Hugging Face API returned malformed JSON.")
    sib = data.get("siblings")
    if not isinstance(sib, list):
        die("unexpected Hugging Face API response (no file list).")
    return [(f.get("rfilename", ""), f.get("size") or 0) for f in sib]


def hf_probe_arch(repo, filename, mb=4):
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    try:
        with hf_request(url, byte_range=f"0-{mb*1024*1024-1}") as r:
            return parse_gguf_meta(io.BytesIO(r.read()))
    except GGUFError as e:
        die(f"cannot read GGUF header of {filename}: {e}")
    except urllib.error.HTTPError as e:
        die(f"cannot download header of {filename}: HTTP {e.code}")
    except urllib.error.URLError as e:
        die(f"network error while probing {filename}: {e.reason}")


def classify_arch(arch, runtime=None):
    a = (arch or "").lower().replace(".", "_").replace("-", "_")
    if runtime and str(runtime).startswith("upstream"):
        if a in UPSTREAM_ARCH_TESTED:
            return "TESTED", ("verified on this hardware with the b10143 "
                              "container (docs/known-bugs.md section 6).")
        if ARCH_BROKEN_MOE.search(a):
            return "EXPECTED", ("MoE computes correctly on upstream runtimes, "
                                "but generation is SYCL-kernel-bound and slow "
                                "(~7 t/s for a 35B-A3B) — pair it with an MTP "
                                "draft head (--draft) where the repo ships one.")
        return "UNKNOWN", ("architecture unverified on this runtime — install "
                           "at your own risk and run `test`.")
    if ARCH_BROKEN_MOE.search(a):
        return "BROKEN", ("MoE architecture: the shipped SYCL build computes "
                          "garbage for every MoE model and quant tested. "
                          "Do not install.")
    if a in ARCH_KNOWN_MISSING:
        return "UNSUPPORTED", "architecture is not present in llama.cpp b8413."
    if a in ARCH_TESTED_OK:
        return "TESTED", "verified end-to-end on this stack."
    if a in ARCH_EXPECTED_OK:
        return "EXPECTED", ("architecture is compiled into b8413 but nobody "
                            "has verified it here yet — run `test` after "
                            "installing and please report the result.")
    return "UNKNOWN", ("architecture is in neither list — it may still work. "
                       "Install at your own risk and run `test`.")


def split_siblings(files, filename):
    """All shards belonging to a split GGUF, or [] if it isn't split."""
    m = SPLIT_RE.search(filename)
    if not m:
        return []
    stem = filename[:m.start()]
    total = int(m.group(2))
    return sorted(n for n, _ in files
                  if n.startswith(stem) and SPLIT_RE.search(n)
                  and int(SPLIT_RE.search(n).group(2)) == total)


# --------------------------------------------------------------------------
# Runtime profiles — vendor b8413 vs. upstream llama.cpp builds
# --------------------------------------------------------------------------

def build_extra_args(ctx, runtime=None, draft_name=None):
    """The llama-server argv fragment for a model config, per runtime."""
    upstream = bool(runtime and str(runtime).startswith("upstream"))
    if draft_name and not upstream:
        die("--draft needs --runtime upstream-*: the vendor build (b8413) "
            "has no speculative decoding support.")
    ea = list(UPSTREAM_EXTRA_ARGS if upstream else SAFE_EXTRA_ARGS)
    ea += ["-c", str(ctx)]
    if draft_name:
        ea += ["--spec-draft-model",
               os.path.join(MODELS_DIR, draft_name, "draft.gguf"),
               "--spec-type", "draft-mtp",
               "--spec-draft-n-max", "4",
               "--spec-draft-ngl", "999"]
    return ea


def dispatcher_script(runtimes_dir=None):
    """The replacement for UGREEN's llama-server wrapper. Routes a model to
    its runtime if (and only if) a marker file opts it in; everything else
    runs the preserved vendor wrapper unchanged. A marker pointing at a
    missing runtime fails fast — falling back to the vendor build would
    crash later on upstream-only flags with a far more confusing error."""
    return f"""#!/bin/bash
{DISPATCHER_TAG} v1 — managed by ugos-llm.py `runtime enable`; do not edit.
DIR="$(cd "$(dirname "$0")" && pwd)"
RUNTIMES="{runtimes_dir or RUNTIMES_DIR}"
MODEL=""
prev=""
for a in "$@"; do
  case "$prev" in -m|--model) MODEL="$a";; esac
  prev="$a"
done
if [ -n "$MODEL" ]; then
  MARKER="$(dirname "$MODEL")/{RUNTIME_MARKER}"
  if [ -f "$MARKER" ]; then
    RT="$(head -c 64 "$MARKER" | tr -cd 'A-Za-z0-9._-')"
    if [ -n "$RT" ]; then
      W="$RUNTIMES/$RT/llama-server-wrapper"
      if [ -x "$W" ]; then
        exec "$W" "$@"
      fi
      echo "ugos-llm dispatcher: runtime '$RT' requested by $MARKER is not installed at $W" >&2
      exit 127
    fi
  fi
fi
exec "$DIR/llama-server.ugreen-orig" "$@"
"""


def runtime_wrapper_script():
    """Entry point inside a deployed runtime directory.

    Verified on UGOS glibc 2.36 (July 2026): the runtime reuses UGREEN's
    own OpenCL GPU userspace (libigdrcl + IGC 2.10 from the vendor bundle)
    — the maximally native choice, and the only one that works:
      * the host level-zero driver (1.3.x) is too old for 2025-era SYCL
        (UR_RESULT_ERROR_UNSUPPORTED_VERSION),
      * every compute-runtime release new enough for oneDNN needs
        GLIBC >= 2.38; the newest one that fits glibc (25.13) segfaults
        under real SYCL compute on this stack,
      * and the OpenCL path is only fast when llama.cpp is built with
        -DGGML_SYCL_DNN=OFF — with oneDNN enabled it re-JITs on every
        prompt batch (~2 t/s prompt processing) or crashes.
    The vendor dir sits AFTER the runtime dir so our libs always win.
    ggml backends load from the executable's directory; GGML_BACKEND_PATH
    must stay unset (llama.cpp treats it as a file path and fails)."""
    return f"""#!/bin/bash
# ugos-llm runtime wrapper — managed by ugos-llm.py `runtime deploy`.
DIR="$(cd "$(dirname "$0")" && pwd)"
export LD_LIBRARY_PATH="$DIR:{VENDOR_BUNDLE_DIR}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"
export OCL_ICD_FILENAMES="{VENDOR_BUNDLE_DIR}/libigdrcl.so"
export ONEAPI_DEVICE_SELECTOR="opencl:gpu"
exec "$DIR/llama-server" "$@"
"""


def dispatcher_state():
    """('missing'|'vendor'|'ours', path) for the bundle's llama-server."""
    path = os.path.join(VENDOR_BUNDLE_DIR, "llama-server")
    if not os.path.isfile(path):
        return "missing", path
    try:
        with open(path, errors="replace") as f:
            head = f.read(4096)
    except OSError:
        return "missing", path
    return ("ours" if DISPATCHER_TAG in head else "vendor"), path


def model_runtime_markers():
    """{model dir -> runtime name} for every installed marker file."""
    markers = {}
    try:
        for d in sorted(os.listdir(MODELS_DIR)):
            p = os.path.join(MODELS_DIR, d, RUNTIME_MARKER)
            if os.path.isfile(p):
                try:
                    markers[d] = open(p).read().strip()
                except OSError:
                    pass
    except OSError:
        pass
    return markers


def installed_runtimes():
    try:
        return sorted(d for d in os.listdir(RUNTIMES_DIR)
                      if not d.startswith("."))
    except OSError:
        return []


def lint_model_config(name, cfg, runtime=None, model_dir=None):
    """Warnings for one model config, aware of which runtime will run it."""
    warns = []
    ea = cfg.get("extra_args") or []
    upstream = bool(runtime and str(runtime).startswith("upstream"))
    if not upstream and "4096" in ea:
        warns.append(f"{name} uses -ub/-b 4096 -> change to 512 "
                     "(docs/known-bugs.md) and reload")
    if upstream and "--no_mmap" in ea:
        warns.append(f"{name}: '--no_mmap' is the vendor spelling — the "
                     "upstream runtime only accepts '--no-mmap' and will "
                     "not start with this config")
    if "--spec-draft-model" in ea:
        p = ea[ea.index("--spec-draft-model") + 1]
        if not os.path.isabs(p) and model_dir:
            p = os.path.join(model_dir, p)
        if not os.path.isfile(p):
            warns.append(f"{name}: draft model {p} is missing")
    return warns


def lint_runtime_setup(markers, dispatcher, runtimes_present):
    """Warnings about the dispatcher/runtime installation as a whole."""
    warns = []
    if markers and dispatcher != "ours":
        warns.append("models with runtime markers ("
                     + ", ".join(sorted(markers))
                     + f") but the dispatcher is '{dispatcher}' — a firmware "
                     "update may have restored the vendor wrapper; re-run "
                     "`runtime enable`")
    for name, rt in sorted(markers.items()):
        if rt not in runtimes_present:
            warns.append(f"{name} requests runtime {rt} which is not "
                         f"installed under {RUNTIMES_DIR}")
    return warns


# --------------------------------------------------------------------------
# Gateway
# --------------------------------------------------------------------------

def gw(path, payload=None, timeout=300):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(GATEWAY + path, data=data)
    if data:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def gw_models():
    try:
        return [m["id"] for m in gw("/v1/models", timeout=10)["data"]]
    except Exception:
        return None


def loaded_servers():
    try:
        out = subprocess.run(["ps", "ax"], capture_output=True,
                             text=True).stdout
    except Exception:
        return []
    hits = []
    for line in out.splitlines():
        if "llama-server" in line and MODELS_DIR in line:
            m = re.search(re.escape(MODELS_DIR) + r"/([^/]+)/", line)
            ub = re.search(r"-ub (\d+)", line)
            if m:
                hits.append((m.group(1), ub.group(1) if ub else "?"))
    return hits


def collapse_score(text):
    """Uniqueness of 4-grams; below 0.5 means degenerated repetition."""
    w = text.split()
    if len(w) < 12:
        return 1.0 if len(set(w)) > 2 else 0.0
    grams = [" ".join(w[i:i + 4]) for i in range(len(w) - 3)]
    return len(set(grams)) / len(grams)


def reload_hint(fresh_install=False):
    log("")
    if fresh_install:
        log("No reload needed: the gateway picks up new models automatically "
            "and loads them on first use (first request takes ~30-60 s). "
            "The Model Manager UI shows the new card after a page refresh.")
        log("Only if you later CHANGE this model's config: toggle it OFF/ON "
            "in Model Manager or reboot — and never kill llama-server "
            "processes (the gateway won't recover until reboot).")
        return
    log("RELOAD REQUIRED — llama-server only reads its config when it spawns:")
    log("  * toggle the model OFF and back ON in Model Manager, or")
    log("  * reboot the NAS.")
    log("  Never kill llama-server processes: the gateway keeps dead socket")
    log("  references and does not recover until the next reboot.")


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

def cmd_check(args):
    files = hf_repo_files(args.repo)
    ggufs = [(n, s) for n, s in files
             if n.endswith(".gguf") and "mmproj" not in n.lower()]
    mmproj = [(n, s) for n, s in files if "mmproj" in n.lower()]
    if not ggufs:
        die(f"{args.repo} contains no GGUF files.")

    log(f"Repo: {args.repo}")
    log(f"  GGUF files: {len(ggufs)}, vision projectors: {len(mmproj)}")

    seen = {}
    for n, s in sorted(ggufs, key=lambda x: x[1]):
        m = re.search(r"(IQ\d\w*|Q\d_K_?\w*|Q\d_\d|BF16|F16|F32)", n)
        seen.setdefault(m.group(1) if m else "?", (n, s))
    for q, (n, s) in seen.items():
        marks = []
        if q.startswith("IQ"):
            marks.append("IQ quant: unverified on this stack")
        if q in ("F16", "F32", "BF16"):
            marks.append("full precision: likely too large")
        if split_siblings(files, n):
            marks.append("SPLIT model: not supported by this tool")
        log(f"    {q:10s} {s/1e9:7.2f} GB  {n}"
            + ("   [" + "; ".join(marks) + "]" if marks else ""))

    probe = args.file or min(ggufs, key=lambda x: x[1])[0]
    shards = split_siblings(files, probe)
    if shards:
        log(f"  NOTE: {probe} is part of a {len(shards)}-way split model; "
            "install would be incomplete. Pick a single-file quant.")
    log(f"  Probing header of: {probe}")
    meta = hf_probe_arch(args.repo, probe)
    arch = meta.get("general.architecture", "?")
    verdict, why = classify_arch(arch, runtime=args.runtime)
    log(f"  architecture = {arch!r}  ->  {verdict}")
    log(f"  {why}")

    try:
        avail = int(re.search(r"MemAvailable:\s+(\d+)",
                              open("/proc/meminfo").read()).group(1))
        log(f"  MemAvailable now: {avail/1e6:.1f} GB — note that models stay "
            "resident until reboot (no idle unload on this stack).")
    except Exception:
        pass
    if verdict in ("BROKEN", "UNSUPPORTED"):
        sys.exit(2)


def _staging_dir():
    for cand in ("/volume1/docker", "/volume1", os.path.expanduser("~")):
        if os.path.isdir(cand) and os.access(cand, os.W_OK):
            d = os.path.join(cand, ".ugos-llm-staging")
            os.makedirs(d, exist_ok=True)
            return d
    die("no writable staging directory found (tried /volume1/docker, "
        "/volume1, $HOME).")


def _download(url, dest, label, expect_size=0):
    tmp = dest + ".part"
    req = urllib.request.Request(url)
    tok = os.environ.get("HF_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    try:
        with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length") or expect_size or 0)
            done, t0 = 0, time.time()
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total and time.time() - t0 > 3:
                    t0 = time.time()
                    log(f"    {label}: {done/1e9:.2f}/{total/1e9:.2f} GB "
                        f"({100*done/total:.0f}%)")
    except urllib.error.HTTPError as e:
        die(f"{label}: download failed with HTTP {e.code}"
            + (" — gated repo? export HF_TOKEN=<token>" if e.code in (401, 403)
               else ""))
    except urllib.error.URLError as e:
        die(f"{label}: network error ({e.reason})")
    if expect_size and os.path.getsize(tmp) != expect_size:
        got = os.path.getsize(tmp)
        os.unlink(tmp)
        die(f"{label}: size mismatch (expected {expect_size}, got {got}) — "
            "download incomplete, please retry.")
    with open(tmp, "rb") as f:
        if f.read(4) != b"GGUF":
            os.unlink(tmp)
            die(f"{label}: downloaded file is not a GGUF.")
    os.rename(tmp, dest)


def _staged(stage, repo, filename, expect_size):
    """Staging cache keyed by repo AND filename; reuse only when the size
    matches and the file really is a GGUF (guards against collisions and
    torn downloads)."""
    key = f"{repo.replace('/', '--')}--{os.path.basename(filename)}"
    p = os.path.join(stage, key)
    if os.path.exists(p):
        ok = expect_size and os.path.getsize(p) == expect_size
        if ok:
            with open(p, "rb") as f:
                ok = f.read(4) == b"GGUF"
        if ok:
            log(f"    reusing staged {key}")
            return p, True
        log(f"    discarding unusable staged {key}")
        os.unlink(p)
    return p, False


def cmd_install(args):
    repo = args.repo
    if not (2048 <= args.ctx <= 65536):
        die(f"--ctx {args.ctx} is outside the sane range 2048..65536.")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.quant or ""):
        die(f"--quant {args.quant!r} is not a plausible quant name.")
    name = validate_name(
        args.name or re.sub(r"-?GGUF$", "", repo.split("/")[-1], flags=re.I))
    target = os.path.join(MODELS_DIR, name)
    if os.path.exists(target) and not args.force:
        die(f"{target} already exists (use --force to replace, or --name).")

    files = hf_repo_files(repo)
    ggufs = [(n, s) for n, s in files
             if n.endswith(".gguf") and "mmproj" not in n.lower()]
    matches = [(n, s) for n, s in ggufs if args.quant.lower() in n.lower()]
    if not matches:
        die(f"no GGUF matching quant {args.quant!r}. Available: "
            + ", ".join(n for n, _ in ggufs))
    if len(matches) > 1:
        log(f"NOTE: {len(matches)} files match {args.quant!r}; using "
            f"{matches[0][0]}. Use --quant with a longer substring to choose "
            "a different one.")
    model_file, model_size = matches[0]

    if split_siblings(files, model_file):
        die(f"{model_file} is part of a split (multi-shard) GGUF. This tool "
            "installs single-file models only — pick a smaller quant that "
            "fits in one file.")

    mmproj_file = mmproj_size = None
    if args.vision:
        cand = [(n, s) for n, s in files if "mmproj" in n.lower()]
        pref = [c for c in cand if "f16" in c[0].lower()] or cand
        if not pref:
            die("--vision requested but this repo ships no mmproj projector.")
        mmproj_file, mmproj_size = pref[0]

    draft_file = draft_size = None
    if args.draft:
        cand = [(n, s) for n, s in files
                if args.draft.lower() in n.lower() and n.endswith(".gguf")
                and "mmproj" not in n.lower() and n != model_file]
        if not cand:
            die(f"--draft: no GGUF matching {args.draft!r} in {repo}. "
                "MTP heads are usually named mtp-*.gguf.")
        draft_file, draft_size = min(cand, key=lambda x: x[1])
        log(f"Draft head: {draft_file} ({draft_size/1e9:.2f} GB)")

    if args.runtime:
        if not args.runtime.startswith("upstream-"):
            die(f"--runtime {args.runtime!r}: only upstream-* runtimes exist "
                "(e.g. upstream-b10143).")
        state, _ = dispatcher_state()
        if args.runtime not in installed_runtimes() or state != "ours":
            log(f"NOTE: runtime {args.runtime} is not fully set up yet "
                f"(dispatcher: {state}, deployed: {installed_runtimes()}). "
                "The model will not load until `runtime deploy` and "
                "`runtime enable` have run — doctor will remind you.")

    meta = hf_probe_arch(repo, model_file)
    verdict, why = classify_arch(meta.get("general.architecture"),
                                 runtime=args.runtime)
    log(f"Architecture {meta.get('general.architecture')!r}: {verdict} — {why}")
    if verdict in ("BROKEN", "UNSUPPORTED") and not args.force:
        die("refusing to install: see docs/known-bugs.md (--force overrides).")

    stage = _staging_dir()
    log(f"Staging in {stage}")
    local_model, have = _staged(stage, repo, model_file, model_size)
    if not have:
        _download(f"https://huggingface.co/{repo}/resolve/main/{model_file}",
                  local_model, "model", model_size)
    local_mm = None
    if mmproj_file:
        local_mm, have = _staged(stage, repo, mmproj_file, mmproj_size)
        if not have:
            _download(f"https://huggingface.co/{repo}/resolve/main/{mmproj_file}",
                      local_mm, "mmproj", mmproj_size)
    local_draft = None
    if draft_file:
        local_draft, have = _staged(stage, repo, draft_file, draft_size)
        if not have:
            _download(f"https://huggingface.co/{repo}/resolve/main/{draft_file}",
                      local_draft, "draft", draft_size)

    cfg = {"num_ctx": args.ctx,
           "context_length": args.ctx,
           "extra_args": build_extra_args(
               args.ctx, runtime=args.runtime,
               draft_name=name if draft_file else None),
           "capabilities": ["completion"] + (["vision"] if mmproj_file else [])}
    if mmproj_file:
        cfg["mmproj"] = "mmproj.gguf"

    if os.path.exists(target):
        log(f"--force: {target} will be replaced atomically (the existing "
            "installation stays in place until the new one is complete)")
    log(f"Installing into {target}")
    try:
        priv_install_files(name, local_model, local_mm,
                           json.dumps(cfg, indent=2),
                           extra_files=([(local_draft, "draft.gguf")]
                                        if local_draft else None),
                           runtime=args.runtime)
    except PrivError as e:
        die(f"install failed: {e}\nNothing was activated; any previous "
            "installation is untouched.")

    if args.ui:
        ui_add(name, cfg, os.path.getsize(local_model))

    if not args.keep_staging:
        for p in (local_model, local_mm, local_draft):
            if p and os.path.exists(p):
                os.unlink(p)

    log("")
    log(f"DONE. Model id for API calls: {name}/{name}")
    reload_hint(fresh_install=True)
    if args.test:
        log("")
        log("--test given: running the acceptance suite now ...")
        cmd_test(argparse.Namespace(name=name, vision=bool(mmproj_file)))
    else:
        log(f"Verify with:  python3 {os.path.basename(sys.argv[0])} "
            f"test {name}")


def ui_add(name, cfg, size_bytes):
    validate_name(name)
    existing = db_read("SELECT id, release_id FROM model_config "
                       "WHERE code = ?", (name,))
    if existing:
        if existing[0][1] != OUR_RELEASE_ID:
            log(f"UI: {name} has a vendor catalog row — leaving it alone.")
            return
        # ours: refresh size/capabilities so a forced reinstall stays accurate
        log(f"UI: catalog backup -> {db_backup()}")
        ext = json.dumps({"num_ctx": cfg["num_ctx"],
                          "context_length": cfg["context_length"],
                          "capabilities": cfg["capabilities"]})
        db_write("UPDATE model_config SET version_size = ?, memory_usage = ?, "
                 "ext = ?, updated_at = datetime('now') "
                 "WHERE code = ? AND release_id = ?",
                 (size_bytes, f"{size_bytes/1e9:.0f} GB", ext, name,
                  OUR_RELEASE_ID))
        log(f"UI: catalog row for {name} updated.")
        return
    log(f"UI: catalog backup -> {db_backup()}")
    donor = db_read("SELECT ext_arch_tools FROM model_config "
                    "WHERE model_type='llm' ORDER BY id LIMIT 1")
    ext_tools = donor[0][0] if donor else ""
    top = db_read("SELECT COALESCE(MAX(id), 99) FROM model_config")[0][0]
    nid = max(100, top + 1)
    ext = json.dumps({"num_ctx": cfg["num_ctx"],
                      "context_length": cfg["context_length"],
                      "capabilities": cfg["capabilities"]})
    i18n = json.dumps([{"description": "Custom model installed by ugos-llm",
                        "language": lang, "modelType": "llm",
                        "modelTypeDesc": "Large Language Models",
                        "modelTypeName": "Large Language Models",
                        "name": name} for lang in ("en-US", "de-DE")])
    db_write(
        "INSERT INTO model_config (id,release_id,code,name,status,\"update\","
        "version,version_num,min_version,model_type,param_value,"
        "response_speed,memory_usage,is_default,version_description,"
        "version_size,ext,ext_i18n,ext_arch_tools,install_paths,created_at,"
        "updated_at) VALUES (?,?,?,?,8,'{\"status\":0}','v1.0.0',1,0,'llm',"
        "'-','-',?,0,'',?,?,?,?,'',datetime('now'),datetime('now'))",
        (nid, OUR_RELEASE_ID, name, name, f"{size_bytes/1e9:.0f} GB",
         size_bytes, ext, i18n, ext_tools))
    log(f"UI: catalog row added (id={nid}).")


def ui_remove(name):
    validate_name(name)
    rows = db_read("SELECT id, release_id FROM model_config WHERE code = ?",
                   (name,))
    if not rows:
        log(f"UI: no catalog row for {name}.")
        return
    if rows[0][1] != OUR_RELEASE_ID:
        die(f"catalog row for {name} was not created by ugos-llm "
            "(release_id mismatch) — refusing to touch vendor rows.")
    log(f"UI: catalog backup -> {db_backup()}")
    db_write("DELETE FROM model_config WHERE code = ? AND release_id = ?",
             (name, OUR_RELEASE_ID))
    log("UI: catalog row removed.")


def cmd_list(args):
    loaded = dict(loaded_servers())
    try:
        cat = {c: s for c, s in db_read(
            "SELECT code, status FROM model_config WHERE model_type='llm'")}
    except sqlite3.Error:
        cat = {}
    try:
        dirs = sorted(d for d in os.listdir(MODELS_DIR)
                      if os.path.isdir(os.path.join(MODELS_DIR, d)))
    except OSError as e:
        die(f"cannot read {MODELS_DIR}: {e}")
    markers = model_runtime_markers()
    log(f"{'model dir':30s} {'catalog':14s} {'runtime':18s} {'loaded (ub)'}")
    for d in dirs:
        if d.startswith((".", "infer_")):
            continue
        state = STATUS.get(cat.get(d, -1), "no UI entry")
        rt = markers.get(d, "vendor")
        lo = f"yes (ub={loaded[d]})" if d in loaded else "-"
        warn = ("  <-- ub != 512, see docs/known-bugs.md"
                if d in loaded and loaded[d] not in ("512", "?")
                and not rt.startswith("upstream") else "")
        log(f"{d:30s} {state:14s} {rt:18s} {lo}{warn}")
    ids = gw_models()
    log(f"\nGateway: {'OK, ' + str(len(ids)) + ' entries' if ids else 'UNREACHABLE'}")


def cmd_test(args):
    name = validate_name(args.name)
    model_id = f"{name}/{name}"
    ids = gw_models()
    if ids is None:
        die("gateway unreachable on 127.0.0.1:62891")
    if not any(i.startswith(name + "/") for i in ids):
        die(f"gateway does not list {name}. Installed? Reloaded?")

    def chat(msgs, **kw):
        p = {"model": model_id, "messages": msgs, "max_tokens": 200,
             "temperature": 0.1, **kw}
        t0 = time.time()
        d = gw("/v1/chat/completions", p, timeout=600)
        ch = d["choices"][0]
        return (ch.get("message", {}).get("content") or "",
                ch.get("message", {}).get("tool_calls"), time.time() - t0)

    results = []
    log("1/4 chat smoke test (loads the model on first call) ...")
    txt, _, dt = chat([{"role": "user",
                        "content": "Antworte in einem Satz: Was ist ein NAS?"}])
    results.append(("chat", collapse_score(txt) > 0.5 and len(txt) > 10, dt,
                    txt[:80]))

    log("2/4 long-prompt stability (the ub=4096 bug detector) ...")
    para = ("Die Lagerhalle wurde 1987 errichtet und mehrfach modernisiert. "
            "Im Erdgeschoss lagern Ersatzteile, das Obergeschoss dient als "
            "Buero. Die Heizung stammt aus dem Jahr 2015. ") * 90
    txt, _, dt = chat([{"role": "user", "content":
                        para + "\n\nIn welchem Jahr wurde die Halle "
                        "errichtet? Antworte in einem Satz."}])
    results.append(("long-prompt", collapse_score(txt) > 0.5, dt, txt[:80]))

    log("3/4 tool-calling ...")
    tool = {"type": "function", "function": {
        "name": "get_system_info",
        "description": "Get system information for a category.",
        "parameters": {"type": "object",
                       "properties": {"category": {
                           "type": "string",
                           "enum": ["storage", "cpu", "memory"]}},
                       "required": ["category"]}}}
    txt, calls, dt = chat([{"role": "user",
                            "content": "Wie viel Speicherplatz ist noch frei?"}],
                          tools=[tool])
    ok = bool(calls) or (collapse_score(txt) > 0.5 and len(txt) > 10)
    results.append(("tools", ok, dt,
                    f"tool_call {calls[0]['function']['name']}" if calls
                    else txt[:80]))

    log("4/4 vision (skipped unless --vision) ...")
    if args.vision:
        import base64
        import zlib
        w = h = 48
        raw = b"".join(b"\x00" + b"\xff\x00\x00" * w for _ in range(h))

        def chunk(t, d):
            c = struct.pack(">I", len(d)) + t + d
            return c + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
        png = (b"\x89PNG\r\n\x1a\n"
               + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
               + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
        uri = "data:image/png;base64," + base64.b64encode(png).decode()
        txt, _, dt = chat([{"role": "user", "content": [
            {"type": "text", "text": "Welche Farbe hat dieses Bild? Ein Wort."},
            {"type": "image_url", "image_url": {"url": uri}}]}])
        results.append(("vision",
                        "rot" in txt.lower() or "red" in txt.lower(),
                        dt, txt[:40]))

    log("")
    fails = 0
    for label, ok, dt, detail in results:
        fails += 0 if ok else 1
        log(f"  {'PASS' if ok else 'FAIL':4s} {label:12s} {dt:5.1f}s  {detail}")

    # copy-pasteable block for a "Model compatibility report" issue
    arch = size_gb = "?"
    try:
        gguf = os.path.join(MODELS_DIR, name, f"{name}.gguf")
        size_gb = f"{os.path.getsize(gguf)/1e9:.1f} GB"
        with open(gguf, "rb") as f:
            arch = parse_gguf_meta(
                io.BytesIO(f.read(4 << 20))).get("general.architecture", "?")
    except (OSError, GGUFError):
        pass
    device = "?"
    try:
        device = open("/sys/class/dmi/id/product_name").read().strip() or "?"
    except OSError:
        pass
    ub = dict(loaded_servers()).get(name, "?")
    log("")
    log("---- report block (paste into a GitHub model report) ----")
    log(f"model: {name} | arch: {arch} | file: {size_gb} | ub: {ub}")
    log(f"device: {device}")
    for label, ok, dt, _ in results:
        log(f"{label}: {'PASS' if ok else 'FAIL'} ({dt:.1f}s)")
    log("---------------------------------------------------------")
    log("")
    if fails:
        log("Some checks failed — see docs/known-bugs.md. Usual suspects: "
            "ub=4096 configs, MoE or unsupported architectures.")
        sys.exit(1)
    log("All checks passed. Please consider filing a model report "
        "(docs/compatibility.md).")


def cmd_doctor(args):
    ids = gw_models()
    log(f"gateway 127.0.0.1:62891 : {'OK' if ids is not None else 'DOWN'}")
    for i in ids or []:
        log(f"  - {i}")
    markers = model_runtime_markers()
    for name, ub in loaded_servers():
        upstream = markers.get(name, "").startswith("upstream")
        bad = ub != "512" and not upstream
        log(f"loaded: {name} (ub={ub})"
            + ("   <-- not 512: long-prompt corruption!" if bad else ""))
    try:
        for d in sorted(os.listdir(MODELS_DIR)):
            cfgp = os.path.join(MODELS_DIR, d, "model_config.json")
            if os.path.isfile(cfgp):
                try:
                    cfg = json.load(open(cfgp))
                except (OSError, json.JSONDecodeError):
                    log(f"config lint: {d} has an unreadable model_config.json")
                    continue
                for w in lint_model_config(d, cfg, runtime=markers.get(d),
                                           model_dir=os.path.join(MODELS_DIR, d)):
                    log(f"config lint: {w}")
    except OSError:
        pass
    if markers:
        state, _ = dispatcher_state()
        log(f"runtime: dispatcher = {state}; deployed = "
            + (", ".join(installed_runtimes()) or "none"))
        for name, rt in sorted(markers.items()):
            log(f"runtime: {name} -> {rt}")
        for w in lint_runtime_setup(markers, state, installed_runtimes()):
            log(f"runtime lint: {w}")
    try:
        for code, st in db_read("SELECT code, status FROM model_config "
                                "WHERE model_type='llm'"):
            log(f"catalog: {code} = {STATUS.get(st, st)}")
    except sqlite3.Error as e:
        log(f"catalog: unreadable ({e})")
    log("")
    log("Reminders: never pkill llama-server; reload = UI toggle or reboot; "
        "model_duration in system_setting has no effect.")


def cmd_ui(args):
    name = validate_name(args.name)
    if args.action == "remove":
        ui_remove(name)
        return
    cfgp = os.path.join(MODELS_DIR, name, "model_config.json")
    if not os.path.isfile(cfgp):
        die(f"{name} is not installed under {MODELS_DIR}")
    cfg = json.load(open(cfgp))
    size = os.path.getsize(os.path.join(MODELS_DIR, name, f"{name}.gguf"))
    ui_add(name, cfg, size)


def runtime_enable():
    _need_privileges()
    state, path = dispatcher_state()
    if state == "missing":
        die(f"{path} not found — is the AI console installed?")
    backup = os.path.join(VENDOR_BUNDLE_DIR, "llama-server.ugreen-orig")
    if state == "ours" and not os.path.exists(backup):
        die(f"dispatcher is installed but the vendor backup {backup} is "
            "missing — restore it manually before changing anything.")
    script = dispatcher_script()
    if is_root():
        if state == "vendor" and not os.path.exists(backup):
            shutil.copy2(path, backup)
        tmp = path + ".ugos-llm-new"
        with open(tmp, "w") as f:
            f.write(script)
        os.chmod(tmp, 0o755)
        os.replace(tmp, path)      # atomic: the gateway never sees a torn file
    else:
        stage = _staging_dir()
        sp = os.path.join(stage, ".dispatcher.sh")
        with open(sp, "w") as f:
            f.write(script)
        mounts = {VENDOR_BUNDLE_DIR: "/b", stage: "/s"}
        steps = []
        if state == "vendor" and not os.path.exists(backup):
            steps.append(["cp", "-p", "/b/llama-server",
                          "/b/llama-server.ugreen-orig"])
        steps += [["cp", "/s/.dispatcher.sh", "/b/.llama-server.dispatcher"],
                  ["chmod", "755", "/b/.llama-server.dispatcher"],
                  ["mv", "/b/.llama-server.dispatcher", "/b/llama-server"]]
        try:
            for argv in steps:
                r = _docker(argv, mounts)
                if r.returncode != 0:
                    die(f"runtime enable failed at {argv[0]}: "
                        f"{r.stderr.strip()[:200]}")
        finally:
            if os.path.exists(sp):
                os.unlink(sp)
    log(f"dispatcher installed at {path}")
    log(f"vendor wrapper preserved at {backup}")
    log("Models without a runtime marker keep running the vendor build, "
        "byte-identically. Already-loaded servers are unaffected until "
        "their next spawn (UI toggle or reboot).")


def runtime_disable():
    _need_privileges()
    state, path = dispatcher_state()
    if state != "ours":
        log(f"nothing to do: the bundle wrapper is '{state}'.")
        return
    backup = os.path.join(VENDOR_BUNDLE_DIR, "llama-server.ugreen-orig")
    if not os.path.exists(backup):
        die(f"vendor backup {backup} is missing — cannot restore.")
    if is_root():
        shutil.copy2(backup, path)
    else:
        r = _docker(["cp", "-p", "/b/llama-server.ugreen-orig",
                     "/b/llama-server"], {VENDOR_BUNDLE_DIR: "/b"})
        if r.returncode != 0:
            die(f"restore failed: {r.stderr.strip()[:200]}")
    log("vendor wrapper restored. Models with runtime markers will fail to "
        "spawn until you re-enable the dispatcher or reinstall them "
        "without --runtime.")


def runtime_deploy(src_dir, rt_name, force=False):
    _need_privileges()
    if not rt_name:
        die("deploy needs --name (e.g. --name upstream-b10143).")
    validate_name(rt_name)
    if not rt_name.startswith("upstream-"):
        die("runtime names must start with 'upstream-' "
            "(e.g. upstream-b10143).")
    src_dir = os.path.abspath(src_dir or "")
    if not os.path.isfile(os.path.join(src_dir, "llama-server")):
        die(f"{src_dir} does not look like a built runtime "
            "(no llama-server binary in it).")
    if os.path.exists(os.path.join(src_dir, "libdnnl.so.3")):
        log("WARNING: the runtime bundles oneDNN — on UGREEN's OpenCL "
            "userspace that build re-JITs every prompt batch or crashes. "
            "Rebuild with -DGGML_SYCL_DNN=OFF (see docs/known-bugs.md).")
    target = os.path.join(RUNTIMES_DIR, rt_name)
    if os.path.exists(target) and not force:
        die(f"{target} already exists (use --force to replace).")
    wrapper = runtime_wrapper_script()
    if is_root():
        work = os.path.join(RUNTIMES_DIR, f".{rt_name}.new")
        shutil.rmtree(work, ignore_errors=True)
        shutil.copytree(src_dir, work)
        wp = os.path.join(work, "llama-server-wrapper")
        with open(wp, "w") as f:
            f.write(wrapper)
        os.chmod(wp, 0o755)
        os.chmod(os.path.join(work, "llama-server"), 0o755)
        if os.path.exists(target):
            shutil.rmtree(target)
        os.rename(work, target)
    else:
        stage = _staging_dir()
        wp = os.path.join(stage, ".runtime-wrapper.sh")
        with open(wp, "w") as f:
            f.write(wrapper)
        parent, base = os.path.dirname(RUNTIMES_DIR), \
            os.path.basename(RUNTIMES_DIR)
        mounts = {parent: "/rt", src_dir: "/src", stage: "/s"}
        work = f"/rt/{base}/.{rt_name}.new"
        steps = [["mkdir", "-p", f"/rt/{base}"],
                 ["rm", "-rf", work],
                 ["cp", "-r", "/src", work],
                 ["cp", "/s/.runtime-wrapper.sh",
                  f"{work}/llama-server-wrapper"],
                 ["chmod", "755", f"{work}/llama-server-wrapper",
                  f"{work}/llama-server"],
                 ["rm", "-rf", f"/rt/{base}/{rt_name}"],
                 ["mv", work, f"/rt/{base}/{rt_name}"]]
        try:
            for argv in steps:
                r = _docker(argv, mounts)
                if r.returncode != 0:
                    die(f"deploy failed at {argv[0]}: "
                        f"{r.stderr.strip()[:200]}")
        finally:
            if os.path.exists(wp):
                os.unlink(wp)
    log(f"runtime {rt_name} deployed to {target}")
    log("Next: `runtime enable` (if not done yet), then install a model "
        f"with --runtime {rt_name}.")


def cmd_runtime(args):
    if args.action == "status":
        state, path = dispatcher_state()
        log(f"bundle wrapper: {state} ({path})")
        log("deployed runtimes: " + (", ".join(installed_runtimes()) or "none"))
        markers = model_runtime_markers()
        for name, rt in sorted(markers.items()):
            log(f"  {name} -> {rt}")
        for w in lint_runtime_setup(markers, state, installed_runtimes()):
            log(f"WARNING: {w}")
        return
    if args.action == "enable":
        runtime_enable()
    elif args.action == "disable":
        runtime_disable()
    elif args.action == "deploy":
        runtime_deploy(args.path, args.name, force=args.force)


def cmd_remove(args):
    name = validate_name(args.name)
    target = os.path.join(MODELS_DIR, name)
    if not os.path.isdir(target):
        die(f"{target} does not exist")
    if not args.yes:
        die(f"refusing without --yes (would delete {target})")
    if not args.keep_ui:
        try:
            ui_remove(name)
        except SystemExit:
            log("(catalog row left untouched)")
    priv_remove_dir(name)
    log(f"{name} removed. A loaded server keeps running until the next "
        "reload (UI toggle or reboot).")


# --------------------------------------------------------------------------

def build_parser():
    ap = argparse.ArgumentParser(
        prog="ugos-llm", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("check", help="pre-flight a Hugging Face GGUF repo")
    p.add_argument("repo")
    p.add_argument("--file", help="probe this specific GGUF file")
    p.add_argument("--runtime",
                   help="judge against an alternative runtime "
                        "(e.g. upstream-b10143) instead of the vendor build")
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("install", help="download, configure and register")
    p.add_argument("repo")
    p.add_argument("--quant", default="Q4_K_M")
    p.add_argument("--name", help="model directory name (default: repo name)")
    p.add_argument("--vision", action="store_true",
                   help="also install the repo's mmproj projector")
    p.add_argument("--ui", action="store_true",
                   help="add a Model Manager card (backs up the catalog DB)")
    p.add_argument("--ctx", type=int, default=16384)
    p.add_argument("--runtime",
                   help="run this model on an alternative runtime "
                        "(e.g. upstream-b10143); writes the "
                        ".ugos-llm-runtime marker for the dispatcher")
    p.add_argument("--draft", nargs="?", const="mtp", default=None,
                   metavar="PATTERN",
                   help="also install a speculative-decoding draft head "
                        "matching PATTERN (default: mtp) as draft.gguf; "
                        "requires --runtime upstream-*")
    p.add_argument("--test", action="store_true",
                   help="run the acceptance suite right after installing")
    p.add_argument("--force", action="store_true")
    p.add_argument("--keep-staging", action="store_true")
    p.set_defaults(fn=cmd_install)

    p = sub.add_parser("list", help="installed models and their state")
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("test", help="acceptance suite for an installed model")
    p.add_argument("name")
    p.add_argument("--vision", action="store_true")
    p.set_defaults(fn=cmd_test)

    p = sub.add_parser("doctor", help="health checks and config lint")
    p.set_defaults(fn=cmd_doctor)

    p = sub.add_parser("ui", help="manage the Model Manager catalog card")
    p.add_argument("action", choices=["add", "remove"])
    p.add_argument("name")
    p.set_defaults(fn=cmd_ui)

    p = sub.add_parser("runtime",
                       help="manage alternative llama.cpp runtimes")
    p.add_argument("action", choices=["status", "enable", "disable",
                                      "deploy"])
    p.add_argument("path", nargs="?",
                   help="deploy: directory containing the built runtime")
    p.add_argument("--name",
                   help="deploy: runtime name (e.g. upstream-b10143)")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_runtime)

    p = sub.add_parser("remove", help="delete an installed model")
    p.add_argument("name")
    p.add_argument("--yes", action="store_true")
    p.add_argument("--keep-ui", action="store_true")
    p.set_defaults(fn=cmd_remove)
    return ap


def main():
    args = build_parser().parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
