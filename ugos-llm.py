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

# Pinned images: privileged file/DB operations must not depend on a moving tag.
IMG_BUSYBOX = "docker.io/library/alpine:3.20"
IMG_PYTHON = "docker.io/library/python:3.12-alpine"

# Catalog rows created by this tool are marked with this release_id so that
# `remove`/`ui remove` can never delete a vendor row.
OUR_RELEASE_ID = 9000

# The single most important deviation from UGREEN's own defaults:
# their `-ub 4096 -b 4096` silently corrupts long-prompt inference on the
# shipped SYCL build (llama.cpp b8413). 512 is correct AND ~2x faster.
SAFE_EXTRA_ARGS = ["-ngl", "999", "-t", "10", "-e", "-lv", "3",
                   "--no-warmup", "--no_mmap", "-fa", "off", "-np", "1",
                   "-ub", "512", "-b", "512"]

ARCH_TESTED_OK = {"qwen3_5", "qwen35", "qwen3"}
ARCH_EXPECTED_OK = {"llama", "qwen2", "qwen2vl", "gemma", "gemma2",
                    "gemma3", "gemma3n", "mistral", "phi3"}
ARCH_BROKEN_MOE = re.compile(r"moe", re.IGNORECASE)
ARCH_KNOWN_MISSING = {"gemma4", "qwen3_6", "qwen36", "glm4moe"}

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


def priv_install_files(target_dir, name, model_src, mmproj_src, config_json):
    """Create <MODELS_DIR>/<name>/ and place model, projector and config."""
    _need_privileges()
    validate_name(name)
    if is_root():
        os.makedirs(target_dir, exist_ok=True)
        shutil.copyfile(model_src, os.path.join(target_dir, f"{name}.gguf"))
        if mmproj_src:
            shutil.copyfile(mmproj_src, os.path.join(target_dir, "mmproj.gguf"))
        with open(os.path.join(target_dir, "model_config.json"), "w") as f:
            f.write(config_json)
        os.chmod(target_dir, 0o755)
        for fn in os.listdir(target_dir):
            p = os.path.join(target_dir, fn)
            if os.path.isfile(p):
                os.chmod(p, 0o755)
        return

    stage = os.path.dirname(os.path.abspath(model_src))
    cfg_path = os.path.join(stage, f".{name}.model_config.json")
    with open(cfg_path, "w") as f:
        f.write(config_json)
    mounts = {MODELS_DIR: "/m", stage: "/s"}
    steps = [["mkdir", "-p", f"/m/{name}"],
             ["cp", f"/s/{os.path.basename(model_src)}", f"/m/{name}/{name}.gguf"]]
    if mmproj_src:
        steps.append(["cp", f"/s/{os.path.basename(mmproj_src)}",
                      f"/m/{name}/mmproj.gguf"])
    steps += [["cp", f"/s/{os.path.basename(cfg_path)}",
               f"/m/{name}/model_config.json"],
              ["chmod", "-R", "755", f"/m/{name}"]]
    for argv in steps:
        r = _docker(argv, mounts)
        if r.returncode != 0:
            os.unlink(cfg_path)
            die(f"install step {argv[0]} failed: {r.stderr.strip()[:300]}")
    os.unlink(cfg_path)


def priv_remove_dir(name):
    _need_privileges()
    validate_name(name)
    target = os.path.join(MODELS_DIR, name)
    if is_root():
        shutil.rmtree(target)
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
    dest = os.path.join(_staging_dir(),
                        f"aiconsole.db.backup-{time.strftime('%Y%m%d-%H%M%S')}")
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


def classify_arch(arch):
    a = (arch or "").lower().replace(".", "_").replace("-", "_")
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


def reload_hint():
    log("")
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
    verdict, why = classify_arch(arch)
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


def _staged(stage, filename, expect_size):
    """Reuse a staged download only if its size matches exactly."""
    p = os.path.join(stage, filename)
    if os.path.exists(p):
        if expect_size and os.path.getsize(p) == expect_size:
            log(f"    reusing staged {filename}")
            return p, True
        log(f"    discarding incomplete staged {filename}")
        os.unlink(p)
    return p, False


def cmd_install(args):
    repo = args.repo
    name = validate_name(
        args.name or re.sub(r"-?GGUF$", "", repo.split("/")[-1], flags=re.I))
    target = os.path.join(MODELS_DIR, name)
    if os.path.exists(target) and not args.force:
        die(f"{target} already exists (use --force to overwrite, or --name).")

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

    meta = hf_probe_arch(repo, model_file)
    verdict, why = classify_arch(meta.get("general.architecture"))
    log(f"Architecture {meta.get('general.architecture')!r}: {verdict} — {why}")
    if verdict in ("BROKEN", "UNSUPPORTED") and not args.force:
        die("refusing to install: see docs/known-bugs.md (--force overrides).")

    stage = _staging_dir()
    log(f"Staging in {stage}")
    local_model, have = _staged(stage, os.path.basename(model_file), model_size)
    if not have:
        _download(f"https://huggingface.co/{repo}/resolve/main/{model_file}",
                  local_model, "model", model_size)
    local_mm = None
    if mmproj_file:
        local_mm, have = _staged(stage, f"{name}-mmproj.gguf", mmproj_size)
        if not have:
            _download(f"https://huggingface.co/{repo}/resolve/main/{mmproj_file}",
                      local_mm, "mmproj", mmproj_size)

    cfg = {"num_ctx": args.ctx,
           "context_length": args.ctx,
           "extra_args": SAFE_EXTRA_ARGS + ["-c", str(args.ctx)],
           "capabilities": ["completion"] + (["vision"] if mmproj_file else [])}
    if mmproj_file:
        cfg["mmproj"] = "mmproj.gguf"

    log(f"Installing into {target}")
    priv_install_files(target, name, local_model, local_mm,
                       json.dumps(cfg, indent=2))

    if args.ui:
        ui_add(name, cfg, os.path.getsize(local_model))

    if not args.keep_staging:
        for p in (local_model, local_mm):
            if p and os.path.exists(p):
                os.unlink(p)

    log("")
    log(f"DONE. Model id for API calls: {name}/{name}")
    reload_hint()
    log(f"Then run:  python3 {os.path.basename(sys.argv[0])} test {name}")


def ui_add(name, cfg, size_bytes):
    validate_name(name)
    if db_read("SELECT id FROM model_config WHERE code = ?", (name,)):
        log(f"UI: a catalog row for {name} already exists.")
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
    log(f"{'model dir':30s} {'catalog':14s} {'loaded (ub)'}")
    for d in dirs:
        if d.startswith((".", "infer_")):
            continue
        state = STATUS.get(cat.get(d, -1), "no UI entry")
        lo = f"yes (ub={loaded[d]})" if d in loaded else "-"
        warn = ("  <-- ub != 512, see docs/known-bugs.md"
                if d in loaded and loaded[d] not in ("512", "?") else "")
        log(f"{d:30s} {state:14s} {lo}{warn}")
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
    log("")
    if fails:
        log("Some checks failed — see docs/known-bugs.md. Usual suspects: "
            "ub=4096 configs, MoE or unsupported architectures.")
        sys.exit(1)
    log("All checks passed. Please consider reporting this model+quant as "
        "working (docs/compatibility.md).")


def cmd_doctor(args):
    ids = gw_models()
    log(f"gateway 127.0.0.1:62891 : {'OK' if ids is not None else 'DOWN'}")
    for i in ids or []:
        log(f"  - {i}")
    for name, ub in loaded_servers():
        log(f"loaded: {name} (ub={ub})"
            + ("" if ub == "512" else "   <-- not 512: long-prompt corruption!"))
    try:
        for d in sorted(os.listdir(MODELS_DIR)):
            cfgp = os.path.join(MODELS_DIR, d, "model_config.json")
            if os.path.isfile(cfgp):
                try:
                    ea = json.load(open(cfgp)).get("extra_args", [])
                except (OSError, json.JSONDecodeError):
                    log(f"config lint: {d} has an unreadable model_config.json")
                    continue
                if "4096" in ea:
                    log(f"config lint: {d} uses -ub/-b 4096 -> change to 512 "
                        "(docs/known-bugs.md) and reload")
    except OSError:
        pass
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
