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

Privileges: `check`, `list`, `test`, `doctor` run as any user. `install`,
`remove` and `ui` write to root-owned paths: run them with sudo, or as a user
in the `docker` group (the tool then routes file operations through a
throwaway Alpine container).

Tested on: UGOS Pro, Model Manager 1.17.0.0055, infer_gateway 1.0.0.0004,
llama.cpp b8413 (SYCL). See docs/known-bugs.md before trusting anything.

License: MIT.
"""

import argparse
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
TESTED_MM_VERSION = "1.17.0.0055"

# The single most important deviation from UGREEN's own defaults:
# their `-ub 4096 -b 4096` silently corrupts long-prompt inference on the
# shipped SYCL build (llama.cpp b8413). 512 is correct AND ~2x faster.
SAFE_EXTRA_ARGS = ["-ngl", "999", "-t", "10", "-e", "-lv", "3",
                   "--no-warmup", "--no_mmap", "-fa", "off", "-np", "1",
                   "-ub", "512", "-b", "512", "-c", "16384"]

# general.architecture values from GGUF headers, matched against the
# architectures compiled into the shipped llama.cpp build (b8413).
ARCH_TESTED_OK = {"qwen3_5", "qwen35", "qwen3"}          # qwen3.5-9B verified end-to-end
ARCH_EXPECTED_OK = {"llama", "qwen2", "qwen2vl", "gemma", "gemma2",
                    "gemma3", "gemma3n", "mistral", "phi3"}
ARCH_BROKEN_MOE = re.compile(r"moe", re.IGNORECASE)       # SYCL computes garbage: verified
ARCH_KNOWN_MISSING = {"gemma4", "qwen3_6", "qwen36", "glm4moe"}

STATUS = {1: "not installed", 3: "disabled", 8: "active"}


def log(msg=""):
    print(msg, flush=True)


def die(msg, code=1):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


# --------------------------------------------------------------------------
# Privileged operations: direct when root, docker-alpine fallback otherwise
# --------------------------------------------------------------------------

def is_root():
    return os.geteuid() == 0


def have_docker():
    return shutil.which("docker") is not None and subprocess.run(
        ["docker", "ps", "-q"], capture_output=True).returncode == 0


def root_sh(script, mounts):
    """Run a small shell script as root. mounts: {host_path: cont_path}."""
    if is_root():
        return subprocess.run(["sh", "-c", script], capture_output=True,
                              text=True)
    if not have_docker():
        die("This action needs root. Re-run with sudo, or add your user to "
            "the docker group.")
    cmd = ["docker", "run", "--rm", "-i"]
    for h, c in mounts.items():
        cmd += ["-v", f"{h}:{c}"]
    cmd += ["alpine:latest", "sh", "-c", script]
    return subprocess.run(cmd, capture_output=True, text=True)


def root_sqlite(sql, readonly=False):
    """Execute SQL against the aiconsole DB. Reads work as any user."""
    if readonly or is_root():
        uri = f"file:{AICONSOLE_DB}?mode={'ro' if readonly else 'rw'}"
        con = sqlite3.connect(uri, uri=True, timeout=10)
        try:
            cur = con.execute(sql) if isinstance(sql, str) else None
            if cur is not None:
                rows = cur.fetchall()
                con.commit()
                return rows
        finally:
            con.close()
    # non-root write path: python inside a container
    if not have_docker():
        die("Writing the Model Manager catalog needs root or docker access.")
    py = ("import sqlite3,sys; con=sqlite3.connect('/db/aiconsole.db',"
          "timeout=10); con.executescript(sys.stdin.read()); con.commit()")
    r = subprocess.run(["docker", "run", "--rm", "-i",
                        "-v", os.path.dirname(AICONSOLE_DB) + ":/db",
                        "python:3.12-alpine", "python3", "-c", py],
                       input=sql, capture_output=True, text=True)
    if r.returncode != 0:
        die(f"catalog DB write failed: {r.stderr.strip()[:300]}")
    return []


# --------------------------------------------------------------------------
# GGUF header parsing (metadata lives at the start of the file)
# --------------------------------------------------------------------------

_GGUF_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
               10: 8, 11: 8, 12: 8}


def parse_gguf_meta(readable, want=("general.architecture", "general.name",
                                    "tokenizer.chat_template")):
    """Parse selected string keys from a GGUF header byte stream."""
    def rstr(f):
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n).decode("utf-8", "replace")

    if readable.read(4) != b"GGUF":
        raise ValueError("not a GGUF file")
    readable.read(4 + 8)  # version, tensor count
    kv = struct.unpack("<Q", readable.read(8))[0]
    out = {}
    for _ in range(kv):
        key = rstr(readable)
        t = struct.unpack("<I", readable.read(4))[0]
        if t == 8:
            v = rstr(readable)
            if key in want:
                out[key] = v
        elif t == 9:
            et = struct.unpack("<I", readable.read(4))[0]
            cnt = struct.unpack("<Q", readable.read(8))[0]
            if et == 8:
                for _ in range(cnt):
                    rstr(readable)
            else:
                readable.read(cnt * _GGUF_SIZES[et])
        else:
            readable.read(_GGUF_SIZES[t])
        if len(out) == len(want):
            break
    return out


# --------------------------------------------------------------------------
# Hugging Face API
# --------------------------------------------------------------------------

def hf_request(url, byte_range=None):
    req = urllib.request.Request(url)
    tok = os.environ.get("HF_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    if byte_range:
        req.add_header("Range", f"bytes={byte_range}")
    return urllib.request.urlopen(req, timeout=60)


def hf_repo_files(repo):
    try:
        with hf_request(f"https://huggingface.co/api/models/{repo}?blobs=true") as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            die(f"{repo} is gated on Hugging Face (HTTP {e.code}). Accept the "
                "license on the website and set HF_TOKEN=<your token>.")
        die(f"Hugging Face API: HTTP {e.code} for {repo}")
    return [(f["rfilename"], f.get("size") or 0)
            for f in data.get("siblings", [])]


def hf_probe_arch(repo, filename):
    """Read the GGUF architecture by fetching only the first 4 MB."""
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    with hf_request(url, byte_range="0-4194303") as r:
        import io
        return parse_gguf_meta(io.BytesIO(r.read()))


def classify_arch(arch):
    a = (arch or "").lower().replace(".", "_").replace("-", "_")
    if ARCH_BROKEN_MOE.search(a):
        return "BROKEN", ("MoE architecture: the shipped SYCL build computes "
                          "garbage output for every MoE model and quant "
                          "tested. Do not install.")
    if a in ARCH_KNOWN_MISSING:
        return "UNSUPPORTED", "architecture not present in llama.cpp b8413."
    if a in ARCH_TESTED_OK:
        return "TESTED", "verified end-to-end on this stack."
    if a in ARCH_EXPECTED_OK:
        return "EXPECTED", ("architecture is compiled into b8413 but nobody "
                            "has verified it here yet — run `test` after "
                            "installing and consider reporting the result.")
    return "UNKNOWN", ("architecture not in the known lists — it may work. "
                       "Install at your own risk and run `test`.")


# --------------------------------------------------------------------------
# Gateway helpers
# --------------------------------------------------------------------------

def gw(path, payload=None, timeout=300):
    url = GATEWAY + path
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data)
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
        out = subprocess.run(["ps", "ax"], capture_output=True, text=True).stdout
    except Exception:
        return []
    hits = []
    for line in out.splitlines():
        if "llama-server" in line and MODELS_DIR in line:
            m = re.search(rf"{MODELS_DIR}/([^/]+)/", line)
            ub = re.search(r"-ub (\d+)", line)
            if m:
                hits.append((m.group(1), ub.group(1) if ub else "?"))
    return hits


def collapse_score(text):
    """4-gram uniqueness; < 0.5 means degenerated repetition."""
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
    log("  Do NOT kill llama-server processes: the gateway holds dead socket")
    log("  references and will not recover until the next reboot.")


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

def cmd_check(args):
    repo = args.repo
    files = hf_repo_files(repo)
    ggufs = [(n, s) for n, s in files if n.endswith(".gguf")
             and "mmproj" not in n.lower()]
    mmproj = [(n, s) for n, s in files if "mmproj" in n.lower()]
    if not ggufs:
        die(f"{repo} contains no GGUF files.")

    log(f"Repo: {repo}")
    log(f"  GGUF files: {len(ggufs)}, vision projectors: {len(mmproj)}")
    quants = {}
    for n, s in sorted(ggufs, key=lambda x: x[1]):
        m = re.search(r"(IQ\d\w*|Q\d_K_?\w*|Q\d_\d|BF16|F16|F32)", n)
        q = m.group(1) if m else "?"
        quants.setdefault(q, (n, s))
    for q, (n, s) in quants.items():
        marks = []
        if q.startswith("IQ"):
            marks.append("IQ-quant: unverified on this stack, prefer K-quants")
        if q in ("F16", "F32", "BF16"):
            marks.append("full precision: likely too large")
        log(f"    {q:10s} {s/1e9:7.2f} GB  {n}" +
            ("   [" + "; ".join(marks) + "]" if marks else ""))

    probe = args.file or min(ggufs, key=lambda x: x[1])[0]
    log(f"  Probing architecture from header of: {probe}")
    meta = hf_probe_arch(repo, probe)
    arch = meta.get("general.architecture", "?")
    verdict, why = classify_arch(arch)
    log(f"  architecture = {arch!r}  ->  {verdict}")
    log(f"  {why}")

    # RAM sanity
    try:
        avail_kb = int(re.search(r"MemAvailable:\s+(\d+)",
                       open("/proc/meminfo").read()).group(1))
        log(f"  MemAvailable now: {avail_kb/1e6:.1f} GB (model stays resident "
            "once loaded; no idle unload on this stack)")
    except Exception:
        pass
    if verdict in ("BROKEN", "UNSUPPORTED"):
        sys.exit(2)


def _download(url, dest, label):
    tmp = dest + ".part"
    req = urllib.request.Request(url)
    tok = os.environ.get("HF_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        t0 = time.time()
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total and time.time() - t0 > 3:
                t0 = time.time()
                log(f"    {label}: {done/1e9:.2f}/{total/1e9:.2f} GB")
    with open(tmp, "rb") as f:
        if f.read(4) != b"GGUF":
            os.unlink(tmp)
            die(f"{label}: downloaded file is not GGUF (gated repo? set HF_TOKEN)")
    os.rename(tmp, dest)


def _staging_dir():
    for cand in ("/volume1/docker", "/volume1", os.path.expanduser("~")):
        if os.path.isdir(cand) and os.access(cand, os.W_OK):
            d = os.path.join(cand, ".ugos-llm-staging")
            os.makedirs(d, exist_ok=True)
            return d
    die("no writable staging directory found")


def cmd_install(args):
    repo = args.repo
    name = args.name or re.sub(r"-?GGUF$", "", repo.split("/")[-1], flags=re.I)
    target = f"{MODELS_DIR}/{name}"
    files = hf_repo_files(repo)
    ggufs = [(n, s) for n, s in files if n.endswith(".gguf")
             and "mmproj" not in n.lower()]
    pick = [n for n, s in ggufs if args.quant.lower() in n.lower()]
    if not pick:
        die(f"no GGUF matching quant {args.quant!r} in {repo}. "
            f"Available: {', '.join(n for n, _ in ggufs)}")
    model_file = pick[0]

    mmproj_file = None
    if args.vision:
        mm = [n for n, s in files if "mmproj" in n.lower()
              and ("f16" in n.lower() or "fp16" in n.lower())] or \
             [n for n, s in files if "mmproj" in n.lower()]
        if not mm:
            die("--vision requested but the repo has no mmproj file.")
        mmproj_file = mm[0]

    # pre-flight
    meta = hf_probe_arch(repo, model_file)
    verdict, why = classify_arch(meta.get("general.architecture"))
    log(f"Architecture {meta.get('general.architecture')!r}: {verdict} — {why}")
    if verdict == "BROKEN" and not args.force:
        die("refusing to install a known-broken architecture (use --force "
            "if you enjoy token soup).")
    if verdict == "UNSUPPORTED" and not args.force:
        die("architecture missing from the shipped llama.cpp build "
            "(--force to try anyway).")

    stage = _staging_dir()
    log(f"Staging in {stage}")
    local_model = os.path.join(stage, os.path.basename(model_file))
    if not os.path.exists(local_model):
        _download(f"https://huggingface.co/{repo}/resolve/main/{model_file}",
                  local_model, "model")
    local_mm = None
    if mmproj_file:
        local_mm = os.path.join(stage, f"{name}-" + os.path.basename(mmproj_file))
        if not os.path.exists(local_mm):
            _download(f"https://huggingface.co/{repo}/resolve/main/{mmproj_file}",
                      local_mm, "mmproj")

    cfg = {
        "num_ctx": args.ctx,
        "context_length": args.ctx,
        "extra_args": [a if a != "16384" else str(args.ctx)
                       for a in SAFE_EXTRA_ARGS],
        "capabilities": ["completion"] + (["vision"] if mmproj_file else []),
    }
    if mmproj_file:
        cfg["mmproj"] = "mmproj.gguf"
    cfg_local = os.path.join(stage, f"{name}.model_config.json")
    with open(cfg_local, "w") as f:
        json.dump(cfg, f, indent=2)

    log(f"Installing into {target}")
    script = (f"set -e; mkdir -p /t; "
              f"cp /s/{os.path.basename(local_model)} /t/{name}.gguf; "
              + (f"cp /s/{os.path.basename(local_mm)} /t/mmproj.gguf; "
                 if local_mm else "")
              + f"cp /s/{os.path.basename(cfg_local)} /t/model_config.json; "
              f"chmod 755 /t /t/*")
    r = root_sh(script, {stage: "/s", target: "/t"})
    if r.returncode != 0:
        die(f"install failed: {r.stderr.strip()[:300]}")

    ids = gw_models()
    model_id = f"{name}/{name}"
    if ids is not None:
        log(f"Gateway sees: {model_id}" if any(name in i for i in ids)
            else "Gateway has not picked it up yet (it rescans on demand).")

    if args.ui:
        _ui_add(name, cfg, os.path.getsize(local_model))

    if not args.keep_staging:
        for p in (local_model, local_mm, cfg_local):
            if p and os.path.exists(p):
                os.unlink(p)

    log("")
    log(f"DONE. Model id for API calls: {model_id}")
    reload_hint()
    log(f"Then run:  python3 {sys.argv[0]} test {name}")


def _ui_add(name, cfg, size_bytes):
    rows = root_sqlite("SELECT id FROM model_config WHERE code = "
                       f"'{name}'", readonly=True)
    if rows:
        log(f"UI: catalog row for {name} already exists.")
        return
    backup = os.path.join(_staging_dir(),
                          f"aiconsole.db.backup-{time.strftime('%Y%m%d-%H%M%S')}")
    shutil.copyfile(AICONSOLE_DB, backup)
    log(f"UI: catalog backup -> {backup}")

    donor = root_sqlite("SELECT ext_arch_tools FROM model_config WHERE "
                        "model_type='llm' ORDER BY id LIMIT 1", readonly=True)
    ext_tools = donor[0][0].replace("'", "''") if donor else ""
    nid = root_sqlite("SELECT COALESCE(MAX(id),99) FROM model_config",
                      readonly=True)[0][0]
    nid = max(100, nid + 1)
    ext = json.dumps({"num_ctx": cfg["num_ctx"],
                      "context_length": cfg["context_length"],
                      "capabilities": cfg["capabilities"]})
    i18n = json.dumps([{"description": f"Custom model installed by ugos-llm",
                        "language": lang, "modelType": "llm",
                        "modelTypeDesc": "Large Language Models",
                        "modelTypeName": "Large Language Models",
                        "name": name}
                       for lang in ("en-US", "de-DE")]).replace("'", "''")
    sql = (f"INSERT INTO model_config (id,release_id,code,name,status,"
           f"\"update\",version,version_num,min_version,model_type,"
           f"param_value,response_speed,memory_usage,is_default,"
           f"version_description,version_size,ext,ext_i18n,ext_arch_tools,"
           f"install_paths,created_at,updated_at) VALUES ({nid},9000,"
           f"'{name}','{name}',8,'{{\"status\":0}}','v1.0.0',1,0,'llm','-',"
           f"'-','{size_bytes/1e9:.0f} GB',0,'',{size_bytes},'{ext}',"
           f"'{i18n}','{ext_tools}','',datetime('now'),datetime('now'));")
    root_sqlite(sql)
    log(f"UI: catalog row added (id={nid}). Restore-backup kept in staging.")


def cmd_list(args):
    log(f"{'model dir':30s} {'catalog':14s} {'loaded (ub)'}")
    loaded = dict(loaded_servers())
    try:
        dirs = sorted(d for d in os.listdir(MODELS_DIR)
                      if os.path.isdir(f"{MODELS_DIR}/{d}"))
    except PermissionError:
        die(f"cannot read {MODELS_DIR}")
    cat = {c: s for c, s in root_sqlite(
        "SELECT code,status FROM model_config WHERE model_type='llm'",
        readonly=True)}
    for d in dirs:
        if d.startswith((".", "infer_")):
            continue
        state = STATUS.get(cat.get(d, -1), "no UI entry")
        lo = f"yes (ub={loaded[d]})" if d in loaded else "-"
        warn = "  <-- ub!=512, see known-bugs!" \
            if d in loaded and loaded[d] not in ("512", "?") else ""
        log(f"{d:30s} {state:14s} {lo}{warn}")
    ids = gw_models()
    log(f"\nGateway models endpoint: "
        f"{'OK, ' + str(len(ids)) + ' entries' if ids else 'UNREACHABLE'}")


def cmd_test(args):
    name = args.name
    model_id = f"{name}/{name}"
    ids = gw_models()
    if ids is None:
        die("gateway unreachable on 127.0.0.1:62891")
    if not any(i.startswith(name + "/") for i in ids):
        die(f"gateway does not list {name}. Installed? Reloaded?")
    results = []

    def chat(msgs, **kw):
        p = {"model": model_id, "messages": msgs, "max_tokens": 200,
             "temperature": 0.1, **kw}
        t0 = time.time()
        d = gw("/v1/chat/completions", p, timeout=600)
        ch = d["choices"][0]
        return (ch.get("message", {}).get("content") or "",
                ch.get("message", {}).get("tool_calls"),
                time.time() - t0)

    log("1/4 chat smoke test (loads the model on first call) ...")
    txt, _, dt = chat([{"role": "user",
                        "content": "Antworte in einem Satz: Was ist ein NAS?"}])
    ok = collapse_score(txt) > 0.5 and len(txt) > 10
    results.append(("chat", ok, dt, txt[:80]))

    log("2/4 long-prompt stability (the ub=4096 bug detector) ...")
    para = ("Die Lagerhalle wurde 1987 errichtet und mehrfach modernisiert. "
            "Im Erdgeschoss lagern Ersatzteile, das Obergeschoss dient als "
            "Buero. Die Heizung stammt aus dem Jahr 2015. ") * 90
    txt, _, dt = chat([{"role": "user", "content":
                        para + "\n\nIn welchem Jahr wurde die Halle errichtet? "
                        "Antworte in einem Satz."}])
    ok = collapse_score(txt) > 0.5
    results.append(("long-prompt", ok, dt, txt[:80]))

    log("3/4 tool-calling ...")
    tool = {"type": "function", "function": {
        "name": "get_system_info",
        "description": "Get system information for a category.",
        "parameters": {"type": "object", "properties": {
            "category": {"type": "string",
                         "enum": ["storage", "cpu", "memory"]}},
            "required": ["category"]}}}
    txt, calls, dt = chat([{"role": "user", "content":
                            "Wie viel Speicherplatz ist noch frei?"}],
                          tools=[tool])
    ok = bool(calls) or (collapse_score(txt) > 0.5 and len(txt) > 10)
    detail = (f"tool_call {calls[0]['function']['name']}" if calls
              else txt[:80])
    results.append(("tools", ok, dt, detail))

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
        results.append(("vision", "rot" in txt.lower() or "red" in txt.lower(),
                        dt, txt[:40]))

    log("")
    fails = 0
    for label, ok, dt, detail in results:
        fails += 0 if ok else 1
        log(f"  {'PASS' if ok else 'FAIL':4s} {label:12s} {dt:5.1f}s  {detail}")
    log("")
    if fails:
        log("Some checks failed. See docs/known-bugs.md — the usual suspects "
            "are ub=4096 configs and MoE/unsupported architectures.")
        sys.exit(1)
    log("All checks passed. Consider reporting this model+quant as working.")


def cmd_doctor(args):
    ids = gw_models()
    log(f"gateway 127.0.0.1:62891 : {'OK' if ids is not None else 'DOWN'}")
    if ids:
        for i in ids:
            log(f"  - {i}")
    for name, ub in loaded_servers():
        flag = "" if ub == "512" else "   <-- not 512: long-prompt corruption!"
        log(f"loaded: {name} (ub={ub}){flag}")
    try:
        for d in sorted(os.listdir(MODELS_DIR)):
            cfgp = f"{MODELS_DIR}/{d}/model_config.json"
            if os.path.isfile(cfgp):
                cfg = json.load(open(cfgp))
                ea = cfg.get("extra_args", [])
                if "4096" in ea:
                    log(f"config lint: {d} uses -ub/-b 4096 "
                        "-> edit to 512 (docs/known-bugs.md) and reload")
    except PermissionError:
        pass
    rows = root_sqlite("SELECT code,status FROM model_config WHERE "
                       "model_type='llm'", readonly=True)
    for code, st in rows:
        log(f"catalog: {code} = {STATUS.get(st, st)}")
    log("")
    log("Reminders: never pkill llama-server; reload = UI toggle or reboot; "
        "model_duration in system_setting has no effect.")


def cmd_ui(args):
    if args.action == "remove":
        backup = os.path.join(_staging_dir(),
                              f"aiconsole.db.backup-{time.strftime('%Y%m%d-%H%M%S')}")
        shutil.copyfile(AICONSOLE_DB, backup)
        root_sqlite(f"DELETE FROM model_config WHERE code='{args.name}' "
                    "AND release_id >= 9000;")
        log(f"UI row removed (backup: {backup}). Only rows created by this "
            "tool (release_id>=9000) are ever deleted.")
    else:
        cfgp = f"{MODELS_DIR}/{args.name}/model_config.json"
        if not os.path.isfile(cfgp):
            die(f"{args.name} is not installed under {MODELS_DIR}")
        cfg = json.load(open(cfgp))
        size = os.path.getsize(f"{MODELS_DIR}/{args.name}/{args.name}.gguf")
        _ui_add(args.name, cfg, size)


def cmd_remove(args):
    target = f"{MODELS_DIR}/{args.name}"
    if not os.path.isdir(target):
        die(f"{target} does not exist")
    if not args.yes:
        die(f"refusing without --yes (would delete {target})")
    r = root_sh("rm -rf /m/" + args.name, {MODELS_DIR: "/m"})
    if r.returncode != 0:
        die(f"remove failed: {r.stderr.strip()[:200]}")
    if not args.keep_ui:
        root_sqlite(f"DELETE FROM model_config WHERE code='{args.name}' "
                    "AND release_id >= 9000;")
    log(f"{args.name} removed. If it was loaded, the server process keeps "
        "running until reload (UI toggle/reboot).")


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(prog="ugos-llm", description=__doc__,
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
                   help="add a Model Manager UI card (writes catalog DB, "
                        "creates a backup first)")
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

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
