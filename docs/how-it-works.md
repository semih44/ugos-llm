# How the UGOS AI stack actually works

Everything below was established by direct inspection on an iDX6011
(UGOS Pro, Model Manager 1.17.0.0055) — process lists, binary strings,
log files, SQLite schema and live probing. No UGREEN source code involved.

## Architecture

```
UGOS apps (Uliya, Universal Search, AI Album, …)
        │                                ┌──────────────────────────────┐
        ▼                                │  your Docker apps            │
aiconsole_serv ──┐                       │  (via openai-bridge, opt.)   │
ai_mcp_serv ─────┤                       └───────────────┬──────────────┘
                 ▼                                       ▼
        infer_gateway_serv  ◄────────────────  127.0.0.1:62891 (HTTP)
                 │   OpenAI-compatible: /v1/models, /v1/chat/completions
                 │   (also partial, non-functional Ollama routes: /api/tags)
                 ▼
        llama-server pool — one process per model, spawned on demand,
        listening on unix sockets /run/ugreen/llama-gw-socks/gw-<port>.sock
        (llama.cpp b8413, IntelLLVM/SYCL build, iGPU via -ngl 999)
```

- The gateway (`infer_gateway_serv`, Go) scans the model directory, spawns
  `llama-server` per model **lazily on first request**, and proxies
  OpenAI-style HTTP to the unix sockets.
- `ai_mcp_serv` (port 127.0.0.1:11540) is an MCP server providing the tool
  catalog for "intelligent commands"; `aiconsole_serv` orchestrates apps,
  catalog DB and downloads.
- The llama.cpp binaries live in `/volume1/@aiconsole/llamacppSycl/` (SYCL,
  the one actually used) and `/volume1/@aiconsole/llamacppGpu/` (CUDA build,
  idle on Intel hardware).
- The gateway is a **Go binary that links only against libc** (`ldd` shows
  vdso, libc, ld-linux — nothing else) and spawns `.llama-server` as a
  separate child process. It therefore does *not* pin you to the bundled
  llama.cpp version: replacing the runtime bundle is a matter of ABI-free
  process invocation, not linking. What must match is the CLI surface
  (`-m`, `--host <unix-socket>`, `-ngl`, `-ub`, …), unix-socket support, and
  the host's glibc (Debian 12 / glibc 2.36 on the iDX6011 — official
  llama.cpp Intel images are built on newer Ubuntu and need a rebuild).

## Model directory layout

Models live in `/opt/ugreen/ai/models/<Name>/` — a dedicated 64 GB ext4
partition on the system SSD (not your storage volumes):

```
/opt/ugreen/ai/models/Qwen3.5-9B/
├── Qwen3.5-9B.gguf          # the model
├── mmproj.gguf              # optional: vision projector
├── model_config.json        # per-model server configuration
└── infer_gateway_cache/     # created by the gateway (HOME for SYCL caches)
```

`model_config.json` (this is the entire integration contract):

```json
{
  "num_ctx": 16384,
  "context_length": 16384,
  "mmproj": "mmproj.gguf",
  "extra_args": ["-ngl","999","-t","10","-e","-lv","3","--no-warmup",
                 "--no_mmap","-fa","off","-np","1",
                 "-ub","512","-b","512","-c","16384"],
  "capabilities": ["completion","vision"]
}
```

`extra_args` are passed to `llama-server` verbatim — this is where the
crucial `-ub 512` lives (UGREEN's default 4096 is broken, see known-bugs).
The gateway's directory watcher notices new model folders **immediately**
(they appear in `/v1/models` without any restart). Model id =
`<DirName>/<gguf-basename-without-extension>`.

UGREEN's own model download flow stages into `/volume1/@aiconsole/models/`
and deploys to `/opt/ugreen/ai/models/` — you can write the target directly.

## The catalog database (what the Model Manager UI shows)

The UI does **not** read the filesystem. It reads SQLite:
`/volume1/@appstore/com.ugreen.aiconsole/db/aiconsole.db`, table
`model_config`. Relevant columns: `code` (must equal the model directory
name), `name`, `status`, `model_type` (`llm`), `ext` (JSON: num_ctx,
capabilities), `ext_i18n` (JSON: per-language display name/description),
`ext_arch_tools` (JSON describing the llama.cpp runtime — clone it from an
existing row).

Status values (observed): `1` = available in catalog, not installed ·
`3` = installed but disabled · `8` = active.

A model **works without any catalog row** (the gateway serves it anyway);
the row only makes it visible/selectable in UGOS UIs. The UI's
enable/disable toggle is also the **only clean way to stop/start** a
model's server process.

## Reload semantics (important!)

- `llama-server` reads `model_config.json` and the chat template **only at
  spawn**.
- Reliable reload paths: UI toggle (off/on) or a NAS reboot. Nothing else:
  - touching/rewriting `model_config.json` triggered a graceful swap exactly
    once in our testing and then never again — unreliable;
  - setting `status` directly in the DB does nothing (the service doesn't
    poll it);
  - `system_setting.model_duration` (idle unload) is ignored — loaded models
    stay resident forever;
  - **killing a llama-server corrupts the gateway** (it keeps dead socket
    references and returns `connection refused` until reboot — and its
    respawned twin fights over the port). Never do this.

## Where the logs are

| What | Where |
|---|---|
| Gateway request log — **only failures**, but with full request payloads | `/var/ugreen/log/infer_gateway_serv.log` |
| llama-server console (timings, per-task token counts, template info) | `/var/ugreen/log/infer_gateway_serv_panic/*.log` (despite the name, these are normal startup/console logs) |
| aiconsole service | `/var/ugreen/log/aiconsole_serv.log` |
| MCP server | `/var/ugreen/log/ai_mcp_serv.log` |

All root-only; read them via `docker run --rm -v /var/ugreen/log:/log:ro
alpine cat …` if you're not root.

## Probing endpoints (gold for debugging)

The unix socket of a running llama-server accepts the full llama.cpp server
API, bypassing the gateway entirely — including endpoints the gateway
doesn't proxy:

```bash
SOCK=/run/ugreen/llama-gw-socks/gw-41080.sock   # root only
curl --unix-socket $SOCK http://localhost/apply-template -d '{...}'  # rendered prompt
curl --unix-socket $SOCK http://localhost/tokenize -d '{"content":"..."}'
curl --unix-socket $SOCK http://localhost/completion -d '{"prompt":"...", ...}'
```

`/apply-template` shows you the **exact** prompt the model receives after
chat-template rendering (tools included) — indispensable when output looks
insane. See [troubleshooting.md](troubleshooting.md).

## Version pinning

Everything here was verified against:

| Component | Version |
|---|---|
| Model Manager (aiconsole) | 1.17.0.0055 (build 2026-06-30) |
| infer_gateway_serv | 1.0.0.0004 (build 2026-06-10) |
| llama.cpp | b8413, IntelLLVM 2025.3.2, SYCL |

UGREEN has announced an Ollama-based runtime (init scripts for it already
ship in `/volume1/@appstore/com.ugreen.aiconsole/init.d/ollama_*`). When
that lands, much of this document becomes historical — happily.
