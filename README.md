# ugos-llm

**Run your own LLMs on UGREEN NASync AI NAS devices — today.**

UGREEN's UGOS Pro ships a locked-down Model Manager with a single built-in model
(Qwen3.5 4B) and says third-party model support is "coming in a future update".
It turns out the underlying stack — a llama.cpp server pool behind a local
OpenAI-compatible gateway — already loads any compatible GGUF you place in the
right directory with the right config. This project documents how the stack
works and ships a CLI that automates the whole thing, including the
**critical stability fix** without which most third-party models produce
garbage output (see [Known bugs](docs/known-bugs.md)).

> **Tested on:** UGREEN NASync **iDX6011** (Core Ultra 5 125H, 32 GB SKU),
> UGOS Pro, Model Manager `1.17.0.0055`, infer_gateway `1.0.0.0004`,
> llama.cpp build `b8413` (SYCL).
> **Expected to work identically** on the iDX6011 64 GB SKU and the
> **NASync iDX6011 Pro** (Core Ultra 7 255H, 64 GB) — same UGOS AI stack,
> more RAM headroom for larger models. Reports welcome.
> **A UGOS update may change any of this.**

## What you get

- `ugos-llm.py` — a single-file, stdlib-only CLI:
  - `check` a Hugging Face GGUF repo for compatibility **before** downloading
  - `install` a model (download, configure, register) with safe defaults
  - `test` an installed model (chat, long-prompt stability, tool calls, vision)
  - `list`, `remove`, `doctor`, `ui` (Model Manager catalog entry)
- [docs/](docs/) — how the UGOS AI stack actually works (gateway, sockets,
  catalog DB, logs), a model compatibility matrix, all known bugs with fixes,
  and a debugging toolkit
- [openai-bridge/](openai-bridge/) *(optional)* — a small reverse proxy that
  exposes the UGOS LLM gateway as a **robust** OpenAI-compatible endpoint for
  your Docker apps, working around gateway bugs (tool-calling requests hang or
  degrade without it)
- [examples/paperless-ngx/](examples/paperless-ngx/) *(optional)* — a complete
  worked example: Paperless-ngx document management with AI suggestions powered
  entirely by your NAS-local model

## Quickstart

The tool always **executes on the NAS** (it needs local paths and the
loopback-only gateway) — but you drive it from any OS over SSH. Enable SSH in
UGOS first: Control Panel → Terminal.

On the NAS (SSH session, user in the `docker` group — or root):

```bash
curl -LO https://raw.githubusercontent.com/semih44/ugos-llm/main/ugos-llm.py

# 1. Pre-flight: is this model going to work?
python3 ugos-llm.py check unsloth/Qwen3.5-9B-GGUF

# 2. Install it (Q4_K_M quant, with vision + Model Manager UI card) and
#    verify it in one go — the gateway loads new models on first use,
#    no reload or reboot needed:
python3 ugos-llm.py install unsloth/Qwen3.5-9B-GGUF \
    --quant Q4_K_M --vision --ui --test
```

After that, the model is selectable in Uliya (chat, knowledge base, intelligent
commands), Universal Search, and reachable for your own apps via the gateway.

### From Windows, macOS or Linux — no install on the NAS needed

Windows 10+ ships an OpenSSH client, so PowerShell works out of the box.
You can even run the tool **without copying it to the NAS** by piping it
through SSH:

```powershell
# PowerShell (Windows)
Get-Content -Raw ugos-llm.py | ssh you@nas-ip "python3 - list"
Get-Content -Raw ugos-llm.py | ssh you@nas-ip "python3 - check unsloth/Qwen3.5-9B-GGUF"
```

```bash
# macOS / Linux
ssh you@nas-ip "python3 - list" < ugos-llm.py
```

For regular use, park it on the NAS once (e.g. `/volume1/docker/ugos-llm.py`)
and call it directly. Copying via `scp` requires the SFTP service to be
enabled in UGOS (Control Panel → Terminal) — otherwise `scp` fails with a
misleading "No such file or directory".

### About root/sudo

`check`, `list`, `test` and `doctor` need no privileges. `install`, `remove`
and `ui` write to root-owned paths; two options:

- **docker group (recommended, no sudo at all):**
  `sudo usermod -aG docker $USER` once, log out/in — the tool then routes
  privileged file operations through pinned containers.
- **sudo:** run those commands as
  `ssh -t you@nas-ip "sudo python3 /volume1/docker/ugos-llm.py install …"`
  (`-t` gives you the password prompt; UGOS sudo always asks).

There is no sudo on the Windows side — privileges are only ever needed on
the NAS, and SSH takes care of that.

## The one fix you must not skip

UGREEN's own server arguments use microbatch `-ub 4096`. On the SYCL build the
device ships, **that setting silently computes wrong results** for long,
structure-heavy prompts (such as the ~4–5k-token tool catalog Uliya sends for
intelligent commands): the model degenerates into endless `- - -` / `1. 1. 1.`
repetition loops, deterministically, while short prompts work fine. UGREEN's
own RL-tuned 4B masks the issue; virtually any vanilla model exposes it.

**`ugos-llm.py` always configures `-ub 512 -b 512`**, which fixes correctness
*and* roughly doubles prompt-processing speed (182 vs 91 t/s measured).
Full analysis: [docs/known-bugs.md](docs/known-bugs.md).

## Compatibility (short version)

| Works | Doesn't |
|---|---|
| Dense Qwen3.5 / Qwen3 / Qwen2.5 (tested: Qwen3.5-9B incl. vision) | **Any MoE model** (SYCL build computes garbage — every quant) |
| Dense Llama 3.x, Gemma 2/3, Mistral (architecture present, untested) | Gemma 4, Qwen 3.6 (architecture missing in b8413) |
| K-quants (Q4_K_M, Q5_K_M, …) | IQ-quants: unverified, treat as risky |

Details and how to qualify new models: [docs/compatibility.md](docs/compatibility.md).

## Runtime profiles (experimental): newer llama.cpp per model

Everything in the table above describes UGREEN's shipped build (b8413). A
current upstream llama.cpp fixes the `-ub 4096` bug, runs MoE models
correctly and adds MTP speculative decoding (measured: Gemma 4 26B-A4B
answers structured/JSON prompts *faster* than a dense 9B on the vendor
build — see [docs/known-bugs.md §6](docs/known-bugs.md)). The `runtime`
command lets individual models opt in to such a build while every other
model — including UGREEN's own — keeps running the vendor runtime:

```bash
# one-time: deploy a self-built glibc-2.36-compatible runtime, then
# swap UGREEN's 258-byte llama-server wrapper for the dispatcher
# (the original is preserved and remains the default path)
python3 ugos-llm.py runtime deploy /path/to/runtime-dir --name upstream-b10143
python3 ugos-llm.py runtime enable

# install a model on that runtime, with its MTP draft head
python3 ugos-llm.py install unsloth/gemma-4-26B-A4B-it-qat-GGUF \
    --quant UD-Q4_K_XL --vision --draft --runtime upstream-b10143 --ui --test

python3 ugos-llm.py runtime status   # dispatcher + marker overview
python3 ugos-llm.py runtime disable  # restore the vendor wrapper
```

Models opt in via a `.ugos-llm-runtime` marker file in their model
directory; the dispatcher routes them at spawn time and fails fast if the
requested runtime is missing (falling back silently would crash on
upstream-only flags). `doctor` warns when a firmware update has restored
the vendor wrapper. How to produce such a runtime build is documented in
[docs/known-bugs.md §6](docs/known-bugs.md) — the short version: build the
pinned llama.cpp commit with `-DGGML_SYCL=ON` against Ubuntu 22.04
(glibc ≤ 2.36) and verify no binary needs `GLIBC_2.37+`.

## Safety

This runs with root privileges on your NAS, so it is built defensively:

- **No shell for privileged work.** File operations run either natively (as
  root) or through pinned containers with explicit `argv` — never `sh -c`.
- **Strict name validation.** Model names must match `[A-Za-z0-9][A-Za-z0-9._-]{0,63}`,
  so no path traversal, no metacharacters.
- **Parameterised SQL only.** The Model Manager catalog is never touched by
  string interpolation, and a **consistent backup** (SQLite backup API, so the
  WAL is included) is written before every change.
- **Vendor rows are untouchable.** Catalog rows created by this tool carry
  `release_id = 9000`; `remove` refuses to delete anything else.
- **Never `pkill`.** The gateway does not survive it (reboot required); the CLI
  only ever prints the safe reload paths.
- **Versioned helper images** (`alpine:3.20`, `python:3.12-alpine`) instead of
  moving `latest` tags. Version tags are still mutable — for bit-for-bit
  reproducibility override them with digests via `UGOS_LLM_IMG_BUSYBOX` /
  `UGOS_LLM_IMG_PYTHON`.
- **Atomic installs with rollback and crash recovery.** A model is built in a
  temporary directory and swapped in only when complete. A previous
  installation is parked as a rollback copy until the swap succeeds, an
  interrupted swap (power loss) is detected and repaired on the next run, and
  a lock directory prevents two installs of the same model from colliding.
- **Data files are installed `644`** (no execute bit), directories `755`.
- This project redistributes **no** UGREEN binaries, models or templates.
  Everything it touches on your NAS stays on your NAS.

Known limitation: split/multi-shard GGUFs (`…-00001-of-00003.gguf`) are
detected and **rejected** rather than half-installed. Pick a single-file quant.

## Development

```bash
python3 -m unittest discover -s tests -v    # 48 unit tests, stdlib only
```

CI runs the suite on Python 3.9 and 3.12 plus a ruff lint. The risky parts —
name validation, GGUF header parsing, architecture rules, split detection,
atomic install/rollback and the proxy's rewrite/wrap logic — are covered by
tests; the NAS-facing parts are verified manually on hardware (see the version
banner above).

Known limits, stated plainly: the bridge checks that a synthetic tool call
carries the schema's *required* top-level keys, but it is **not** a JSON Schema
validator (no type/enum/nested checks) — validate tool arguments in your
application, as you would with any model output.

## Disclaimer

Not affiliated with or endorsed by UGREEN. This modifies the behavior of an
appliance in ways the vendor doesn't officially support yet; a firmware update
may undo or break it (the models and configs survive in practice — the UI
catalog entry is the fragile part, and `ui add` restores it in seconds).
Use at your own risk.

## License

MIT
