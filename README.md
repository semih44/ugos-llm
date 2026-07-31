# ugos-llm

**Run your own LLMs on UGREEN NASync AI NAS devices — up to Gemma 4 26B, natively.**

UGREEN's UGOS Pro ships a locked-down Model Manager with a single built-in model
(Qwen3.5 4B) and says third-party model support is "coming in a future update".
The underlying stack — a llama.cpp server pool behind a local OpenAI-compatible
gateway — will in fact load any compatible GGUF you put in the right directory
with the right config. This project documents that stack and ships a CLI that
drives it.

The headline result: **Gemma 4 26B-A4B runs natively on an iDX6011.** It shows up
as a card in the Model Manager, answers from Uliya, Universal Search and your own
apps, reads images, and returns **real tool calls** through the gateway. On
structured output it generates *faster* than the dense 9B that the shipped
runtime tops out at — because the llama.cpp underneath it supports **MTP
speculative decoding**.

The runtime UGREEN ships cannot do any of that: it has no Gemma 4 architecture
compiled in and miscomputes every MoE model. So this project works at two levels,
and you can stop after the first:

- **Any compatible GGUF on the runtime your NAS already has.** Five minutes, no
  build. Includes the **critical stability fix** without which most third-party
  models produce garbage output.
- **Or a newer llama.cpp you build yourself**, deployed *alongside* UGREEN's and
  selected per model. That is the Gemma 4 / MoE / MTP path.

> **Tested on:** UGREEN NASync **iDX6011** (Core Ultra 5 125H, 32 GB SKU),
> UGOS Pro, Model Manager `1.17.0.0055`, infer_gateway `1.0.0.0004`,
> llama.cpp build `b8413` (SYCL, vendor) and a self-built `b10143`
> (SYCL, upstream — see the Gemma 4 section).
> **Expected to work identically** on the iDX6011 64 GB SKU and the
> **NASync iDX6011 Pro** (Core Ultra 7 255H, 64 GB) — same UGOS AI stack,
> more RAM headroom for larger models. Reports welcome.
> **A UGOS update may change any of this.**

## Which path is yours?

| | **Any GGUF, stock runtime** | **Gemma 4 26B, your own runtime** |
|---|---|---|
| Effort | ~5 min, no build | one container build (~30 min) + ~15 GB download |
| Unlocks | dense models up to ~9B, vision, tools via the bridge | Gemma 4, MoE, MTP speculative decoding, native tool calls |
| Touches | nothing vendor-owned — models and a catalog row are added | replaces one vendor wrapper script (backed up, one command to revert) |
| Survives a firmware update | yes, in practice | the dispatcher may be overwritten; `doctor` detects it, `runtime enable` restores it |
| Start at | [Quickstart](#quickstart) | [Gemma 4 26B-A4B, natively](#gemma-4-26b-a4b-natively) |

## What you get

- `ugos-llm.py` — a single-file, stdlib-only CLI:
  - `check` a Hugging Face GGUF repo for compatibility **before** downloading
  - `install` a model (download, configure, register) with safe defaults,
    optionally with a vision projector and an MTP draft head
  - `test` an installed model (chat, long-prompt stability, tool calls, vision)
  - `runtime` deploy/enable a newer llama.cpp alongside the vendor's
  - `list`, `remove`, `doctor`, `ui` (Model Manager catalog entry)
- [scripts/build-runtime.sh](scripts/build-runtime.sh) — the reproducible,
  ELF-gated recipe for building that newer llama.cpp so it actually loads on
  UGOS (glibc 2.36)
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

The five-minute path, on the runtime your NAS already has. The tool always
**executes on the NAS** (it needs local paths and the loopback-only gateway) —
but you drive it from any OS over SSH. Enable SSH in UGOS first:
Control Panel → Terminal.

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

For Gemma 4 or any MoE model, keep reading — those need the second path.

## Gemma 4 26B-A4B, natively

The shipped b8413 (April 2026 vintage) has no Gemma 4 architecture and
miscomputes every MoE model. A current upstream llama.cpp fixes both, shows
no sign of the `-ub 4096` corruption in our tests (a 1.9k-token structured
prompt and the CLI's long-prompt check both came out correct at `-ub 4096`,
and twice as fast), and adds **MTP speculative decoding** — a small draft head
that proposes several tokens at once, which the main model then verifies in a
single batch.

Measured on Gemma 4 26B-A4B (Q4 QAT, 14.2 GB, 26B total / ~4B active):

| Workload (Gemma 4 26B-A4B + MTP) | Generation | Draft acceptance |
|---|---|---|
| JSON / structured output | **14.8 t/s** | 79 % |
| Short answers | 13.4 t/s | 74 % |
| Free-form prose | 9.9 t/s | 44 % |
| Long context (2.5k tokens) | 6.3 t/s | 43 % |

For reference, the dense Qwen3.5-9B generates at ~12.3 t/s on the vendor
runtime. So: a 26B-class model that answers *structured* prompts faster than
a dense 9B, runs about 20 % slower on free prose, and roughly halves
throughput at long context — speculative decoding buys nothing where the
output is unpredictable, because drafts only pay off when they survive
verification. That sweep was measured in a container; on the self-built
runtime this project deploys, the JSON case came out at 16.6 t/s at 90 %
acceptance, with prompt processing at 115.8 t/s over 2.9k tokens.

Bonus: the upstream server's **native tool calling survives the gateway**,
which b8413 never managed — that is what [openai-bridge/](openai-bridge/)
exists to work around. Full numbers, and the five traps that cost us a day
each, are in [docs/known-bugs.md §6](docs/known-bugs.md).

### Building the runtime

One container run. [scripts/build-runtime.sh](scripts/build-runtime.sh) pins the
exact commit, builds against Ubuntu 22.04 for glibc ≤ 2.36, and **fails the
build** if any binary would need `GLIBC_2.37+` or if a dependency stays
unresolved:

```bash
# fetch the recipe onto the NAS, then build (~30 min, needs ~20 GB scratch)
mkdir -p /volume1/docker/ugos-llm-build
curl -Lo /volume1/docker/ugos-llm-build/build-runtime.sh \
  https://raw.githubusercontent.com/semih44/ugos-llm/main/scripts/build-runtime.sh

docker run --rm -v /volume1/docker/ugos-llm-build:/out \
  intel/oneapi-basekit:2025.3.2-0-devel-ubuntu22.04 bash /out/build-runtime.sh

# result: /volume1/docker/ugos-llm-build/ugos-llm-runtime-b10143/
```

Two build flags are load-bearing and explained in the script: `GGML_SYCL=ON`
(obviously) and `GGML_SYCL_DNN=OFF` — with oneDNN compiled in, the build
either crashes or re-JITs kernels on every prompt batch, which drags prompt
processing from 115 t/s down to under 2.

### Deploying it and installing Gemma

```bash
# one-time: install the runtime alongside the vendor's, then swap UGREEN's
# 254-byte llama-server wrapper for the dispatcher (the original is
# preserved and stays the default path for every other model)
python3 ugos-llm.py runtime deploy \
    /volume1/docker/ugos-llm-build/ugos-llm-runtime-b10143 --name upstream-b10143
python3 ugos-llm.py runtime enable

# install Gemma 4 on that runtime, with vision and its MTP draft head;
# --thinking off reproduces the benchmarked configuration
python3 ugos-llm.py install unsloth/gemma-4-26B-A4B-it-qat-GGUF \
    --quant UD-Q4_K_XL --vision --draft --thinking off \
    --runtime upstream-b10143 --ui --test

python3 ugos-llm.py runtime status   # dispatcher + marker overview
python3 ugos-llm.py runtime disable  # restore the vendor wrapper
```

Models opt in via a `.ugos-llm-runtime` marker file in their model directory;
the dispatcher reads the `-m` argument at spawn time, routes marked models to
their runtime, and hands everything else to the preserved vendor wrapper
unchanged. A marker pointing at a missing runtime fails fast and loud rather
than falling back silently — a silent fallback would hand upstream-only flags
to b8413 and crash confusingly.

### What this costs you

**This is the experimental part of the project.** It replaces one vendor file
(a 254-byte shell script, backed up alongside), and a firmware update can
overwrite it. The failure mode is contained: your Gemma model stops loading
while everything else keeps running, `doctor` names the cause, and
`runtime enable` repairs it in one command. `runtime disable` reverts the whole
thing. No vendor binary or library is ever modified.

Two practical notes for 32 GB devices. Models stay resident until reboot on
this stack, so a 14 GB model plus containers is most of your RAM. And UGOS'
own task scheduler refuses to start an assistant job unless *available* memory
exceeds the model's **file size** — with a big model resident you can fall
below that line and Uliya queues forever while everything else keeps working.
Watch for `Memory check failed` in `/var/ugreen/log/aiconsole_serv.log`.

## Using it from your desktop — no install on the NAS needed

Windows 10+ ships an OpenSSH client, so PowerShell works out of the box.
You can run the tool **without copying it to the NAS** by piping it through
SSH — fetch it to your own machine first:

```powershell
# PowerShell (Windows)
Invoke-WebRequest -OutFile ugos-llm.py `
  https://raw.githubusercontent.com/semih44/ugos-llm/main/ugos-llm.py

Get-Content -Raw ugos-llm.py | ssh you@nas-ip "python3 - list"
Get-Content -Raw ugos-llm.py | ssh you@nas-ip "python3 - check unsloth/Qwen3.5-9B-GGUF"
```

```bash
# macOS / Linux
curl -LO https://raw.githubusercontent.com/semih44/ugos-llm/main/ugos-llm.py
ssh you@nas-ip "python3 - list" < ugos-llm.py
```

For regular use, park it on the NAS once (e.g. `/volume1/docker/ugos-llm.py`)
and call it directly. Copying via `scp` requires the SFTP service to be
enabled in UGOS (Control Panel → Terminal) — otherwise `scp` fails with a
misleading "No such file or directory".

### About root/sudo

`check`, `list`, `test`, `doctor` and `runtime status` need no privileges.
`install`, `remove`, `ui` and the other `runtime` subcommands write to
root-owned paths; two options:

- **docker group (recommended, no sudo at all):**
  `sudo usermod -aG docker $USER` once, log out/in — the tool then routes
  privileged file operations through pinned containers.
- **sudo:** run those commands as
  `ssh -t you@nas-ip "sudo python3 /volume1/docker/ugos-llm.py install …"`
  (`-t` gives you the password prompt; UGOS sudo always asks).

There is no sudo on the Windows side — privileges are only ever needed on
the NAS, and SSH takes care of that.

## The one fix you must not skip

This one is about the **stock** runtime, and it is why the five-minute path
needs a tool at all.

UGREEN's own server arguments use microbatch `-ub 4096`. On the SYCL build the
device ships, **that setting silently computes wrong results** for long,
structure-heavy prompts (such as the ~4–5k-token tool catalog Uliya sends for
intelligent commands): the model degenerates into endless `- - -` / `1. 1. 1.`
repetition loops, deterministically, while short prompts work fine. UGREEN's
own RL-tuned 4B masks the issue; virtually any vanilla model exposes it.

**On the vendor runtime, `ugos-llm.py` always configures `-ub 512 -b 512`**,
which fixes correctness *and* roughly doubles prompt-processing speed there
(182 vs 91 t/s measured). Models pinned to an upstream runtime deliberately
keep `-ub 4096` — that build computes correctly at the larger microbatch and
is faster for it.
Full analysis: [docs/known-bugs.md](docs/known-bugs.md).

## Compatibility (short version)

Two runtimes, two answers. **Vendor** is the llama.cpp b8413 your NAS ships
with; **upstream** is the newer build from the Gemma 4 section, which
individual models opt into.

| Model class | Vendor runtime (b8413) | Upstream runtime (b10143) |
|---|---|---|
| Dense Qwen3.5 | ✅ tested (9B incl. vision + tools) | ✅ **tested** — 64k ctx, native tools, vision |
| Dense Qwen3 / Qwen2.5 / Llama 3.x / Gemma 2-3 / Mistral | ✅ expected, untested | ✅ expected, untested |
| **Gemma 4** (incl. 26B-A4B) | ❌ architecture not in b8413 | ✅ **tested** — vision, tools, MTP |
| **MoE** (Qwen3.5-35B-A3B, …) | ❌ token soup, every quant tried | ✅ correct, but slow without MTP |
| K-quants (Q4_K_M, Q5_K_M, …) | ✅ | ✅ |
| IQ-quants | ⚠️ unverified | ⚠️ unverified |

These are the verdicts `check` actually prints. Only Gemma 4 and the Qwen
MoE were re-verified on the upstream build, so dense models sit at
"expected" there even where the vendor runtime has them as tested.

Details and how to qualify new models: [docs/compatibility.md](docs/compatibility.md).

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
  an `flock` prevents two installs of the same model from colliding (the
  kernel drops it on crash, so a stale lock can never block recovery). The
  same guarantees apply to `runtime deploy`.
- **Data files are installed `644`** (no execute bit), directories `755`.
- This project redistributes **no** UGREEN binaries, models or templates.
  Everything it touches on your NAS stays on your NAS.

Known limitation: split/multi-shard GGUFs (`…-00001-of-00003.gguf`) are
detected and **rejected** rather than half-installed. Pick a single-file quant.

## Development

```bash
python3 -m unittest discover -s tests -v    # 108 unit tests, stdlib only
```

CI runs the suite on Python 3.9 and 3.12 plus a ruff lint. The risky parts —
name validation, GGUF header parsing, architecture rules, split detection,
atomic install/rollback, the runtime dispatcher's routing (executed as real
bash) and deploy's park-before-replace ordering, plus the proxy's
rewrite/wrap logic — are covered by tests; the NAS-facing parts are verified
manually on hardware (see the version banner above).

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
