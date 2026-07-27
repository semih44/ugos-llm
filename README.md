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

On your NAS (SSH, a user in the `docker` group — or root):

```bash
curl -LO https://raw.githubusercontent.com/OWNER/ugos-llm/main/ugos-llm.py

# 1. Pre-flight: is this model going to work?
python3 ugos-llm.py check unsloth/Qwen3.5-9B-GGUF

# 2. Install it (Q4_K_M quant, with vision, with a Model Manager UI card)
sudo python3 ugos-llm.py install unsloth/Qwen3.5-9B-GGUF --quant Q4_K_M --vision --ui

# 3. Reload: toggle the model OFF/ON in Model Manager, or reboot the NAS
#    (there is no other reliable way — see docs/known-bugs.md)

# 4. Verify
python3 ugos-llm.py test Qwen3.5-9B
```

After that, the model is selectable in Uliya (chat, knowledge base, intelligent
commands), Universal Search, and reachable for your own apps via the gateway.

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

## Safety

- The only system-owned file this tool ever writes is the Model Manager catalog
  DB (`--ui` flag only), and it **creates a timestamped backup first**.
  Models themselves live in their own directories and are trivially removable.
- Never `pkill` a UGOS llama-server. The gateway does not recover until reboot.
  The CLI never does this and tells you the safe reload paths.
- This project redistributes **no** UGREEN binaries, models or templates.
  Everything it touches on your NAS stays on your NAS.

## Disclaimer

Not affiliated with or endorsed by UGREEN. This modifies the behavior of an
appliance in ways the vendor doesn't officially support yet; a firmware update
may undo or break it (the models and configs survive in practice — the UI
catalog entry is the fragile part, and `ui add` restores it in seconds).
Use at your own risk.

## License

MIT
