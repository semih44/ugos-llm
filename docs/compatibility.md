# Model compatibility

The shipped llama.cpp is **b8413** (April 2026 vintage). Anything newer than
its architecture list cannot load, and its SYCL backend has correctness
limits (see [known-bugs.md](known-bugs.md)). `ugos-llm check <repo>` reads
the GGUF header straight from Hugging Face (a 4 MB range, widened to 16/64 MB
for models with oversized metadata such as QAT builds — never a full
download) and applies the rules below.

Since July 2026 the verdict depends on **which runtime the model will run
on**: UGREEN's b8413, or a newer upstream build deployed alongside it via
`ugos-llm runtime` (see [known-bugs.md §6](known-bugs.md)). Pass
`--runtime upstream-b10143` to `check`/`install` to be judged against the
latter.

## Matrix — vendor runtime (b8413)

| Architecture (GGUF `general.architecture`) | Verdict | Notes |
|---|---|---|
| `qwen3_5` dense (0.8B–27B) | ✅ **TESTED** | Qwen3.5-9B Q4_K_M verified end-to-end incl. vision (mmproj) and tool calls |
| `qwen3`, `qwen2`, `qwen2vl` dense | ✅ expected | compiled in; untested here — `qwen3` is a different family from the verified `qwen3_5` |
| `llama` (Llama 3.x, Mistral, and most finetunes) | ✅ expected | compiled in; untested |
| `gemma`, `gemma2`, `gemma3`, `gemma3n` | ✅ expected | HF repos are license-gated → `HF_TOKEN` needed |
| **any MoE** (`qwen35moe`, `qwen3moe`, `qwen2moe`, `glm4moe`, …) | ❌ **BROKEN** | SYCL build emits token soup — verified with two quant families |
| `gemma4`, `qwen3_6` and anything newer than ~04/2026 | ❌ missing | not in b8413 |

## Matrix — upstream runtime (b10143, self-built)

| Architecture | Verdict | Notes |
|---|---|---|
| `gemma4` (incl. 26B-A4B MoE) | ✅ **TESTED** | end-to-end through the gateway: chat, long prompts at `-ub 4096`, **native tool calls**, vision, MTP draft head |
| `qwen3_5` dense | ✅ **TESTED** | Qwen3.5-9B re-verified end-to-end on this build (Jul 2026): chat, long-prompt, native tool calls, vision — running at `-c 65536 -ub 512` |
| `qwen35moe` and MoE generally | ✅ correct | no more token soup — but generation is SYCL-kernel-bound (~7 t/s for a 35B-A3B); only worth it with an MTP head |
| every other architecture the vendor build loads (`qwen3`, `qwen2`, `llama`, `gemma`–`gemma3n`, `mistral`, `phi3`) | ✅ expected | a newer llama.cpp is an architectural superset of b8413, so these are present — just not re-verified on this build |
| anything else | ⚠️ unknown | `check --runtime` reports UNKNOWN — install and run `test` |

One deliberate conservatism in these verdicts: `TESTED` is granted only for
the exact runtime build we verified — deploy a different one and `check`
downgrades to `EXPECTED`, because a name like `upstream-…` proves nothing
about what was compiled in. Architectures earn their upstream `TESTED` the
same way they earn the vendor one: somebody runs the full `test` suite on
this hardware and reports it (as happened for Qwen3.5-9B in July 2026).

## Quantization guidance

- **Q4_K_M** is the sweet spot and the only quant family battle-tested here.
  Q5_K_M / Q6_K should be equally safe (same kernel family), cost more RAM.
- **IQ-quants (IQ4_XS, …): unverified.** Our only IQ test was on a (broken
  anyway) MoE model, so no honest conclusion exists. If you test a dense
  IQ-quant, please report the result.
- Full precision (F16/BF16) barely fits and gains little; skip.

## Sizing rule of thumb

`RAM needed ≈ GGUF file size + ~1–1.5 GB` (KV cache at 16k context)
— and remember: **models never unload** on this stack (no idle timeout),
so whatever you load stays resident until reboot. On a 32 GB device with
UGOS (~4 GB) plus your containers, a single 7–8 GB model is comfortable;
two mid-size models are the practical ceiling.

Speed expectations (Core Ultra 5 125H iGPU, `-ub 512`): prompt processing
~180 t/s, generation ~12 t/s for a 9B Q4. First request after reload loads
the model (~30–60 s).

On the upstream runtime, Gemma 4 26B-A4B Q4 (14.2 GB) with an MTP draft head
generates ~14.8 t/s on JSON output and ~13.4 t/s on short answers, but only
~9.9 t/s on free prose and ~6.3 t/s at 2.5k context — speculative decoding
helps in proportion to how predictable the output is. Those four are one
coherent sweep measured in a container; the self-built runtime reached
16.6 t/s on JSON and 115.8 t/s prompt processing over 2.9k tokens. Loading
the model takes ~28 s.

Qwen3.5-9B on the upstream runtime at `-c 65536 -ub 512` (July 2026):
~11.3 t/s generation at small context, prompt processing a steady
**179–190 t/s even across a 44k-token prompt** — but plan for the two
long-context realities: filling 44k takes ~4 minutes on the first shot
(follow-ups reuse the prompt cache and only pay for the delta), and
generation **degrades to ~4.7 t/s once ~44k of context is resident**,
because attention over the long history dominates. Loading takes ~26 s.

One sizing trap on 32 GB devices: UGOS' own task scheduler refuses to start
an Uliya job unless *available* RAM exceeds the model's **file size** (a
14.2 GB model demands 13588 MiB free). With a big model resident plus
containers you can fall below that and Uliya queues forever while Paperless
and direct gateway calls keep working. Watch for `Memory check failed` in
`/var/ugreen/log/aiconsole_serv.log`; a reboot clears it, a smaller quant
fixes it for good.

## Vision

Qwen3.5 dense models are natively multimodal — grab the `mmproj-F16.gguf`
**from the same repo** as your GGUF (the projector must match the vision
tower of the exact model size), install with `--vision`. Verified working
through the whole chain (gateway + Uliya UI).

## Chat templates & thinking

The gateway may pass `chat_template_kwargs` and models ship different
`enable_thinking` polarities in their templates. In practice with the
embedded template of unsloth's Qwen3.5 GGUFs, thinking defaults to **off**
(closed `<think>` block), which is what you want for app integrations. If a
model thinks too much (slow responses full of reasoning), you can point
`extra_args` to a patched template file via `--chat-template-file` — but try
the embedded template first; it's the one the model was trained with.

On the upstream runtime there is a cleaner lever: `install --thinking off`
writes `--chat-template-kwargs '{"enable_thinking":false}'`, which settles
the question server-side for every client. That matters for apps like
Paperless-ngx, which never send `chat_template_kwargs` themselves — and for
Gemma 4, whose thinking block happily consumes a whole small token budget
before producing any answer.

## Qualifying a new model (please contribute!)

```bash
python3 ugos-llm.py check  <repo>            # header + rules
sudo python3 ugos-llm.py install <repo> --quant Q4_K_M
# no reload needed — the gateway picks up new models and loads them on the
# first request (~30-60 s). A reload (UI toggle or reboot) is only required
# after CHANGING an already-installed model's config. Then:
python3 ugos-llm.py test <Name>              # chat + long-prompt + tools
python3 ugos-llm.py test <Name> --vision     # if you installed a projector
```

The `test` long-prompt check is specifically designed to expose the ub-4096
class of corruption; if it passes together with the tool check, the model is
genuinely usable including Uliya's intelligent commands.

`test` ends with a ready-made **report block** (model, architecture, file
size, ub, per-check results, device). Paste it into a
["Model compatibility report" issue](../../issues/new?template=model-report.yml)
— the matrix above grows from exactly such reports, for working *and* for
failing models.
