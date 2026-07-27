# Model compatibility

The shipped llama.cpp is **b8413** (April 2026 vintage). Anything newer than
its architecture list cannot load, and its SYCL backend has correctness
limits (see [known-bugs.md](known-bugs.md)). `ugos-llm check <repo>` reads
the GGUF header straight from Hugging Face (first 4 MB, no full download)
and applies the rules below.

## Matrix

| Architecture (GGUF `general.architecture`) | Verdict | Notes |
|---|---|---|
| `qwen3_5` dense (0.8B–27B) | ✅ **TESTED** | Qwen3.5-9B Q4_K_M verified end-to-end incl. vision (mmproj) and tool calls |
| `qwen3`, `qwen2`, `qwen2vl` dense | ✅ expected | compiled in; untested here |
| `llama` (Llama 3.x, Mistral, and most finetunes) | ✅ expected | compiled in; untested |
| `gemma`, `gemma2`, `gemma3`, `gemma3n` | ✅ expected | HF repos are license-gated → `HF_TOKEN` needed |
| **any MoE** (`qwen35moe`, `qwen3moe`, `qwen2moe`, `glm4moe`, …) | ❌ **BROKEN** | SYCL build emits token soup — verified with two quant families |
| `gemma4`, `qwen3_6` and anything newer than ~04/2026 | ❌ missing | not in b8413 |

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

## Qualifying a new model (please contribute!)

```bash
python3 ugos-llm.py check  <repo>            # header + rules
sudo python3 ugos-llm.py install <repo> --quant Q4_K_M
# reload (UI toggle or reboot), then:
python3 ugos-llm.py test <Name>              # chat + long-prompt + tools
python3 ugos-llm.py test <Name> --vision     # if you installed a projector
```

The `test` long-prompt check is specifically designed to expose the ub-4096
class of corruption; if it passes together with the tool check, the model is
genuinely usable including Uliya's intelligent commands. Open an issue with
the model, quant, and test output — the matrix above grows from exactly
such reports.
