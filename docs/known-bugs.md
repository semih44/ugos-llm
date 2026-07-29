# Known bugs in the UGOS AI stack (and their workarounds)

All findings verified on llama.cpp b8413 (SYCL) / infer_gateway 1.0.0.0004 /
Model Manager 1.17.0.0055. Each section states the evidence, because several
of these look unbelievable until you watch them happen.

---

## 1. THE BIG ONE: `-ub 4096` silently computes garbage on long prompts

**Symptom.** A vanilla model (e.g. Qwen3.5-9B Q4_K_M) answers short prompts
perfectly, but on long, structure-heavy prompts — the ~4–5k-token tool
catalog that Uliya's "intelligent commands" injects is the natural trigger —
output degenerates into infinite repetition: `- - - - -`, `1. 1. 1.`,
`determined determined …`, or coherent-sounding text that treats its own
context as garbage ("The text you provided appears to be a continuous block
of characters…").

**What it is NOT (each ruled out experimentally):**
- not the chat template (rendering verified byte-for-byte via `/apply-template`),
- not the gateway (identical collapse when talking to the llama-server unix
  socket directly),
- not KV-cache reuse (`cache_prompt: false` changes nothing),
- not sampling (greedy `top_k=1` collapses **deterministically** → the
  garbage is in the logits),
- not prompt length alone (tool-free prompts of 4,900 tokens stay coherent),
- not fixable by penalties (`repeat_penalty 1.3` breaks the loop but the
  model remains context-blind).

The behavior is chaotic in the strict sense: trivially perturbing the prompt
(one apostrophe escaped differently, a doubled space) flips between collapse
and coherence unpredictably. Classic numerical instability.

**Root cause.** The microbatch size UGREEN ships (`-ub 4096 -b 4096`) makes
the b8413 SYCL backend compute incorrect results on this class of input.
UGREEN's own RL-tuned 4B model appears simply to have been trained/validated
on exactly these prompts and shapes, masking the bug.

**Fix.** `-ub 512 -b 512` in the model's `model_config.json` `extra_args`.
Verified result: with all 18 MCP tools loaded, correct German answers when
no tool applies and clean tool calls
(`get_system_info {"category":"storage"}`) within ~7 s. Bonus: prompt
processing got ~2× faster (182 vs 91 tokens/s — big batches are counter-
productive on this iGPU anyway).

`ugos-llm.py` applies this automatically; `doctor` flags any model config
still carrying 4096.

**Fixed upstream.** Current llama.cpp (July 2026, `server-intel` container on
the same iGPU, see §6) shows no sign of this: a structured 1.9k-token prompt
processed as a single microbatch under `-ub 4096` came out correct — and 2×
faster than `-ub 512` (102 vs 55 t/s prompt processing, measured on the MoE
below). The full 4k-microbatch case wasn't re-tested; treat `-ub 512` as a
b8413 rule, not a law of nature.

---

## 2. `-fa on` (flash attention) is unusable

With `-ub 4096`, enabling flash attention makes the SYCL kernel fail its
workspace allocation in an endless retry loop:

```
Scratch buffer not enough to allocate 196558848 bytes   (repeated forever)
```

Prompt processing then takes >700 s for ~4k tokens. With `-ub 512` FA does
work correctly, but is *slower* than fa=off (131 vs 182 t/s prompt
processing). Conclusion: keep `-fa off`, fix correctness with `-ub 512`.

---

## 3. MoE models produce pure token soup

Every Mixture-of-Experts model tested (`qwen35moe` architecture; both
IQ4_XS and Q3_K_M quants) loads fine and then emits multilingual garbage
from the first token, independent of prompt, template, temperature or ubatch.
The dense models from the same family work. Assume **all MoE models are
broken** on this build. (`ugos-llm check` refuses them.)

**Fixed upstream.** Current llama.cpp (July 2026, `server-intel` container,
same iGPU) runs Qwen3.5-35B-A3B (UD-Q3_K_XL) correctly: coherent German
answers, a faithful 5-point summary of a 1.9k-token document, valid JSON on
a Paperless-style "answer ONLY with JSON" prompt. The catch is speed — see
the container notes in §6.

---

## 4. Gateway tool-calling: hangs and prompt-injection-only

Affects any external app talking OpenAI-style to `127.0.0.1:62891`:

- `tool_choice: "required"` → the request **never returns** (this is what
  llama-index/Paperless-ngx send; their requests hang for the full timeout).
  Named tool_choice without `parallel_tool_calls` hangs too.
- The only shape that reliably returns is **named tool_choice +
  `parallel_tool_calls: true`**.
- `response_format`/`json_schema` is silently stripped by the gateway — you
  cannot get grammar-constrained output through it.
- Tools are injected into the prompt via the chat template; nothing forces
  the model to actually emit a tool call.

**Workaround** for your own apps: [openai-bridge](../openai-bridge/) — it
removes `tools` from the request, appends an explicit "answer ONLY with JSON
matching this schema" instruction, caps `max_tokens`, and re-wraps the JSON
answer as a synthetic `tool_call`. Client libraries (llama-index, LangChain,
the OpenAI SDK) never notice.

---

## 5. Reload / lifecycle quirks

- llama-server reads config + chat template **only at spawn**; reload =
  Model Manager UI toggle or reboot. Config-file touches are unreliable
  (worked once, then never again). DB status flips do nothing.
- `system_setting.model_duration` (idle unload) is ignored: loaded models
  occupy RAM until reboot. Budget accordingly (Q4 9B ≈ 7 GB resident).
- **Never `pkill` a llama-server.** The gateway keeps dead unix-socket
  references, answers `connection refused`, spawns a second gateway that
  loses the port fight, and only a reboot recovers. Ask us how we know.

---

## 6. Third-party llama.cpp builds need glibc <= 2.36

UGOS is Debian 12 (glibc 2.36). The official `ghcr.io/ggml-org/llama.cpp:server-intel`
image is built on Ubuntu 24.04: `llama-server` itself only needs GLIBC_2.34, but
`libllama.so` and `libggml-base.so` require **GLIBC_2.38** — they cannot simply be
dropped into the UGOS runtime bundle. A SYCL build against Ubuntu 22.04 (glibc 2.35)
is required for native integration.

Note that the gateway itself is *not* the obstacle: it links only against libc and
spawns `.llama-server` as a child process (see how-it-works.md), so replacing the
runtime bundle is a matter of matching the CLI surface and the host glibc, not ABI
compatibility.

Verified in a container on an iDX6011 (July 2026): current llama.cpp runs Gemma 4
(`gemma4_assistant` architecture present) on the Arc iGPU via SYCL, loads its mmproj
automatically and answers correctly — so the hardware and driver side is fine.
Measured ~4.7 tok/s for Gemma 4 12B Q4_K_M with roughly three quarters of the output
budget consumed by its thinking block; a dense 9B on the vendor build reaches
~12 tok/s. Pick your model accordingly.

Same setup, Qwen3.5-35B-A3B MoE (UD-Q3_K_XL, 16.6 GB, fully offloaded — SYCL
reports 28.5/29.3 GiB used): output is correct (see §3), but generation runs at
only ~7.0 tok/s regardless of ubatch size — *slower* than the dense 9B on the
vendor build (12.3 tok/s) despite 3 B active parameters, so the SYCL MoE path,
not memory bandwidth, is the bottleneck. Prompt processing: 55 t/s at `-ub 512`,
102 t/s at `-ub 4096` (correct on this build, see §1). Two practical notes:
`chat_template_kwargs: {"enable_thinking": false}` works and skips the thinking
block entirely; with thinking on, a trivial "answer in three sentences" prompt
burned its whole 1024-token budget before producing any content. RAM while
loaded: ~23 GiB used, 2.2 GiB swap touched — a 22 GB Q4_K_M would not have fit
next to the running Paperless stack.

The way out of the slow-MoE trap is **Gemma 4's MTP draft head** (multi-token
prediction, supported since b9549 via `--model-draft mtp-*.gguf --spec-type
draft-mtp --spec-draft-n-max 4`). Measured on Gemma 4 26B-A4B qat-UD-Q4_K_XL
(14.2 GB + 0.25 GB MTP head + 1.2 GB mmproj, image b10143, same container
setup, warm, `-ub 4096`):

| workload                     | baseline | with MTP | draft acceptance |
|------------------------------|----------|----------|------------------|
| short list-style answer      | 9.9 t/s  | 13.4 t/s | 74 %             |
| JSON-only extraction         | 10.1 t/s | 14.8 t/s | 79 %             |
| free-form German prose       | 9.7 t/s  | 9.9 t/s  | 44 %             |
| 2.5k-token summary           | 6.8 t/s  | 6.3 t/s  | 43 %             |

Speculative decoding verifies drafts at prompt-processing speed (~90 t/s
here), so the win tracks output predictability: structured output lands
*above* the dense 9B on the vendor build (12.3 t/s), free prose stays at
baseline, and at long context the rejected drafts cost slightly more than
they save. `enable_thinking: false` via `chat_template_kwargs` works for
Gemma 4 as well. Output quality was correct throughout (German answers,
faithful summary, valid JSON — dates not ISO-normalized unless asked).
Untuned knobs worth a look: `--spec-draft-n-max` below 4 might soften the
long-context penalty.

**Native deployment (verified end-to-end, July 2026).** The b10143 commit
built via [scripts/build-runtime.sh](../scripts/build-runtime.sh) runs
natively under the UGOS gateway (`runtime deploy` + `runtime enable` +
`install --runtime upstream-b10143 --draft`): the Model Manager card
loads Gemma 4 26B-A4B through the dispatcher, all four acceptance tests
pass (chat, long-prompt at `-ub 4096`, tools, vision), and the gateway
even returns **real `tool_calls`** — the upstream server's native
tool-calling survives the gateway pipeline, which b8413 never managed.
Measured on the isolated runtime: PP 115.8 t/s at 2.9k tokens, JSON
16.6 t/s at 90 % draft acceptance. The traps below cost a day each to
find and are baked into wrapper + build script; the full story:

1. Every GPU userspace new enough for oneDNN needs GLIBC ≥ 2.38 (host
   level-zero 1.3.x is too old, compute-runtime ≥ 25.18 too new, 25.13
   segfaults under SYCL compute). The way out: build with
   `-DGGML_SYCL_DNN=OFF` and run on **UGREEN's own OpenCL userspace**
   (igdrcl + IGC 2.10 from the vendor bundle).
2. With oneDNN compiled in, the OpenCL path re-JITs kernels on every
   prompt batch: 0.4–2 t/s prompt processing. Without it: 115 t/s.
3. `/etc/OpenCL/vendors` registers UGOS' old system driver — the same
   iGPU appears twice and ggml's multi-GPU peer-access path crashes the
   OpenCL adapter. The wrapper points `OCL_ICD_VENDORS` into the void.
4. `GGML_BACKEND_PATH` is treated as a file path by llama.cpp — leave it
   unset; backends load from the executable's directory.
5. Bonus: unsloth qat GGUFs carry > 4 MB of header metadata; fixed-size
   header probes must widen their byte range.

**Context is capped by memory, not by the model** (measured July 2026 while
trying to grow Gemma 4 26B-A4B beyond 8k on a 32 GB device). Three attempts,
three distinct failures, all reproducible:

| Config | Result |
|---|---|
| `-c 8192 -ub 4096 -fa off` | works — the shipped configuration |
| `-c 16384 -ub 4096 -fa off` | `failed to allocate compute pp buffers`, exits after 17 s |
| `-c 16384 -ub 2048 -ctk q8_0 -ctv q8_0 -fa off` | same allocation failure |
| `-c 16384 -fa on` | loads, but slower than the gateway's patience |

Two lessons. **`-fa on` is still unusable**, now for a different reason than
on b8413: the model loads (the V-cache padding warning does disappear, so
flash attention really engages) but takes longer than the gateway's *hard,
non-configurable* 2-minute readiness timeout — `server on unix socket … not
ready after 240 attempts (2m0s)` — after which the gateway kills it. A
warm SYCL JIT cache on the second attempt does not save it.

And the ceiling is the **compute buffers**, not the KV cache: halving the
microbatch and quantizing K and V to `q8_0` together were still not enough
for 16k. On a 32 GB device a 14 GB model simply does not leave room to
double the context.

Worth knowing before you chase a bigger number anyway: prompt processing
runs at ~115 t/s here, so filling 16k costs ~2.5 minutes and 32k ~4.5. The
practical ceiling on this hardware is set by patience as much as by RAM.

**What does work, at no memory cost:** `--chat-template-kwargs
'{"enable_thinking":false}'` in `extra_args` (or `install --thinking off`).
Verified: `reasoning_content` drops to 0 characters, answers start
immediately, and the whole output budget goes to the actual answer — which
also shortens the queue when several clients share the `-np 1` slot.

---

## 7. Assorted smaller surprises

- The gateway request log (`infer_gateway_serv.log`) records **only failed**
  requests — but those with full request payloads. Great for debugging,
  useless for auditing successes.
- The llama-server console output lands in
  `/var/ugreen/log/infer_gateway_serv_panic/` — the files are not panics.
- The gateway exposes fragments of an Ollama API (`/api/tags` returns an
  empty list; `/api/chat` accepts requests and never answers). Ignore it.
- `scp` to the NAS fails with a misleading "No such file or directory"
  unless the SFTP service is enabled in UGOS (Control Panel → Terminal).
- UGOS's root filesystem is immutable — even root cannot `mkdir /home/x`.
  User homes require enabling the "personal folder" in the Files app.
