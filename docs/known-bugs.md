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
- The CLI refuses `install --force` and `remove` while that model is loaded.
  Otherwise a forced install could test the old process, and removing the UI
  card would take away the clean stop control. Disable the model in Model
  Manager (or reboot), then retry.

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

The full FA verdict, measured later on Qwen3.5-9B (July 2026). The load
problem is actually *solvable*: FA kernel compilation took 137 s — 17 s
over the gateway's window — but the gateway sets `HOME` to the model dir's
`infer_gateway_cache/`, and **warming that cache by running the same
llama-server once OUTSIDE the gateway** (own container, same runtime, same
OpenCL stack, `HOME` pointed at that directory, no time limit) brought the
subsequent gateway load down to 13 s. Remember this trick — it applies to
anything whose first load busts the 2-minute window.

Then the benchmark killed the idea anyway. Identical ~14k-token prompt,
`-c 65536 -ub 512`, same runtime:

| | `-fa off` | `-fa on` (2 runs) |
|---|---|---|
| prompt processing | **221.8 t/s** | 100.8 / 86.8 t/s |
| generation at ~14k ctx | 7.34 t/s | 7.97 / 7.39 t/s |

FA on this SYCL/OpenCL path **halves prompt processing and buys nothing on
generation** — the attention kernels are clearly unoptimized for this
iGPU. Since prefill dominates every large-context request, `-fa on` makes
the box strictly worse. Keep `-fa off`; re-measure after major llama.cpp
SYCL releases before believing otherwise.

And the ceiling is the **compute buffers**, not the KV cache: halving the
microbatch and quantizing K and V to `q8_0` together were still not enough
for 16k. On a 32 GB device a 14 GB model simply does not leave room to
double the context.

Refinement, found while bringing Qwen3.5-9B up at 32k on the upstream
runtime: the hard wall is the iGPU's **maximum single allocation of just
under 4 GiB**, not total memory. The attention compute buffer scales with
`microbatch x context`; at `-c 32768` the device refused exactly 2^33 bytes
at `-ub 4096` and exactly 2^32 at `-ub 2048` (`can't allocate ... Bytes of
memory on device`), while `-ub 1024` loads and runs fine — 11.3 t/s
generation, all four acceptance tests pass, ~14 GiB still available. Rule
of thumb: when growing `-c`, shrink `-ub` so their product stays where it
was. This also recasts the Gemma 16k failure above — it may have hit the
same per-allocation wall rather than exhaustion.

Worth knowing before you chase a bigger number anyway: prompt processing
runs at ~115 t/s here, so filling 16k costs ~2.5 minutes and 32k ~4.5. The
practical ceiling on this hardware is set by patience as much as by RAM.

**What does work, at no memory cost:** `--chat-template-kwargs
'{"enable_thinking":false}'` in `extra_args` (or `install --thinking off`).
Verified: `reasoning_content` drops to 0 characters, answers start
immediately, and the whole output budget goes to the actual answer — which
also shortens the queue when several clients share the `-np 1` slot.

**Runtime refresh: b10621 (August 2026, verified on-device).** The same
recipe with `COMMIT`/`BUILD` bumped to the `b10621` release tag
(`c1d0e7a0…`, 2026-08-25) builds and runs natively under the gateway,
side-by-side with b10143 — models migrate per marker file, an unloaded
model migrates without even a reload. What the window b10143→b10621 adds
for this stack: the gated-delta-net writeback fusion, MTP autodetect and
the MTP memory-fit fix, and ESIMD Q2_K/Q3_K/Q5_K kernels. Two upstream
cautions if you pick your own pin: **avoid b10552–b10575** (broken Q2_K
ESIMD land, reverted and re-landed in b10576), and recent builds moved
SYCL host memory to pinned allocations — if you see load-time OOM or a
CPU spin that older builds didn't have, try
`GGML_SYCL_ENABLE_HOST_PINNED_MEM=0` in the wrapper.

Measured on b10621, Qwen3.5-35B-A3B UD-IQ3_S (15.3 GB + mmproj, native
through the gateway, `-c 65536 -ub 512 -fa off`, no MTP): **all four
acceptance tests pass including vision**, and server-side prompt caching
visibly works (the identical long-prompt check drops from 67.4 s to 3.8 s
on repeat). But generation is a flat **5.9 t/s** — identical for free
prose and JSON-only extraction, i.e. purely kernel-bound — and prompt
processing ~62 t/s over a 4k prompt. The upstream issue for the Qwen3.5
architecture malus on Intel iGPUs (llama.cpp #22001, ~40 % tg deficit)
was closed *not planned* in June 2026, so do not wait for a fix. We did
not A/B the embedded MTP head: on SYCL, MoE MTP measured *negative* on an
Arc dGPU (discussion #26184) and IQ3-class quants regress hard with MTP
(#27373). Verdict: the 35B MoE is correct and fully functional on this
box, and roughly half the speed of the dense 9B at everything
interactive. We installed it, measured it, and removed it again.

**Update, October 2026 — the 5.9 t/s was a bug, and the next limit is
memory.** llama.cpp #28476 (merged 2026-09-09, in b10868) added the IQ
quant types to the SYCL MoE kernel. Before that, every IQ-quantized expert
(our UD-IQ3_S) went through a host-serialized fallback. The same file on
a b11790 runtime:

| configuration (b11790, `-fa off`) | iGPU memory | generation | prompt (13.9k tokens) |
|---|---|---|---|
| container, `-c 16384`, no mmproj, `-ub 512` | (not measured) | **14.7 t/s** | 60 t/s |
| same, `-ub 1024` | larger | 11.5 t/s | 90 t/s |
| same, `-ub 2048` | larger | 11.5 t/s | 124 t/s |
| gateway, `-c 32768`, mmproj on the iGPU | 19.0 GB | 8.7 t/s | 56 t/s |
| gateway, `-c 16384`, `--no-mmproj-offload` | 17.5 GB | 10–12 t/s | 60 t/s |

Every configuration passed the acceptance suite and recalled a fact
planted in a 13.9k-token prompt exactly. Three findings:
- **Speed falls with iGPU footprint.** Generation drops once the iGPU
  holds more than ~15 GB. It is not ordinary swapping: `vmstat` showed no
  swap-out and a few KB/s of swap-in during generation. The 13.6 GB of
  35B weights leave too little room on a 32 GB box. The dense 9B
  (14.9 GB at `-c 65536`) stays below that line.
- **Microbatch: above 512 trades generation for prefill on MoE.**
- **`-fa on` never became ready.** The first spawn compiles kernels for
  longer than the gateway's 2-minute readiness window, and the gateway
  killed the server (`signal: killed`). A pre-warmed kernel cache might
  get around it; untested.

Same day, same runtime, the dense Qwen3.5-9B Q4_K_M generates at
**14.05 t/s** with 176 t/s prefill. That is +6 % and +3 % over b10621,
from the post-August SYCL fusion work. **Verdict: the 35B-A3B is correct
and no longer slow in isolation, but on a 32 GB iDX6011 it does not beat
the 9B in daily use.** We kept the 9B. Re-test on a box with more RAM or
with a smaller MoE.

**Qwen3.5-9B with its embedded MTP head: slower, not faster (b10621,
August 2026).** unsloth ships `Qwen3.5-9B-MTP-GGUF` — identical weights
plus a multi-token-prediction head baked into the GGUF, enabled with
`--spec-type draft-mtp --spec-draft-n-max 4` (no separate draft file;
b10621 autodetects the head). All four acceptance tests pass with it,
including vision — MTP and `--mmproj` do coexist, whatever older model
cards say. But the speed went the wrong way, measured against the plain
9B through the same gateway:

| workload | plain 9B | 9B + MTP | draft acceptance |
|---|---|---|---|
| JSON extraction (390 tok) | 11.1 t/s | **5.7 t/s** | 78 %, mean draft 4.1 |
| Python code (500 tok) | 12.1 t/s | **7.9 t/s** | 47 %, mean 2.9 |
| free German prose | 12.0 t/s | **5.5 t/s** | 25 %, mean 2.0 |

Note the first row: 78 % acceptance and *still* half the speed. Qwen's
MTP head is a full transformer layer (~0.5 GB in the file), and on this
SYCL path its forward pass plus the batched verification of a 5-token
draft costs more than five single-token steps. Gemma 4's win in the table
above came from a 241 MB head verified at a much better ratio. Verdict:
**no MTP for Qwen3.5 dense on this stack**; the numbers would only get
worse at long context, so that was not measured. (Also a reminder that
two 64k-context 9B instances do not fit next to each other on a 32 GB
device — running the A/B side by side pushed the box 15 GiB into swap.)

**UGREEN's own clients on the upstream runtime.** With the Qwen3.5-9B
card enabled in the Model Manager, Uliya answered through the dispatcher
→ b10621 path without any special handling (August 2026). Note that the
CLI cannot enable a card it did not create: a model whose catalog row
predates `ugos-llm` (or was made by UGOS itself) carries a vendor
`release_id`, and `ui add`/`ui remove` refuse to touch it by design — use
the UI toggle for those.

---

## 7. Vulkan (Mesa ANV) on the Meteor Lake iGPU: tested, rejected

The SYCL MoE malus (#22001 above) made the Vulkan backend the obvious
thing to try — it is often faster than SYCL on Arc, and its
gated-delta-net path is under active optimization. Short version: the
runtime-bundling technique carries a whole foreign GPU driver just fine,
and it is still not worth it on this silicon.

**What works.** `scripts/build-runtime-vulkan.sh` builds llama.cpp with
`GGML_VULKAN=ON` (same commit as the SYCL pin, ubuntu:22.04 base) and
bundles mesa's ANV driver from the kisak PPA (stock jammy mesa predates
Meteor Lake) plus the Vulkan loader. ANV needs **no vendor OpenCL and no
level-zero** — it talks straight to the kernel DRM, and UGOS' 6.12 kernel
is new enough. The ICD manifest is rewritten to a bare
`"library_path": "libvulkan_intel.so"` so the loader resolves it via
`LD_LIBRARY_PATH`; the runtime wrapper exports `VK_ICD_FILENAMES`,
`VK_DRIVER_FILES` and a stable `MESA_SHADER_CACHE_DIR`. Device
enumeration confirms `Vulkan0: Intel(R) Arc(tm) Graphics (MTL)`.

Three traps found on the way, all now baked into the script/wrapper:

1. **The ldd-closure gate is blind to libraries the build container has
   system-wide.** `libvulkan_intel.so` pulls libbsd → libmd; the closure
   copied libbsd on the last pass and never rescanned it, and the final
   `ldd` check resolved libmd from the *container's* system dir — green
   build, dead driver on the host (`libmd.so.0: cannot open shared object
   file`, ICD silently skipped, "no devices"). Debug with
   `VK_LOADER_DEBUG=error,warn,driver`. libmd is now bundled explicitly.
2. **First start compiles shaders for ~3.5 minutes — the gateway kills it
   at 2.** Same watchdog as the FA story in §6, same cure: run the server
   once *outside* the gateway with the same `MESA_SHADER_CACHE_DIR`, then
   the gateway start hits the warm cache. (The cache dir must be a stable
   host path for that to work — not the container's own `/tmp`.)
3. `--no-mmap` + full offload is a bad idea on Vulkan/iGPU — host staging
   plus device-local heap both live in the same 32 GB. Drop it and let
   mmap stream.

**Why it was rejected anyway.** With all of that solved: `qwen35moe`
(Qwen3.5-35B-A3B) **hangs forever on its first inference** — 100 % of one
core (the backend busy-waits on a fence), shader cache no longer growing,
a 4-token prompt with `max_tokens: 30` still nothing after 4 minutes.
Reproduced with cooperative-matrix paths disabled
(`GGML_VK_DISABLE_COOPMAT=1`, `GGML_VK_DISABLE_COOPMAT2=1`) and at
`-c 8192` — so an ANV/Xe-LPG shader bug in the MoE path, not a memory or
config issue. The dense 9B *does* run — at **6.7 t/s** generation (warm,
repeated) versus ~11–12 t/s for the same model on the SYCL runtime:
Vulkan/ANV is ~45 % slower on this iGPU even where it works.

Verdict: **SYCL on UGREEN's OpenCL userspace stays the stack.** The build
script is kept because it reproduces the experiment in one container run —
re-run it when mesa or llama.cpp's Vulkan MoE path makes news, and expect
this verdict to age.

---

## 8. OpenVINO Model Server on the same iGPU: measured, not adopted

The other way around the SYCL kernel problem is Intel's own stack. OpenVINO
2026.3 lists Qwen3.5/3.6 linear attention among its optimized paths and
ships official INT4 IRs for both Qwen3.5-9B and 35B-A3B, and the Model
Server (OVMS) speaks OpenAI chat completions — so it slots in next to the
gateway without touching UGOS. Measured August 2026, same iDX6011, same
prompts as the llama.cpp numbers above.

**Setup that works.** `openvino/model_server:latest-gpu` with
`--device /dev/dri --network host`, the server pulling
`OpenVINO/Qwen3.5-9B-int4-ov` itself (`--source_model … --task
text_generation --target_device GPU --rest_port …`). Three things to
know: the image's default user cannot write a UGOS volume (ACLs —
`Permission denied` on the model dir; run with `--user 0:0`); the
generated `graph.pbtxt` defaults to `max_num_seqs: 256` (tune it down,
along with `cache_size`, on a shared-memory box); and OVMS answers on
`/v1/chat/completions` as well as `/v3/…`, so the bridge can point at it
via `UPSTREAM` unchanged. `chat_template_kwargs` is honored, prefix
caching is on by default, vision works via `image_url`. The 9B needs
~7.4 GiB resident and noticeably more during load — a 14 GB cgroup cap
killed it, 22 GB was fine.

**Qwen3.5-9B, OVMS INT4 vs llama.cpp b10621 Q4_K_M:**

| | OVMS (GPU) | llama.cpp (SYCL) |
|---|---|---|
| generation, free prose | ~15 t/s | 13.3 t/s |
| cold 4k-token prefill (turn 1) | **15.5 s** | 25.6 s |
| follow-up turns on the cached 4k prefix | 7.0 / 6.8 s | **3.4 / 3.2 s** |
| vision round-trip (64×64 image) | 1.3 s | 3.7 s |
| 4 German Paperless-style extractions | all valid JSON; one missing `due_date`, clunkier titles | all valid, schema-exact, cleaner titles |

So: a real win on cold prefill (~+60 %) and vision latency, a real loss on
multi-turn reuse (llama.cpp's prompt cache halves the follow-up cost, OVMS'
prefix caching only takes ~55 % off), a slight but consistent quality edge
for the Q4_K_M side, and OVMS is a second always-on service the UGOS UI
cannot use. For a box whose interactive lane is multi-turn (VS Code) and
whose batch lane cares about schema fidelity (Paperless), that nets out to
**staying on llama.cpp** — re-measure when OpenVINO's hybrid-model prefix
caching improves.

**Qwen3.5-35B-A3B: does not fit, full stop.** The only official IR is
23.1 GB (INT4 gs128, INT8 attention). Under a 25 GB no-swap cgroup with
Paperless and the bridges stopped it was **OOM-killed after 21 s**, still
in "Initializing … servable" — on an iGPU the weights are pinned host
memory, and 23 GB plus a few for UGOS leaves nothing on 30.9 GiB. The
first attempt, run without a cap next to Paperless, left the host with
349 MiB and unresponsive; the lesson generalizes: **always run model
experiments in a container with `--memory=X --memory-swap=X`** so the
experiment dies instead of the NAS. The MoE-to-disk offloading mentioned
in the 2026.3 release notes is not exposed as an OVMS option as far as we
could find.

---

## 9. Assorted smaller surprises

- The gateway request log (`infer_gateway_serv.log`) records **only failed**
  requests — but those with full request payloads. Great for debugging,
  useless for auditing successes.
- The llama-server console output lands in
  `/var/ugreen/log/infer_gateway_serv_panic/` — the files are not panics.
- The gateway exposes fragments of an Ollama API (`/api/tags` returns an
  empty list; `/api/chat` accepts requests and never answers). Ignore it.
- A server-side `--chat-template-kwargs` default can be swallowed on the
  gateway path: with Qwen3.5-9B the flag arrives intact in the spawned
  server's argv, yet `/props` reports empty kwargs and requests without
  their own `chat_template_kwargs` behave as if no default were set —
  while the same flag did take effect for Gemma 4. Request-level kwargs
  win in every case, so set thinking defaults at the request level (the
  bridge's `THINKING_DEFAULT` does exactly that) rather than in
  `extra_args` if you need them to hold.
- `scp` to the NAS fails with a misleading "No such file or directory"
  unless the SFTP service is enabled in UGOS (Control Panel → Terminal).
- UGOS's root filesystem is immutable — even root cannot `mkdir /home/x`.
  User homes require enabling the "personal folder" in the Files app.

---

## 10. `--no-mmap` doubles the model's memory on the iGPU (and upstream dropped the flag)

An iGPU has no VRAM of its own. The "device" buffers are i915 GEM objects
in system RAM. They are shmem-backed, so they show up as `Shmem` in
`/proc/meminfo` and per process as `drm-total-system0` in
`/proc/<pid>/fdinfo/*`. Shmem can be swapped like any other anonymous
memory. With `--no-mmap`, llama-server *additionally* reads the whole GGUF,
mmproj included, into anonymous host memory and keeps it. That is a
second copy that nothing touches after the upload to the iGPU.

**Setup:** measured October 2026 on an iDX6011 after 25 days of uptime.
The model was Qwen3.5-9B Q4_K_M (5.68 GB) plus mmproj F16 (0.92 GB) on
b10621, run with `-c 65536 -ub 512 -fa off`:

| | with `--no-mmap` | without it, fresh boot |
|---|---|---|
| iGPU buffers (`drm-total-system0`) | 13.6 GB | 14.6 GB (after a 14k-token prompt) |
| llama-server anonymous memory | 8.8 GB, 4.7 GB of it swapped | 33 MB RSS, nothing swapped |
| swap (4×4 GB zram + 6 GB on disk) | 21 of 21 GB used | 0, later 1.2 GB |
| generation, 300-token prose | 10.7 t/s | 13.3 t/s |
| first request after an idle period | +20 s (prompt phase at 1.8 t/s) | normal |

**What happened:** UGOS ships four 4 GB zram devices at swappiness 60.
1. Under memory pressure the kernel swapped out the idle host copy first.
2. It then swapped the iGPU buffers themselves.
3. Model weights barely compress (~1.29:1). zram therefore spent ~11.6 GB
   of real RAM holding ~15 GB of swapped data, which fed the pressure.
4. Every request after an idle period first had to fault the GPU buffers
   back in, which is the 20-second prompt phase above.

To be precise about the evidence: the right-hand column combines two
changes, a reboot and dropping the flag. A reboot alone also empties the
swap, but only for a while. The 8.8 GB of anonymous memory that does not
come back is what keeps it empty. Long-prompt correctness was checked
after the change: a fact planted in a 13.9k-token prompt was recalled
exactly, with prompt processing at 170 t/s.

**Fix: drop the flag.**
- **Default behaviour:** the server maps the file instead. The host side
  becomes clean page cache, which the kernel drops without swapping.
- **Upstream removal:** upstream removed `--no-mmap`, `--mmap` and
  `--mlock` in September 2026 (llama.cpp #28334, around b10875) in favour
  of `--load-mode {auto,none,mmap,mlock,mmap+mlock,dio}`. A newer server
  refuses to start when it sees `--no-mmap`.
- **This tool:** `ugos-llm.py` no longer emits the flag for upstream
  runtimes, and `doctor` flags it in either spelling. The vendor runtime
  (b8413) keeps its `--no_mmap`, which is untested there.

**Related: UGREEN's semantic search loads at boot.** The semantic-search
service (`ugreen_rag_serv@aiconsole`, process `rag_ai_sdk_tool`) loads
its ~2.1 GB embedding model onto the iGPU at every boot. It does so even
when that model is switched off in the Model Manager UI. Stopping the unit
frees the memory until the next boot. Do not mask it blindly: the same
service also runs the Postgres database named `uliya`.

**Diagnostics:**
- `grep -E 'Shmem|Swap' /proc/meminfo`: iGPU buffers and swap at a glance.
- `/proc/<pid>/fdinfo/*`: the `drm-total-system0` line gives iGPU memory
  per process.
- `/proc/<pid>/smaps_rollup`: resident anonymous memory versus swapped.
- `/sys/block/zram*/mm_stat`: original size versus compressed size,
  i.e. how well your swap really compresses.

Reading other processes needs root. On UGOS:
`docker run --rm --privileged --pid=host alpine:3.20 …`.
