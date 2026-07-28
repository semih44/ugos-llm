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
