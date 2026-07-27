# Troubleshooting toolkit

Start with `python3 ugos-llm.py doctor`. When that isn't enough, this is the
debugging arsenal that cracked every problem we hit, in escalating order.

## 0. Ground rules

- Reload = Model Manager UI toggle or reboot. Nothing else works reliably.
- **Never kill llama-server processes** — gateway death until reboot.
- `ps` lies about GPU work: during iGPU inference the llama-server process
  shows ~5 % CPU. "Low CPU" does **not** mean "hung" — check response
  timings instead.

## 1. Is the model listed and loaded?

```bash
curl -s http://127.0.0.1:62891/v1/models | python3 -m json.tool
ps ax | grep llama-server        # which models loaded, which -ub value
```

Model appears in `/v1/models` but errors on use → its server failed to
spawn; read the newest console log (step 3).

## 2. Failed requests are logged WITH payloads

```bash
docker run --rm -v /var/ugreen/log:/log:ro alpine \
  grep -A3 'POST /v1/chat' /log/infer_gateway_serv.log | tail -40
```

The gateway logs every non-2xx response together with the **complete
request JSON** — this is how you see exactly what an app (Uliya, Paperless,
anything) actually sends. Successful requests are not logged.

## 3. llama-server console logs

```bash
docker run --rm -v /var/ugreen/log:/log:ro alpine \
  sh -c 'ls -t /log/infer_gateway_serv_panic/ | head -3'
```

Not panics — these are the live console logs of spawned servers: model
loading, chat-template detection (`Chat format: …`), per-request token
counts and timings (`prompt eval time … tokens per second`), and error spam
like the flash-attention `Scratch buffer` failure. A request that "hangs"
while these logs show steady token progress is the ub-4096 corruption
generating garbage forever.

## 4. The unix-socket probes (root)

Every running llama-server exposes the full llama.cpp API on its socket,
bypassing the gateway:

```bash
SOCK=$(ls /run/ugreen/llama-gw-socks/*.sock | head -1)

# What prompt does the template REALLY produce? (tools included)
curl -s --unix-socket $SOCK http://localhost/apply-template \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hi"}],"tools":[]}' \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["prompt"])'

# Exact token count of anything
curl -s --unix-socket $SOCK http://localhost/tokenize \
  -d '{"content":"some text"}'

# Raw completion without chat template or gateway interference
curl -s --unix-socket $SOCK http://localhost/completion \
  -d '{"prompt":"...","n_predict":80,"temperature":0.1,"top_k":1}'
```

Isolation logic: gateway path broken but socket path fine → gateway bug.
Both broken with a clean rendered prompt → model/backend bug. `top_k: 1`
makes collapses deterministic and therefore bisectable.

## 5. Catalog DB (read-only is safe for everyone)

```bash
python3 - <<'EOF'
import sqlite3
db = sqlite3.connect("file:/volume1/@appstore/com.ugreen.aiconsole/db/"
                     "aiconsole.db?mode=ro", uri=True)
for r in db.execute("SELECT id,code,status FROM model_config"):
    print(r)
EOF
```

Status: 1 = catalog-only, 3 = disabled, 8 = active. Writes: only through
`ugos-llm.py ui` (it backs the DB up first).

## 6. Symptom → likely cause

| Symptom | Cause / fix |
|---|---|
| Endless `- - -` / `1. 1. 1.` repetition on long prompts | ub=4096 corruption → set `-ub 512 -b 512`, reload ([known-bugs #1](known-bugs.md)) |
| App using tool-calling hangs forever | gateway `tool_choice:"required"` bug → route through [openai-bridge](../openai-bridge/) |
| Multilingual token soup from the first word | MoE model → unsupported, remove it |
| `image input is not supported` | server spawned before mmproj was configured → reload |
| Everything `connection refused` on 62891 | somebody killed a llama-server → reboot |
| Model answers but ignores its context on huge prompts | you're near 16k `-c` limit, or ub-4096 again |
| Response arrives after exactly your client timeout | model was cold-loading (30–60 s) — retry |
