# openai-bridge

Exposes the UGOS LLM gateway as a **robust** OpenAI-compatible endpoint for
your Docker containers — with working tool-calling.

## Why you need it

The UGOS gateway (`127.0.0.1:62891`) is loopback-only, so containers can't
reach it at all. And even bridged, its tool-calling is broken for real-world
clients: `tool_choice: "required"` (what llama-index, LangChain & friends
send) **hangs forever**, `response_format` is stripped, and nothing forces
the model to emit a tool call — vanilla models drift off into text. Details:
[docs/known-bugs.md](../docs/known-bugs.md) #4.

This proxy listens on the Docker bridge IP (`172.17.0.1:11434` — reachable
by every container, invisible to your LAN), forwards everything to the
gateway, and **emulates tool-calling**: tools are converted into an explicit
JSON-schema instruction, the answer is parsed and re-wrapped as a synthetic
`tool_call`. Your client library sees a perfectly normal OpenAI response.

## Run it

```yaml
# add to any docker-compose.yml
services:
  llm-bridge:
    image: docker.io/library/python:3.12-alpine
    container_name: llm-bridge
    restart: unless-stopped
    network_mode: host          # needed to reach 127.0.0.1:62891
    volumes:
      - /volume1/docker/openai-bridge:/bridge:ro
    command: ["python3", "/bridge/proxy.py"]
```

Put `proxy.py` into `/volume1/docker/openai-bridge/` first. Containers then
use:

```
OPENAI_BASE_URL = http://172.17.0.1:11434/v1
OPENAI_API_KEY  = anything-nonempty        # the gateway ignores it
model           = <DirName>/<DirName>      # e.g. Qwen3.5-9B/Qwen3.5-9B
```

## What is rewritten, and what is not

| Request | Behaviour |
|---|---|
| no `tools` | passed through verbatim (including SSE streaming) |
| `tool_choice: "auto"` / `"none"` | **passed through** — the gateway handles these natively |
| `tool_choice: "required"`, one tool | emulated against that tool's schema |
| `tool_choice: "required"`, several tools | emulated: the model picks one (`{"tool_name":…,"arguments":…}`), the answer is mapped back and validated against the offered names |
| `tool_choice: {"function": {"name": …}}` | emulated against exactly that tool (a name that isn't in `tools` → HTTP 400, because passing it on would hit the gateway's named-tool hang) |

The bridge verifies that a synthetic tool call carries the schema's
**required top-level keys** — it is deliberately *not* a full JSON Schema
validator (no type, enum, nested or array checks). Validate tool arguments in
your application, as you would with any model output.

Emulated requests get `stream` removed (the answer must be buffered to be
wrapped) and `max_tokens` clamped to `MAX_TOKENS` — a **hard cap**, larger
client values are reduced.

## Reaching it from another machine (and locking it down)

The gateway has **no authentication of its own**, so the moment you bind
this bridge to something your LAN can reach, every device on the network can
use your model. The bridge therefore **refuses to start** on a non-local
address unless `API_KEY` is set:

```
openai-bridge: refusing to bind '192.168.1.221' without authentication.
```

Set a key and it becomes the authentication boundary — it validates
`Authorization: Bearer <key>` in constant time, answers `401` otherwise, and
does not pass the client's credential upstream:

```bash
docker run -d --name ugos-llm-bridge --restart unless-stopped \
  --network host \
  -v /volume1/docker/ugos-llm-bridge:/bridge:ro \
  -e LISTEN_HOST=192.168.1.221 \
  -e LISTEN_PORT=11436 \
  -e API_KEY="$(python3 -c 'import secrets;print(secrets.token_urlsafe(32))')" \
  -e MAX_TOKENS=4096 \
  python:3.12-alpine python3 /bridge/proxy.py
```

Run this as a **second** instance rather than re-purposing a container your
apps already depend on. `ALLOW_UNAUTHENTICATED=1` exists to override the
guard; there is no good reason to use it.

### VS Code (Copilot BYOK, "Custom Endpoint")

`Chat: Manage Language Models` → **Custom Endpoint**, base URL
`http://<nas-ip>:11436/v1`, the API key from above, model id
`Gemma4-26B-A4B/Gemma4-26B-A4B`. Works without a GitHub account or Copilot
plan. Streaming and native tool calls both survive the chain — verified
end-to-end from a MacBook.

Two limits worth knowing before you set expectations. **Inline completions
are out of reach**: VS Code's docs state plainly that "you cannot connect to
a local model for inline suggestions" — BYOK drives chat only, so no amount
of server-side work produces ghost text. And **agent-style extensions will
struggle**: the gateway serialises at `-np 1`, and a model configured with
an 8k context runs out of room quickly once an agent starts passing files
around.

## Ops notes

- One log line per request (`docker logs llm-bridge`). Healthy:
  `[emulate:DocumentClassifierSchema,wrapped] -> 200 in 21.5s`.
  `[tools-passthrough]` means auto/none went straight to the gateway.
  `wrap-failed:*` means the model answered non-JSON — the client then gets an
  honest **HTTP 502**, never a silent bad result.
  `[unauthorized] -> 401 from <ip>` logs the caller's address.
- Requests larger than `MAX_BODY` (32 MiB) are rejected with 413.
- The gateway serializes per model (`-np 1`): parallel callers queue.
- Environment: `LISTEN_HOST`, `LISTEN_PORT`, `UPSTREAM`, `API_KEY`,
  `ALLOW_UNAUTHENTICATED`, `MAX_TOKENS`, `TIMEOUT`, `MAX_BODY`.
