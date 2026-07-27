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
| `tool_choice: {"function": {"name": …}}` | emulated against exactly that tool (unknown name → passthrough) |

Emulated requests get `stream` removed (the answer must be buffered to be
wrapped) and `max_tokens` clamped to `MAX_TOKENS` — a **hard cap**, larger
client values are reduced.

## Ops notes

- One log line per request (`docker logs llm-bridge`). Healthy:
  `[emulate:DocumentClassifierSchema,wrapped] -> 200 in 21.5s`.
  `[tools-passthrough]` means auto/none went straight to the gateway.
  `wrap-failed:*` means the model answered non-JSON — the client then gets an
  honest **HTTP 502**, never a silent bad result.
- Requests larger than `MAX_BODY` (32 MiB) are rejected with 413.
- The gateway serializes per model (`-np 1`): parallel callers queue.
- Environment: `LISTEN_HOST`, `LISTEN_PORT`, `UPSTREAM`, `MAX_TOKENS`,
  `TIMEOUT`, `MAX_BODY`.
