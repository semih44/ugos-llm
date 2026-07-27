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

## Ops notes

- One log line per request (`docker logs llm-bridge`):
  `[tool-emulation:Name,wrapped] -> 200 in 21.5s` is the healthy shape.
  `wrap-failed:*` means the model answered non-JSON — the client gets a
  normal error instead of a hang.
- `MAX_TOKENS` (default 1024) caps generation as a safety net against the
  infinite-generation failure modes of the stack.
- Remember the gateway serializes per model (`-np 1`): parallel callers
  queue. Plan batch jobs accordingly.
