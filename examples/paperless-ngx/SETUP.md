# Paperless-ngx with NAS-local AI on UGOS

End result: Paperless-ngx 3.x with AI title/tag/correspondent suggestions and
document chat, running entirely on your NAS — the LLM is the one you
installed with `ugos-llm.py`. Nothing leaves the device.

## Prerequisites

1. A model installed and reloaded (`ugos-llm.py install … && test …`).
2. Docker installed from the UGOS App Center; your user in the `docker`
   group (`sudo usermod -aG docker <you>`).
3. Shared folders: one on your fast/ext4 volume (e.g. `docker`), optionally
   one on a Btrfs volume for the archive (checksums against bitrot).
   Create shares in the **Files app**, not Control Panel. Disable the share
   recycle bin for the archive share (Paperless renames files constantly).

## Steps

```bash
mkdir -p /volume1/docker/paperless/{data,consume,db,redis,bridge}
mkdir -p /volume2/paperless/{media,export}

# the tool-calling bridge (required, see below)
cp openai-bridge/proxy.py /volume1/docker/paperless/bridge/

cp examples/paperless-ngx/docker-compose.template.yml \
   /volume1/docker/paperless/docker-compose.yml
# edit it: replace every CHANGE_ME, set your model id and volume paths

cd /volume1/docker/paperless && docker compose up -d
```

Or import the compose file as a "project" in the UGOS Docker app if you
prefer UI management (name it, pick the folder, deploy). Note the app saves
its copy as `docker-compose.yaml` — keep `.yml`/`.yaml` identical if both
exist.

First start pulls ~2 GB of images and runs DB migrations; the admin account
comes from the `PAPERLESS_ADMIN_*` variables. Change that password after
login — it sits in plaintext in the compose file.

## Why the bridge is mandatory

Paperless (via llama-index) requests suggestions through **forced tool
calling** (`tool_choice: "required"`). The UGOS gateway hangs on that shape
forever, and even in working shapes it doesn't enforce tool output — vanilla
models drift. The bridge emulates the whole mechanism with schema-guided
JSON and synthetic tool_calls; with it, suggestions complete in ~30–60 s
(two LLM passes when `PAPERLESS_AI_LLM_OUTPUT_LANGUAGE` is set).
Healthy log line (`docker logs paperless-llm-bridge`):

```
POST /v1/chat/completions [tool-emulation:DocumentClassifierSchema,wrapped] -> 200 in 24.2s
```

## Expectations & tips

- AI suggestions are **on-demand per document** (button in the UI) — bulk
  auto-tagging is Paperless's classic matching + the learning classifier,
  which need no LLM at all. Configure those first; use AI for the odd cases.
- First AI call after a model reload takes ~1 min (model load). Subsequent
  calls are much faster.
- The nightly embedding index (03:10 by default) runs on CPU inside the
  Paperless container; with a large archive expect some load.
- `HF_HUB_OFFLINE: 1` requires the embedding model to be cached — it
  downloads on the very first AI call, so either trigger one suggestion
  before enabling it, or start with it unset and add it later.
- Document chat quality scales with model size; 9B-class is the sweet spot
  on 32 GB devices.
