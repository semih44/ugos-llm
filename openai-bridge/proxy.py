#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""openai-bridge — a robust OpenAI-compatible endpoint in front of the UGOS
LLM gateway, for your Docker apps.

Why this exists: the UGOS infer_gateway (127.0.0.1:62891) hangs forever on
tool-calling requests in the shape most client libraries send
(tool_choice:"required"), silently strips response_format, and only injects
tools into the prompt without enforcing the output. Vanilla models routinely
fail that. This proxy makes tool-calling WORK by emulating it:

  request:  strip tools/tool_choice, append an explicit "answer ONLY with
            one JSON object matching this schema" instruction, cap max_tokens
  response: extract the JSON, re-wrap it as a synthetic tool_call
            (finish_reason "tool_calls")

Clients (openai SDK, llama-index, LangChain, Paperless-ngx, ...) never
notice. Non-tool requests pass through untouched.

Configuration (env):
  LISTEN_HOST   default 172.17.0.1  (docker bridge IP: reachable by
                containers, NOT by your LAN)
  LISTEN_PORT   default 11434
  UPSTREAM      default http://127.0.0.1:62891
  MAX_TOKENS    default 1024   (cap; prevents infinite generation)
  TIMEOUT       default 570    (seconds; keep below your client timeout)

Run it with network_mode: host (see docker-compose.snippet.yml).
Logs go to stdout: one line per request with rewrite notes and duration.

License: MIT.
"""
import json
import os
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LISTEN = (os.environ.get("LISTEN_HOST", "172.17.0.1"),
          int(os.environ.get("LISTEN_PORT", "11434")))
UPSTREAM = os.environ.get("UPSTREAM", "http://127.0.0.1:62891")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1024"))
TIMEOUT = int(os.environ.get("TIMEOUT", "570"))

JSON_INSTR = (
    "\n\nRespond with EXACTLY ONE valid JSON object conforming to this JSON "
    "schema. No markdown, no code fences, no text before or after — only "
    "the JSON object itself:\n{schema}"
)


def extract_json(text):
    """Peel a JSON object out of a model answer (handles code fences)."""
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j <= i:
        raise ValueError("no JSON object found")
    candidate = text[i:j + 1]
    json.loads(candidate)
    return candidate


def rewrite(body, path):
    """Returns (new_body, note, tool_name_or_None)."""
    if not path.startswith("/v1/chat/completions"):
        return body, "passthrough", None
    try:
        data = json.loads(body)
    except Exception:
        return body, "no-json", None
    tools = data.get("tools")
    if not tools:
        return body, "no-tools", None
    try:
        name = tools[0]["function"]["name"]
        schema = tools[0]["function"].get("parameters", {"type": "object"})
    except Exception:
        return body, "tool-shape-unknown", None

    def scrub(node):  # OpenAI-isms the gateway/model don't need
        if isinstance(node, dict):
            node.pop("strict", None)
            node.pop("additionalProperties", None)
            for v in node.values():
                scrub(v)
        elif isinstance(node, list):
            for v in node:
                scrub(v)
    scrub(schema)

    data.pop("tools", None)
    data.pop("tool_choice", None)
    data.pop("parallel_tool_calls", None)
    msgs = data.get("messages", [])
    if msgs and isinstance(msgs[-1].get("content"), str):
        msgs[-1]["content"] += JSON_INSTR.format(schema=json.dumps(schema))
    if not data.get("max_tokens"):
        data["max_tokens"] = MAX_TOKENS
    return (json.dumps(data).encode("utf-8"),
            f"tool-emulation:{name}", name)


def wrap_as_tool_call(payload, tool_name):
    resp = json.loads(payload)
    choice = resp["choices"][0]
    content = choice.get("message", {}).get("content") or ""
    args = extract_json(content)
    choice["message"] = {"role": "assistant", "content": None,
                         "tool_calls": [{"id": "call_bridge_0",
                                         "type": "function",
                                         "function": {"name": tool_name,
                                                      "arguments": args}}]}
    choice["finish_reason"] = "tool_calls"
    return json.dumps(resp).encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _proxy(self, body=None):
        t0 = time.time()
        note, tool_name = "-", None
        if body is not None:
            body, note, tool_name = rewrite(body, self.path)
        req = urllib.request.Request(UPSTREAM + self.path, data=body,
                                     method=self.command)
        for h in ("Content-Type", "Authorization", "Accept"):
            if self.headers.get(h):
                req.add_header(h, self.headers[h])
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                payload, status = r.read(), r.status
                ctype = r.headers.get("Content-Type", "application/json")
        except urllib.error.HTTPError as e:
            payload, status = e.read(), e.code
            ctype = e.headers.get("Content-Type", "application/json")
        except Exception as e:
            payload = json.dumps(
                {"error": {"message": f"openai-bridge: {e}"}}).encode()
            status, ctype = 502, "application/json"
        if tool_name and status == 200:
            try:
                payload = wrap_as_tool_call(payload, tool_name)
                note += ",wrapped"
            except Exception as e:
                note += f",wrap-failed:{type(e).__name__}"
        try:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except BrokenPipeError:
            note += ",client-gone"
        print(f"{self.command} {self.path} [{note}] -> {status} "
              f"in {time.time()-t0:.1f}s", flush=True)

    def do_GET(self):
        self._proxy()

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        self._proxy(self.rfile.read(n))


if __name__ == "__main__":
    print(f"openai-bridge {LISTEN} -> {UPSTREAM} "
          f"(max_tokens cap {MAX_TOKENS})", flush=True)
    ThreadingHTTPServer(LISTEN, Handler).serve_forever()
