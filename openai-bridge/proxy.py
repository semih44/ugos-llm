#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""openai-bridge — a robust OpenAI-compatible endpoint in front of the UGOS
LLM gateway, for your Docker apps.

Why this exists: the UGOS infer_gateway (127.0.0.1:62891) never answers
tool-calling requests in the shape most client libraries send
(tool_choice:"required"), silently strips response_format, and only injects
tools into the prompt without enforcing the output — vanilla models drift.

What it does, and only when tool use is actually FORCED
(tool_choice "required" or a named function):

  request:  remove tools/tool_choice, append an explicit "answer with ONE
            JSON object matching this schema" instruction, cap max_tokens
  response: extract the JSON and re-wrap it as a synthetic tool_call

`tool_choice: "auto"` and `"none"` are passed through untouched — the
gateway handles those natively. Non-tool requests, including streaming, are
proxied verbatim.

The gateway itself has NO authentication, so if you bind this bridge
anywhere reachable from your network, set API_KEY — the bridge then becomes
the authentication boundary and refuses to start on a non-local address
without one.

Configuration (env):
  LISTEN_HOST   default 172.17.0.1  (docker bridge IP: reachable by
                containers, not by your LAN)
  LISTEN_PORT   default 11434
  UPSTREAM      default http://127.0.0.1:62891
  API_KEY       default empty (no auth). When set, every request must carry
                `Authorization: Bearer <key>`. Required to bind anything
                other than loopback or a docker bridge address.
  ALLOW_UNAUTHENTICATED  set to 1 to bind a LAN address without a key.
                Do not. It publishes an open LLM endpoint to your network.
  MAX_TOKENS    default 1024   (hard cap, clamps larger client values)
  TIMEOUT       default 570    (seconds; keep below your client timeout)
  MAX_BODY      default 33554432 (32 MiB request limit)

Run with network_mode: host (see README.md). One log line per request.

License: MIT.
"""
import hmac
import ipaddress
import json
import os
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LISTEN = (os.environ.get("LISTEN_HOST", "172.17.0.1"),
          int(os.environ.get("LISTEN_PORT", "11434")))
UPSTREAM = os.environ.get("UPSTREAM", "http://127.0.0.1:62891").rstrip("/")
API_KEY = os.environ.get("API_KEY", "")
# "on" | "off" | "" (leave requests alone). Injected at request level
# because that is the only level that reliably reaches the model: the UGOS
# gateway swallows llama-server's own --chat-template-kwargs default
# (verified: argv intact, /props kwargs empty), while request-level
# chat_template_kwargs demonstrably win. Clients that send their own kwargs
# always take precedence; emulated tool requests are always pinned off.
THINKING_DEFAULT = os.environ.get("THINKING_DEFAULT", "").lower()
ALLOW_UNAUTHENTICATED = (os.environ.get("ALLOW_UNAUTHENTICATED", "").lower()
                         not in ("", "0", "false", "no"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1024"))
TIMEOUT = int(os.environ.get("TIMEOUT", "570"))
MAX_BODY = int(os.environ.get("MAX_BODY", str(32 * 1024 * 1024)))

SINGLE_INSTR = (
    "\n\nRespond with EXACTLY ONE valid JSON object conforming to this JSON "
    "schema. No markdown, no code fences, no text before or after — only the "
    "JSON object itself:\n{schema}"
)
MULTI_INSTR = (
    "\n\nChoose exactly one of the following functions and respond with "
    "EXACTLY ONE valid JSON object of the form "
    '{{"tool_name": "<one of: {names}>", "arguments": {{...}}}} where '
    "\"arguments\" matches that function's schema. No markdown, no code "
    "fences, no text before or after. Functions:\n{catalog}"
)


class BadRequest(ValueError):
    """Client sent something we can answer with a clean HTTP 400."""


def check_auth(header, key):
    """May this request proceed?

    An empty key disables authentication entirely — that is the shipped
    default and what the docker-only deployment relies on. With a key set,
    exactly one shape is accepted: `Authorization: Bearer <key>`. Being
    strict here is deliberate: silently accepting a bare key or another
    scheme would make a misconfigured client look like it works.
    """
    if not key:
        return True
    if not header:
        return False
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return False
    # bytes, not str: compare_digest rejects non-ASCII str outright, and a
    # plain == would leak the key's length and prefix through timing
    return hmac.compare_digest(token.encode("utf-8"), key.encode("utf-8"))


def _is_local_bind(host):
    """True for addresses only this machine (or its containers) can reach."""
    if host == "localhost":
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False                      # "" and 0.0.0.0-style wildcards
    if ip.is_loopback:
        return True
    # docker bridge networks: reachable by containers on this host but not
    # from the LAN. This is the shipped default and must keep working.
    return ip.version == 4 and ip in ipaddress.ip_network("172.16.0.0/12")


def guard_bind(host, key, allow_unauth):
    """Refuse to publish an unauthenticated LLM endpoint to the network.

    The upstream gateway has no authentication of its own, so binding this
    bridge to a LAN address without a key hands every device on the network
    a free, unmetered model. Raises SystemExit rather than warning, because
    a warning in a container log is a warning nobody reads.
    """
    if key or allow_unauth or _is_local_bind(host):
        return
    raise SystemExit(
        f"openai-bridge: refusing to bind {host!r} without authentication.\n"
        f"  The UGOS gateway has no auth, so this would expose an open LLM "
        f"endpoint to your network.\n"
        f"  Set API_KEY=<something long and random> and send it as "
        f"'Authorization: Bearer <key>',\n"
        f"  or set ALLOW_UNAUTHENTICATED=1 if you genuinely want this.")


def scrub_schema(node, in_properties=False):
    """Drop OpenAI-only keywords ("strict") at schema level. Keys inside a
    "properties" map are property NAMES and must never be touched — a schema
    may legitimately define properties called "strict" or similar."""
    if isinstance(node, dict):
        if not in_properties:
            node.pop("strict", None)
        for k, v in node.items():
            scrub_schema(v, in_properties=(not in_properties
                                           and k == "properties"))
    elif isinstance(node, list):
        for v in node:
            scrub_schema(v, in_properties=False)
    return node


def plan(data):
    """Decide whether to emulate. Returns None (passthrough) or
    ("single", tool) / ("multi", tools). Raises BadRequest on malformed
    tool definitions."""
    tools = data.get("tools")
    if not tools:
        return None
    if not isinstance(tools, list):
        raise BadRequest("'tools' must be an array")
    for t in tools:
        if (not isinstance(t, dict)
                or not isinstance(t.get("function"), dict)
                or not isinstance(t["function"].get("name"), str)):
            raise BadRequest("malformed entry in 'tools' (need "
                             "{type, function:{name,...}})")
    choice = data.get("tool_choice", "auto")
    if choice in ("none", "auto", None):
        return None                      # gateway handles these natively
    if isinstance(choice, dict):
        want = (choice.get("function") or {}).get("name")
        for t in tools:
            if t["function"]["name"] == want:
                return ("single", t)
        # Passing this on would hit the gateway's named-tool hang. Fail fast.
        raise BadRequest(f"tool_choice names {want!r}, which is not in "
                         "'tools'")
    if choice == "required":
        return ("single", tools[0]) if len(tools) == 1 else ("multi", tools)
    return None


def extract_json(text):
    """Peel a JSON object out of a model answer (handles code fences)."""
    text = (text or "").strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j <= i:
        raise ValueError("no JSON object in model output")
    candidate = text[i:j + 1]
    json.loads(candidate)
    return candidate


def rewrite(body, path):
    """Returns (new_body, note, plan_or_None)."""
    if not path.startswith("/v1/chat/completions"):
        return body, "passthrough", None
    try:
        data = json.loads(body)
    except Exception:
        return body, "no-json", None
    p = plan(data)
    if p is None:
        note = "tools-passthrough" if data.get("tools") else "-"
        if THINKING_DEFAULT in ("on", "off") \
                and "chat_template_kwargs" not in data:
            data["chat_template_kwargs"] = {
                "enable_thinking": THINKING_DEFAULT == "on"}
            return (json.dumps(data).encode("utf-8"),
                    note + f",think-{THINKING_DEFAULT}", None)
        return body, note, None

    kind, payload = p
    if kind == "single":
        fn = payload["function"]
        schema = scrub_schema(fn.get("parameters") or {"type": "object"})
        instr = SINGLE_INSTR.format(schema=json.dumps(schema))
        note = f"emulate:{fn['name']}"
    else:
        names = ", ".join(t["function"]["name"] for t in payload)
        catalog = "\n".join(
            f"- {t['function']['name']}: {t['function'].get('description','')}"
            f"\n  schema: "
            f"{json.dumps(scrub_schema(t['function'].get('parameters') or {}))}"
            for t in payload)
        instr = MULTI_INSTR.format(names=names, catalog=catalog)
        note = f"emulate:{len(payload)}-tools"

    data.pop("tools", None)
    data.pop("tool_choice", None)
    data.pop("parallel_tool_calls", None)
    # Emulated requests are budget-capped JSON extractions. If the server's
    # default is enable_thinking:true, reasoning would eat that budget before
    # any JSON appears — pin it off here, unless the client asked otherwise.
    data.setdefault("chat_template_kwargs", {"enable_thinking": False})
    msgs = data.get("messages")
    if not isinstance(msgs, list):
        raise BadRequest("'messages' must be an array")
    if msgs and isinstance(msgs[-1], dict) \
            and isinstance(msgs[-1].get("content"), str):
        msgs[-1]["content"] += instr
    else:
        msgs.append({"role": "user", "content": instr})

    # hard cap, not a default — and strictly a positive JSON integer
    requested = data.get("max_tokens")
    if requested is None:
        requested = MAX_TOKENS
    elif isinstance(requested, bool) or not isinstance(requested, int):
        raise BadRequest("'max_tokens' must be a positive integer")
    elif requested < 1:
        raise BadRequest("'max_tokens' must be >= 1")
    data["max_tokens"] = min(requested, MAX_TOKENS)
    # emulation needs the whole answer before it can be wrapped
    if data.pop("stream", False):
        note += ",stream-disabled"
    return json.dumps(data).encode("utf-8"), note, p


def check_required(args_json, tool):
    """Presence check for required top-level keys.

    Deliberately NOT a JSON Schema validator: types, enum, nested
    requirements, arrays and additionalProperties are not enforced. Treat
    tool arguments coming out of this proxy like any other model output —
    validate them in your application before acting on them.
    """
    params = tool.get("function", {}).get("parameters") or {}
    required = params.get("required") or []
    if not required:
        return
    obj = json.loads(args_json)
    if not isinstance(obj, dict):
        raise ValueError("tool arguments are not a JSON object")
    missing = [k for k in required if k not in obj]
    if missing:
        raise ValueError(f"tool arguments missing required keys: {missing}")


def wrap_as_tool_call(payload, p):
    """Turn a content answer into a synthetic tool_call response."""
    resp = json.loads(payload)
    choice = resp["choices"][0]
    raw = extract_json(choice.get("message", {}).get("content"))
    kind, spec = p
    if kind == "single":
        name, args = spec["function"]["name"], raw
        check_required(args, spec)
    else:
        obj = json.loads(raw)
        name = obj.get("tool_name")
        chosen = next((t for t in spec
                       if t["function"]["name"] == name), None)
        if chosen is None:
            raise ValueError(f"model picked unknown tool {name!r}")
        args = json.dumps(obj.get("arguments") or {})
        check_required(args, chosen)
    choice["message"] = {"role": "assistant", "content": None,
                         "tool_calls": [{"id": "call_bridge_0",
                                         "type": "function",
                                         "function": {"name": name,
                                                      "arguments": args}}]}
    choice["finish_reason"] = "tool_calls"
    return json.dumps(resp).encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, status, payload, ctype="application/json", extra=()):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        for k, v in extra:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(payload)

    def _unauthorized(self):
        self._send(401, json.dumps(
            {"error": {"message": "missing or invalid API key",
                       "type": "invalid_request_error",
                       "code": "invalid_api_key"}}).encode(),
            extra=(("WWW-Authenticate", 'Bearer realm="openai-bridge"'),))
        print(f"{self.command} {self.path} [unauthorized] -> 401 "
              f"from {self.client_address[0]}", flush=True)

    def _proxy(self, body=None):
        t0 = time.time()
        note, p = "-", None
        if body is not None:
            try:
                body, note, p = rewrite(body, self.path)
            except BadRequest as e:
                self._send(400, json.dumps(
                    {"error": {"message": str(e),
                               "type": "invalid_request_error"}}).encode())
                print(f"{self.command} {self.path} [bad-request] -> 400 "
                      f"({e})", flush=True)
                return
            except Exception as e:  # never tear down the connection
                self._send(400, json.dumps(
                    {"error": {"message": f"unprocessable request: {e}",
                               "type": "invalid_request_error"}}).encode())
                print(f"{self.command} {self.path} [rewrite-error] -> 400 "
                      f"({type(e).__name__}: {e})", flush=True)
                return
        req = urllib.request.Request(UPSTREAM + self.path, data=body,
                                     method=self.command)
        # When we authenticate, we ARE the boundary: the client's credential
        # is ours to check and stops here rather than travelling upstream.
        forward = ("Content-Type", "Accept") if API_KEY \
            else ("Content-Type", "Authorization", "Accept")
        for h in forward:
            if self.headers.get(h):
                req.add_header(h, self.headers[h])
        status = 502
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                status = r.status
                ctype = r.headers.get("Content-Type", "application/json")
                if p is None and "text/event-stream" in ctype:
                    # Stream passthrough. Without Content-Length the client
                    # can only detect the end via connection close — announce
                    # and enforce that, or HTTP/1.1 keep-alive hangs forever.
                    self.close_connection = True
                    self.send_response(status)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    # readline() returns as soon as an event line is complete,
                    # so events reach the client while they happen. read(n)
                    # would block until the buffer fills or upstream closes.
                    for line in r:
                        self.wfile.write(line)
                        self.wfile.flush()
                    print(f"{self.command} {self.path} [{note},sse] -> "
                          f"{status} in {time.time()-t0:.1f}s", flush=True)
                    return
                payload = r.read()
        except urllib.error.HTTPError as e:
            payload, status = e.read(), e.code
            ctype = e.headers.get("Content-Type", "application/json")
        except Exception as e:
            self._send(502, json.dumps(
                {"error": {"message": f"openai-bridge: {e}",
                           "type": "upstream_error"}}).encode())
            print(f"{self.command} {self.path} [{note}] -> 502 "
                  f"in {time.time()-t0:.1f}s ({e})", flush=True)
            return

        if p is not None and status == 200:
            try:
                payload = wrap_as_tool_call(payload, p)
                note += ",wrapped"
            except Exception as e:
                note += f",wrap-failed:{type(e).__name__}"
                status = 502
                payload = json.dumps(
                    {"error": {"message": f"openai-bridge: model did not "
                                          f"return usable JSON ({e})",
                               "type": "tool_emulation_failed"}}).encode()
            ctype = "application/json"
        try:
            self._send(status, payload, ctype)
        except BrokenPipeError:
            note += ",client-gone"
        print(f"{self.command} {self.path} [{note}] -> {status} "
              f"in {time.time()-t0:.1f}s", flush=True)

    def do_GET(self):
        if not check_auth(self.headers.get("Authorization"), API_KEY):
            self._unauthorized()
            return
        self._proxy()

    def do_POST(self):
        if not check_auth(self.headers.get("Authorization"), API_KEY):
            self._unauthorized()
            return
        try:
            n = int(self.headers.get("Content-Length", 0) or 0)
        except ValueError:
            self._send(400, b'{"error":{"message":"bad Content-Length"}}')
            return
        if n < 0:
            # read(-1) would block until EOF and pin a handler thread
            self._send(400, b'{"error":{"message":"bad Content-Length"}}')
            return
        if n > MAX_BODY:
            self._send(413, b'{"error":{"message":"request too large"}}')
            return
        self._proxy(self.rfile.read(n))


if __name__ == "__main__":
    guard_bind(LISTEN[0], API_KEY, ALLOW_UNAUTHENTICATED)
    auth = "API key required" if API_KEY else "NO AUTH"
    print(f"openai-bridge {LISTEN} -> {UPSTREAM} [{auth}] "
          f"(max_tokens cap {MAX_TOKENS}, body limit {MAX_BODY//1048576} MiB)",
          flush=True)
    ThreadingHTTPServer(LISTEN, Handler).serve_forever()
