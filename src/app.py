"""Thin proxy for the RAG Android app.

Holds the Claude and Voyage API keys server-side (SSM Parameter Store,
SecureString) so the Android client never ships them in the APK.

Routes (HTTP API, payload format 2.0):
  POST /claude/messages    -> https://api.anthropic.com/v1/messages
  POST /voyage/embeddings  -> https://api.voyageai.com/v1/embeddings
"""

import base64
import http.client
import json
import logging
import os
import urllib.error
import urllib.request

import boto3

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Upper bound for one upstream call; stays under the HTTP API integration timeout (30 s).
UPSTREAM_TIMEOUT_S = 28
# Lambda's total timeout is 29 s; leave room to send the response back.
RESPONSE_MARGIN_S = 1.5

CLAUDE_KEY_PARAM = os.environ["CLAUDE_KEY_PARAM"]
VOYAGE_KEY_PARAM = os.environ["VOYAGE_KEY_PARAM"]
ALLOWED_MODELS = {m.strip() for m in os.environ.get("ALLOWED_MODELS", "").split(",") if m.strip()}
MAX_TOKENS_CAP = int(os.environ.get("MAX_TOKENS_CAP", "1024"))

# Abuse limits. Sized ~5-10x above what the app really sends (800-char chunks,
# a handful per document) so normal use never trips them.
MAX_BODY_BYTES = 256 * 1024
CLAUDE_FIELDS = ("model", "max_tokens", "system", "messages", "tools")
ALLOWED_VOYAGE_MODELS = {"voyage-4"}
MAX_VOYAGE_INPUTS = 100
MAX_VOYAGE_CHARS_PER_INPUT = 4000

_ssm = boto3.client("ssm")

# Module-level cache: fetched once per container, reused across warm invocations.
_keys = None


def _load_keys():
    global _keys
    if _keys is None:
        resp = _ssm.get_parameters(
            Names=[CLAUDE_KEY_PARAM, VOYAGE_KEY_PARAM],
            WithDecryption=True,
        )
        if resp.get("InvalidParameters"):
            # Log the parameter names only, never the values.
            raise RuntimeError(f"Missing SSM parameters: {resp['InvalidParameters']}")
        by_name = {p["Name"]: p["Value"] for p in resp["Parameters"]}
        _keys = {
            "claude": by_name[CLAUDE_KEY_PARAM].strip(),
            "voyage": by_name[VOYAGE_KEY_PARAM].strip(),
        }
    return _keys


def _upstream_timeout(context):
    if context is None:
        return UPSTREAM_TIMEOUT_S
    remaining_s = context.get_remaining_time_in_millis() / 1000
    return max(1.0, min(UPSTREAM_TIMEOUT_S, remaining_s - RESPONSE_MARGIN_S))


def _response(status, payload, extra_headers=None):
    body = payload if isinstance(payload, str) else json.dumps(payload)
    return {
        "statusCode": status,
        "headers": {"content-type": "application/json", **(extra_headers or {})},
        "body": body,
    }


def _error(status, message):
    return _response(status, {"error": {"type": "proxy_error", "message": message}})


class _BadRequest(Exception):
    pass


def _parse_body(event):
    raw = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        raw = base64.b64decode(raw).decode("utf-8")
    if len(raw.encode("utf-8")) > MAX_BODY_BYTES:
        raise _BadRequest("Request body too large")
    return json.loads(raw)


def _forward(url, headers, body, timeout):
    """POST to the provider and return a ready-made Lambda response."""
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={**headers, "content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return _response(resp.status, resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as e:
        # urllib raises for any non-2xx. Pass upstream errors (400, 429, 529...)
        # through so the client can react.
        logger.warning("Upstream %s answered %s", url, e.code)
        retry_after = e.headers.get("retry-after")
        extra = {"retry-after": retry_after} if retry_after else None
        return _response(e.code, e.read().decode("utf-8", errors="replace"), extra)
    except (OSError, http.client.HTTPException) as e:
        # URLError wraps the real cause in .reason; a connect timeout lands here too.
        timed_out = isinstance(getattr(e, "reason", e), TimeoutError)
        outcome = "timed out" if timed_out else "unreachable"
        logger.warning("Upstream %s %s: %s", url, outcome, type(e).__name__)
        if timed_out:
            return _error(504, "Upstream timeout")
        return _error(502, "Upstream unreachable")


def _handle_claude(body, keys, timeout):
    model = body.get("model")
    if not isinstance(model, str) or model not in ALLOWED_MODELS:
        return _error(400, f"Model not allowed: {model}")
    if body.get("stream"):
        return _error(400, "Streaming is not supported by this proxy")

    max_tokens = body.get("max_tokens", MAX_TOKENS_CAP)
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        return _error(400, "max_tokens must be a positive integer")

    # Server-side tools (e.g. web search) are billed separately, so allow only
    # client-defined tools, which have no "type" or type "custom".
    tools = body.get("tools") or []
    if not isinstance(tools, list) or any(
        not isinstance(t, dict) or t.get("type", "custom") != "custom" for t in tools
    ):
        return _error(400, "Only custom tools are allowed")

    # Forward only the fields the app uses; anything else is dropped.
    forwarded = {k: body[k] for k in CLAUDE_FIELDS if k in body}
    # Cap output so a leaked proxy URL can't be used for expensive long generations.
    forwarded["max_tokens"] = min(max_tokens, MAX_TOKENS_CAP)

    return _forward(
        ANTHROPIC_URL,
        {"x-api-key": keys["claude"], "anthropic-version": ANTHROPIC_VERSION},
        forwarded,
        timeout,
    )


def _handle_voyage(body, keys, timeout):
    model = body.get("model")
    if not isinstance(model, str) or model not in ALLOWED_VOYAGE_MODELS:
        return _error(400, f"Model not allowed: {model}")

    texts = body.get("input")
    if (
        not isinstance(texts, list)
        or not 1 <= len(texts) <= MAX_VOYAGE_INPUTS
        or any(not isinstance(t, str) or len(t) > MAX_VOYAGE_CHARS_PER_INPUT for t in texts)
    ):
        return _error(
            400,
            f"input must be 1-{MAX_VOYAGE_INPUTS} strings of at most "
            f"{MAX_VOYAGE_CHARS_PER_INPUT} characters",
        )

    return _forward(
        VOYAGE_URL,
        {"Authorization": f"Bearer {keys['voyage']}"},
        {"input": texts, "model": model},
        timeout,
    )


ROUTES = {
    "/claude/messages": _handle_claude,
    "/voyage/embeddings": _handle_voyage,
}


def handler(event, context):
    route = ROUTES.get(event.get("rawPath", ""))
    if route is None:
        return _error(404, "Unknown route")

    try:
        body = _parse_body(event)
    except _BadRequest as e:
        return _error(400, str(e))
    except ValueError:
        return _error(400, "Request body must be valid JSON")
    if not isinstance(body, dict):
        return _error(400, "Request body must be a JSON object")

    try:
        keys = _load_keys()
    except Exception as e:
        # Parameter names / ARNs only; boto errors don't contain parameter values.
        logger.error("Could not load API keys: %s: %s", type(e).__name__, e)
        return _error(500, "Proxy is misconfigured")

    return route(body, keys, _upstream_timeout(context))