"""Unit tests for the proxy handler. They run offline: boto3 and the outgoing HTTP call are replaced."""
import base64
import http.client
import io
import json
import logging
import os
import sys
import types
import unittest
import urllib.error

sys.modules["boto3"] = types.SimpleNamespace(client=lambda *_: None)
os.environ.update(CLAUDE_KEY_PARAM="/p/claude", VOYAGE_KEY_PARAM="/p/voyage", ALLOWED_MODELS="m", MAX_TOKENS_CAP="1024")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import app  # noqa: E402

CLAUDE = "/claude/messages"
VOYAGE = "/voyage/embeddings"
MESSAGES = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]


class FakeResponse:
    status = 200

    def __init__(self, body=b'{"ok":true}'):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class HandlerTestCase(unittest.TestCase):
    def setUp(self):
        app._keys = {"claude": "claude-key", "voyage": "voyage-key"}
        self.sent = {}

        def fake_forward(url, headers, body, timeout):
            self.sent.update(url=url, headers=headers, body=body, timeout=timeout)
            return app._response(200, "{}")

        self._real_forward = app._forward
        app._forward = fake_forward

    def tearDown(self):
        app._forward = self._real_forward

    def call(self, path, body=None, raw=None, base64_encoded=False, context=None):
        event = {"rawPath": path, "body": raw if raw is not None else json.dumps(body)}
        if base64_encoded:
            event["isBase64Encoded"] = True
        return app.handler(event, context)


class ClaudeRouteTest(HandlerTestCase):
    def test_valid_request_is_forwarded_with_the_key_and_version(self):
        r = self.call(CLAUDE, {"model": "m", "max_tokens": 10, "messages": MESSAGES})

        self.assertEqual(r["statusCode"], 200)
        self.assertEqual(self.sent["url"], app.ANTHROPIC_URL)
        self.assertEqual(self.sent["headers"]["x-api-key"], "claude-key")
        self.assertEqual(self.sent["headers"]["anthropic-version"], app.ANTHROPIC_VERSION)

    def test_only_known_fields_are_forwarded(self):
        self.call(CLAUDE, {"model": "m", "max_tokens": 10, "messages": MESSAGES, "extra": 1, "stream": False})

        self.assertEqual(set(self.sent["body"]), {"model", "max_tokens", "messages"})

    def test_max_tokens_is_capped(self):
        self.call(CLAUDE, {"model": "m", "max_tokens": 5000, "messages": MESSAGES})

        self.assertEqual(self.sent["body"]["max_tokens"], 1024)

    def test_missing_max_tokens_defaults_to_the_cap(self):
        self.call(CLAUDE, {"model": "m", "messages": MESSAGES})

        self.assertEqual(self.sent["body"]["max_tokens"], 1024)

    def test_invalid_max_tokens_is_a_400(self):
        for bad in ("abc", 0, -1, True, None, 1.5):
            r = self.call(CLAUDE, {"model": "m", "max_tokens": bad, "messages": MESSAGES})
            self.assertEqual(r["statusCode"], 400, bad)

    def test_unknown_model_is_a_400(self):
        r = self.call(CLAUDE, {"model": "other", "max_tokens": 5, "messages": MESSAGES})

        self.assertEqual(r["statusCode"], 400)
        self.assertEqual(self.sent, {})

    def test_model_that_is_not_a_string_is_a_400(self):
        r = self.call(CLAUDE, {"model": ["m"], "max_tokens": 5, "messages": MESSAGES})

        self.assertEqual(r["statusCode"], 400)

    def test_streaming_is_rejected(self):
        r = self.call(CLAUDE, {"model": "m", "stream": True, "messages": MESSAGES})

        self.assertEqual(r["statusCode"], 400)

    def test_server_side_tools_are_rejected(self):
        tools = [{"type": "web_search_20250305", "name": "web_search"}]
        r = self.call(CLAUDE, {"model": "m", "messages": MESSAGES, "tools": tools})

        self.assertEqual(r["statusCode"], 400)

    def test_custom_tools_are_allowed(self):
        tools = [{"name": "t", "description": "d", "input_schema": {}}]
        r = self.call(CLAUDE, {"model": "m", "messages": MESSAGES, "tools": tools})

        self.assertEqual(r["statusCode"], 200)
        self.assertEqual(self.sent["body"]["tools"], tools)


class VoyageRouteTest(HandlerTestCase):
    def test_valid_request_is_forwarded_with_the_bearer_token(self):
        r = self.call(VOYAGE, {"model": "voyage-4", "input": ["a", "b"]})

        self.assertEqual(r["statusCode"], 200)
        self.assertEqual(self.sent["url"], app.VOYAGE_URL)
        self.assertEqual(self.sent["headers"]["Authorization"], "Bearer voyage-key")
        self.assertEqual(self.sent["body"], {"input": ["a", "b"], "model": "voyage-4"})

    def test_unknown_model_is_a_400(self):
        self.assertEqual(self.call(VOYAGE, {"model": "voyage-3", "input": ["a"]})["statusCode"], 400)

    def test_input_must_be_a_non_empty_list_of_short_strings(self):
        cases = [[], "a", [1], ["a" * (app.MAX_VOYAGE_CHARS_PER_INPUT + 1)], ["a"] * (app.MAX_VOYAGE_INPUTS + 1)]
        for bad in cases:
            r = self.call(VOYAGE, {"model": "voyage-4", "input": bad})
            self.assertEqual(r["statusCode"], 400)

    def test_input_at_the_limits_is_accepted(self):
        r = self.call(VOYAGE, {"model": "voyage-4", "input": ["a" * 800] * app.MAX_VOYAGE_INPUTS})

        self.assertEqual(r["statusCode"], 200)

    def test_other_fields_are_dropped(self):
        self.call(VOYAGE, {"model": "voyage-4", "input": ["a"], "input_type": "query", "truncation": False})

        self.assertEqual(set(self.sent["body"]), {"input", "model"})


class RequestHandlingTest(HandlerTestCase):
    def test_unknown_route_is_a_404(self):
        self.assertEqual(self.call("/nope", {})["statusCode"], 404)

    def test_invalid_json_is_a_400(self):
        self.assertEqual(self.call(CLAUDE, raw="{oops")["statusCode"], 400)

    def test_invalid_utf8_is_a_400(self):
        raw = base64.b64encode(b"\xff\xfe").decode()

        self.assertEqual(self.call(CLAUDE, raw=raw, base64_encoded=True)["statusCode"], 400)

    def test_json_that_is_not_an_object_is_a_400(self):
        self.assertEqual(self.call(CLAUDE, raw="[1]")["statusCode"], 400)

    def test_a_body_over_the_size_limit_is_a_400(self):
        r = self.call(VOYAGE, {"model": "voyage-4", "input": ["я" * 4000] * 100})

        self.assertEqual(r["statusCode"], 400)

    def test_upstream_timeout_is_derived_from_the_remaining_lambda_time(self):
        context = types.SimpleNamespace(get_remaining_time_in_millis=lambda: 27000)
        self.call(VOYAGE, {"model": "voyage-4", "input": ["a"]}, context=context)

        self.assertEqual(self.sent["timeout"], 25.5)

    def test_timeout_never_drops_below_one_second(self):
        context = types.SimpleNamespace(get_remaining_time_in_millis=lambda: 500)
        self.call(VOYAGE, {"model": "voyage-4", "input": ["a"]}, context=context)

        self.assertEqual(self.sent["timeout"], 1.0)


class KeyLoadingTest(unittest.TestCase):
    def tearDown(self):
        app._keys = None

    def test_keys_are_stripped_of_whitespace(self):
        class Ssm:
            def get_parameters(self, Names, WithDecryption):
                return {
                    "Parameters": [{"Name": "/p/claude", "Value": "SECRET_C\n"}, {"Name": "/p/voyage", "Value": " SECRET_V "}],
                    "InvalidParameters": [],
                }

        app._ssm, app._keys = Ssm(), None

        self.assertEqual(app._load_keys(), {"claude": "SECRET_C", "voyage": "SECRET_V"})

    def test_a_failure_returns_500_and_logs_no_secret(self):
        class Ssm:
            def get_parameters(self, **kwargs):
                raise RuntimeError("AccessDenied for /p/claude")

        app._ssm, app._keys = Ssm(), None
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        app.logger.addHandler(handler)
        try:
            r = app.handler({"rawPath": CLAUDE, "body": json.dumps({"model": "m"})}, None)
        finally:
            app.logger.removeHandler(handler)

        self.assertEqual(r["statusCode"], 500)
        self.assertIn("RuntimeError", stream.getvalue())


class ForwardTest(unittest.TestCase):
    def forward(self, effect):
        real = app.urllib.request.urlopen
        app.urllib.request.urlopen = effect
        try:
            return app._forward("https://x", {}, {}, 5)
        finally:
            app.urllib.request.urlopen = real

    def test_success_passes_the_body_through(self):
        r = self.forward(lambda req, timeout: FakeResponse())

        self.assertEqual(r["statusCode"], 200)
        self.assertEqual(json.loads(r["body"]), {"ok": True})

    def test_upstream_errors_keep_their_status_and_retry_after(self):
        def raise_429(req, timeout):
            raise urllib.error.HTTPError("https://x", 429, "Too Many", {"retry-after": "7"}, io.BytesIO(b'{"e":1}'))

        r = self.forward(raise_429)

        self.assertEqual(r["statusCode"], 429)
        self.assertEqual(r["headers"]["retry-after"], "7")

    def test_error_without_retry_after_has_no_such_header(self):
        def raise_400(req, timeout):
            raise urllib.error.HTTPError("https://x", 400, "Bad", {}, io.BytesIO(b"{}"))

        self.assertNotIn("retry-after", self.forward(raise_400)["headers"])

    def test_timeouts_are_a_504(self):
        def connect_timeout(req, timeout):
            raise urllib.error.URLError(TimeoutError())

        def read_timeout(req, timeout):
            raise TimeoutError()

        self.assertEqual(self.forward(connect_timeout)["statusCode"], 504)
        self.assertEqual(self.forward(read_timeout)["statusCode"], 504)

    def test_other_network_failures_are_a_502(self):
        def reset(req, timeout):
            raise ConnectionResetError()

        def incomplete(req, timeout):
            raise http.client.IncompleteRead(b"x")

        def dns(req, timeout):
            raise urllib.error.URLError(OSError("dns"))

        for effect in (reset, incomplete, dns):
            self.assertEqual(self.forward(effect)["statusCode"], 502)


if __name__ == "__main__":
    unittest.main()
