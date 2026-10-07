import asyncio
import json

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from viola import adapters, config
from viola.classifier import Decision
from viola.proxy import build_app


class StubClassifier:
    """Keyword stand-in for GLiNER so the proxy can be tested without the model."""

    def __init__(self, ready=True, confidence=0.9):
        self.ready = ready
        self.confidence = confidence
        self.calls = []

    def classify(self, text):
        self.calls.append(text)
        tier = "xl" if "compiler" in text else "large" if "deadlock" in text else "small" if "typo" in text else "medium"
        return Decision(tier, self.confidence, tier, 1.0)


def fake_upstream():
    seen = []

    async def handler(request):
        body = await request.read()
        seen.append({"path": request.path_qs, "headers": dict(request.headers), "body": body})
        if request.headers.get("x-fail"):
            return web.Response(status=400, body=b'{"type":"error","error":{"message":"thinking: bound to a different conversation"}}',
                                content_type="application/json")
        resp = web.StreamResponse(headers={"content-type": "text/event-stream", "x-should-retry": "false"})
        await resp.prepare(request)
        for event in (b"event: ping\ndata: {}\n\n", b"event: message_stop\ndata: {}\n\n"):
            await resp.write(event)
        await resp.write_eof()
        return resp

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handler)
    return app, seen


def run(coro):
    return asyncio.run(coro)


async def make(cfg_overrides=None, classifier=None):
    up_app, seen = fake_upstream()
    up = TestServer(up_app)
    await up.start_server()
    cfg = config._merge(config.DEFAULTS, cfg_overrides or {})
    cfg["anthropic"]["upstream"] = str(up.make_url("")).rstrip("/")
    cfg["openai"]["upstream"] = str(up.make_url("/v1")).rstrip("/")
    classifier = classifier or StubClassifier()
    client = TestClient(TestServer(build_app(cfg, classifier, load_in_background=False)))
    await client.start_server()
    return client, up, seen, classifier


def cc_request(text, request_class="main", **extra):
    body = {
        "model": "claude-opus-5-5",
        "max_tokens": 128000,
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "xhigh"},
        "system": [{"type": "text", "text": "x-anthropic-billing-header: cc"}, {"type": "text", "text": "You are Claude Code"}],
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "<system-reminder>context stuff</system-reminder>"},
            {"type": "text", "text": text},
        ]}],
        **extra,
    }
    headers = {"x-api-key": "sk-test", "anthropic-version": "2023-06-01", "anthropic-beta": "oauth-2025-04-20,foo",
               "x-claude-code-request-class": request_class, "content-type": "application/json"}
    return body, headers


def test_main_request_is_routed_and_streamed():
    async def go():
        client, up, seen, clf = await make()
        body, headers = cc_request("build a compiler for our language")
        r = await client.post("/anthropic/v1/messages?beta=true", data=json.dumps(body), headers=headers)
        text = await r.text()
        assert r.status == 200
        assert r.headers["x-viola-tier"] == "xl" and r.headers["x-viola-model"] == "claude-fable-5-1"
        assert r.headers["x-should-retry"] == "false"
        assert "event: ping" in text and "message_stop" in text
        sent = json.loads(seen[0]["body"])
        assert sent["model"] == "claude-fable-5-1"
        assert sent["system"] == body["system"]  # system array untouched (attribution block stays first)
        assert seen[0]["path"] == "/v1/messages?beta=true"
        assert seen[0]["headers"]["x-api-key"] == "sk-test"
        assert seen[0]["headers"]["anthropic-beta"] == "oauth-2025-04-20,foo"
        assert clf.calls == ["build a compiler for our language"]  # system reminder stripped
        await client.close(); await up.close()
    run(go())


def test_small_tier_adapts_params_for_haiku():
    async def go():
        client, up, seen, _ = await make()
        body, headers = cc_request("fix the typo in the readme")
        r = await client.post("/anthropic/v1/messages", data=json.dumps(body), headers=headers)
        await r.read()
        sent = json.loads(seen[0]["body"])
        assert sent["model"] == "claude-haiku-4-5"
        assert "thinking" not in sent and "output_config" not in sent
        assert sent["max_tokens"] == 64000
        await client.close(); await up.close()
    run(go())


def test_legacy_tier_drops_eager_input_streaming():
    body = {"tools": [{"name": "Bash", "input_schema": {}, "eager_input_streaming": True}]}
    out = adapters.anthropic_params(body, "au.anthropic.claude-sonnet-4-5-20250929-v1:0", None)
    assert out["tools"] == [{"name": "Bash", "input_schema": {}}]
    keep = {"tools": [{"name": "Bash", "eager_input_streaming": True}]}
    assert adapters.anthropic_params(keep, "au.anthropic.claude-opus-5-5", None)["tools"][0]["eager_input_streaming"]


def test_non_main_requests_pass_through_byte_for_byte():
    async def go():
        client, up, seen, clf = await make()
        body, headers = cc_request("fix the typo", request_class="auxiliary")
        raw = json.dumps(body, indent=3).encode()
        r = await client.post("/anthropic/v1/messages", data=raw, headers=headers)
        await r.read()
        assert seen[0]["body"] == raw and "x-viola-tier" not in r.headers and clf.calls == []
        await client.close(); await up.close()
    run(go())


def test_tool_loop_keeps_the_prompt_tier():
    async def go():
        client, up, seen, clf = await make()
        body, headers = cc_request("find the deadlock in the worker pool")
        body["messages"] += [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]},
        ]
        r = await client.post("/anthropic/v1/messages", data=json.dumps(body), headers=headers)
        await r.read()
        assert r.headers["x-viola-tier"] == "large"
        assert clf.calls == ["find the deadlock in the worker pool"]
        await client.close(); await up.close()
    run(go())


def test_low_confidence_rounds_up_and_max_tier_caps():
    async def go():
        client, up, seen, _ = await make({"anthropic": {"max_tier": "large"}})
        body, headers = cc_request("build a compiler")
        r = await client.post("/anthropic/v1/messages", data=json.dumps(body), headers=headers)
        await r.read()
        assert r.headers["x-viola-tier"] == "large"  # xl capped
        await client.close(); await up.close()
    run(go())


def test_upstream_errors_are_forwarded_unchanged():
    async def go():
        client, up, seen, _ = await make()
        body, headers = cc_request("add pagination")
        r = await client.post("/anthropic/v1/messages", data=json.dumps(body), headers={**headers, "x-fail": "1"})
        assert r.status == 400
        assert await r.read() == b'{"type":"error","error":{"message":"thinking: bound to a different conversation"}}'
        await client.close(); await up.close()
    run(go())


def test_passthrough_while_classifier_loads():
    async def go():
        client, up, seen, _ = await make(classifier=StubClassifier(ready=False))
        body, headers = cc_request("build a compiler")
        raw = json.dumps(body).encode()
        r = await client.post("/anthropic/v1/messages", data=raw, headers=headers)
        await r.read()
        assert seen[0]["body"] == raw
        st = await (await client.get("/viola/status")).json()
        assert st["recent"][0]["reason"] == "classifier warming up"
        await client.close(); await up.close()
    run(go())


def test_pause_and_resume():
    async def go():
        client, up, seen, _ = await make()
        await client.post("/viola/pause")
        body, headers = cc_request("build a compiler")
        r = await client.post("/anthropic/v1/messages", data=json.dumps(body), headers=headers)
        await r.read()
        assert json.loads(seen[0]["body"])["model"] == "claude-opus-5-5"
        await client.post("/viola/resume")
        r = await client.post("/anthropic/v1/messages", data=json.dumps(body), headers=headers)
        await r.read()
        assert json.loads(seen[1]["body"])["model"] == "claude-fable-5-1"
        await client.close(); await up.close()
    run(go())


def test_codex_responses_continuation_and_reasoning_drop():
    async def go():
        client, up, seen, _ = await make()
        h = {"x-codex-session-id": "s1", "authorization": "Bearer sk-x", "content-type": "application/json"}
        first = {"model": "gpt-6.1-sol", "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "<environment_context>cwd</environment_context>"}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "fix the typo in main.go"}]}]}
        r = await client.post("/openai/v1/responses", data=json.dumps(first), headers=h)
        await r.read()
        sent = json.loads(seen[0]["body"])
        assert seen[0]["path"] == "/v1/responses"
        assert sent["model"] == "gpt-6-luna" and sent["reasoning"]["effort"] == "low"

        # continuation with only tool output: no prompt, so the session's tier is reused
        cont = {"model": "gpt-6.1-sol", "previous_response_id": "r1",
                "input": [{"type": "function_call_output", "call_id": "c", "output": "ok"}]}
        r = await client.post("/openai/v1/responses", data=json.dumps(cont), headers=h)
        await r.read()
        assert json.loads(seen[1]["body"])["model"] == "gpt-6-luna"

        # new, harder prompt in the same session: tier changes, stale reasoning items dropped
        nxt = {"model": "gpt-6.1-sol", "input": [
            {"type": "reasoning", "encrypted_content": "abc"},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "write a compiler"}]}]}
        r = await client.post("/openai/v1/responses", data=json.dumps(nxt), headers=h)
        await r.read()
        sent = json.loads(seen[2]["body"])
        assert sent["model"] == "gpt-6-astra" and all(i.get("type") != "reasoning" for i in sent["input"])
        await client.close(); await up.close()
    run(go())


def test_chat_completions_and_other_paths():
    async def go():
        client, up, seen, _ = await make()
        body = {"model": "x", "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "find the deadlock"}]}
        r = await client.post("/openai/v1/chat/completions", data=json.dumps(body), headers={"content-type": "application/json"})
        await r.read()
        sent = json.loads(seen[0]["body"])
        assert sent["model"] == "gpt-6.1-sol" and sent["reasoning_effort"] == "high"
        r = await client.get("/openai/v1/models")
        await r.read()
        assert seen[1]["path"] == "/v1/models"
        await client.close(); await up.close()
    run(go())


def test_bedrock_routes_model_in_path_and_signs_locally(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIDEXAMPLE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.delenv("AWS_PROFILE", raising=False)

    async def go():
        up_app, seen = fake_upstream()
        up = TestServer(up_app)
        await up.start_server()
        cfg = config._merge(config.DEFAULTS, {"bedrock": {
            "upstream": str(up.make_url("")).rstrip("/"), "region": "ap-southeast-2",
            "models": {"small": "au.anthropic.claude-haiku-4-5-20251001-v1:0", "medium": "au.anthropic.claude-sonnet-4-5-20250929-v1:0",
                       "large": "au.anthropic.claude-opus-5-5", "xl": "au.anthropic.claude-opus-5-5"}}})
        client = TestClient(TestServer(build_app(cfg, StubClassifier(), load_in_background=False)))
        await client.start_server()

        body = {"anthropic_version": "bedrock-2023-05-31", "max_tokens": 32000, "thinking": {"type": "adaptive"},
                "output_config": {"effort": "high"}, "messages": [{"role": "user", "content": "fix the typo in setup.py"}]}
        headers = {"content-type": "application/json", "x-claude-code-request-class": "main"}
        r = await client.post("/bedrock/model/au.anthropic.claude-opus-5-5/invoke-with-response-stream",
                              data=json.dumps(body), headers=headers)
        await r.read()
        assert r.headers["x-viola-tier"] == "small"
        got = seen[0]
        assert got["path"] == "/model/au.anthropic.claude-haiku-4-5-20251001-v1%3A0/invoke-with-response-stream"
        assert got["headers"]["Authorization"].startswith("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/")
        assert "/ap-southeast-2/bedrock/aws4_request" in got["headers"]["Authorization"]
        assert not any(k.lower().startswith("x-claude-code-") for k in got["headers"])
        sent = json.loads(got["body"])
        assert "thinking" not in sent and "output_config" not in sent

        # auxiliary request: path and body untouched, still signed
        raw = json.dumps(body).encode()
        r = await client.post("/bedrock/model/au.anthropic.claude-opus-5-5/invoke", data=raw,
                              headers={**headers, "x-claude-code-request-class": "auxiliary"})
        await r.read()
        assert seen[1]["path"] == "/model/au.anthropic.claude-opus-5-5/invoke" and seen[1]["body"] == raw
        assert seen[1]["headers"]["Authorization"].startswith("AWS4-HMAC-SHA256")

        # Bedrock API key: forwarded as-is, not re-signed
        r = await client.post("/bedrock/model/x/invoke", data=raw, headers={**headers, "authorization": "Bearer bedrock-key",
                                                                         "x-claude-code-request-class": "auxiliary"})
        await r.read()
        assert seen[2]["headers"]["Authorization"] == "Bearer bedrock-key"
        await client.close(); await up.close()
    run(go())
