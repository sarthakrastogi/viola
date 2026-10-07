"""The local routing proxy.

    /anthropic/...   -> [anthropic].upstream   (Anthropic Messages: Claude Code, opencode, Zed, ...)
    /bedrock/...     -> Bedrock runtime         (Claude Code on Amazon Bedrock; Viola signs with SigV4)
    /openai/v1/...   -> [openai].upstream      (OpenAI Responses / Chat Completions: Codex, Aider, ...)
    /viola/...       -> local control API      (status, pause, resume, classify)

Only inference requests are rewritten, and only their `model` (plus the few parameters the new
model needs, see adapters.py). Everything else, including streams, auth headers and error
bodies, is passed through byte for byte. Prompt text is classified in-process and never logged
to disk or sent anywhere except, unchanged, to the upstream the client was already using.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor

from aiohttp import ClientSession, ClientTimeout, web
from multidict import CIMultiDict

from yarl import URL

from . import __version__, adapters, bedrock, extract
from .config import TIERS, tier_target

log = logging.getLogger("viola")

HOP_BY_HOP = {
    "host", "content-length", "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "trailers", "transfer-encoding", "upgrade",
}
ROUTER = web.AppKey("router", "Router")
SESSION = web.AppKey("session", ClientSession)
SIGNER = web.AppKey("signer", bedrock.Signer)
SESSION_HEADERS = ("x-claude-code-session-id", "x-codex-session-id", "session_id", "x-session-id")


class Router:
    def __init__(self, cfg: dict, classifier):
        self.cfg = cfg
        self.classifier = classifier
        self.paused = False
        self.load_error: str | None = None
        self.started = time.time()
        self.recent: deque = deque(maxlen=50)
        self.counts: Counter = Counter()
        self.session_tier: dict[str, str] = {}  # last tier per conversation, for prompt-less continuations
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="viola-classify")

    # ------------------------------------------------------------------ decisions
    async def _classify(self, text: str):
        return await asyncio.get_running_loop().run_in_executor(self._executor, self.classifier.classify, text)

    def _cap(self, section: str, tier: str) -> str:
        cap = self.cfg[section]["max_tier"]
        return tier if TIERS.index(tier) <= TIERS.index(cap) else cap

    @staticmethod
    def _session_key(headers, body: dict) -> str | None:
        for h in SESSION_HEADERS:
            if headers.get(h):
                return headers[h]
        return body.get("prompt_cache_key") or None

    def _record(self, protocol: str, tier: str | None, model: str | None, reason: str, decision=None, text: str = ""):
        self.counts[tier or "passthrough"] += 1
        self.recent.append({
            "time": time.strftime("%H:%M:%S"), "protocol": protocol, "tier": tier, "model": model,
            "reason": reason,
            "confidence": round(decision.confidence, 2) if decision else None,
            "classify_ms": round(decision.ms) if decision else None,
            "preview": " ".join(text.split())[:70] if text else None,  # memory only, never written to disk
        })

    async def route(self, protocol: str, path: str, headers, body: dict, raw_len: int, client_model: str | None = None):
        """Return (new_body, tier, model) or (None, None, None) to pass the request through untouched.

        protocol is "anthropic", "bedrock" (model in the URL: client_model) or "openai".
        """
        section = protocol
        if self.paused:
            return self._skip(protocol, "paused")

        if protocol in ("anthropic", "bedrock"):
            rc = headers.get("x-claude-code-request-class")
            if rc and rc not in self.cfg[section]["route_request_classes"]:
                return self._skip(protocol, f"{rc} request")
            asked = client_model or str(body.get("model", ""))
            if not rc and any(s in asked for s in self.cfg[section]["passthrough_model_substrings"]):
                return self._skip(protocol, "client chose a small model")
            text = extract.anthropic_prompt(body)
        elif path.endswith("responses"):
            text = extract.responses_prompt(body)
        else:
            text = extract.chat_prompt(body)

        key = self._session_key(headers, body)
        previous = self.session_tier.get(key) if key else None
        decision = None
        if text is None:
            if not previous:
                return self._skip(protocol, "no human prompt in request")
            tier, reason = previous, "continuation"
        elif not self.classifier.ready:
            return self._skip(protocol, "classifier warming up" if not self.load_error else "classifier failed to load")
        else:
            decision = await self._classify(text)
            tier, reason = decision.tier, "cached" if decision.cached else "classified"
            if decision.tier != decision.raw_tier:
                reason += f" ({decision.raw_tier}, low confidence: rounded up)"

        tier = self._cap(section, tier)
        model, effort = tier_target(self.cfg, section, tier)
        if protocol in ("anthropic", "bedrock") and adapters.is_legacy_anthropic(model) and raw_len / 4 > 0.85 * adapters.LEGACY_CONTEXT_TOKENS:
            tier = "medium"  # conversation too long for a 200K-context model
            model, effort = tier_target(self.cfg, section, tier)
            reason += "; too long for small tier"
        if key:
            self.session_tier[key] = tier
            if len(self.session_tier) > 1000:
                self.session_tier.pop(next(iter(self.session_tier)))

        if protocol == "anthropic":
            body = adapters.anthropic(body, model, effort)
        elif protocol == "bedrock":
            body = adapters.anthropic_params(body, model, effort)
        elif path.endswith("responses"):
            switched = previous is not None and previous != tier
            body = adapters.responses(body, model, effort, switched and self.cfg["openai"]["drop_reasoning_on_switch"])
        else:
            body = adapters.chat(body, model, effort)
        self._record(protocol, tier, model, reason, decision, text or "")
        return body, tier, model

    def _skip(self, protocol: str, reason: str):
        self._record(protocol, None, None, reason)
        return None, None, None

    def status(self) -> dict:
        return {
            "version": __version__,
            "ready": self.classifier.ready,
            "load_error": self.load_error,
            "paused": self.paused,
            "uptime_s": round(time.time() - self.started),
            "counts": dict(self.counts),
            "recent": list(self.recent)[::-1],
        }


# ---------------------------------------------------------------------- HTTP
def _upstream_url(cfg: dict, protocol: str, tail: str, query: str) -> str:
    base = cfg[protocol]["upstream"].rstrip("/")
    return f"{base}/{tail}" + (f"?{query}" if query else "")


def _is_inference(protocol: str, method: str, tail: str) -> bool:
    if method != "POST":
        return False
    if protocol == "anthropic":
        return tail == "v1/messages"
    return tail in ("responses", "chat/completions")


async def _forward(request: web.Request, protocol: str, tail: str) -> web.StreamResponse:
    router: Router = request.app[ROUTER]
    cfg = router.cfg
    raw = await request.read()
    body_bytes, tier, model = raw, None, None

    invoke = bedrock.parse_invoke(tail) if protocol == "bedrock" and request.method == "POST" else None
    if invoke or _is_inference(protocol, request.method, tail):
        try:
            body = json.loads(raw)
        except ValueError:
            body = None
        if isinstance(body, dict):
            try:
                new_body, tier, model = await router.route(protocol, tail, request.headers, body, len(raw),
                                                           client_model=invoke[0] if invoke else None)
            except Exception:  # a routing bug must never break the user's request
                log.exception("routing failed; passing request through")
                new_body = None
            if new_body is not None:
                body_bytes = json.dumps(new_body, ensure_ascii=False, separators=(",", ":")).encode()
                if invoke:
                    tail = bedrock.with_model(tail, model)

    headers = CIMultiDict((k, v) for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP)
    session: ClientSession = request.app[SESSION]
    if protocol == "bedrock":
        signer: bedrock.Signer = request.app[SIGNER]
        url = signer.url(tail + (f"?{request.query_string}" if request.query_string else ""))
        try:
            headers = await signer.headers(request.method, url, dict(headers), body_bytes)
        except Exception as e:
            log.error("bedrock signing failed: %s", e)
            return web.json_response({"message": f"viola: could not sign the Bedrock request: {e}"}, status=403)
        url = URL(url, encoded=True)  # sent exactly as signed
    else:
        url = _upstream_url(cfg, protocol, tail, request.query_string)
    async with session.request(request.method, url, headers=headers, data=body_bytes, allow_redirects=False) as up:
        resp = web.StreamResponse(status=up.status, reason=up.reason)
        for k, v in up.headers.items():
            if k.lower() not in HOP_BY_HOP:
                resp.headers.add(k, v)
        if tier:
            resp.headers["x-viola-tier"] = tier
            resp.headers["x-viola-model"] = model
        await resp.prepare(request)
        async for chunk in up.content.iter_any():
            await resp.write(chunk)
        await resp.write_eof()
        return resp


async def anthropic_handler(request: web.Request):
    return await _forward(request, "anthropic", request.match_info["tail"])


async def bedrock_handler(request: web.Request):
    # raw path: the model ID stays percent-encoded exactly as Claude Code sent it
    tail = request.raw_path.split("?", 1)[0].split("/bedrock/", 1)[1]
    return await _forward(request, "bedrock", tail)


async def openai_handler(request: web.Request):
    return await _forward(request, "openai", request.match_info["tail"])


async def status_handler(request: web.Request):
    return web.json_response(request.app[ROUTER].status())


async def pause_handler(request: web.Request):
    request.app[ROUTER].paused = True
    return web.json_response({"paused": True})


async def resume_handler(request: web.Request):
    request.app[ROUTER].paused = False
    return web.json_response({"paused": False})


async def classify_handler(request: web.Request):
    router: Router = request.app[ROUTER]
    if not router.classifier.ready:
        return web.json_response({"error": router.load_error or "classifier warming up"}, status=503)
    text = (await request.json()).get("text", "")
    d = await router._classify(text)
    out = {"tier": d.tier, "raw_tier": d.raw_tier, "confidence": round(d.confidence, 3), "ms": round(d.ms, 1)}
    for section in ("anthropic", "bedrock", "openai"):
        out[f"{section}_model"] = tier_target(router.cfg, section, router._cap(section, d.tier))[0]
    return web.json_response(out)


def build_app(cfg: dict, classifier, load_in_background: bool = True) -> web.Application:
    app = web.Application(client_max_size=256 * 1024 * 1024)
    router = Router(cfg, classifier)
    app[ROUTER] = router
    app[SIGNER] = bedrock.Signer(cfg)

    async def on_startup(app):
        app[SESSION] = ClientSession(
            auto_decompress=False,  # pass compressed bodies through untouched
            trust_env=True,         # honour HTTPS_PROXY etc. on corporate networks
            timeout=ClientTimeout(total=None, sock_connect=30, sock_read=None),
        )
        if load_in_background and not classifier.ready:
            def load():
                try:
                    t0 = time.time()
                    classifier.load()
                    log.info("classifier ready in %.1fs", time.time() - t0)
                except Exception as e:  # keep proxying (pass-through) even if the model can't load
                    router.load_error = f"{type(e).__name__}: {e}"
                    log.error("classifier failed to load: %s", router.load_error)
            threading.Thread(target=load, daemon=True, name="viola-load").start()

    async def on_cleanup(app):
        await app[SESSION].close()

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_get("/viola/status", status_handler)
    app.router.add_post("/viola/pause", pause_handler)
    app.router.add_post("/viola/resume", resume_handler)
    app.router.add_post("/viola/classify", classify_handler)
    app.router.add_route("*", "/anthropic/{tail:.*}", anthropic_handler)
    app.router.add_route("*", "/bedrock/{tail:.*}", bedrock_handler)
    app.router.add_route("*", "/openai/v1/{tail:.*}", openai_handler)
    return app


def serve(cfg: dict) -> None:
    from .classifier import Classifier

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if cfg["host"] not in ("127.0.0.1", "localhost", "::1"):
        log.warning("listening on %s: anyone who can reach this port can use the credentials sent through it", cfg["host"])
    app = build_app(cfg, Classifier(cfg))
    web.run_app(app, host=cfg["host"], port=int(cfg["port"]), print=lambda *_: log.info(
        "viola %s listening on http://%s:%s", __version__, cfg["host"], cfg["port"]), access_log=None)
