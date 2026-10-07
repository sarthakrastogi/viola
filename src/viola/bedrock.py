"""Amazon Bedrock (Invoke API) support.

Claude Code puts the model ID in the URL path and SigV4-signs path and body, so a router can't
change the model of a signed request. With CLAUDE_CODE_SKIP_BEDROCK_AUTH=1 Claude Code sends
the request unsigned to Viola, which picks the model and then signs locally with your AWS
credentials (botocore default chain / profile). Credentials never leave the machine except as
the normal request signature to AWS.
"""
from __future__ import annotations

import asyncio
import re
from urllib.parse import quote, unquote

from .config import bedrock_region

INVOKE = re.compile(r"^model/([^/]+)/(invoke|invoke-with-response-stream)$")
MODEL_PATH = re.compile(r"^model/([^/]+)/(.+)$")
# Headers that must not reach AWS: our own hints, and anything the new signature replaces.
DROP = ("x-amz-date", "x-amz-security-token", "x-amz-content-sha256")


class Signer:
    def __init__(self, cfg: dict):
        b = cfg["bedrock"]
        self.profile = b["profile"] or None
        self.region = bedrock_region(cfg)
        self.sign = bool(b["sign"])
        self.upstream = b["upstream"].rstrip("/")
        self._session = None

    def _botocore(self):
        if self._session is None:
            import botocore.session

            s = botocore.session.Session(profile=self.profile)
            self.region = self.region or s.get_config_variable("region") or "us-east-1"
            self._session = s
        return self._session

    def url(self, tail: str) -> str:
        """`tail` is the raw (still percent-encoded) path after /bedrock/."""
        if self.upstream:
            return f"{self.upstream}/{tail}"
        self._botocore()
        host = "bedrock" if tail.startswith(("inference-profiles", "foundation-models")) else "bedrock-runtime"
        return f"https://{host}.{self.region}.amazonaws.com/{tail}"

    def _sign_sync(self, method: str, url: str, headers: dict, body: bytes) -> dict:
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest

        creds = self._botocore().get_credentials()
        if creds is None:
            raise PermissionError("no AWS credentials found (set AWS_PROFILE or run `aws sso login`)")
        req = AWSRequest(method=method, url=url, data=body, headers=headers)
        SigV4Auth(creds.get_frozen_credentials(), "bedrock", self.region).add_auth(req)
        return dict(req.headers.items())

    async def headers(self, method: str, url: str, headers: dict, body: bytes) -> dict:
        headers = {k: v for k, v in headers.items() if not k.lower().startswith("x-claude-code-") and k.lower() not in DROP}
        auth = next((v for k, v in headers.items() if k.lower() == "authorization"), "")
        if not self.sign or auth.lower().startswith("bearer "):
            return headers  # Bedrock API key (AWS_BEARER_TOKEN_BEDROCK) or an upstream gateway
        headers = {k: v for k, v in headers.items() if k.lower() != "authorization"}
        # credential refresh can do network I/O (STS/SSO), so keep it off the event loop
        return await asyncio.get_running_loop().run_in_executor(None, self._sign_sync, method, url, headers, body)


def parse_invoke(tail: str) -> tuple[str, str] | None:
    """(model_id, action) for an invoke path, model ID decoded."""
    m = INVOKE.match(tail)
    return (unquote(m.group(1)), m.group(2)) if m else None


def with_model(tail: str, model: str) -> str:
    return MODEL_PATH.sub(lambda m: f"model/{quote(model, safe='')}/{m.group(2)}", tail, count=1)
