"""viola command line."""
from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import __version__, config
from .integrations import base_url, claude_code, codex

TOOLS = {"claude-code": claude_code, "codex": codex}

# Talk to the local daemon directly, never through HTTP(S)_PROXY.
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _url(cfg: dict, path: str) -> str:
    host = "127.0.0.1" if cfg["host"] in ("0.0.0.0", "::") else cfg["host"]
    return f"http://{host}:{cfg['port']}{path}"


def _call(cfg: dict, path: str, payload: dict | None = None, timeout: float = 3.0):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(_url(cfg, path), data=data, method="POST" if payload is not None else "GET",
                                 headers={"content-type": "application/json"})
    with _opener.open(req, timeout=timeout) as r:
        return json.loads(r.read())


def _running(cfg: dict) -> dict | None:
    try:
        return _call(cfg, "/viola/status", timeout=1.0)
    except (urllib.error.URLError, OSError, ValueError):
        return None


def _pidfile() -> Path:
    return config.data_dir() / "viola.pid"


def _logfile() -> Path:
    return config.data_dir() / "viola.log"


# ---------------------------------------------------------------------- commands
def cmd_setup(cfg, args):
    path = config.write_default()
    print(f"config: {path}")
    from .classifier import download

    c = cfg["classifier"]
    print(f"downloading {c['model']} (~2 GB, one time) ...")
    print(f"model: {download(c['model'], c['revision'])}")
    print("done. Next: `viola install claude-code` and/or `viola install codex`, then `viola start`.")


def cmd_serve(cfg, args):
    from .proxy import serve

    serve(cfg)


def cmd_start(cfg, args):
    if _running(cfg):
        if not args.quiet:
            print(f"viola already running on {_url(cfg, '')}")
        return
    config.data_dir().mkdir(parents=True, exist_ok=True)
    log = open(_logfile(), "ab")
    kwargs = {"creationflags": 0x00000008 | 0x00000200} if os.name == "nt" else {"start_new_session": True}
    proc = subprocess.Popen([sys.executable, "-m", "viola", "serve"], stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, **kwargs)
    _pidfile().write_text(str(proc.pid))
    for _ in range(50):  # the port opens in ~1s; the model keeps loading in the background
        if _running(cfg):
            break
        if proc.poll() is not None:
            sys.exit(f"viola exited during startup; see {_logfile()}")
        time.sleep(0.2)
    if not args.quiet:
        print(f"viola started (pid {proc.pid}) on {_url(cfg, '')}; the classifier loads in ~20s, "
              f"until then requests pass through unrouted. Log: {_logfile()}")


def cmd_stop(cfg, args):
    pid = _pidfile()
    if not pid.exists():
        print("viola is not running (no pid file)")
        return
    try:
        os.kill(int(pid.read_text()), signal.SIGTERM)
        print("viola stopped")
    except ProcessLookupError:
        print("viola was not running")
    pid.unlink(missing_ok=True)


def cmd_status(cfg, args):
    st = _running(cfg)
    if not st:
        print(f"viola is not running on {_url(cfg, '')}. Start it with `viola start`.")
        sys.exit(1)
    if args.json:
        print(json.dumps(st, indent=2))
        return
    state = "paused" if st["paused"] else ("ready" if st["ready"] else (f"ERROR: {st['load_error']}" if st["load_error"] else "loading model"))
    print(f"viola {st['version']} on {_url(cfg, '')}: {state}, up {st['uptime_s']}s")
    if st["counts"]:
        print("routed: " + ", ".join(f"{k}={v}" for k, v in sorted(st["counts"].items())))
    for r in st["recent"][: args.n]:
        tier = f"{r['tier']:<6} -> {r['model']}" if r["tier"] else "pass-through"
        conf = f" conf={r['confidence']}" if r["confidence"] is not None else ""
        ms = f" {r['classify_ms']}ms" if r.get("classify_ms") else ""
        print(f"  {r['time']} {r['protocol']:<9} {tier:<34} {r['reason']}{conf}{ms}")
        if r.get("preview") and r["reason"].startswith("classified"):
            print(f"           \"{r['preview']}\"")


def cmd_classify(cfg, args):
    text = " ".join(args.text) if args.text else sys.stdin.read()
    if _running(cfg):
        try:
            print(json.dumps(_call(cfg, "/viola/classify", {"text": text}, timeout=60), indent=2))
            return
        except urllib.error.HTTPError as e:
            if e.code != 503:
                raise
    from .classifier import Classifier

    clf = Classifier(cfg)
    clf.load()
    d = clf.classify(text)
    print(json.dumps({"tier": d.tier, "raw_tier": d.raw_tier, "confidence": round(d.confidence, 3),
                      "anthropic_model": config.tier_target(cfg, "anthropic", d.tier)[0],
                      "openai_model": config.tier_target(cfg, "openai", d.tier)[0]}, indent=2))


def cmd_pause(cfg, args):
    _call(cfg, "/viola/pause", {})
    print("viola paused: requests pass through with the model your tool chose")


def cmd_resume(cfg, args):
    _call(cfg, "/viola/resume", {})
    print("viola resumed")


def cmd_install(cfg, args):
    for note in TOOLS[args.tool].install(config.load()):
        print(note)
    print("Restart the tool to pick up the change. Make sure the daemon runs: `viola start` "
          "(or `viola autostart` to start it at login).")


def cmd_uninstall(cfg, args):
    for note in TOOLS[args.tool].uninstall(cfg):
        print(note)


def cmd_env(cfg, args):
    a, o, b = base_url(cfg, "anthropic"), base_url(cfg, "openai"), base_url(cfg, "bedrock")
    print(f"""Anthropic Messages base URL: {a}
Bedrock Invoke base URL:     {b}
OpenAI-compatible base URL:  {o}

  Claude Code   viola install claude-code      (detects Anthropic API vs Amazon Bedrock)
                  manual: ANTHROPIC_BASE_URL={a} CLAUDE_CODE_GATEWAY_HINT_HEADERS=1
                  Bedrock: ANTHROPIC_BEDROCK_BASE_URL={b} CLAUDE_CODE_SKIP_BEDROCK_AUTH=1 CLAUDE_CODE_GATEWAY_HINT_HEADERS=1
  Codex         viola install codex
  opencode      provider.<id>.options.baseURL = "{a}"  (npm @ai-sdk/anthropic)  or "{o}" (@ai-sdk/openai)
  Aider         OPENAI_API_BASE={o} aider --model openai/<any>
  Continue      apiBase: {o}   (provider: openai)
  Cline / Roo   "OpenAI Compatible" provider, Base URL {o}
  Goose         OPENAI_HOST=http://127.0.0.1:{cfg['port']} OPENAI_BASE_PATH=openai/v1/chat/completions
  Zed           language_models.anthropic.api_url = "{a}"
  anything      point its Anthropic or OpenAI base URL at the matching URL above""")


def cmd_autostart(cfg, args):
    exe = [sys.executable, "-m", "viola", "serve"]
    if platform.system() == "Linux":
        unit = Path.home() / ".config/systemd/user/viola.service"
        if args.disable:
            subprocess.run(["systemctl", "--user", "disable", "--now", "viola"], check=False)
            unit.unlink(missing_ok=True)
            print("autostart disabled")
            return
        unit.parent.mkdir(parents=True, exist_ok=True)
        unit.write_text(f"""[Unit]
Description=Viola on-device model router

[Service]
ExecStart={' '.join(exe)}
Restart=on-failure

[Install]
WantedBy=default.target
""")
        cmd_stop(cfg, args) if _pidfile().exists() else None
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        subprocess.run(["systemctl", "--user", "enable", "--now", "viola"], check=True)
        print(f"autostart enabled ({unit})")
    elif platform.system() == "Darwin":
        plist = Path.home() / "Library/LaunchAgents/dev.viola.router.plist"
        if args.disable:
            subprocess.run(["launchctl", "unload", str(plist)], check=False)
            plist.unlink(missing_ok=True)
            print("autostart disabled")
            return
        plist.parent.mkdir(parents=True, exist_ok=True)
        args_xml = "".join(f"<string>{a}</string>" for a in exe)
        plist.write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>Label</key><string>dev.viola.router</string>
<key>ProgramArguments</key><array>{args_xml}</array>
<key>RunAtLoad</key><true/><key>KeepAlive</key><true/>
<key>StandardOutPath</key><string>{_logfile()}</string><key>StandardErrorPath</key><string>{_logfile()}</string>
</dict></plist>
""")
        subprocess.run(["launchctl", "load", str(plist)], check=True)
        print(f"autostart enabled ({plist})")
    else:
        sys.exit("autostart supports Linux (systemd) and macOS (launchd); on Windows add `viola start` to Startup.")


def cmd_doctor(cfg, args):
    ok = True
    def check(cond, good, bad):
        nonlocal ok
        print(("  ok   " if cond else "  FAIL ") + (good if cond else bad))
        ok &= bool(cond)

    print(f"viola {__version__}, python {platform.python_version()}, {platform.system()}")
    check(config.config_path().exists(), f"config {config.config_path()}", "no config yet (run `viola setup`)")
    check((config.model_dir() / "config.json").exists(), f"model {config.model_dir()}", "model not downloaded (run `viola setup`)")
    try:
        import gliner2  # noqa: F401
        import torch
        check(True, f"torch {torch.__version__}, gliner2 installed", "")
    except ImportError as e:
        check(False, "", f"classifier dependencies missing ({e.name}); reinstall with install.sh")
    st = _running(cfg)
    check(st, f"daemon running on {_url(cfg, '')} ({'ready' if st and st['ready'] else 'loading'})", "daemon not running (`viola start`)")
    if st and st.get("load_error"):
        check(False, "", f"classifier failed to load: {st['load_error']}")
    sys.exit(0 if ok else 1)


def main(argv=None):
    p = argparse.ArgumentParser(prog="viola", description="On-device model routing for coding agents.")
    p.add_argument("--version", action="version", version=f"viola {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup", help="write the default config and download the classifier model")
    sub.add_parser("serve", help="run the router in the foreground")
    s = sub.add_parser("start", help="run the router in the background")
    s.add_argument("--quiet", action="store_true")
    s = sub.add_parser("ensure-running", help="start the router if it isn't running (for hooks)")
    s.add_argument("--quiet", action="store_true")
    sub.add_parser("stop", help="stop the background router")
    s = sub.add_parser("status", help="show recent routing decisions")
    s.add_argument("-n", type=int, default=10)
    s.add_argument("--json", action="store_true")
    s = sub.add_parser("classify", help="classify a prompt (argument or stdin)")
    s.add_argument("text", nargs="*")
    sub.add_parser("pause", help="pass every request through unrouted")
    sub.add_parser("resume", help="resume routing")
    for name in ("install", "uninstall"):
        s = sub.add_parser(name, help=f"{name} the integration for a tool")
        s.add_argument("tool", choices=sorted(TOOLS))
    sub.add_parser("env", help="base URLs and settings for other tools")
    s = sub.add_parser("autostart", help="start the router at login (systemd / launchd)")
    s.add_argument("--disable", action="store_true")
    sub.add_parser("doctor", help="check the installation")

    args = p.parse_args(argv)
    cfg = config.load()
    handler = {"ensure-running": cmd_start}.get(args.cmd) or globals()[f"cmd_{args.cmd.replace('-', '_')}"]
    handler(cfg, args)


if __name__ == "__main__":
    main()
