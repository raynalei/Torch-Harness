"""Drive the real TUI/launcher/gateway/AgentLoop under a pseudo-terminal.

All key schedules are observation-driven. Gate files hold the scripted LLM
at a known point until a scenario has asserted what the terminal displayed.
"""

from __future__ import annotations

import argparse
import codecs
import errno
import fcntl
import hashlib
import importlib.util
import json
import os
import pty
import re
import select
import shlex
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import termios
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pyte

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DEFAULT_TIMEOUT = 20.0
SCENARIO_SPEC = importlib.util.spec_from_file_location("tui_e2e_scenarios", HERE / "scenarios.py")
SCENARIO_MODULE = importlib.util.module_from_spec(SCENARIO_SPEC)
SCENARIO_SPEC.loader.exec_module(SCENARIO_MODULE)
SCENARIOS = SCENARIO_MODULE.SCENARIOS
# Remove complete CSI, OSC, DCS/APC and ordinary two-byte escape sequences.
# Called on the accumulated decoded capture, so chunk boundaries are harmless.
ESCAPES = re.compile(
    r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)"
    r"|[P_^][\s\S]*?(?:\x07|\x1b\\)|[@-_])"
)


ENTER_ALT = "\x1b[?1049h"
LEAVE_ALT = "\x1b[?1049l"
# The offer a single Ctrl+C makes at an idle prompt, defined with the scenarios.
QUIT_HINT = SCENARIO_MODULE.QUIT_HINT


def strip_ansi(text: str) -> str:
    return ESCAPES.sub("", text).replace("\r", "")


def source_state(repo: Path) -> dict:
    executable = shutil.which("git")
    if not executable:
        raise RuntimeError("git is required to record the tested source revision")

    def git(*args):
        return subprocess.check_output([executable, "-C", str(repo), *args], text=True).strip()

    digest = hashlib.sha256()
    for path in sorted((repo / "ui-tui" / "src").rglob("*.ts")):
        digest.update(str(path.relative_to(repo)).encode())
        digest.update(path.read_bytes())
    return {
        "commit": git("rev-parse", "HEAD"),
        "status": git("status", "--short"),
        "tui_src_sha256": digest.hexdigest(),
    }


# `(12s · esc to interrupt)`, the status line's trailing group.
_WORKING_RE = re.compile(r"\(\d+[hms].*to interrupt\)")


class Fixture:
    def __init__(self, root: Path, repo: Path = REPO, timeout: float = DEFAULT_TIMEOUT):
        self.root, self.repo, self.timeout = root, repo.resolve(), timeout
        self.home = root / "home"
        self.workspace = root / "workspace"
        self.workspace.mkdir(parents=True)
        self.home.mkdir(mode=0o700)
        self.journal = root / "journal.jsonl"
        self.config = self.home / ".opendde_harness" / "config.json"
        self.config.parent.mkdir(mode=0o700)
        self.task_root = self.config.parent / "protein_design"
        self.task_root.mkdir(mode=0o700)
        self.script = root / "script.json"
        self.token_cache = root / "tokenizer-cache"
        self.token_cache.mkdir()
        # tiktoken otherwise downloads its vocabulary on first use. Copy the
        # public cached asset, never credentials, into this isolated runtime.
        cached = (
            Path(
                os.environ.get("TIKTOKEN_CACHE_DIR")
                or os.environ.get("DATA_GYM_CACHE_DIR")
                or str(Path(tempfile.gettempdir()) / "data-gym-cache")
            )
            / "9b5ad71b2ce5302211f9c61530b329a4922fc6a4"
        )
        if not cached.is_file():
            raise RuntimeError("cl100k_base is not cached; provide its existing cache via TIKTOKEN_CACHE_DIR")
        shutil.copyfile(cached, self.token_cache / cached.name)
        self.starts = 0
        self.source_before = source_state(self.repo)
        if not (self.repo / "ui-tui/node_modules/.bin/tsx").is_file():
            raise RuntimeError("Install frontend dependencies first: cd ui-tui && npm ci")
        if not shutil.which("node") or not shutil.which("npx"):
            raise RuntimeError("Node >=22.19 and npx must already be installed; e2e never downloads a runtime")
        # All real paths, all temporary. No production HOME/config is inherited.
        payload = {
            "agents": {"defaults": {"workspace": str(self.workspace), "model": "scripted/e2e", "provider": "scripted"}},
            "providers": {
                "scripted": {
                    "apiKey": "e2e-not-a-network-credential",
                    "models": ["scripted/e2e"],
                    "modelOverlay": {"e2e": {"contextWindowTokens": 1_000_000, "maxOutputTokens": 4096}},
                }
            },
            "memory": {"backend": None},
            "plugins": {"disabled": ["long-term-memory", "protein-design"]},
            "skillForge": {"enabled": False},
            "runtime": {"checkpoint": {"policy": "never"}},
            "tracing": {"enabled": False},
            "context": {"serverCompactRatio": 0},
            "tools": {"restrictToWorkspace": True, "mcpServers": {}},
        }
        self.config.write_text(json.dumps(payload, indent=2))
        self.config.chmod(0o600)
        sample = self.workspace / "fixture.txt"
        sample.write_text("E2E_REAL_TOOL_PAYLOAD\nThis text came from the real read_file tool.\n")
        # The approval scenarios may delete only this disposable fixture file.
        self.approval_target = self.workspace / "approval-target.txt"
        self.approval_target.write_text("E2E_APPROVAL_TARGET\n")
        substitutions = {
            "fixture_file": str(sample),
            "hold_gate": str(root / "release-hold"),
            "slow_gate": str(root / "release-slow"),
            "approval_command": shlex.join(["rm", "--", str(self.approval_target)]),
        }

        def substitute(value):
            if isinstance(value, str):
                for key, replacement in substitutions.items():
                    value = value.replace("{{" + key + "}}", replacement)
            elif isinstance(value, list):
                value = [substitute(v) for v in value]
            elif isinstance(value, dict):
                value = {k: substitute(v) for k, v in value.items()}
            return value

        self.script.write_text(json.dumps(substitute(json.loads((HERE / "script.json").read_text())), indent=2))

    def design_task(self, task_id: str, *, log: str | None = None, mtime: float | None = None, **fields) -> Path:
        """Write a historical task snapshot for the TUI monitor.

        `mtime` pins the snapshot's timestamp. That is what the monitor records
        as the task's last observation and orders its rows on, and it also makes
        a rewrite a change the unchanged-file check has to look at rather than
        skip.
        """
        payload = {
            "task_id": task_id,
            "status": "running",
            "cycle": 1,
            "total_cycles": 8,
            "phase": "design_cycle",
            "selected_skill": "binder_design",
            "compute_url": "http://e2e.invalid:9000",
            "compute_worker_id": "e2e-worker",
            "best_candidate": {"candidate_id": "cand-1", "sequence": "MKTAYIAKQR", "objective": 0.8125},
            **fields,
        }
        directory = self.task_root / task_id
        directory.mkdir(exist_ok=True)
        snapshot = directory / "snapshot.json"
        snapshot.write_text(json.dumps(payload))
        if log is not None:
            (directory / "worker.log").write_text(log)
        if mtime is not None:
            os.utime(snapshot, (mtime, mtime))
        return directory

    def records(self) -> list[dict]:
        if not self.journal.exists():
            return []
        records = []
        # A concurrently appended final line may still be incomplete.
        for line in self.journal.read_text().splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records

    @contextmanager
    def launch(self, resume: str | None = None, *, fullscreen: bool = False) -> Iterator[PtySession]:
        self.starts += 1
        # tsx puts a Unix socket below TMPDIR; artifact paths can exceed its
        # 108-byte address limit. Clean this owned directory after PTY teardown.
        with tempfile.TemporaryDirectory(prefix="e2e-", dir="/tmp") as tmpdir:
            with PtySession(self, self.starts, resume, tmpdir=tmpdir, fullscreen=fullscreen) as ui:
                yield ui

    def verify(self):
        bad = [r for r in self.records() if r["kind"] in {"network_blocked", "script_mismatch"}]
        if bad:
            raise AssertionError(f"Test isolation/script failure: {bad}")
        if not any(r["kind"] == "injection_ready" for r in self.records()):
            raise AssertionError("The scripted-provider injection never initialized")


class PtySession:
    def __init__(self, fixture: Fixture, index: int, resume: str | None, *, tmpdir: str, fullscreen: bool = False):
        self.fixture, self.index = fixture, index
        self.tmpdir = tmpdir
        self.fullscreen = fullscreen
        self.raw = bytearray()
        self.decoded = ""
        self.decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self.screen = pyte.HistoryScreen(120, 40, history=2000)
        self.stream = pyte.Stream(self.screen)
        self.exit_code = None
        self.actions: list[dict] = []
        self.start = time.monotonic()
        self.record_start = len(fixture.records())
        self.command = [sys.executable, "-m", "opendde_harness", "tui", "--dev"]
        env = {
            "HOME": str(fixture.home),
            "USER": "tui-e2e",
            "LOGNAME": "tui-e2e",
            "SHELL": "/bin/sh",
            "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
            "LANG": "C.UTF-8",
            "TERM": "xterm-256color",
            "TMPDIR": self.tmpdir,
            "TIKTOKEN_CACHE_DIR": str(fixture.token_cache),
            "XDG_CACHE_HOME": str(fixture.home / ".cache"),
            "PYTHONPATH": os.pathsep.join([str(HERE), str(fixture.repo)]),
            "PYTHONUNBUFFERED": "1",
            "OPENDDE_HARNESS_HOME": str(fixture.config.parent),
            "OPENDDE_HARNESS_E2E_CONFIG": str(fixture.config),
            "OPENDDE_HARNESS_PROTEIN_DESIGN_ROOT": str(fixture.task_root),
            "OPENDDE_HARNESS_E2E_SCRIPT": str(fixture.script),
            "OPENDDE_HARNESS_E2E_JOURNAL": str(fixture.journal),
            "OPENDDE_HARNESS_TRACING": "0",
            "OPENDDE_HARNESS_NO_NODE_INSTALL": "1",
            "OPENDDE_HARNESS_TUI_COLOR": "none",
            "OPENDDE_HARNESS_TUI_THEME": "dark",
            # The product defaults to the alternate screen. Scenarios assert on
            # the accumulated main-buffer transcript, which the alternate
            # screen does not produce, so they pin the main screen and the one
            # fullscreen scenario asks for the default explicitly.
            "OPENDDE_HARNESS_TUI_FULLSCREEN": "1" if fullscreen else "0",
            "NO_COLOR": "1",
            "npm_config_offline": "true",
            "npm_config_update_notifier": "false",
        }
        if resume:
            env["OPENDDE_HARNESS_TUI_RESUME"] = resume
        self.pid, self.fd = pty.fork()
        if self.pid == 0:
            os.chdir(fixture.workspace)
            os.execvpe(self.command[0], self.command, env)
        fcntl.ioctl(self.fd, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 120, 0, 0))
        self.initial_flags = termios.tcgetattr(self.fd)[3]

    @property
    def text(self):
        return strip_ansi(self.decoded)

    @property
    def screen_text(self):
        return "\n".join(self.screen.display)

    @property
    def raw_text(self):
        return self.raw.decode("utf-8", "replace")

    @property
    def alt_screen_entries(self):
        return self.raw_text.count(ENTER_ALT)

    def main_buffer_after_alt_exit(self):
        """What reached the main buffer after the alternate screen was left.

        Boot resets terminal modes and the exit reset does the same, and both
        emit a bare 1049l of their own, so the window starts at the last
        alternate-screen entry rather than at the last exit.
        """
        raw = self.raw_text
        if ENTER_ALT not in raw:
            return ""
        tail = raw[raw.rindex(ENTER_ALT) :]
        if LEAVE_ALT not in tail:
            return ""
        return strip_ansi(tail.split(LEAVE_ALT, 1)[1])

    @property
    def working(self):
        # The status row above the editor is the only thing that says a turn is
        # running: a face, the turn's activity verb, the clock and the way out.
        # Matched on the clock-plus-hint group, which no transcript text has and
        # which does not depend on which verb or indicator style was drawn.
        return any(_WORKING_RE.search(line) for line in self.screen.display)

    def records(self):
        return self.fixture.records()[self.record_start :]

    def pump(self, timeout=0.05):
        data = b""
        readable, _, _ = select.select([self.fd], [], [], max(0, timeout))
        if readable:
            try:
                data = os.read(self.fd, 65536)
            except OSError as exc:
                if exc.errno != errno.EIO:
                    raise
                data = b""
            self.raw.extend(data)
            decoded = self.decoder.decode(data)
            self.decoded += decoded
            self.stream.feed(decoded)
        if self.exit_code is None:
            pid, status = os.waitpid(self.pid, os.WNOHANG)
            if pid:
                self.exit_code = os.waitstatus_to_exitcode(status)
        return bool(data)

    def wait(self, predicate, description, timeout=None):
        deadline = time.monotonic() + (self.fixture.timeout if timeout is None else timeout)
        while True:
            if result := predicate():
                return result
            if self.exit_code is not None:
                raise AssertionError(f"Exited {self.exit_code} waiting for {description}\n{self.text[-5000:]}")
            if time.monotonic() >= deadline:
                raise AssertionError(f"Timed out waiting for {description}\n{self.text[-5000:]}")
            self.pump(min(0.05, deadline - time.monotonic()))

    def expect(self, text: str, *, after: int = 0, timeout=None):
        return self.wait(lambda: text in self.text[after:], repr(text), timeout)

    def expect_record(self, predicate, description, *, after: int = 0):
        return self.wait(lambda: next((r for r in self.records()[after:] if predicate(r)), None), description)

    def event(self, name, *, after=0):
        return self.expect_record(
            lambda r: (
                r["kind"] == "rpc_sent"
                and r["frame"].get("method") == "event"
                and r["frame"].get("params", {}).get("event", {}).get("type") == name
            ),
            name,
            after=after,
        )

    def ready(self):
        self.expect("scripted/e2e")
        self.expect_record(lambda r: r["kind"] == "rpc_result" and r.get("method") == "turn.subscribe", "subscription")
        self.expect("Describe the antibody")

    def keys(self, data: bytes):
        self.actions.append({"at": time.monotonic() - self.start, "keys_hex": data.hex()})
        os.write(self.fd, data)

    def submit(self, text: str):
        # One paste event prevents completion/keybinding interpretation of the
        # body. Enter remains a real pi Editor submission event.
        self.keys(b"\x1b[200~" + text.encode() + b"\x1b[201~\r")

    def idle(self, after=0):
        self.event("message.complete", after=after)
        self.wait(lambda: not self.working, "idle status line")

    def cancelled(self, after=0):
        self.expect_record(
            lambda r: (
                r["kind"] == "rpc_sent"
                and r["frame"].get("params", {}).get("event", {}).get("payload", {}).get("reason")
                == "cancelled_by_client"
            ),
            "real cancellation notification",
            after=after,
        )
        self.expect("turn cancelled")
        self.wait(lambda: not self.working, "cancelled editor is idle")

    def quit(self):
        # Ctrl+C offers to exit before it exits, so this is two presses from an
        # idle prompt and one when the scenario has already made the offer
        # itself. Which of the two just happened is read from what this press
        # produced, never from the screen: an offer the scenario made is still
        # painted there, and taking it for this press's own would send a third
        # Ctrl+C into an already-exiting process.
        mark = len(self.text)
        self.keys(b"\x03")
        self.wait(
            lambda: self.exit_code is not None or QUIT_HINT in self.text[mark:],
            "the exit, or the offer to exit",
        )
        if self.exit_code is None:
            self.keys(b"\x03")
            self.wait(lambda: self.exit_code is not None, "clean process exit")
        if self.exit_code != 0:
            raise AssertionError(f"Expected exit 0, got {self.exit_code}")
        flags = termios.tcgetattr(self.fd)[3]
        mask = termios.ICANON | termios.ECHO | termios.ISIG
        if flags & mask != self.initial_flags & mask:
            raise AssertionError("Terminal canonical/echo/signal modes were not restored")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        unfinished = self.exit_code is None
        try:
            if unfinished:
                try:
                    os.killpg(self.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                deadline = time.monotonic() + 2
                while self.exit_code is None and time.monotonic() < deadline:
                    self.pump()
                # The test owns the whole process group, including npx/Node.
                # Clean orphan children even if the Python launcher exited.
                try:
                    os.killpg(self.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                if self.exit_code is None:
                    _, status = os.waitpid(self.pid, 0)
                    self.exit_code = os.waitstatus_to_exitcode(status)
            while self.pump(0):
                pass
        finally:
            prefix = self.fixture.root / f"terminal-{self.index}"
            prefix.with_suffix(".bin").write_bytes(self.raw)
            prefix.with_suffix(".txt").write_text(self.text)
            prefix.with_suffix(".screen.txt").write_text(self.screen_text)
            prefix.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "command": self.command,
                        "exit_code": self.exit_code,
                        "actions": self.actions,
                        "tmpdir": self.tmpdir,
                    },
                    indent=2,
                )
            )
            os.close(self.fd)
        if exc_type is None and unfinished:
            raise AssertionError("Scenario left the TUI running instead of verifying clean exit")


def run_scenario(name, artifacts: Path, repo=REPO, timeout=DEFAULT_TIMEOUT):
    artifacts.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix=name + "-", dir=artifacts))
    started = time.monotonic()
    result = {
        "scenario": name,
        "artifacts": str(root),
        "status": "failed",
        "harness_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(HERE.iterdir())
            if p.suffix in {".py", ".json"}
        },
    }
    fixture = None
    try:
        fixture = Fixture(root, repo, timeout)
        SCENARIOS[name](fixture)
        fixture.verify()
        result["status"] = "passed"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result["seconds"] = round(time.monotonic() - started, 3)
        if fixture is not None:
            result["source_before"] = fixture.source_before
            result["source_after"] = source_state(fixture.repo)
        (root / "result.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=["all", *SCENARIOS], default="all")
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    args = parser.parse_args()
    artifacts = args.artifacts or Path(tempfile.mkdtemp(prefix="tui-e2e-"))
    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    results = []
    failed = False
    for name in names:
        try:
            result = run_scenario(name, artifacts, args.repo, args.timeout)
            print(f"PASS {name} {result['seconds']:.3f}s {result['artifacts']}", flush=True)
            results.append(result)
        except Exception as exc:
            failed = True
            print(f"FAIL {name}: {exc}\nArtifacts: {artifacts}", file=sys.stderr, flush=True)
            results.append({"scenario": name, "status": "failed", "error": str(exc)})
    (artifacts / "summary.json").write_text(json.dumps(results, indent=2))
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
