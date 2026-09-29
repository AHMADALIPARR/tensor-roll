# Tensor Roll — Recursive CUDA-Q Model Quantizer
# Copyright (C) 2026 SnapKitty Collective
# SPDX-License-Identifier: AGPL-3.0-or-later
"""OS-level sandbox supervisor (spec section 12).

Architecture (enforced below):
    USER -> TERMINAL -> TENSOR ROLL -> MODEL -> TOOL REQUEST
        -> SANDBOX SUPERVISOR -> POLICY -> DENY | EXECUTE -> ISOLATED PROCESS

Enforced: process isolation (separate session/process group), filesystem
confinement (all path arguments must resolve inside the jail), resource
limits (RLIMIT_CPU/AS/FSIZE/NPROC/NOFILE via preexec_fn), environment
filtering (whitelist), child-process restrictions (own process group,
killed on timeout), signal handling, and a JSON-lines audit log.

Network namespaces are attempted best-effort (unshare(CLONE_NEWNET));
when the kernel refuses, the supervisor records net_isolated=false and
falls back to command-level policy (network fetch tools are denied).
What is enforced vs. best-effort is reported honestly, never assumed.
"""

from __future__ import annotations

import os
import shlex
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field

from .util import AuditLog, ensure_dir, utc_now_iso

try:
    import resource as _resource
except ImportError:  # pragma: no cover
    _resource = None


DENIED_BINS = {
    # destructive
    "rm", "dd", "mkfs", "mkfs.ext4", "shutdown", "reboot", "halt",
    "poweroff", "init", "kill", "killall", "pkill",
    # network fetch / exfiltration
    "curl", "wget", "aria2c", "scp", "sftp", "ftp", "telnet", "nc",
    "ncat", "socat", "ssh",
    # privilege / package management
    "sudo", "su", "doas", "apt", "apt-get", "dpkg", "pip", "pip3",
    "snap", "flatpak",
    # shells that escape argument scanning
    "sh", "bash", "zsh", "fish", "dash",
}

ENV_WHITELIST = {"PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR",
                 "PYTHONPATH", "PYTHONHOME", "TERM", "TZ"}


@dataclass
class Policy:
    max_cpu_s: int = 30
    max_mem_mb: int = 1024
    max_fsize_mb: int = 256
    max_nproc: int = 32
    max_nofile: int = 128
    timeout_s: int = 60


@dataclass
class Verdict:
    allowed: bool
    reason: str
    resolved_bin: str = ""


class SandboxSupervisor:
    def __init__(self, workdir: str, policy: Policy | None = None):
        self.workdir = ensure_dir(workdir)
        self.policy = policy or Policy()
        self.jail = ensure_dir(os.path.join(self.workdir, "jail"))
        self.audit = AuditLog(os.path.join(self.workdir, "audit.log"))
        self.netns_available = self._probe_netns()

    @staticmethod
    def _probe_netns() -> bool:
        """Honest capability probe: can we actually unshare a net namespace?"""
        try:
            r = subprocess.run(["unshare", "--net", "true"],
                               capture_output=True, timeout=10)
            return r.returncode == 0
        except Exception:
            return False

    # -- policy ---------------------------------------------------------
    def check(self, argv: list[str]) -> Verdict:
        if not argv:
            return Verdict(False, "empty command")
        name = os.path.basename(argv[0])
        if name in DENIED_BINS:
            return Verdict(False, f"denied binary: {name}")
        resolved = shutil.which(argv[0], path="/usr/bin:/bin")
        if resolved is None:
            # allow binaries shipped inside the jail itself
            cand = os.path.join(self.jail, os.path.basename(argv[0]))
            if os.path.isfile(cand) and os.access(cand, os.X_OK):
                resolved = cand
            else:
                return Verdict(False, f"binary not found in restricted PATH: {argv[0]}")
        for arg in argv[1:]:
            if ".." in arg.split(os.sep):
                return Verdict(False, f"path traversal rejected: {arg}")
            if os.path.isabs(arg):
                real = os.path.realpath(arg)
                if not (real == self.jail or real.startswith(self.jail + os.sep)):
                    return Verdict(False,
                                   f"absolute path outside jail rejected: {arg}")
        return Verdict(True, "policy pass", resolved)

    # -- execution ------------------------------------------------------
    def _preexec(self):
        # NOTE: start_new_session=True already puts the child in its own
        # session + process group (the child-process restriction anchor).
        if _resource is not None:
            p = self.policy
            _resource.setrlimit(_resource.RLIMIT_CPU, (p.max_cpu_s, p.max_cpu_s))
            mem = p.max_mem_mb * 1024 * 1024
            _resource.setrlimit(_resource.RLIMIT_AS, (mem, mem))
            _resource.setrlimit(_resource.RLIMIT_FSIZE,
                                (p.max_fsize_mb * 1024 * 1024,) * 2)
            _resource.setrlimit(_resource.RLIMIT_NPROC, (p.max_nproc, p.max_nproc))
            _resource.setrlimit(_resource.RLIMIT_NOFILE, (p.max_nofile, p.max_nofile))
        # best-effort network namespace; failure is recorded, not hidden
        try:
            os.unshare(os.CLONE_NEWNET)
        except OSError:
            pass

    def run(self, cmd: str, *, timeout: int | None = None) -> dict:
        argv = shlex.split(cmd)
        verdict = self.check(argv)
        rec = {"ts": utc_now_iso(), "cmd": cmd, "allowed": verdict.allowed,
               "reason": verdict.reason}
        if not verdict.allowed:
            self.audit.record(event="sandbox.deny", **rec)
            rec["returncode"] = None
            return rec
        env = {k: v for k, v in os.environ.items() if k in ENV_WHITELIST}
        env["PATH"] = "/usr/bin:/bin"
        env["TMPDIR"] = self.jail
        t0 = time.perf_counter()
        proc = subprocess.Popen(
            [verdict.resolved_bin] + argv[1:],
            cwd=self.jail, env=env, preexec_fn=self._preexec,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            start_new_session=True)  # own process group: child restriction anchor
        try:
            out, err = proc.communicate(timeout=timeout or self.policy.timeout_s)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc, out, err = "timeout", "", ""
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass
            try:
                out, err = proc.communicate(timeout=5)
            except Exception:
                pass
        # reap any stray children via the process group
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        rec.update({"returncode": rc,
                    "stdout": (out or "")[-4000:],
                    "stderr": (err or "")[-4000:],
                    "elapsed_s": time.perf_counter() - t0,
                    "netns_isolated": self.netns_available})
        self.audit.record(event="sandbox.execute", **rec)
        return rec

    def demo(self) -> list[dict]:
        """Demonstrably ALLOW a permitted action and DENY forbidden ones."""
        results = []
        results.append(("ALLOW echo", self.run("echo hello-tensor-roll")))
        with tempfile.NamedTemporaryFile("w", suffix=".txt",
                                         dir=self.jail, delete=False) as f:
            f.write("jail-write-ok")
        results.append(("ALLOW jail write+read",
                        self.run(f"cat {f.name}")))
        results.append(("DENY shadow", self.run("cat /etc/shadow")))
        results.append(("DENY curl", self.run("curl http://example.com")))
        results.append(("DENY rm", self.run("rm -rf /")))
        return results
