"""Governed Semgrep launcher attestation and bounded runtime health checks.

This module owns launcher validation, exact version probing, local engine smoke
checks, and dynamic-loader classification.  It deliberately emits only stable
identity fields and hashes; callers must never persist its lexical paths or raw
process output.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping


EXPECTED_SEMGREP_VERSION = "1.174.0"
IDENTITY_SCHEMA_VERSION = 1
_MAX_OUTPUT_BYTES = 65_536
_MAX_DEPENDENCY_CLOSURE_ENTRIES = 128
_VERSION_LINE = re.compile(
    r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)"
    r"(?:[-+][0-9A-Za-z][0-9A-Za-z.-]*)?\Z"
)
_SARIF_STATUSES = frozenset({
    "full_valid", "invalid", "full_validation_unavailable", "missing", "not_run",
})
_SYSTEM_LIBRARY_PREFIXES = ("/usr/lib/", "/System/Library/")


class SemgrepRuntimeError(ValueError):
    """Raised when an explicit Semgrep launcher violates the contract."""


@dataclass(frozen=True)
class VerifiedSemgrepLauncher:
    """A validated lexical launcher and its stable resolved executable."""

    lexical_path: Path
    resolved_path: Path
    private_root: Path | None
    launcher_lstat_mode: str
    launcher_symlink: bool
    path_kind: str

    def base_identity(self) -> dict[str, object]:
        return {
            "identity_schema_version": IDENTITY_SCHEMA_VERSION,
            "launcher_basename": self.lexical_path.name or "semgrep",
            "launcher_string_sha256": _sha256_text(str(self.lexical_path)),
            "launcher_lstat_mode": self.launcher_lstat_mode,
            "launcher_symlink": self.launcher_symlink,
            "resolved_executable_sha256": _sha256_file(self.resolved_path),
            "path_kind": self.path_kind,
        }


def verify_explicit_launcher(value: Path | str) -> VerifiedSemgrepLauncher:
    """Validate a governed, current-user-owned private launcher once.

    The lexical path remains the invocation path even when it is a trusted
    symlink.  The resolved target supplies the executable digest.
    """
    lexical = Path(os.fspath(value))
    if not lexical.is_absolute():
        raise SemgrepRuntimeError("semgrep launcher must be an absolute path")
    lexical = Path(os.path.abspath(os.fspath(lexical)))
    try:
        lexical_stat = os.lstat(lexical)
        resolved = lexical.resolve(strict=True)
        resolved_stat = os.stat(resolved)
    except OSError as exc:
        raise SemgrepRuntimeError("semgrep launcher is unavailable") from exc

    lexical_is_symlink = stat.S_ISLNK(lexical_stat.st_mode)
    if not lexical_is_symlink and not stat.S_ISREG(lexical_stat.st_mode):
        raise SemgrepRuntimeError("semgrep launcher must be a regular file or symlink")
    if not stat.S_ISREG(resolved_stat.st_mode):
        raise SemgrepRuntimeError("semgrep resolved target must be a regular file")
    if not os.access(resolved, os.X_OK):
        raise SemgrepRuntimeError("semgrep resolved target is not executable")

    _require_current_user_owner(lexical_stat, "semgrep launcher")
    _require_current_user_owner(resolved_stat, "semgrep resolved target")
    if not lexical_is_symlink:
        _require_not_group_or_world_writable(lexical_stat, "semgrep launcher")
    _require_not_group_or_world_writable(resolved_stat, "semgrep resolved target")

    private_root = _private_runtime_root(lexical)
    try:
        resolved.relative_to(private_root)
        lexical_parent = lexical.parent.resolve(strict=True)
    except (OSError, ValueError) as exc:
        raise SemgrepRuntimeError("semgrep launcher escapes private runtime") from exc
    _validate_private_ancestors(lexical_parent / lexical.name, private_root)
    _validate_private_ancestors(resolved, private_root)

    return VerifiedSemgrepLauncher(
        lexical_path=lexical,
        resolved_path=resolved,
        private_root=private_root,
        launcher_lstat_mode=f"{stat.S_IMODE(lexical_stat.st_mode):04o}",
        launcher_symlink=lexical_is_symlink,
        path_kind="governed_private",
    )


def resolve_default_launcher() -> VerifiedSemgrepLauncher:
    """Resolve the legacy scanner fallback without granting it governed status."""
    import shutil

    candidate = shutil.which("semgrep") or "/opt/homebrew/bin/semgrep"
    lexical = Path(candidate)
    try:
        lexical_stat = os.lstat(lexical)
        resolved = lexical.resolve(strict=True)
        resolved_stat = os.stat(resolved)
    except OSError:
        return VerifiedSemgrepLauncher(
            lexical_path=lexical,
            resolved_path=lexical,
            private_root=None,
            launcher_lstat_mode="unknown",
            launcher_symlink=False,
            path_kind="unknown",
        )
    lexical_is_symlink = stat.S_ISLNK(lexical_stat.st_mode)
    executable = stat.S_ISREG(resolved_stat.st_mode) and os.access(resolved, os.X_OK)
    return VerifiedSemgrepLauncher(
        lexical_path=lexical,
        resolved_path=resolved,
        private_root=None,
        launcher_lstat_mode=f"{stat.S_IMODE(lexical_stat.st_mode):04o}",
        launcher_symlink=lexical_is_symlink,
        path_kind=_path_kind(resolved) if executable else "unknown",
    )


def collect_runtime_health(
    launcher: VerifiedSemgrepLauncher,
    *,
    environment: Mapping[str, str],
    engine_runner: Callable[..., object],
    runner: Callable[..., object] = subprocess.run,
    sarif_validator: Callable[[str], str] | None = None,
    workspace_root: Path | None = None,
    version_timeout: float = 10,
    engine_timeout: float = 20,
) -> dict[str, object]:
    """Return a bounded health record for one already-resolved launcher."""
    identity = launcher.base_identity()
    validator = sarif_validator or _validate_sarif_text
    version_capture = _run_bounded(
        [str(launcher.lexical_path), "--version"],
        environment=environment,
        timeout=version_timeout,
        runner=runner,
    )
    version = None
    parse_source = "none"
    if version_capture.return_code == 0 and not version_capture.timed_out:
        version = _parse_exact_version(version_capture.stdout)
        if version is not None:
            parse_source = "stdout"

    closure = inspect_dependency_closure(launcher)
    base: dict[str, object] = {
        **identity,
        "version_probe_return_code": version_capture.return_code,
        "version_probe_timed_out": version_capture.timed_out,
        "version_probe_stdout_sha256": _sha256_bytes(version_capture.stdout),
        "version_probe_stderr_sha256": _sha256_bytes(version_capture.stderr),
        "version": version,
        "version_parse_source": parse_source,
        "engine_smoke_return_code": None,
        "engine_smoke_timed_out": False,
        "engine_smoke_stdout_sha256": None,
        "engine_smoke_stderr_sha256": None,
        "engine_smoke_sarif_status": "not_run",
        "engine_smoke_raw_output_persisted": False,
        "dependency_closure_sha256": closure.get("dependency_closure_sha256"),
        "dependency_manifest": closure.get("dependency_manifest"),
        "semgrep_core_sha256": closure.get("semgrep_core_sha256"),
        "linker_family": None,
        "missing_library_basename": None,
        "failure_class": None,
        "healthy": False,
    }

    version_process_failure = classify_semgrep_process_failure(
        version_capture.return_code,
        _decode_bounded(version_capture.stderr),
    )
    if version_process_failure is not None:
        base.update(_failure_fields(version_process_failure, version_capture.stderr))
        return base
    if version_capture.timed_out or version_capture.return_code != 0:
        base["failure_class"] = "semgrep_runtime_version_probe_failed"
        return base
    if version is None or parse_source != "stdout":
        base["failure_class"] = "semgrep_runtime_version_unparseable"
        return base
    if version != EXPECTED_SEMGREP_VERSION:
        base["failure_class"] = "semgrep_runtime_version_probe_failed"
        return base
    if closure.get("failure_class") is not None:
        base["failure_class"] = closure["failure_class"]
        return base

    smoke = _run_engine_smoke(
        launcher,
        environment=environment,
        runner=engine_runner,
        sarif_validator=validator,
        workspace_root=workspace_root,
        timeout=engine_timeout,
    )
    base.update({
        "engine_smoke_return_code": smoke.return_code,
        "engine_smoke_timed_out": smoke.timed_out,
        "engine_smoke_stdout_sha256": _sha256_bytes(smoke.stdout),
        "engine_smoke_stderr_sha256": _sha256_bytes(smoke.stderr),
        "engine_smoke_sarif_status": smoke.sarif_status,
    })
    smoke_process_failure = classify_semgrep_process_failure(
        smoke.return_code,
        _decode_bounded(smoke.stderr),
    )
    if smoke_process_failure is not None:
        base.update(_failure_fields(smoke_process_failure, smoke.stderr))
        return base
    if (
        smoke.timed_out
        or smoke.return_code not in (0, 1)
        or smoke.sarif_status != "full_valid"
    ):
        base["failure_class"] = "semgrep_runtime_engine_smoke_failed"
        return base
    base["healthy"] = True
    return base

def classify_dynamic_loader_failure(stderr: object) -> dict[str, str] | None:
    """Classify only evidenced dynamic-loader failures without retaining paths."""
    text = str(stderr or "")
    lowered = text.lower()
    if "library not loaded" in lowered or "dyld" in lowered and "image not found" in lowered:
        basename = _missing_library_basename(
            text,
            r"(?im)library\s+not\s+loaded\s*:\s*([^\r\n]+)",
            r"(?i)([A-Za-z0-9_.-]+\.dylib)\b",
        )
        if basename is not None:
            return {"linker_family": "dyld", "missing_library_basename": basename}
    if "error while loading shared libraries" in lowered or "cannot open shared object file" in lowered:
        basename = _missing_library_basename(
            text,
            r"(?im)shared\s+libraries\s*:\s*([^:\s]+)",
            r"(?i)([A-Za-z0-9_.-]+\.so(?:\.\d+)*)\b",
        )
        if basename is not None:
            return {"linker_family": "ld.so", "missing_library_basename": basename}
    if "dll load failed" in lowered or "specified module could not be found" in lowered:
        basename = _missing_library_basename(
            text,
            r"(?i)([A-Za-z0-9_.-]+\.dll)\b",
        )
        if basename is not None:
            return {
                "linker_family": "windows_loader",
                "missing_library_basename": basename,
            }
    return None


def classify_semgrep_process_failure(
    raw_exit_code: int | None,
    stderr: object,
) -> str | None:
    """Classify loader and abort failures before SARIF interpretation."""
    if classify_dynamic_loader_failure(stderr) is not None:
        return "semgrep_runtime_linker_dependency_missing"
    if raw_exit_code == 134:
        return "semgrep_runtime_process_aborted"
    return None


def inspect_dependency_closure(launcher: VerifiedSemgrepLauncher) -> dict[str, object]:
    """Attest the complete non-system Mach-O closure of Semgrep-core."""
    core = _locate_semgrep_core(launcher)
    if core is None:
        return _closure_failure(None)
    core_digest = _sha256_file(core)
    private_root = launcher.private_root
    if core_digest is None or private_root is None:
        return _closure_failure(core_digest)
    try:
        private_root = private_root.resolve(strict=True)
        core.relative_to(private_root)
        observed = os.stat(core)
        if not stat.S_ISREG(observed.st_mode):
            return _closure_failure(core_digest)
        _require_current_user_owner(observed, "Semgrep core")
        _require_not_group_or_world_writable(observed, "Semgrep core")
        _validate_private_ancestors(core, private_root)
    except (OSError, SemgrepRuntimeError, ValueError):
        return _closure_failure(core_digest)

    if sys.platform != "darwin":
        manifest: list[dict[str, str]] = []
        encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return {
            "dependency_closure_sha256": _sha256_bytes(encoded),
            "dependency_manifest": manifest,
            "semgrep_core_sha256": core_digest,
            "failure_class": None,
        }

    pending = [core]
    inspected: set[Path] = set()
    manifest_by_path: dict[Path, dict[str, str]] = {}
    while pending:
        binary = pending.pop()
        try:
            binary = binary.resolve(strict=True)
        except OSError:
            return _closure_failure(core_digest)
        if binary in inspected:
            continue
        if len(inspected) >= _MAX_DEPENDENCY_CLOSURE_ENTRIES:
            return _closure_failure(core_digest)
        inspected.add(binary)
        dependencies = _macho_dependencies(binary)
        if dependencies is None:
            return _closure_failure(core_digest)
        for dependency in dependencies:
            if dependency.startswith(_SYSTEM_LIBRARY_PREFIXES):
                continue
            resolved = _resolve_macho_dependency(binary, dependency)
            if resolved is None:
                return _closure_failure(core_digest)
            if _is_system_library(resolved):
                continue
            try:
                resolved.relative_to(private_root)
                observed = os.stat(resolved)
                if not stat.S_ISREG(observed.st_mode):
                    return _closure_failure(core_digest)
                _require_current_user_owner(observed, "Semgrep dependency")
                _require_not_group_or_world_writable(observed, "Semgrep dependency")
                _validate_private_ancestors(resolved, private_root)
                digest = _sha256_file(resolved)
            except (OSError, SemgrepRuntimeError, ValueError):
                return _closure_failure(core_digest)
            if digest is None:
                return _closure_failure(core_digest)
            manifest_by_path[resolved] = {"basename": resolved.name, "sha256": digest}
            if resolved not in inspected:
                pending.append(resolved)

    manifest = sorted(
        manifest_by_path.values(),
        key=lambda item: (item["basename"], item["sha256"]),
    )
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "dependency_closure_sha256": _sha256_bytes(encoded),
        "dependency_manifest": manifest,
        "semgrep_core_sha256": core_digest,
        "failure_class": None,
    }

@dataclass(frozen=True)
class _Capture:
    return_code: int | None
    timed_out: bool
    stdout: bytes
    stderr: bytes


@dataclass(frozen=True)
class _SmokeCapture(_Capture):
    sarif_status: str


def _run_engine_smoke(
    launcher: VerifiedSemgrepLauncher,
    *,
    environment: Mapping[str, str],
    runner: Callable[..., object],
    sarif_validator: Callable[[str], str],
    workspace_root: Path | None,
    timeout: float,
) -> _SmokeCapture:
    root = str(workspace_root) if workspace_root is not None else None
    with tempfile.TemporaryDirectory(prefix="semgrep-runtime-smoke-", dir=root) as raw_root:
        smoke_root = Path(raw_root)
        smoke_root.chmod(0o700)
        rule = smoke_root / "rule.yml"
        source = smoke_root / "fixture.py"
        rule.write_text(
            "rules:\n"
            "  - id: raptor-runtime-smoke\n"
            "    languages: [python]\n"
            "    severity: INFO\n"
            "    message: runtime smoke\n"
            "    pattern: raise RuntimeError(...)\n",
            encoding="utf-8",
        )
        source.write_text("value = 1\n", encoding="utf-8")
        capture = _run_bounded(
            [
                str(launcher.lexical_path),
                "scan",
                "--config", str(rule),
                "--quiet",
                "--metrics", "off",
                "--error",
                "--sarif",
                "--timeout", "10",
                str(source),
            ],
            environment=environment,
            timeout=timeout,
            runner=runner,
        )
    sarif_status = "not_run"
    if not capture.timed_out and capture.return_code in (0, 1):
        try:
            candidate = sarif_validator(_decode_bounded(capture.stdout))
        except Exception:  # noqa: BLE001 - runtime health fails closed
            candidate = "invalid"
        sarif_status = candidate if candidate in _SARIF_STATUSES else "invalid"
    return _SmokeCapture(
        return_code=capture.return_code,
        timed_out=capture.timed_out,
        stdout=capture.stdout,
        stderr=capture.stderr,
        sarif_status=sarif_status,
    )


def _run_bounded(
    argv: list[str],
    *,
    environment: Mapping[str, str],
    timeout: float,
    runner: Callable[..., object],
) -> _Capture:
    try:
        completed = runner(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            env=dict(environment),
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return _Capture(
            return_code=None,
            timed_out=True,
            stdout=_bounded_bytes(getattr(exc, "stdout", b"")),
            stderr=_bounded_bytes(getattr(exc, "stderr", b"")),
        )
    except OSError as exc:
        return _Capture(
            return_code=None,
            timed_out=False,
            stdout=b"",
            stderr=_bounded_bytes(str(exc)),
        )
    return _Capture(
        return_code=getattr(completed, "returncode", None),
        timed_out=False,
        stdout=_bounded_bytes(getattr(completed, "stdout", b"")),
        stderr=_bounded_bytes(getattr(completed, "stderr", b"")),
    )


def _validate_sarif_text(value: str) -> str:
    try:
        payload = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return "invalid"
    if not isinstance(payload, dict):
        return "invalid"
    schema_path = Path(__file__).resolve().parents[2] / "engine/schemas/sarif-2.1.0.json"
    try:
        import jsonschema
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        jsonschema.validate(payload, schema)
    except (ImportError, OSError, json.JSONDecodeError):
        return "full_validation_unavailable"
    except Exception:  # noqa: BLE001 - schema validation is fail closed
        return "invalid"
    return "full_valid"


def _parse_exact_version(stdout: bytes) -> str | None:
    lines = _decode_bounded(stdout).splitlines()
    if len(lines) != 1:
        return None
    candidate = lines[0].strip()
    return candidate if _VERSION_LINE.fullmatch(candidate) else None


def _failure_fields(failure_class: str, stderr: bytes) -> dict[str, object]:
    loader = classify_dynamic_loader_failure(_decode_bounded(stderr))
    if loader is None:
        return {"failure_class": failure_class}
    return {"failure_class": failure_class, **loader}


def _private_runtime_root(lexical: Path) -> Path:
    if lexical.parent.name != "bin":
        raise SemgrepRuntimeError("semgrep launcher must live under a private bin directory")
    root = lexical.parent.parent
    try:
        observed = os.lstat(root)
        canonical = root.resolve(strict=True)
        canonical_observed = os.lstat(canonical)
    except OSError as exc:
        raise SemgrepRuntimeError("semgrep private runtime root is unavailable") from exc
    for candidate in (observed, canonical_observed):
        if stat.S_ISLNK(candidate.st_mode) or not stat.S_ISDIR(candidate.st_mode):
            raise SemgrepRuntimeError("semgrep private runtime root is not a real directory")
        _require_current_user_owner(candidate, "semgrep private runtime root")
        if stat.S_IMODE(candidate.st_mode) != 0o700:
            raise SemgrepRuntimeError("semgrep private runtime root is not private")
    return canonical
def _validate_private_ancestors(path: Path, private_root: Path) -> None:
    current = path.parent
    while True:
        try:
            observed = os.lstat(current)
        except OSError as exc:
            raise SemgrepRuntimeError("semgrep launcher ancestor is unavailable") from exc
        if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
            raise SemgrepRuntimeError("semgrep launcher ancestor is not a real directory")
        _require_current_user_owner(observed, "semgrep launcher ancestor")
        if stat.S_IMODE(observed.st_mode) != 0o700:
            raise SemgrepRuntimeError("semgrep launcher ancestor is not private")
        if current == private_root:
            return
        try:
            current.relative_to(private_root)
        except ValueError as exc:
            raise SemgrepRuntimeError("semgrep launcher escapes private runtime") from exc
        current = current.parent


def _require_current_user_owner(observed: os.stat_result, label: str) -> None:
    if observed.st_uid != _current_uid():
        raise SemgrepRuntimeError(f"{label} owner is not the current user")


def _require_not_group_or_world_writable(observed: os.stat_result, label: str) -> None:
    if stat.S_IMODE(observed.st_mode) & 0o022:
        raise SemgrepRuntimeError(f"{label} is group or world writable")


def _current_uid() -> int:
    getter = getattr(os, "geteuid", None) or getattr(os, "getuid", None)
    if getter is None:
        return -1
    return getter()


def _path_kind(path: Path) -> str:
    rendered = str(path)
    if rendered.startswith(("/bin/", "/sbin/", "/usr/", "/opt/", "/System/Library/", "/Library/Frameworks/")):
        return "system"
    return "unknown"


def _locate_semgrep_core(launcher: VerifiedSemgrepLauncher) -> Path | None:
    root = launcher.private_root
    if root is None:
        return None
    try:
        root = root.resolve(strict=True)
    except OSError:
        return None
    candidates = list(root.glob("lib/python*/site-packages/semgrep/bin/semgrep-core"))
    candidates.extend(root.glob("Lib/site-packages/semgrep/bin/semgrep-core"))
    for candidate in sorted(candidates):
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
            observed = os.stat(resolved)
        except (OSError, ValueError):
            continue
        if stat.S_ISREG(observed.st_mode):
            return resolved
    return None


def _macho_dependencies(binary: Path) -> list[str] | None:
    try:
        proc = subprocess.run(
            ["/usr/bin/otool", "-L", str(binary)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    output = _bounded_bytes(proc.stdout).decode("utf-8", errors="replace")
    dependencies: list[str] = []
    for raw_line in output.splitlines()[1:]:
        dependency = raw_line.strip().split(" (", 1)[0].strip()
        if dependency:
            dependencies.append(dependency)
    return dependencies


def _macho_rpaths(binary: Path) -> list[str] | None:
    try:
        proc = subprocess.run(
            ["/usr/bin/otool", "-l", str(binary)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    lines = _bounded_bytes(proc.stdout).decode("utf-8", errors="replace").splitlines()
    rpaths: list[str] = []
    for index, raw_line in enumerate(lines):
        if raw_line.strip() != "cmd LC_RPATH":
            continue
        for candidate_line in lines[index + 1:index + 6]:
            match = re.match(r"\s*path\s+(.+?)\s+\(offset", candidate_line)
            if match is not None:
                rpaths.append(match.group(1))
                break
            if candidate_line.strip().startswith("Load command"):
                break
    return rpaths


def _resolve_macho_dependency(binary: Path, dependency: str) -> Path | None:
    if dependency.startswith("@rpath/"):
        suffix = dependency.removeprefix("@rpath/")
        rpaths = _macho_rpaths(binary)
        if rpaths is None:
            return None
        for rpath in rpaths:
            root = _resolve_macho_token(binary, rpath)
            if root is None:
                continue
            try:
                return (root / suffix).resolve(strict=True)
            except OSError:
                continue
        return None
    return _resolve_macho_token(binary, dependency)


def _resolve_macho_token(binary: Path, value: str) -> Path | None:
    if value.startswith("@loader_path/"):
        candidate = binary.parent / value.removeprefix("@loader_path/")
    elif value.startswith("@executable_path/"):
        candidate = binary.parent / value.removeprefix("@executable_path/")
    else:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = binary.parent / candidate
    try:
        return candidate.resolve(strict=True)
    except OSError:
        return None


def _is_system_library(path: Path) -> bool:
    return str(path).startswith(_SYSTEM_LIBRARY_PREFIXES)


def _closure_failure(core_digest: str | None) -> dict[str, object]:
    return {
        "dependency_closure_sha256": None,
        "dependency_manifest": None,
        "semgrep_core_sha256": core_digest,
        "failure_class": "semgrep_runtime_dependency_closure_invalid",
    }


def _missing_library_basename(
    text: str,
    primary: str,
    fallback: str | None = None,
) -> str | None:
    match = re.search(primary, text)
    candidate = match.group(1).strip().strip("'\" ") if match else ""
    if not candidate and fallback is not None:
        fallback_match = re.search(fallback, text)
        candidate = fallback_match.group(1) if fallback_match else ""
    basename = Path(candidate).name
    if not basename or len(basename) > 255 or not re.fullmatch(r"[A-Za-z0-9_.-]+", basename):
        return None
    return basename


def _bounded_bytes(value: object) -> bytes:
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value[:_MAX_OUTPUT_BYTES]
    return str(value).encode("utf-8", errors="replace")[:_MAX_OUTPUT_BYTES]


def _decode_bounded(value: bytes) -> str:
    return value.decode("utf-8", errors="replace")


def _sha256_file(path: Path) -> str | None:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))
