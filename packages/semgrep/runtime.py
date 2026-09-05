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


def environment_for_launcher(
    launcher_path: Path | str,
    environment: Mapping[str, str],
) -> dict[str, str]:
    """Bind subprocess lookup to the launcher's private bin directory."""
    bound = dict(environment)
    private_bin = str(Path(launcher_path).parent)
    existing_path = bound.get("PATH", "")
    entries = [private_bin]
    entries.extend(
        entry
        for entry in existing_path.split(os.pathsep)
        if entry and entry != private_bin
    )
    bound["PATH"] = os.pathsep.join(entries)
    bound["SEMGREP_ENABLE_VERSION_CHECK"] = "0"
    return bound


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
    runtime_environment = environment_for_launcher(launcher.lexical_path, environment)
    version_capture = _run_bounded(
        [str(launcher.lexical_path), "--version"],
        environment=runtime_environment,
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
        environment=runtime_environment,
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
            resolved = _resolve_macho_dependency(
                binary,
                dependency,
                executable_path=core,
            )
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
    lines = output.splitlines()[1:]
    if binary.suffix == ".dylib":
        # otool -L reports a dylib install name before its dependencies.
        lines = lines[1:]
    dependencies: list[str] = []
    for raw_line in lines:
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


def _resolve_macho_dependency(
    binary: Path,
    dependency: str,
    *,
    executable_path: Path | None = None,
) -> Path | None:
    if dependency.startswith("@rpath/"):
        suffix = dependency.removeprefix("@rpath/")
        rpaths = _macho_rpaths(binary)
        if rpaths is None:
            return None
        for rpath in rpaths:
            root = _resolve_macho_token(
                binary,
                rpath,
                executable_path=executable_path,
            )
            if root is None:
                continue
            try:
                return (root / suffix).resolve(strict=True)
            except OSError:
                continue
        return None
    return _resolve_macho_token(
        binary,
        dependency,
        executable_path=executable_path,
    )


def _resolve_macho_token(
    binary: Path,
    value: str,
    *,
    executable_path: Path | None = None,
) -> Path | None:
    if value.startswith("@loader_path/"):
        candidate = binary.parent / value.removeprefix("@loader_path/")
    elif value.startswith("@executable_path/"):
        executable = executable_path or binary
        candidate = executable.parent / value.removeprefix("@executable_path/")
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


# ---------------------------------------------------------------------------
# Standalone identity schema v2.  The historical v1 API above is deliberately
# unchanged.  v2 returns only hashes, booleans, and closed categories: no
# launcher paths, environment values, shebang text, or process output.

_V2_SAFE_ENV_KEYS = frozenset({
    "PATH", "HOME", "TMP", "TEMP", "TMPDIR", "LANG", "LANGUAGE",
    "LC_ALL", "LC_CTYPE", "LC_MESSAGES", "TERM", "TERM_PROGRAM",
    "SYSTEMROOT", "WINDIR",
})
_V2_TOKEN = re.compile(r"[A-Za-z0-9_.:+@=\-\[\],]{1,255}\Z")
_V2_HEX = re.compile(r"[0-9a-f]{64}\Z")


def collect_runtime_identity_v2(
    launcher: Path | str | VerifiedSemgrepLauncher,
    *,
    environment: Mapping[str, str],
    engine_runner: Callable[..., object],
    runner: Callable[..., object] = subprocess.run,
    sarif_validator: Callable[[str], str] | None = None,
    workspace_root: Path | None = None,
    version_timeout: float = 10,
    engine_timeout: float = 20,
) -> dict[str, object]:
    """Collect a complete standalone identity using the launcher's interpreter.

    Unsafe launcher or interpreter input is represented by a failed record,
    rather than raising, so callers can safely retain the result.  Package
    inspection runs inside the exact interpreter named by the shebang; this
    process never imports the inspected distribution.
    """
    lexical, resolved, lexical_stat = _v2_launcher_paths(launcher)
    launcher_identity = _v2_empty_launcher_identity()
    verified: VerifiedSemgrepLauncher | None = None
    interpreter_path: Path | None = None
    shebang_safe = False
    if lexical is not None and resolved is not None and lexical_stat is not None:
        launcher_identity.update({
            "basename": lexical.name or "semgrep",
            "file_sha256": _sha256_file(lexical),
            "resolved_file_sha256": _sha256_file(resolved),
            "mode": f"{stat.S_IMODE(lexical_stat.st_mode):04o}",
            "symlink": stat.S_ISLNK(lexical_stat.st_mode),
        })
        shebang_kind, shebang_digest, interpreter_path, interpreter_basename = (
            _v2_shebang_identity(lexical, resolved)
        )
        launcher_identity.update({
            "shebang_kind": shebang_kind,
            "shebang_sha256": shebang_digest,
            "shebang_interpreter_basename": interpreter_basename,
        })
        shebang_safe = interpreter_path is not None
        try:
            verified = (
                launcher
                if isinstance(launcher, VerifiedSemgrepLauncher)
                else verify_explicit_launcher(lexical)
            )
        except (OSError, SemgrepRuntimeError, TypeError, ValueError):
            verified = None

    interpreter = _v2_empty_interpreter_identity()
    environment_map = _v2_safe_environment(
        environment, lexical.parent if lexical is not None else None
    )
    if interpreter_path is not None:
        interpreter = _v2_interpreter_file_identity(interpreter_path)

    version_capture = _Capture(
        return_code=None, timed_out=False, stdout=b"", stderr=b""
    )
    if verified is not None and shebang_safe and interpreter_path is not None:
        version_capture = _run_bounded_v2(
            [str(lexical)],
            environment=environment_map,
            timeout=version_timeout,
            runner=runner,
            cwd=workspace_root,
            extra_args=("--version",),
        )
    parsed_version = None
    if version_capture.return_code == 0 and not version_capture.timed_out:
        parsed_version = _parse_exact_version(version_capture.stdout)

    distribution = _v2_empty_distribution_identity()
    probe_ok = False
    if verified is not None and shebang_safe and interpreter_path is not None:
        probe_capture = _run_bounded_v2(
            [str(interpreter_path), "-c", _V2_INTERPRETER_PROBE],
            environment=environment_map,
            timeout=version_timeout,
            runner=runner,
            cwd=workspace_root,
        )
        probe = _v2_parse_probe(probe_capture)
        if probe is not None:
            probe_interpreter = probe.get("interpreter")
            if isinstance(probe_interpreter, Mapping):
                _v2_merge_interpreter_identity(interpreter, probe_interpreter)
            probe_distribution = probe.get("distribution")
            if isinstance(probe_distribution, Mapping):
                _v2_merge_distribution_identity(distribution, probe_distribution)
            probe_ok = bool(probe.get("probe_ok"))

    native_detail = _v2_native_runtime(verified)
    native = {
        "semgrep_core_sha256": native_detail.get("semgrep_core_sha256"),
        "dependency_closure_sha256": native_detail.get("dependency_closure_sha256"),
    }
    version_rc3 = version_capture.return_code == 3 and not version_capture.timed_out
    health = {
        "version_probe_return_code": version_capture.return_code,
        "version_probe_timed_out": version_capture.timed_out,
        "parsed_version": parsed_version,
        "version_probe_failure_category": (
            "unexpected_cli_exit" if version_rc3 else None
        ),
        "version_probe_cli_exit_label": (
            "target_parse_failure" if version_rc3 else None
        ),
        "version_probe_root_cause_proven": False,
        "version_probe_redacted_stderr_tail": "",
        "engine_smoke_return_code": None,
        "engine_smoke_timed_out": False,
        "engine_smoke_sarif_status": "not_run",
        "healthy": False,
    }
    distribution_ready = (
        distribution["version"] == EXPECTED_SEMGREP_VERSION
        and distribution["importable"] is True
        and distribution["metadata_sha256"] is not None
        and distribution["record_sha256"] is not None
        and distribution["entry_points_sha256"] is not None
        and distribution["package_tree_manifest_sha256"] is not None
        and distribution["python_distribution_manifest_sha256"] is not None
        and distribution["record_missing_count"] == 0
        and distribution["record_hash_mismatch_count"] == 0
    )
    native_ready = (
        native["semgrep_core_sha256"] is not None
        and native_detail.get("failure_class") is None
    )
    prerequisites_ready = (
        shebang_safe
        and interpreter["present"] is True
        and probe_ok
        and version_capture.return_code == 0
        and not version_capture.timed_out
        and parsed_version == EXPECTED_SEMGREP_VERSION
        and distribution_ready
        and native_ready
    )
    if prerequisites_ready and lexical is not None:
        validator = sarif_validator or _validate_sarif_text
        smoke = _run_engine_smoke(
            verified,
            environment=environment_map,
            runner=engine_runner,
            sarif_validator=validator,
            workspace_root=workspace_root,
            timeout=engine_timeout,
        )
        health.update({
            "engine_smoke_return_code": smoke.return_code,
            "engine_smoke_timed_out": smoke.timed_out,
            "engine_smoke_sarif_status": smoke.sarif_status,
        })
        health["healthy"] = bool(
            not smoke.timed_out
            and smoke.return_code in (0, 1)
            and smoke.sarif_status == "full_valid"
        )

    contract_material = {
        "identity_schema_version": 2,
        "policy": "v2-safe-runtime-environment",
        "environment": _v2_environment_material(environment_map),
    }
    python_distribution_manifest = distribution.get(
        "python_distribution_manifest_sha256"
    )
    return {
        "identity_schema_version": 2,
        "launcher": launcher_identity,
        "interpreter": interpreter,
        "semgrep_distribution": distribution,
        "python_distribution_manifest_sha256": python_distribution_manifest,
        "native_runtime": native,
        "health": health,
        "runtime_environment_contract_sha256": _sha256_bytes(
            json.dumps(
                contract_material,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ),
    }


def classify_runtime_diagnosis(
    profiles: object,
    prior_failure: bool = True,
) -> dict[str, object]:
    """Classify raw version profiles using only bounded, closed categories.

    profiles may be a mapping with current_safe and minimal_private outcomes,
    or a two-item sequence in that order.  An outcome uses return_code,
    timed_out, and parsed_version and may include nested health or smoke
    results.  Caller-provided category strings are ignored, and no stderr is
    inspected.
    """
    named: dict[str, Mapping[str, object]] = {}
    context: Mapping[str, object] | None = profiles if isinstance(profiles, Mapping) else None
    if isinstance(profiles, Mapping):
        for name in ("current_safe", "minimal_private"):
            value = profiles.get(name)
            if isinstance(value, Mapping):
                named[name] = value
    elif isinstance(profiles, (list, tuple)):
        values = [value for value in profiles if isinstance(value, Mapping)]
        if values:
            named["current_safe"] = values[0]
        if len(values) > 1:
            named["minimal_private"] = values[1]

    current = named.get("current_safe")
    minimal = named.get("minimal_private")
    if current is not None and minimal is not None:
        current_state = _v2_outcome_state(current)
        minimal_state = _v2_outcome_state(minimal)
        if minimal_state == "pass" and current_state == "fail":
            label = "target_parse_failure" if _v2_outcome_return_code(current) == 3 else "runtime_health_failure"
            return {
                "category": "qualification_environment_failed",
                "label": label,
                "root_cause_proven": False,
            }
        if current_state == "pass" and minimal_state == "pass":
            if prior_failure:
                return {
                    "category": "runtime_nondeterministic",
                    "label": "historical_failure_unreproduced",
                    "root_cause_proven": False,
                }
            if _v2_outcome_smoke_healthy(current, context) and _v2_outcome_smoke_healthy(minimal, context):
                return {
                    "category": "runtime_healthy",
                    "label": "stable_runtime",
                    "root_cause_proven": False,
                }
    return {
        "category": "diagnosis_incomplete",
        "label": "insufficient_profiles",
        "root_cause_proven": False,
    }


def _v2_outcome_state(row: Mapping[str, object]) -> str:
    return "pass" if _v2_outcome_exact(row) else (
        "fail"
        if (
            _v2_outcome_return_code(row) is not None
            or _v2_outcome_timed_out(row)
            or isinstance(_v2_outcome_parsed_version(row), str)
        )
        else "unknown"
    )


def _v2_empty_launcher_identity() -> dict[str, object]:
    return {
        "basename": None,
        "file_sha256": None,
        "resolved_file_sha256": None,
        "mode": None,
        "symlink": False,
        "shebang_kind": "missing",
        "shebang_sha256": None,
        "shebang_interpreter_basename": None,
    }


def _v2_empty_interpreter_identity() -> dict[str, object]:
    return {
        "present": False,
        "resolved_sha256": None,
        "implementation": None,
        "version": None,
        "sys_executable_sha256": None,
        "sys_prefix_sha256": None,
        "sys_base_prefix_sha256": None,
        "virtual_environment_active": False,
        "pyvenv_cfg_present": False,
        "pyvenv_cfg_sha256": None,
    }


def _v2_empty_distribution_identity() -> dict[str, object]:
    return {
        "version": None,
        "importable": False,
        "entrypoint_identifier": None,
        "metadata_sha256": None,
        "record_sha256": None,
        "entry_points_sha256": None,
        "record_entry_count": 0,
        "record_verified_count": 0,
        "record_missing_count": 0,
        "record_hash_mismatch_count": 0,
        "package_tree_manifest_sha256": None,
        "python_distribution_manifest_sha256": None,
    }


def _v2_launcher_paths(
    launcher: Path | str | VerifiedSemgrepLauncher,
) -> tuple[Path | None, Path | None, os.stat_result | None]:
    if isinstance(launcher, VerifiedSemgrepLauncher):
        lexical = launcher.lexical_path
        resolved = launcher.resolved_path
    else:
        try:
            lexical = Path(os.fspath(launcher))
            if not lexical.is_absolute():
                return None, None, None
            lexical = Path(os.path.abspath(os.fspath(lexical)))
            resolved = lexical.resolve(strict=True)
        except (OSError, TypeError, ValueError):
            return None, None, None
    try:
        lexical_stat = os.lstat(lexical)
        resolved_stat = os.stat(resolved)
    except OSError:
        return None, None, None
    if not stat.S_ISLNK(lexical_stat.st_mode) and not stat.S_ISREG(lexical_stat.st_mode):
        return None, None, None
    if not stat.S_ISREG(resolved_stat.st_mode) or not os.access(resolved, os.X_OK):
        return None, None, None
    if stat.S_IMODE(resolved_stat.st_mode) & 0o022:
        return None, None, None
    return lexical, resolved, lexical_stat


def _v2_shebang_identity(
    lexical: Path,
    resolved: Path,
) -> tuple[str, str | None, Path | None, str | None]:
    try:
        with resolved.open("rb") as handle:
            line = handle.readline(4097)
    except OSError:
        return "none", None, None, None
    if not line.startswith(b"#!"):
        if line[:4] in {
            b"\x7fELF", b"\xcf\xfa\xed\xfe", b"\xfe\xed\xfa\xcf", b"MZ"
        }:
            return "native_binary", None, None, None
        return "none", None, None, None
    digest = _sha256_bytes(line)
    if len(line) > 4096:
        return "malformed", digest, None, None
    try:
        import shlex

        text = line[2:].decode("ascii").strip()
        parts = shlex.split(text, posix=True)
    except (UnicodeDecodeError, ValueError):
        return "malformed", digest, None, None
    if not parts:
        return "malformed", digest, None, None
    command = parts[0]
    basename = Path(command).name or None
    if basename == "env":
        env_interpreter = next((part for part in parts[1:] if not part.startswith("-")), None)
        return "env_interpreter", digest, None, Path(env_interpreter).name if env_interpreter else None
    if len(parts) != 1 or not command.startswith("/"):
        return "malformed", digest, None, basename
    lexical_interpreter = Path(command)
    try:
        resolved_interpreter = lexical_interpreter.resolve(strict=True)
        observed = os.stat(resolved_interpreter)
    except OSError:
        return "absolute_interpreter", digest, None, lexical_interpreter.name
    if (
        not stat.S_ISREG(observed.st_mode)
        or not os.access(resolved_interpreter, os.X_OK)
        or stat.S_IMODE(observed.st_mode) & 0o022
        or not _v2_safe_ancestors(resolved_interpreter)
    ):
        return "absolute_interpreter", digest, None, lexical_interpreter.name
    return "absolute_interpreter", digest, lexical_interpreter, lexical_interpreter.name


def _v2_safe_ancestors(path: Path) -> bool:
    current = path
    uid = _current_uid()
    while True:
        try:
            observed = os.stat(current)
        except OSError:
            return False
        mode = stat.S_IMODE(observed.st_mode)
        if stat.S_ISDIR(observed.st_mode) and mode & 0o002:
            sticky = bool(mode & stat.S_ISVTX)
            if not sticky or observed.st_uid not in {0, uid}:
                return False
        elif mode & 0o022:
            return False
        if current.parent == current:
            return True
        current = current.parent




def _v2_interpreter_file_identity(path: Path) -> dict[str, object]:
    result = _v2_empty_interpreter_identity()
    digest = _sha256_file(path)
    result.update({"present": digest is not None, "resolved_sha256": digest})
    return result




def _v2_safe_environment(
    environment: Mapping[str, str],
    launcher_parent: Path | None = None,
) -> dict[str, str]:
    safe: dict[str, str] = {}
    for key, value in environment.items():
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        if key in _V2_SAFE_ENV_KEYS or key.startswith("XDG_"):
            safe[key] = value
    if launcher_parent is not None:
        private_bin = str(launcher_parent)
        existing_path = safe.get("PATH", "")
        entries = [private_bin]
        entries.extend(
            entry for entry in existing_path.split(os.pathsep)
            if entry and entry != private_bin
        )
        safe["PATH"] = os.pathsep.join(entries)
    safe.update({
        "SEMGREP_ENABLE_VERSION_CHECK": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    return safe


def _v2_environment_material(environment: Mapping[str, str]) -> list[dict[str, str]]:
    return [
        {"key": key, "value_sha256": _sha256_text(value)}
        for key, value in sorted(environment.items())
        if isinstance(key, str) and isinstance(value, str)
    ]


def _run_bounded_v2(
    argv: list[str],
    *,
    environment: Mapping[str, str],
    timeout: float,
    runner: Callable[..., object],
    cwd: Path | None = None,
    extra_args: tuple[str, ...] = (),
) -> _Capture:
    actual_argv = [*argv, *extra_args] if argv else []
    if not actual_argv:
        return _Capture(return_code=None, timed_out=False, stdout=b"", stderr=b"")
    kwargs: dict[str, object] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": False,
        "env": dict(environment),
        "timeout": timeout,
        "check": False,
    }
    if cwd is not None:
        kwargs["cwd"] = str(cwd)
    try:
        completed = runner(actual_argv, **kwargs)
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


def _v2_parse_probe(capture: _Capture) -> Mapping[str, object] | None:
    if capture.timed_out or capture.return_code != 0:
        return None
    try:
        payload = json.loads(_decode_bounded(capture.stdout).strip())
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, Mapping) else None


def _v2_merge_interpreter_identity(
    target: dict[str, object],
    source: Mapping[str, object],
) -> None:
    for key in target:
        value = source.get(key)
        if key in {"present", "virtual_environment_active", "pyvenv_cfg_present"}:
            if isinstance(value, bool):
                target[key] = value
        elif key in {
            "resolved_sha256", "sys_executable_sha256", "sys_prefix_sha256",
            "sys_base_prefix_sha256", "pyvenv_cfg_sha256",
        }:
            if value is None or (isinstance(value, str) and _V2_HEX.fullmatch(value)):
                target[key] = value
        elif key in {"implementation", "version"}:
            if value is None or (isinstance(value, str) and _V2_TOKEN.fullmatch(value)):
                target[key] = value


def _v2_merge_distribution_identity(
    target: dict[str, object],
    source: Mapping[str, object],
) -> None:
    for key in ("version", "entrypoint_identifier"):
        value = source.get(key)
        if value is None or (isinstance(value, str) and _V2_TOKEN.fullmatch(value)):
            target[key] = value
    if isinstance(source.get("importable"), bool):
        target["importable"] = source["importable"]
    for key in (
        "metadata_sha256", "record_sha256", "entry_points_sha256",
        "package_tree_manifest_sha256", "python_distribution_manifest_sha256",
    ):
        value = source.get(key)
        if value is None or (isinstance(value, str) and _V2_HEX.fullmatch(value)):
            target[key] = value
    for key in (
        "record_entry_count", "record_verified_count", "record_missing_count",
        "record_hash_mismatch_count",
    ):
        value = source.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            target[key] = value


def _v2_native_runtime(
    launcher: VerifiedSemgrepLauncher | None,
) -> dict[str, object]:
    if launcher is None:
        return {
            "semgrep_core_sha256": None,
            "dependency_closure_sha256": None,
            "failure_class": "semgrep_runtime_dependency_closure_invalid",
        }
    try:
        closure = inspect_dependency_closure(launcher)
    except Exception:  # noqa: BLE001 - native identity fails closed
        closure = {}
    return {
        "semgrep_core_sha256": closure.get("semgrep_core_sha256"),
        "dependency_closure_sha256": closure.get("dependency_closure_sha256"),
        "failure_class": closure.get("failure_class"),
    }


def _v2_outcome_return_code(row: Mapping[str, object]) -> int | None:
    for key in ("return_code", "exit_code", "rc", "version_probe_return_code"):
        value = row.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    health = row.get("health")
    if isinstance(health, Mapping):
        return _v2_outcome_return_code(health)
    return None


def _v2_outcome_timed_out(row: Mapping[str, object]) -> bool:
    for key in ("timed_out", "version_probe_timed_out"):
        value = row.get(key)
        if isinstance(value, bool):
            return value
    health = row.get("health")
    if isinstance(health, Mapping):
        return _v2_outcome_timed_out(health)
    return False


def _v2_outcome_parsed_version(row: Mapping[str, object]) -> object:
    for key in ("parsed_version", "version"):
        if key in row:
            return row[key]
    health = row.get("health")
    if isinstance(health, Mapping):
        return _v2_outcome_parsed_version(health)
    return None


def _v2_outcome_exact(row: Mapping[str, object]) -> bool:
    return (
        _v2_outcome_return_code(row) == 0
        and not _v2_outcome_timed_out(row)
        and _v2_outcome_parsed_version(row) == EXPECTED_SEMGREP_VERSION
    )


def _v2_outcome_smoke_healthy(
    row: Mapping[str, object],
    context: Mapping[str, object] | None = None,
) -> bool:
    for key in ("smoke_healthy", "healthy"):
        value = row.get(key)
        if isinstance(value, bool):
            return value
    for key in ("smoke", "health"):
        nested = row.get(key)
        if isinstance(nested, Mapping):
            if isinstance(nested.get("healthy"), bool):
                return nested["healthy"]
            if key == "smoke" and nested.get("sarif_status") == "full_valid":
                return True
    if context is not None:
        nested = context.get("smoke")
        if isinstance(nested, Mapping):
            if isinstance(nested.get("healthy"), bool):
                return nested["healthy"]
            if nested.get("sarif_status") == "full_valid":
                return True
        elif isinstance(nested, bool):
            return nested
    return False


_V2_INTERPRETER_PROBE = r'''
import base64
import csv
import hashlib
import importlib
import importlib.metadata as metadata
import importlib.util
import io
import json
import os
import platform
import stat
from pathlib import Path, PurePosixPath
import re
import sys

_MAX_FILE = 128 * 1024 * 1024


def _digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _read_file(path):
    try:
        stat_result = path.stat()
        if not stat.S_ISREG(stat_result.st_mode) or stat_result.st_size > _MAX_FILE:
            return None
        with path.open("rb") as handle:
            return handle.read()
    except OSError:
        return None


def _digest_file(path):
    value = _read_file(path)
    return _digest_bytes(value) if value is not None else None


def _trusted_roots():
    roots = []
    for raw in [sys.prefix, sys.base_prefix, os.getcwd(), *sys.path]:
        if not raw:
            continue
        try:
            root = Path(raw).resolve()
        except OSError:
            continue
        if root not in roots:
            roots.append(root)
    return roots


_ROOTS = _trusted_roots()


def _safe_resolved(path):
    try:
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            return None
        for root in _ROOTS:
            try:
                resolved.relative_to(root)
                return resolved
            except ValueError:
                continue
    except OSError:
        return None
    return None


def _path_digest(path):
    try:
        return _digest_bytes(str(path.resolve()).encode("utf-8"))
    except OSError:
        return None


def _parse_requirement(value):
    try:
        from packaging.requirements import Requirement
        requirement = Requirement(value)
        if requirement.marker is not None and not requirement.marker.evaluate():
            return None
        return requirement.name
    except Exception:
        match = re.match(r"\s*([A-Za-z0-9_.-]+)", value)
        return match.group(1) if match is not None else ""


def _normalized_name(value):
    return re.sub(r"[-_.]+", "-", value).lower()


def _record_for_distribution(dist):
    root = Path(dist._path).resolve()
    record = root / "RECORD"
    raw = _read_file(record)
    return root, record, raw


def _inside(root, path):
    try:
        path.resolve().relative_to(root)
        return True
    except (OSError, ValueError):
        return False


def _verify_record(dist, root, raw):
    result = {"entry_count": 0, "verified_count": 0, "missing_count": 0, "mismatch_count": 0, "valid": True}
    if raw is None:
        result["valid"] = False
        return result
    try:
        rows = list(csv.reader(io.StringIO(raw.decode("utf-8", errors="strict"))))
    except (UnicodeDecodeError, csv.Error):
        result["valid"] = False
        result["mismatch_count"] = 1
        return result
    result["entry_count"] = len(rows)
    for row in rows:
        if len(row) < 3:
            result["mismatch_count"] += 1
            continue
        relative, encoded, size_text = row[0], row[1], row[2]
        if not relative or Path(relative).is_absolute() or "\\" in relative:
            result["mismatch_count"] += 1
            continue
        candidate = _safe_resolved(root.parent / PurePosixPath(relative))
        if candidate is None:
            result["missing_count"] += 1
            continue
        if not encoded:
            continue
        try:
            algorithm, value = encoded.split("=", 1)
            if algorithm not in ("sha256", "sha384", "sha512"):
                raise ValueError
            padding = "=" * ((4 - len(value) % 4) % 4)
            expected = base64.urlsafe_b64decode((value + padding).encode("ascii"))
            actual_bytes = _read_file(candidate)
            if actual_bytes is None:
                raise OSError
            actual = hashlib.new(algorithm, actual_bytes).digest()
            if actual != expected or (size_text and int(size_text) != len(actual_bytes)):
                result["mismatch_count"] += 1
            else:
                result["verified_count"] += 1
        except (ValueError, TypeError, OSError, UnicodeError):
            result["mismatch_count"] += 1
    result["valid"] = result["missing_count"] == 0 and result["mismatch_count"] == 0
    return result


def _tree_files(dist, root):
    files = set()
    package_roots = set()
    for entry in dist.files or []:
        relative = PurePosixPath(str(entry))
        if not relative.parts or relative.parts[0].endswith((".dist-info", ".egg-info")):
            continue
        first = relative.parts[0]
        if first in (".", ".."):
            # Prefix/bin scripts are individual files, never tree roots.
            candidate = _safe_resolved(root.parent / relative)
            if candidate is not None:
                files.add(candidate)
            continue
        package_roots.add(root.parent / first)
    for package in package_roots:
        if package.is_dir():
            candidates = package.rglob("*")
        else:
            candidates = (package,)
        for child in candidates:
            if child.is_dir():
                continue
            resolved = _safe_resolved(child)
            if resolved is not None:
                files.add(resolved)
            elif child.exists() or child.is_symlink():
                raise ValueError("unreadable package-tree entry")
    return files


def _distribution_record(dist):
    root, record_path, record_raw = _record_for_distribution(dist)
    metadata_raw = _read_file(root / "METADATA")
    entry_points_raw = _read_file(root / "entry_points.txt")
    verification = _verify_record(dist, root, record_raw)
    files = []
    package_files = []
    for path in sorted(_tree_files(dist, root), key=lambda item: str(item)):
        digest = _digest_file(path)
        if digest is None:
            verification["valid"] = False
            continue
        try:
            relative = path.relative_to(root.parent).as_posix()
        except ValueError:
            relative = path.name
        row = {"path": relative, "sha256": digest}
        files.append(row)
        try:
            package_relative = path.relative_to(root.parent)
        except ValueError:
            continue
        if package_relative.parts and not package_relative.parts[0].endswith((".dist-info", ".egg-info")):
            package_files.append(row)
    return {
        "name": _normalized_name(dist.metadata.get("Name") or dist.name),
        "version": dist.version,
        "metadata_sha256": _digest_bytes(metadata_raw) if metadata_raw is not None else None,
        "record_sha256": _digest_bytes(record_raw) if record_raw is not None else None,
        "entry_points_sha256": _digest_bytes(entry_points_raw) if entry_points_raw is not None else None,
        "record": verification,
        "files": files,
        "package_files": package_files,
        "requires": list(dist.requires or []),
    }


def _entrypoint_identifier(dist):
    try:
        choices = [
            entry for entry in dist.entry_points
            if entry.group == "console_scripts" and entry.name == "semgrep"
        ]
        if not choices:
            return None
        loaded = choices[0].load()
        if not callable(loaded):
            return None
    except Exception:
        return None
    value = choices[0].value
    identifier = "semgrep=" + value
    return identifier if re.fullmatch(r"[A-Za-z0-9_.+:\[\],=-]+", identifier) else None


probe_ok = True
interpreter_path = _safe_resolved(Path(sys.executable))
interpreter_digest = _digest_file(interpreter_path) if interpreter_path is not None else None
prefix_path = Path(sys.prefix)
base_prefix_path = Path(sys.base_prefix)
cfg_path = prefix_path / "pyvenv.cfg"
cfg_resolved = _safe_resolved(cfg_path)
cfg_digest = _digest_file(cfg_resolved) if cfg_resolved is not None else None
interpreter_identity = {
    "present": interpreter_digest is not None,
    "resolved_sha256": interpreter_digest,
    "implementation": platform.python_implementation(),
    "version": platform.python_version(),
    "sys_executable_sha256": _digest_bytes(str(sys.executable).encode("utf-8")),
    "sys_prefix_sha256": _path_digest(prefix_path),
    "sys_base_prefix_sha256": _path_digest(base_prefix_path),
    "virtual_environment_active": str(prefix_path.resolve()) != str(base_prefix_path.resolve()),
    "pyvenv_cfg_present": cfg_digest is not None,
    "pyvenv_cfg_sha256": cfg_digest,
}
if not interpreter_identity["present"]:
    probe_ok = False

distribution_identity = {
    "version": None,
    "importable": False,
    "entrypoint_identifier": None,
    "metadata_sha256": None,
    "record_sha256": None,
    "entry_points_sha256": None,
    "record_entry_count": 0,
    "record_verified_count": 0,
    "record_missing_count": 0,
    "record_hash_mismatch_count": 0,
    "package_tree_manifest_sha256": None,
    "python_distribution_manifest_sha256": None,
}
try:
    spec = importlib.util.find_spec("semgrep")
    if spec is None:
        probe_ok = False
    else:
        import semgrep
        semgrep_bin = importlib.import_module("semgrep.bin")
        distribution = metadata.distribution("semgrep")
        semgrep_root = Path(distribution._path).resolve().parent
        package_origin = _safe_resolved(Path(getattr(semgrep, "__file__", "")))
        package_location_ok = package_origin is not None and _inside(semgrep_root, package_origin)
        bin_location_ok = False
        for raw_bin in getattr(semgrep_bin, "__path__", ()):
            try:
                bin_path = Path(raw_bin).resolve(strict=True)
                if bin_path.is_dir() and _inside(semgrep_root, bin_path):
                    bin_location_ok = True
                    break
            except OSError:
                continue
        semgrep_record = _distribution_record(distribution)
        verification = semgrep_record["record"]
        entrypoint_identifier = _entrypoint_identifier(distribution)
        distribution_identity.update({
            "version": distribution.version,
            "importable": bool(package_location_ok and bin_location_ok),
            "entrypoint_identifier": entrypoint_identifier,
            "metadata_sha256": semgrep_record["metadata_sha256"],
            "record_sha256": semgrep_record["record_sha256"],
            "entry_points_sha256": semgrep_record["entry_points_sha256"],
            "record_entry_count": verification["entry_count"],
            "record_verified_count": verification["verified_count"],
            "record_missing_count": verification["missing_count"],
            "record_hash_mismatch_count": verification["mismatch_count"],
            "package_tree_manifest_sha256": _digest_bytes(
                json.dumps(
                    semgrep_record["package_files"],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ),
        })
        distributions = {_normalized_name(distribution.name): distribution}
        queue = [distribution]
        closure_valid = bool(
            package_location_ok
            and bin_location_ok
            and entrypoint_identifier is not None
            and semgrep_record["record"]["valid"]
            and semgrep_record["metadata_sha256"] is not None
            and semgrep_record["record_sha256"] is not None
            and semgrep_record["entry_points_sha256"] is not None
        )
        while queue:
            current = queue.pop(0)
            for requirement in current.requires or []:
                dependency_name = _parse_requirement(requirement)
                if dependency_name is None:
                    continue
                if not dependency_name:
                    closure_valid = False
                    continue
                normalized = _normalized_name(dependency_name)
                if normalized in distributions:
                    continue
                try:
                    dependency = metadata.distribution(dependency_name)
                except metadata.PackageNotFoundError:
                    closure_valid = False
                    continue
                distributions[normalized] = dependency
                queue.append(dependency)
        rows = []
        for name in sorted(distributions):
            record = _distribution_record(distributions[name])
            rows.append(record)
            if (
                not record["record"]["valid"]
                or record["metadata_sha256"] is None
                or record["record_sha256"] is None
                or not record["files"]
            ):
                closure_valid = False
        manifest = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
        distribution_identity["python_distribution_manifest_sha256"] = _digest_bytes(manifest)
        if not closure_valid:
            probe_ok = False
except Exception:
    probe_ok = False

print(json.dumps({
    "probe_ok": probe_ok,
    "interpreter": interpreter_identity,
    "distribution": distribution_identity,
}, sort_keys=True, separators=(",", ":")))
'''
