"""Focused offline contracts for governed Semgrep runtime health."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from packages.semgrep import runtime
from packages.semgrep.runtime import (
    EXPECTED_SEMGREP_VERSION,
    SemgrepRuntimeError,
    classify_dynamic_loader_failure,
    classify_semgrep_process_failure,
    collect_runtime_health,
    inspect_dependency_closure,
    verify_explicit_launcher,
)


_VALID_SARIF = '{"version":"2.1.0","runs":[]}'
_HEALTHY_CLOSURE = {
    "dependency_closure_sha256": "a" * 64,
    "dependency_manifest": [],
    "semgrep_core_sha256": "b" * 64,
    "failure_class": None,
}


def _private_directory(path: Path) -> Path:
    path.mkdir(mode=0o700, exist_ok=True)
    path.chmod(0o700)
    return path


def _launcher(tmp_path: Path):
    root = _private_directory(tmp_path / "private-runtime")
    launcher = _private_directory(root / "bin") / "semgrep"
    launcher.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    launcher.chmod(0o700)
    return verify_explicit_launcher(launcher)


def _runtime_with_core(tmp_path: Path):
    launcher = _launcher(tmp_path)
    assert launcher.private_root is not None
    directory = launcher.private_root
    for part in ("lib", "python3.13", "site-packages", "semgrep", "bin"):
        directory = _private_directory(directory / part)
    core = directory / "semgrep-core"
    core.write_bytes(b"semgrep-core")
    core.chmod(0o700)
    libraries = _private_directory(launcher.private_root / "runtime-libs")
    return launcher, core, libraries


def _completed(returncode: int, stdout: str = "", stderr: str = ""):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def test_launcher_rejects_nonprivate_descendant(tmp_path: Path) -> None:
    root = _private_directory(tmp_path / "private-runtime")
    bin_dir = root / "bin"
    bin_dir.mkdir()
    bin_dir.chmod(0o755)
    launcher = bin_dir / "semgrep"
    launcher.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    launcher.chmod(0o700)

    with pytest.raises(SemgrepRuntimeError, match="ancestor is not private"):
        verify_explicit_launcher(launcher)


def test_launcher_owner_must_match_current_user() -> None:
    foreign_uid = runtime._current_uid() + 1
    with pytest.raises(SemgrepRuntimeError, match="current user"):
        runtime._require_current_user_owner(
            SimpleNamespace(st_uid=foreign_uid), "Semgrep launcher",
        )


def test_loader_abort_never_parses_competing_version_token(tmp_path: Path) -> None:
    """R6's dyld abort must fail before a scan can dispatch."""
    launcher = _launcher(tmp_path)
    calls: list[list[str]] = []

    def runner(argv, **_kwargs):
        calls.append(list(argv))
        return _completed(
            134,
            stdout=f"{EXPECTED_SEMGREP_VERSION}\n",
            stderr=(
                "dyld: Library not loaded: "
                "/private/runtime/libtree-sitter.0.26.dylib\n"
            ),
        )

    health = collect_runtime_health(
        launcher,
        environment={},
        runner=runner,
        engine_runner=runner,
        sarif_validator=lambda _payload: "full_valid",
    )

    assert calls == [[str(launcher.lexical_path), "--version"]]
    assert health["version_probe_return_code"] == 134
    assert health["version"] is None
    assert health["version_parse_source"] == "none"
    assert health["healthy"] is False
    assert health["failure_class"] == "semgrep_runtime_linker_dependency_missing"
    assert health["linker_family"] == "dyld"
    assert health["missing_library_basename"] == "libtree-sitter.0.26.dylib"
    assert health["engine_smoke_return_code"] is None


def test_healthy_exact_stdout_version_and_engine_smoke_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _launcher(tmp_path)
    monkeypatch.setattr(runtime, "inspect_dependency_closure", lambda _launcher: dict(_HEALTHY_CLOSURE))

    def runner(argv, **_kwargs):
        if argv[-1] == "--version":
            return _completed(0, stdout=f"{EXPECTED_SEMGREP_VERSION}\n")
        assert argv[1] == "scan"
        assert "--config" in argv
        assert "--sarif" in argv
        return _completed(0, stdout=_VALID_SARIF)

    health = collect_runtime_health(
        launcher,
        environment={},
        runner=runner,
        engine_runner=runner,
        sarif_validator=lambda payload: "full_valid" if payload == _VALID_SARIF else "invalid",
    )

    assert health["version"] == EXPECTED_SEMGREP_VERSION
    assert health["version_parse_source"] == "stdout"
    assert health["version_probe_return_code"] == 0
    assert health["engine_smoke_return_code"] == 0
    assert health["engine_smoke_timed_out"] is False
    assert health["engine_smoke_sarif_status"] == "full_valid"
    assert health["engine_smoke_raw_output_persisted"] is False
    assert health["healthy"] is True


def test_fixed_version_cannot_be_overridden(tmp_path: Path) -> None:
    launcher = _launcher(tmp_path)

    health = collect_runtime_health(
        launcher,
        environment={},
        runner=lambda _argv, **_kwargs: _completed(0, stdout="1.174.1\n"),
        engine_runner=lambda _argv, **_kwargs: _completed(0, stdout=_VALID_SARIF),
        sarif_validator=lambda _payload: "full_valid",
    )

    assert health["version"] == "1.174.1"
    assert health["healthy"] is False
    assert health["failure_class"] == "semgrep_runtime_version_probe_failed"


@pytest.mark.parametrize(
    ("returncode", "stdout", "stderr"),
    [
        (1, EXPECTED_SEMGREP_VERSION + "\n", ""),
        (1, "", EXPECTED_SEMGREP_VERSION + "\n"),
        (0, "", EXPECTED_SEMGREP_VERSION + "\n"),
    ],
)
def test_nonzero_or_stderr_version_tokens_are_never_accepted(
    tmp_path: Path,
    returncode: int,
    stdout: str,
    stderr: str,
) -> None:
    launcher = _launcher(tmp_path)
    health = collect_runtime_health(
        launcher,
        environment={},
        runner=lambda _argv, **_kwargs: _completed(returncode, stdout, stderr),
        engine_runner=lambda _argv, **_kwargs: _completed(0, stdout=_VALID_SARIF),
        sarif_validator=lambda _payload: "full_valid",
    )

    assert health["version"] is None
    assert health["version_parse_source"] == "none"
    assert health["healthy"] is False
    assert health["engine_smoke_return_code"] is None


def test_generic_abort_is_not_reclassified_as_sarif_invalid(tmp_path: Path) -> None:
    launcher = _launcher(tmp_path)
    health = collect_runtime_health(
        launcher,
        environment={},
        runner=lambda _argv, **_kwargs: _completed(134, stderr="fatal allocator abort"),
        engine_runner=lambda _argv, **_kwargs: _completed(0, stdout=_VALID_SARIF),
        sarif_validator=lambda _payload: "full_valid",
    )

    assert health["failure_class"] == "semgrep_runtime_process_aborted"
    assert health["missing_library_basename"] is None
    assert classify_semgrep_process_failure(134, "fatal allocator abort") == (
        "semgrep_runtime_process_aborted"
    )


@pytest.mark.parametrize(
    ("stderr", "family", "basename"),
    [
        (
            "dyld: Library not loaded: /opt/x/libtree-sitter.0.26.dylib\nimage not found",
            "dyld",
            "libtree-sitter.0.26.dylib",
        ),
        (
            "error while loading shared libraries: libtree-sitter.so.0: cannot open shared object file",
            "ld.so",
            "libtree-sitter.so.0",
        ),
        (
            "DLL load failed: The specified module could not be found: tree-sitter.dll",
            "windows_loader",
            "tree-sitter.dll",
        ),
    ],
)
def test_loader_signatures_produce_only_bounded_diagnostics(
    stderr: str,
    family: str,
    basename: str,
) -> None:
    assert classify_dynamic_loader_failure(stderr) == {
        "linker_family": family,
        "missing_library_basename": basename,
    }
    assert classify_semgrep_process_failure(134, stderr) == (
        "semgrep_runtime_linker_dependency_missing"
    )


def test_missing_semgrep_core_fails_dependency_closure(tmp_path: Path) -> None:
    closure = inspect_dependency_closure(_launcher(tmp_path))

    assert closure["failure_class"] == "semgrep_runtime_dependency_closure_invalid"
    assert closure["dependency_closure_sha256"] is None


def test_symlinked_semgrep_core_cannot_escape_private_runtime(tmp_path: Path) -> None:
    launcher, core, _libraries = _runtime_with_core(tmp_path)
    external = tmp_path / "outside-semgrep-core"
    external.write_bytes(b"outside")
    external.chmod(0o700)
    core.unlink()
    core.symlink_to(external)

    assert runtime._locate_semgrep_core(launcher) is None
    assert inspect_dependency_closure(launcher)["failure_class"] == (
        "semgrep_runtime_dependency_closure_invalid"
    )


def test_darwin_closure_recurses_rpaths_without_retaining_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher, core, libraries = _runtime_with_core(tmp_path)
    library_a = libraries / "liba.dylib"
    library_b = libraries / "libb.dylib"
    for library in (library_a, library_b):
        library.write_bytes(library.name.encode("utf-8"))
        library.chmod(0o700)

    def fake_otool(argv, **_kwargs):
        binary = Path(argv[-1])
        if argv[1] == "-L":
            dependencies = {
                core: ["@rpath/liba.dylib", "/usr/lib/libSystem.B.dylib"],
                library_a: ["@loader_path/libb.dylib"],
                library_b: [],
            }[binary]
            output = "\n".join([f"{binary}:", *[
                f"\t{dependency} (compatibility version 1.0.0)"
                for dependency in dependencies
            ]])
            return _completed(0, stdout=output)
        assert argv[1] == "-l"
        return _completed(
            0,
            stdout=(
                "Load command 1\n"
                "          cmd LC_RPATH\n"
                "      cmdsize 48\n"
                f"         path {libraries} (offset 12)\n"
            ),
        )

    monkeypatch.setattr(runtime.sys, "platform", "darwin")
    monkeypatch.setattr(runtime.subprocess, "run", fake_otool)

    closure = inspect_dependency_closure(launcher)

    assert closure["failure_class"] is None
    assert closure["semgrep_core_sha256"] is not None
    assert closure["dependency_closure_sha256"] is not None
    assert closure["dependency_manifest"] == [
        {"basename": "liba.dylib", "sha256": runtime._sha256_file(library_a)},
        {"basename": "libb.dylib", "sha256": runtime._sha256_file(library_b)},
    ]


def test_darwin_closure_rejects_missing_transitive_library(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher, core, libraries = _runtime_with_core(tmp_path)
    library = libraries / "liba.dylib"
    library.write_bytes(b"a")
    library.chmod(0o700)

    def fake_otool(argv, **_kwargs):
        binary = Path(argv[-1])
        if argv[1] == "-L":
            dependencies = {
                core: ["@rpath/liba.dylib"],
                library: ["@loader_path/missing.dylib"],
            }[binary]
            output = "\n".join([f"{binary}:", *[
                f"\t{dependency} (compatibility version 1.0.0)"
                for dependency in dependencies
            ]])
            return _completed(0, stdout=output)
        return _completed(
            0,
            stdout=(
                "Load command 1\n"
                "          cmd LC_RPATH\n"
                "      cmdsize 48\n"
                f"         path {libraries} (offset 12)\n"
            ),
        )

    monkeypatch.setattr(runtime.sys, "platform", "darwin")
    monkeypatch.setattr(runtime.subprocess, "run", fake_otool)

    closure = inspect_dependency_closure(launcher)

    assert closure["failure_class"] == "semgrep_runtime_dependency_closure_invalid"
