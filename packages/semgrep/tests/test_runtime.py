"""Focused offline contracts for governed Semgrep runtime health."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import venv

import pytest

from packages.semgrep import runtime
from packages.semgrep.runtime import (
    EXPECTED_SEMGREP_VERSION,
    SemgrepRuntimeError,
    classify_dynamic_loader_failure,
    classify_runtime_diagnosis,
    classify_semgrep_process_failure,
    collect_runtime_health,
    collect_runtime_identity_v2,
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
    environments: list[dict[str, str]] = []

    def runner(argv, **kwargs):
        environments.append(dict(kwargs["env"]))
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
    assert len(environments) == 2
    expected_bin = str(launcher.lexical_path.parent)
    assert environments[0]["PATH"].split(os.pathsep)[0] == expected_bin
    assert environments[1]["PATH"].split(os.pathsep)[0] == expected_bin
    assert environments[0]["SEMGREP_ENABLE_VERSION_CHECK"] == "0"
    assert environments[1]["SEMGREP_ENABLE_VERSION_CHECK"] == "0"


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
            install_names = {
                library_a: "/build/lib/liba.dylib",
                library_b: "/build/lib/libb.dylib",
            }
            rows = [f"{binary}:"]
            if binary.suffix == ".dylib":
                rows.append(
                    f"\t{install_names[binary]} (compatibility version 1.0.0)"
                )
            rows.extend(
                f"\t{dependency} (compatibility version 1.0.0)"
                for dependency in dependencies
            )
            output = "\n".join(rows)
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

def test_darwin_executable_path_is_anchored_at_core(tmp_path: Path) -> None:
    core = tmp_path / "bin" / "semgrep-core"
    nested = tmp_path / "nested" / "libnested.dylib"
    target = tmp_path / "bin" / "runtime-libs" / "libdep.dylib"
    core.parent.mkdir()
    nested.parent.mkdir()
    target.parent.mkdir()
    for path in (core, nested, target):
        path.write_bytes(path.name.encode("utf-8"))

    assert runtime._resolve_macho_token(
        nested,
        "@executable_path/runtime-libs/libdep.dylib",
        executable_path=core,
    ) == target.resolve()


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
            rows = [f"{binary}:"]
            if binary.suffix == ".dylib":
                rows.append("\t/build/lib/liba.dylib (compatibility version 1.0.0)")
            rows.extend(
                f"\t{dependency} (compatibility version 1.0.0)"
                for dependency in dependencies
            )
            output = "\n".join(rows)
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


def _tighten_private_tree(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_symlink():
            continue
        if path.is_dir():
            path.chmod(0o700)
        else:
            path.chmod(0o600)
    root.chmod(0o700)


def _write_private_text(path: Path, content: str, *, mode: int = 0o600) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    path.write_text(content, encoding="utf-8")
    path.chmod(mode)


def _record_digest(data: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest())
    return "sha256=" + encoded.decode("ascii").rstrip("=")


def _rewrite_synthetic_record(distribution_root: Path) -> Path:
    dist_info = distribution_root / "semgrep-1.174.0.dist-info"
    record = dist_info / "RECORD"
    rows: list[str] = []
    for path in sorted(distribution_root.rglob("*")):
        if not path.is_file() or path == record:
            continue
        relative = path.relative_to(distribution_root).as_posix()
        data = path.read_bytes()
        rows.append(f"{relative},{_record_digest(data)},{len(data)}")

    runtime_root = distribution_root.parents[2]
    launcher_candidates = (
        runtime_root / "bin" / "semgrep",
        runtime_root / "Scripts" / "semgrep",
    )
    launcher = next((path for path in launcher_candidates if path.is_file()), None)
    if launcher is not None:
        relative = os.path.relpath(launcher, distribution_root).replace(os.sep, "/")
        data = launcher.read_bytes()
        rows.append(f"{relative},{_record_digest(data)},{len(data)}")

    rows.append(f"{record.relative_to(distribution_root).as_posix()},,")
    _write_private_text(record, "\n".join(rows) + "\n")
    return record


def _synthetic_runtime(tmp_path: Path) -> dict[str, object]:
    workspace = _private_directory(tmp_path / "workspace")
    root = tmp_path / "synthetic-venv"
    venv.EnvBuilder(with_pip=False, clear=True, symlinks=False).create(root)
    _tighten_private_tree(root)

    if os.name == "nt":
        interpreter = root / "Scripts" / "python.exe"
        site_packages = root / "Lib" / "site-packages"
        bin_dir = root / "Scripts"
    else:
        interpreter = root / "bin" / "python"
        site_packages = next(root.glob("lib/python*/site-packages"))
        bin_dir = root / "bin"
    assert interpreter.is_file()
    assert not interpreter.is_symlink()

    package = site_packages / "semgrep"
    _write_private_text(
        package / "__init__.py",
        "import sys\n"
        "sys.dont_write_bytecode = True\n"
        "__version__ = '1.174.0'\n"
        "",
    )
    _write_private_text(package / "bin" / "__init__.py", "")
    _write_private_text(package / "console_scripts" / "__init__.py", "")
    _write_private_text(
        package / "console_scripts" / "entrypoint.py",
        "def main():\n"
        "    return 0\n",
    )
    dist_info = site_packages / "semgrep-1.174.0.dist-info"
    _write_private_text(
        dist_info / "METADATA",
        "Metadata-Version: 2.1\nName: semgrep\nVersion: 1.174.0\n",
    )
    _write_private_text(
        dist_info / "entry_points.txt",
        "[console_scripts]\n"
        "semgrep = semgrep.console_scripts.entrypoint:main\n",
    )
    _write_private_text(dist_info / "top_level.txt", "semgrep\n")

    native_core = package / "bin" / "semgrep-core"
    native_core.write_bytes(b"synthetic semgrep-core\n")
    native_core.chmod(0o700)
    record = _rewrite_synthetic_record(site_packages)

    launcher = bin_dir / "semgrep"
    _write_private_text(
        launcher,
        f"#!{interpreter}\n"
        "import importlib.metadata\n"
        "import sys\n"
        "if sys.argv[1:] == ['--version']:\n"
        "    print(importlib.metadata.version('semgrep'))\n"
        "else:\n"
        f"    print({_VALID_SARIF!r})\n",
        mode=0o700,
    )
    interpreter.chmod(0o700)
    launcher.chmod(0o700)
    record = _rewrite_synthetic_record(site_packages)
    return {
        "root": root,
        "workspace": workspace,
        "site_packages": site_packages,
        "launcher": verify_explicit_launcher(launcher),
        "interpreter": interpreter,
        "record": record,
        "module": package / "console_scripts" / "entrypoint.py",
        "core": native_core,
    }


def _synthetic_sarif_validator(payload: str) -> str:
    return "full_valid" if payload == _VALID_SARIF else "invalid"


def _synthetic_engine_runner(_argv, **_kwargs):
    return _completed(0, stdout=_VALID_SARIF.encode("utf-8"), stderr=b"")


def _patch_synthetic_native(
    monkeypatch: pytest.MonkeyPatch, fixture: dict[str, object],
) -> None:
    core = fixture["core"]
    assert isinstance(core, Path)
    closure = {
        "dependency_closure_sha256": hashlib.sha256(
            b"synthetic-native-closure",
        ).hexdigest(),
        "dependency_manifest": [],
        "semgrep_core_sha256": runtime._sha256_file(core),
        "failure_class": None,
    }
    monkeypatch.setattr(
        runtime,
        "inspect_dependency_closure",
        lambda _launcher: dict(closure),
    )


def _collect_v2(
    fixture: dict[str, object],
    *,
    runner=subprocess.run,
) -> dict[str, object]:
    workspace = fixture["workspace"]
    launcher = fixture["launcher"]
    assert isinstance(workspace, Path)
    return collect_runtime_identity_v2(
        launcher,
        environment={},
        engine_runner=_synthetic_engine_runner,
        runner=runner,
        sarif_validator=_synthetic_sarif_validator,
        workspace_root=workspace,
    )


def _collect_v1(fixture: dict[str, object]) -> dict[str, object]:
    def runner(_argv, **_kwargs):
        return _completed(0, stdout=f"{EXPECTED_SEMGREP_VERSION}\n")

    return collect_runtime_health(
        fixture["launcher"],
        environment={},
        runner=runner,
        engine_runner=lambda _argv, **_kwargs: _completed(
            0, stdout=_VALID_SARIF,
        ),
        sarif_validator=lambda _payload: "full_valid",
    )


@pytest.fixture
def synthetic_runtime(tmp_path: Path) -> dict[str, object]:
    return _synthetic_runtime(tmp_path)


def test_v1_identity_ignores_changed_shebang_interpreter_bytes(
    synthetic_runtime: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_synthetic_native(monkeypatch, synthetic_runtime)
    interpreter = synthetic_runtime["interpreter"]
    assert isinstance(interpreter, Path)

    before = _collect_v1(synthetic_runtime)
    interpreter.write_bytes(interpreter.read_bytes() + b"\nfixture-byte-change\n")
    interpreter.chmod(0o700)
    after = _collect_v1(synthetic_runtime)

    assert before == after
    assert before["resolved_executable_sha256"] == after["resolved_executable_sha256"]
    assert before["semgrep_core_sha256"] == after["semgrep_core_sha256"]


def test_v2_identity_detects_changed_shebang_interpreter_bytes(
    synthetic_runtime: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_synthetic_native(monkeypatch, synthetic_runtime)
    interpreter = synthetic_runtime["interpreter"]
    assert isinstance(interpreter, Path)

    before = _collect_v2(synthetic_runtime)
    interpreter.write_bytes(interpreter.read_bytes() + b"\nfixture-byte-change\n")
    interpreter.chmod(0o700)
    after = _collect_v2(synthetic_runtime)

    assert before["launcher"] == after["launcher"]
    assert before["native_runtime"] == after["native_runtime"]
    assert before["interpreter"]["resolved_sha256"] != after["interpreter"]["resolved_sha256"]
    assert before["health"]["healthy"] is True


def test_v2_missing_shebang_interpreter_is_unhealthy(
    synthetic_runtime: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_synthetic_native(monkeypatch, synthetic_runtime)
    interpreter = synthetic_runtime["interpreter"]
    assert isinstance(interpreter, Path)
    interpreter.unlink()

    identity = _collect_v2(synthetic_runtime)

    assert identity["interpreter"]["present"] is False
    assert identity["health"]["healthy"] is False
def test_v2_symlinked_interpreter_retains_virtual_environment_prefix(
    synthetic_runtime: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_synthetic_native(monkeypatch, synthetic_runtime)
    direct = _collect_v2(synthetic_runtime)
    root = synthetic_runtime["root"]
    site_packages = synthetic_runtime["site_packages"]
    interpreter = synthetic_runtime["interpreter"]
    launcher = synthetic_runtime["launcher"]
    assert isinstance(root, Path)
    assert isinstance(site_packages, Path)
    assert isinstance(interpreter, Path)
    assert isinstance(launcher, runtime.VerifiedSemgrepLauncher)

    alias = root / ("Scripts" if os.name == "nt" else "bin") / "python-alias"
    alias.symlink_to(interpreter)
    launcher_path = launcher.lexical_path
    source = launcher_path.read_text(encoding="utf-8")
    source = source.replace(f"#!{interpreter}\n", f"#!{alias}\n", 1)
    launcher_path.write_text(source, encoding="utf-8")
    launcher_path.chmod(0o700)
    _rewrite_synthetic_record(site_packages)
    synthetic_runtime["launcher"] = verify_explicit_launcher(launcher_path)

    aliased = _collect_v2(synthetic_runtime)
    direct_interpreter = direct["interpreter"]
    aliased_interpreter = aliased["interpreter"]
    assert aliased_interpreter["virtual_environment_active"] is True
    assert aliased_interpreter["sys_prefix_sha256"] == direct_interpreter["sys_prefix_sha256"]
    assert aliased_interpreter["sys_base_prefix_sha256"] == direct_interpreter["sys_base_prefix_sha256"]


def test_v2_pyvenv_cfg_changes_and_missing_file_change_identity(
    synthetic_runtime: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_synthetic_native(monkeypatch, synthetic_runtime)
    root = synthetic_runtime["root"]
    assert isinstance(root, Path)
    cfg = root / "pyvenv.cfg"

    before = _collect_v2(synthetic_runtime)
    cfg.write_text(cfg.read_text(encoding="utf-8") + "# fixture change\n", encoding="utf-8")
    changed = _collect_v2(synthetic_runtime)
    cfg.unlink()
    missing = _collect_v2(synthetic_runtime)

    assert before["interpreter"]["pyvenv_cfg_present"] is True
    assert changed["interpreter"]["pyvenv_cfg_present"] is True
    assert before["interpreter"]["pyvenv_cfg_sha256"] != changed["interpreter"]["pyvenv_cfg_sha256"]
    assert missing["interpreter"]["pyvenv_cfg_present"] is False
    assert missing["interpreter"]["pyvenv_cfg_sha256"] is None
    assert changed["interpreter"] != missing["interpreter"]


def test_v2_missing_semgrep_record_file_is_unhealthy(
    synthetic_runtime: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_synthetic_native(monkeypatch, synthetic_runtime)
    module = synthetic_runtime["module"]
    assert isinstance(module, Path)
    module.unlink()

    identity = _collect_v2(synthetic_runtime)
    distribution = identity["semgrep_distribution"]

    assert distribution["record_missing_count"] > 0
    assert distribution["record_hash_mismatch_count"] == 0
    assert identity["health"]["healthy"] is False


def test_v2_semgrep_record_hash_mismatch_is_unhealthy(
    synthetic_runtime: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_synthetic_native(monkeypatch, synthetic_runtime)
    module = synthetic_runtime["module"]
    assert isinstance(module, Path)
    module.write_text("VALUE = 'record-mismatch'\n", encoding="utf-8")
    module.chmod(0o600)

    identity = _collect_v2(synthetic_runtime)
    distribution = identity["semgrep_distribution"]

    assert distribution["record_missing_count"] == 0
    assert distribution["record_hash_mismatch_count"] > 0
    assert identity["health"]["healthy"] is False


def test_v2_semgrep_module_change_updates_identity_without_native_change(
    synthetic_runtime: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_synthetic_native(monkeypatch, synthetic_runtime)
    module = synthetic_runtime["module"]
    site_packages = synthetic_runtime["site_packages"]
    assert isinstance(module, Path)
    assert isinstance(site_packages, Path)

    before = _collect_v2(synthetic_runtime)
    module.write_text(
        "def main():\n"
        "    return 0\n\n"
        "VALUE = 'synthetic-module-v2'\n",
        encoding="utf-8",
    )
    module.chmod(0o600)
    _rewrite_synthetic_record(site_packages)
    after = _collect_v2(synthetic_runtime)

    assert before["launcher"] == after["launcher"]
    assert before["native_runtime"] == after["native_runtime"]
    assert before["semgrep_distribution"]["metadata_sha256"] == after["semgrep_distribution"]["metadata_sha256"]
    assert before["semgrep_distribution"]["package_tree_manifest_sha256"] != after["semgrep_distribution"]["package_tree_manifest_sha256"]
    assert after["semgrep_distribution"]["record_hash_mismatch_count"] == 0
    assert after["health"]["healthy"] is True


def test_v2_healthy_synthetic_semgrep_1_174_0_passes(
    synthetic_runtime: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_synthetic_native(monkeypatch, synthetic_runtime)

    identity = _collect_v2(synthetic_runtime)
    distribution = identity["semgrep_distribution"]
    interpreter = identity["interpreter"]
    interpreter_path = synthetic_runtime["interpreter"]
    native = identity["native_runtime"]
    health = identity["health"]
    assert isinstance(interpreter_path, Path)

    assert identity["identity_schema_version"] == 2
    assert distribution["version"] == EXPECTED_SEMGREP_VERSION
    assert distribution["importable"] is True
    assert distribution["record_missing_count"] == 0
    assert distribution["record_hash_mismatch_count"] == 0
    assert interpreter["present"] is True
    assert interpreter["implementation"] == "CPython"
    assert interpreter["sys_executable_sha256"] == hashlib.sha256(
        str(interpreter_path).encode("utf-8"),
    ).hexdigest()
    assert native["semgrep_core_sha256"] is not None
    assert native["dependency_closure_sha256"] is not None
    assert health["parsed_version"] == EXPECTED_SEMGREP_VERSION
    assert health["version_probe_return_code"] == 0
    assert health["engine_smoke_return_code"] == 0
    assert health["engine_smoke_sarif_status"] == "full_valid"
    assert health["healthy"] is True
    assert identity["runtime_environment_contract_sha256"]


def test_v2_runtime_diagnostics_redact_paths_secrets_and_cookies(
    synthetic_runtime: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_synthetic_native(monkeypatch, synthetic_runtime)
    hostile = (
        b"/Users/alice/project /home/alice/project /private/tmp/fixture "
        b"/var/folders/xx/token API_KEY=sk-live-secret "
        b"Bearer super-secret Authorization: Basic abc123 "
        b"Cookie: session=private-cookie"
    )

    def hostile_runner(argv, **kwargs):
        if argv[-1] == "--version":
            return _completed(3, stdout=b"", stderr=hostile)
        return subprocess.run(argv, **kwargs)

    identity = _collect_v2(synthetic_runtime, runner=hostile_runner)
    rendered = json.dumps(identity, sort_keys=True)

    for token in (
        "/Users",
        "/home",
        "/private",
        "/var/folders",
        "sk-live-secret",
        "Bearer",
        "Authorization",
        "Cookie",
        "private-cookie",
    ):
        assert token not in rendered


def test_v2_rc3_unexpected_cli_exit_label_does_not_prove_root_cause(
    synthetic_runtime: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_synthetic_native(monkeypatch, synthetic_runtime)

    def rc3_runner(argv, **kwargs):
        if argv[-1] == "--version":
            return _completed(
                3,
                stdout=b"",
                stderr=b"unexpected CLI exit while parsing target",
            )
        return subprocess.run(argv, **kwargs)

    identity = _collect_v2(synthetic_runtime, runner=rc3_runner)
    health = identity["health"]

    assert health["version_probe_return_code"] == 3
    assert health["version_probe_failure_category"] == "unexpected_cli_exit"
    assert health["version_probe_cli_exit_label"] == "target_parse_failure"
    assert health["version_probe_root_cause_proven"] is False
    assert health["healthy"] is False
    redacted_tail = health.get("version_probe_redacted_stderr_tail")
    if redacted_tail is not None:
        assert isinstance(redacted_tail, str)
        assert len(redacted_tail) <= 512
        assert "unexpected CLI exit" not in redacted_tail


def test_runtime_diagnosis_minimal_pass_current_fail_is_qualification_environment_failed() -> None:
    profiles = {
        "current_safe": {
            "return_code": 3,
            "timed_out": False,
            "parsed_version": None,
        },
        "minimal_private": {
            "return_code": 0,
            "timed_out": False,
            "parsed_version": EXPECTED_SEMGREP_VERSION,
        },
    }

    diagnosis = classify_runtime_diagnosis(profiles)

    assert diagnosis["category"] == "qualification_environment_failed"
    assert diagnosis["root_cause_proven"] is False


def test_runtime_diagnosis_both_pass_after_historical_failure_is_nondeterministic() -> None:
    profiles = {
        "current_safe": {
            "return_code": 0,
            "timed_out": False,
            "parsed_version": EXPECTED_SEMGREP_VERSION,
        },
        "minimal_private": {
            "return_code": 0,
            "timed_out": False,
            "parsed_version": EXPECTED_SEMGREP_VERSION,
        },
    }

    diagnosis = classify_runtime_diagnosis(profiles, prior_failure=True)

    assert diagnosis["category"] == "runtime_nondeterministic"
    assert diagnosis["root_cause_proven"] is False


def test_version_exit_alone_cannot_establish_runtime_diagnosis() -> None:
    diagnosis = classify_runtime_diagnosis({
        "current_safe": {
            "return_code": 3,
            "timed_out": False,
            "parsed_version": None,
        },
    })

    assert diagnosis["category"] == "diagnosis_incomplete"
    assert diagnosis["root_cause_proven"] is False


def test_v2_package_identity_covers_unlisted_bytecode(
    synthetic_runtime: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_synthetic_native(monkeypatch, synthetic_runtime)
    module = synthetic_runtime["module"]
    assert isinstance(module, Path)
    bytecode = module.parent / "cache_only.pyc"
    bytecode.write_bytes(b"first synthetic bytecode")
    before = _collect_v2(synthetic_runtime)
    bytecode.write_bytes(b"changed synthetic bytecode")
    after = _collect_v2(synthetic_runtime)

    assert before["launcher"] == after["launcher"]
    assert before["native_runtime"] == after["native_runtime"]
    assert (
        before["semgrep_distribution"]["package_tree_manifest_sha256"]
        != after["semgrep_distribution"]["package_tree_manifest_sha256"]
    )
