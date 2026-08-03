"""Exercise the launcher-to-worker credential boundary with a real child process."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import raptor
import raptor_agentic

class _CanaryDispatcher:
    def __init__(self, root: Path) -> None:
        self.root = root

    def allocate_worker(self, *, label: str) -> tuple[str, int]:
        del label
        read_fd, write_fd = os.pipe()
        try:
            os.write(write_fd, b"fixture-token")
        finally:
            os.close(write_fd)
        return str(self.root / "dispatcher.sock"), read_fd


def _canary_script(path: Path, report: Path) -> None:
    path.write_text(
        "import json\n"
        "import os\n"
        "from pathlib import Path\n"
        f"Path({str(report)!r}).write_text(json.dumps({{\n"
        "    'gemini_present': bool(os.environ.get('GEMINI_API_KEY')),\n"
        "    'unrelated_secret_present': bool(os.environ.get('UNRELATED_DISPATCH_SECRET')),\n"
        "    'python_user_base_present': bool(os.environ.get('PYTHONUSERBASE')),\n"
        "    'tmpdir': os.environ.get('TMPDIR'),\n"
        "    'dispatcher_socket_present': bool(os.environ.get('RAPTOR_LLM_SOCKET')),\n"
        "    'dispatcher_token_fd_present': bool(os.environ.get('RAPTOR_LLM_TOKEN_FD')),\n"
        "}))\n"
    )


def _agentic_canary_script(path: Path, report: Path) -> None:
    path.write_text(
        "import json\n"
        "import os\n"
        "from pathlib import Path\n"
        f"Path({str(report)!r}).write_text(json.dumps({{\n"
        "    'gemini_present': bool(os.environ.get('GEMINI_API_KEY')),\n"
        "    'unrelated_secret_present': bool(os.environ.get('UNRELATED_DISPATCH_SECRET')),\n"
        "    'home': os.environ.get('HOME'),\n"
        "    'xdg_config_home': os.environ.get('XDG_CONFIG_HOME'),\n"
        "    'raptor_config_present': bool(os.environ.get('RAPTOR_CONFIG')),\n"
        "    'tmpdir': os.environ.get('TMPDIR'),\n"
        "}))\n"
    )
def _sandboxed_target_canary_script(path: Path, report: Path) -> None:
    path.write_text(
        "import json\n"
        "import os\n"
        "import tempfile\n"
        "from pathlib import Path\n"
        f"report = Path({str(report)!r})\n"
        "temporary = Path(tempfile.mkdtemp(prefix='raptor-sandbox-canary-'))\n"
        "try:\n"
        "    report.write_text(json.dumps({\n"
        "        'gemini_present': bool(os.environ.get('GEMINI_API_KEY')),\n"
        "        'unrelated_secret_present': bool(os.environ.get('UNRELATED_DISPATCH_SECRET')),\n"
        "        'dispatcher_socket_present': bool(os.environ.get('RAPTOR_LLM_SOCKET')),\n"
        "        'dispatcher_token_fd_present': bool(os.environ.get('RAPTOR_LLM_TOKEN_FD')),\n"
        "        'raptor_config_present': bool(os.environ.get('RAPTOR_CONFIG')),\n"
        "        'isolation_mode': os.environ.get('RAPTOR_REQUIRE_CREDENTIAL_ISOLATION'),\n"
        "        'isolated_temp_marker': os.environ.get('RAPTOR_PRIVATE_TMPDIR'),\n"
        "        'tmpdir': os.environ.get('TMPDIR'),\n"
        "        'tmp': os.environ.get('TMP'),\n"
        "        'temp': os.environ.get('TEMP'),\n"
        "        'tempfile_parent': str(temporary.parent),\n"
        "    }))\n"
        "finally:\n"
        "    temporary.rmdir()\n"
    )




def test_dispatcher_worker_never_receives_provider_or_parent_secret(tmp_path, monkeypatch):
    report = tmp_path / "canary.json"
    child = tmp_path / "canary.py"
    _canary_script(child, report)
    monkeypatch.setenv("GEMINI_API_KEY", "fixture-gemini-key")
    monkeypatch.setenv("UNRELATED_DISPATCH_SECRET", "fixture-parent-secret")
    monkeypatch.setenv("RAPTOR_REQUIRE_CREDENTIAL_ISOLATION", "1")
    monkeypatch.setenv("PYTHONUSERBASE", "/fixture/user-base")
    private_temp = tmp_path / "private-temp"
    private_temp.mkdir(mode=0o700)
    private_temp.chmod(0o700)
    private_temp_text = str(private_temp.resolve())
    monkeypatch.setenv("RAPTOR_PRIVATE_TMPDIR", private_temp_text)
    monkeypatch.setenv("TMPDIR", private_temp_text)
    monkeypatch.setenv("TMP", private_temp_text)
    monkeypatch.setenv("TEMP", private_temp_text)
    monkeypatch.setattr(raptor, "_get_or_start_dispatcher", lambda: _CanaryDispatcher(tmp_path))

    assert raptor._run_script(child, []) == 0

    assert json.loads(report.read_text()) == {
        "gemini_present": False,
        "unrelated_secret_present": False,
        "python_user_base_present": False,
        "tmpdir": private_temp_text,
        "dispatcher_socket_present": True,
        "dispatcher_token_fd_present": True,
    }



def test_target_facing_agentic_child_keeps_private_home_without_provider_key(tmp_path, monkeypatch):
    report = tmp_path / "agentic-canary.json"
    child = tmp_path / "agentic-canary.py"
    private_home = tmp_path / "private-home"
    private_config = tmp_path / "private-config"
    private_home.mkdir()
    private_config.mkdir()
    private_temp = tmp_path / "private-temp"
    private_temp.mkdir(mode=0o700)
    private_temp.chmod(0o700)
    private_temp_text = str(private_temp.resolve())
    _agentic_canary_script(child, report)
    monkeypatch.setenv("HOME", str(private_home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(private_config))
    monkeypatch.setenv("RAPTOR_CONFIG", str(private_config / "models.json"))
    monkeypatch.setenv("GEMINI_API_KEY", "fixture-gemini-key")
    monkeypatch.setenv("UNRELATED_DISPATCH_SECRET", "fixture-parent-secret")
    monkeypatch.setenv("RAPTOR_REQUIRE_CREDENTIAL_ISOLATION", "1")
    monkeypatch.setenv("RAPTOR_PRIVATE_TMPDIR", private_temp_text)
    monkeypatch.setenv("TMPDIR", private_temp_text)
    monkeypatch.setenv("TMP", private_temp_text)
    monkeypatch.setenv("TEMP", private_temp_text)
    monkeypatch.delenv("RAPTOR_LLM_SOCKET", raising=False)
    monkeypatch.delenv("RAPTOR_LLM_TOKEN_FD", raising=False)

    returncode, _stdout, _stderr = raptor_agentic.run_command_streaming(
        [sys.executable, str(child)],
        "credential-isolation canary",
        timeout=10,
    )

    assert returncode == 0
    assert json.loads(report.read_text()) == {
        "gemini_present": False,
        "unrelated_secret_present": False,
        "home": str(private_home),
        "xdg_config_home": str(private_config),
        "raptor_config_present": False,
        "tmpdir": private_temp_text,
    }


def test_strict_sandboxed_target_child_keeps_validated_temp_and_no_credentials(
    tmp_path, monkeypatch, request
):
    private_temp_root = tempfile.TemporaryDirectory(
        prefix="raptor-isolated-temp-",
        dir="/tmp",
    )
    request.addfinalizer(private_temp_root.cleanup)
    private_temp = Path(private_temp_root.name).resolve()
    private_temp.chmod(0o700)
    private_temp_text = str(private_temp)
    target = tmp_path / "target"
    output = tmp_path / "output"
    target.mkdir()
    output.mkdir()
    report = output / "sandbox-canary.json"
    child = target / "sandbox-canary.py"
    _sandboxed_target_canary_script(child, report)

    monkeypatch.setenv("GEMINI_API_KEY", "fixture-gemini-key")
    monkeypatch.setenv("UNRELATED_DISPATCH_SECRET", "fixture-parent-secret")
    monkeypatch.setenv("RAPTOR_REQUIRE_CREDENTIAL_ISOLATION", "1")
    monkeypatch.setenv("RAPTOR_PRIVATE_TMPDIR", private_temp_text)
    monkeypatch.setenv("TMPDIR", private_temp_text)
    monkeypatch.setenv("TMP", private_temp_text)
    monkeypatch.setenv("TEMP", private_temp_text)

    from core.config import RaptorConfig
    from core.sandbox import (
        check_mount_available,
        check_net_available,
        check_seatbelt_available,
        sandbox,
        state as sandbox_state,
    )

    monkeypatch.setattr(sandbox_state, "_cli_sandbox_disabled", False)
    monkeypatch.setattr(sandbox_state, "_cli_sandbox_profile", None)

    if sys.platform == "darwin":
        if not check_seatbelt_available():
            pytest.skip("macOS Seatbelt sandbox is unavailable")
    elif not (check_net_available() and check_mount_available()):
        pytest.skip("Linux strict sandbox prerequisites are unavailable")

    hostile_temp = tmp_path / "hostile-temp"
    hostile_temp.mkdir(mode=0o700)
    hostile_temp.chmod(0o700)
    hostile_temp_text = str(hostile_temp.resolve())
    child_env = RaptorConfig.get_safe_env()
    child_env.update(
        {
            "GEMINI_API_KEY": "caller-supplied-key",
            "RAPTOR_LLM_SOCKET": "caller-supplied-socket",
            "RAPTOR_LLM_TOKEN_FD": "99",
            "RAPTOR_CONFIG": "caller-supplied-config",
            "RAPTOR_PRIVATE_TMPDIR": hostile_temp_text,
            "TMPDIR": hostile_temp_text,
            "TMP": hostile_temp_text,
            "TEMP": hostile_temp_text,
        }
    )

    with sandbox(
        profile="strict",
        target=str(target),
        output=str(output),
        block_network=True,
    ) as run:
        result = run(
            [sys.executable, str(child)],
            env=child_env,
            strict_env=True,
            capture_output=True,
            text=True,
            timeout=30,
        )

    assert result.returncode == 0, result.stderr
    assert json.loads(report.read_text()) == {
        "gemini_present": False,
        "unrelated_secret_present": False,
        "dispatcher_socket_present": False,
        "dispatcher_token_fd_present": False,
        "raptor_config_present": False,
        "isolation_mode": "1",
        "isolated_temp_marker": private_temp_text,
        "tmpdir": private_temp_text,
        "tmp": private_temp_text,
        "temp": private_temp_text,
        "tempfile_parent": private_temp_text,
    }
    if sys.platform == "darwin":
        assert getattr(result, "_setup_status", "missing") is None
    else:
        assert result.sandbox_info["mount_ns_active"] is True

def test_required_isolation_refuses_environment_key_fallback(tmp_path, monkeypatch, capsys):
    report = tmp_path / "canary.json"
    child = tmp_path / "canary.py"
    _canary_script(child, report)
    monkeypatch.setenv("GEMINI_API_KEY", "fixture-gemini-key")
    monkeypatch.setenv("RAPTOR_REQUIRE_CREDENTIAL_ISOLATION", "1")
    monkeypatch.setattr(raptor, "_get_or_start_dispatcher", lambda: None)

    assert raptor._run_script(child, []) == 2
    assert not report.exists()
    assert "refusing environment credential fallback" in capsys.readouterr().err
