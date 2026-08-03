"""Exercise the launcher-to-worker credential boundary with a real child process."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

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
        "}))\n"
    )


def test_dispatcher_worker_never_receives_provider_or_parent_secret(tmp_path, monkeypatch):
    report = tmp_path / "canary.json"
    child = tmp_path / "canary.py"
    _canary_script(child, report)
    monkeypatch.setenv("GEMINI_API_KEY", "fixture-gemini-key")
    monkeypatch.setenv("UNRELATED_DISPATCH_SECRET", "fixture-parent-secret")
    monkeypatch.setenv("RAPTOR_REQUIRE_CREDENTIAL_ISOLATION", "1")
    monkeypatch.setenv("PYTHONUSERBASE", "/fixture/user-base")
    monkeypatch.setattr(raptor, "_get_or_start_dispatcher", lambda: _CanaryDispatcher(tmp_path))

    assert raptor._run_script(child, []) == 0

    assert json.loads(report.read_text()) == {
        "gemini_present": False,
        "unrelated_secret_present": False,
        "python_user_base_present": False,
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
    _agentic_canary_script(child, report)
    monkeypatch.setenv("HOME", str(private_home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(private_config))
    monkeypatch.setenv("RAPTOR_CONFIG", str(private_config / "models.json"))
    monkeypatch.setenv("GEMINI_API_KEY", "fixture-gemini-key")
    monkeypatch.setenv("UNRELATED_DISPATCH_SECRET", "fixture-parent-secret")
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
    }

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
