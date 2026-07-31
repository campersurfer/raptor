"""Tests for libexec/raptor-study-loop orchestrator."""

import importlib.machinery
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

RAPTOR_DIR = Path(__file__).resolve().parents[3]
STUDY_LOOP = str(RAPTOR_DIR / "libexec" / "raptor-study-loop")


def _load_loop_module() -> ModuleType:
    loader = importlib.machinery.SourceFileLoader("raptor_study_loop", STUDY_LOOP)
    spec = importlib.util.spec_from_file_location(
        "raptor_study_loop", STUDY_LOOP, loader=loader,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_loop = _load_loop_module()


def _run_loop(args: list[str], env_extra: dict | None = None) -> subprocess.CompletedProcess:
    """Run study-loop with trust marker set."""
    import os
    env = os.environ.copy()
    env["_RAPTOR_TRUSTED"] = "1"
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, STUDY_LOOP] + args,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


class TestStudyLoopCLI:
    def test_help(self):
        result = _run_loop(["--help"])
        assert result.returncode == 0
        assert "Multi-pass" in result.stdout

    def test_missing_target_produces_no_items(self, tmp_path):
        result = _run_loop(["/nonexistent/target", str(tmp_path)])
        # study-prep handles missing dirs gracefully (0 files → 0 items)
        assert "no study items" in result.stderr.lower() or result.returncode == 0

    def test_creates_output_dir(self, tmp_path):
        out = tmp_path / "loop-out"
        # Will fail because target has no C files, but output dir should exist
        target = tmp_path / "src"
        target.mkdir()
        _run_loop([str(target), str(out)])
        assert out.is_dir()


class TestStudyLoopConvergence:
    """Test that the loop terminates correctly."""

    def test_empty_target_terminates(self, tmp_path):
        target = tmp_path / "src"
        target.mkdir()
        out = tmp_path / "out"
        result = _run_loop([str(target), str(out)])
        assert "no study items" in result.stderr.lower() or result.returncode == 0


class TestReadingListDrain:
    """Test reading-list integration in the loop."""

    def test_reading_list_items_extracted(self):
        """Verify the helper that extracts identifiers from reading list."""
        import importlib.util
        import os
        os.environ["_RAPTOR_TRUSTED"] = "1"
        sys.path.insert(0, str(RAPTOR_DIR))

        spec = importlib.util.spec_from_file_location(
            "study_loop", STUDY_LOOP,
            submodule_search_locations=[],
        )
        if spec is None or spec.loader is None:
            # Fallback: exec the file directly and grab the function
            import types
            mod = types.ModuleType("study_loop")
            mod.__file__ = STUDY_LOOP
            exec(
                compile(
                    Path(STUDY_LOOP).read_text(encoding="utf-8"),
                    STUDY_LOOP, "exec",
                ),
                mod.__dict__,
            )
        else:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

        pending = [
            {"question": "struct page ownership", "resolution": "identifier",
             "context": "Need to understand `page` lifecycle"},
            {"question": "memory aliasing semantics", "resolution": "concept",
             "context": ""},
        ]
        idents, concepts = mod._extract_rl_identifiers(pending)
        assert "page" in idents
        assert "memory aliasing semantics" in concepts


class TestResolveCoveredItems:
    def _write_rl(self, out_dir, items):
        rl_path = out_dir / "reading-list.json"
        rl_path.write_text(json.dumps({"items": items}), encoding="utf-8")
        return rl_path

    def _write_dm(self, out_dir, concepts=None, contracts=None):
        dm = {
            "concepts": concepts or [],
            "contracts": contracts or [],
            "invariants": [],
        }
        (out_dir / "domain-model.json").write_text(
            json.dumps(dm), encoding="utf-8")

    def test_resolves_by_local_domain_model(self, tmp_path):
        self._write_dm(tmp_path, concepts=[
            {"id": "page_ownership", "description": "page ownership semantics"},
        ])
        self._write_rl(tmp_path, [
            {"question": "what is `page_ownership`?", "resolved": False},
        ])
        count = _loop._resolve_covered_items(tmp_path)
        assert count == 1
        data = json.loads((tmp_path / "reading-list.json").read_text())
        assert data["items"][0]["resolved"] is True

    def test_skips_already_resolved(self, tmp_path):
        self._write_dm(tmp_path, concepts=[
            {"id": "foo", "description": "foo thing"},
        ])
        self._write_rl(tmp_path, [
            {"question": "what is `foo`?", "resolved": True},
        ])
        count = _loop._resolve_covered_items(tmp_path)
        assert count == 0

    def test_sage_fallback_resolves(self, tmp_path):
        self._write_dm(tmp_path)
        (tmp_path / "study-list.json").write_text(
            json.dumps({"target": "/repos/myproj"}), encoding="utf-8")
        self._write_rl(tmp_path, [
            {"question": "what is the lifetime of crypto_alg?",
             "resolved": False},
        ])
        mock_client = MagicMock()
        mock_client.query.return_value = [{"confidence": 0.9, "content": "known"}]
        mock_hooks = MagicMock()
        mock_hooks._get_client.return_value = mock_client
        mock_hooks._concepts_domain.return_value = "test-domain"

        with patch.dict("sys.modules", {"core.sage.hooks": mock_hooks}):
            count = _loop._resolve_covered_items(tmp_path)
        assert count == 1
        data = json.loads((tmp_path / "reading-list.json").read_text())
        assert data["items"][0]["resolved"] is True

    def test_no_sage_when_local_covers(self, tmp_path):
        """SAGE is not queried when local domain-model already resolves."""
        self._write_dm(tmp_path, concepts=[
            {"id": "crypto_alg", "description": "crypto algorithm structure"},
        ])
        (tmp_path / "study-list.json").write_text(
            json.dumps({"target": "/repos/myproj"}), encoding="utf-8")
        self._write_rl(tmp_path, [
            {"question": "what is `crypto_alg`?", "resolved": False},
        ])
        mock_client = MagicMock()
        mock_hooks = MagicMock()
        mock_hooks._get_client.return_value = mock_client
        mock_hooks._concepts_domain.return_value = "test-domain"

        with patch.dict("sys.modules", {"core.sage.hooks": mock_hooks}):
            count = _loop._resolve_covered_items(tmp_path)
        assert count == 1
        mock_client.query.assert_not_called()

    def test_no_resolution_when_nothing_matches(self, tmp_path):
        self._write_dm(tmp_path)
        self._write_rl(tmp_path, [
            {"question": "what is `totally_unknown`?", "resolved": False},
        ])
        count = _loop._resolve_covered_items(tmp_path)
        assert count == 0

    def test_sage_unavailable_degrades_gracefully(self, tmp_path):
        self._write_dm(tmp_path)
        (tmp_path / "study-list.json").write_text(
            json.dumps({"target": "/repos/myproj"}), encoding="utf-8")
        self._write_rl(tmp_path, [
            {"question": "what is the lifetime of crypto_alg?",
             "resolved": False},
        ])
        with patch.dict("sys.modules", {"core.sage.hooks": None}):
            count = _loop._resolve_covered_items(tmp_path)
        assert count == 0
