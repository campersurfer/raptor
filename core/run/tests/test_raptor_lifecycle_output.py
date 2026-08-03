"""Regression tests for explicit Raptor lifecycle output roots."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[3]
RAPTOR_PATH = REPO_ROOT / "raptor.py"
SPEC = importlib.util.spec_from_file_location("raptor_launcher_under_test", RAPTOR_PATH)
assert SPEC and SPEC.loader
RAPTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RAPTOR)


class RaptorLifecycleOutputTests(unittest.TestCase):
    def _assert_explicit_output_is_preserved(self, args: list[str], expected: Path) -> None:
        forwarded: list[list[str]] = []
        with (
            mock.patch.object(RAPTOR, "get_output_dir", return_value=expected) as get_output_dir,
            mock.patch.object(RAPTOR, "safe_run_mkdir"),
            mock.patch.object(RAPTOR, "start_run"),
            mock.patch.object(RAPTOR, "complete_run"),
            mock.patch.object(
                RAPTOR,
                "_run_script",
                side_effect=lambda _script, command_args: forwarded.append(command_args) or 0,
            ),
        ):
            result = RAPTOR._run_with_lifecycle("fixture", Path("fixture.py"), args, "fixture")

        self.assertEqual(0, result)
        get_output_dir.assert_called_once_with(
            "fixture",
            explicit_out=str(expected),
            target_path=None,
        )
        self.assertEqual([args], forwarded)

    def test_space_separated_out_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            explicit = Path(temporary_directory) / "scanner-output"
            self._assert_explicit_output_is_preserved(["--out", str(explicit)], explicit)

    def test_equals_out_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            explicit = Path(temporary_directory) / "scanner-output"
            self._assert_explicit_output_is_preserved([f"--out={explicit}"], explicit)


if __name__ == "__main__":
    unittest.main()
