#!/usr/bin/env python3
"""Fail-closed regression tests for the SAGE_HOME migration."""

import os
import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


class TestSageSetupMigration(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.checkout = Path(self.tempdir.name)
        (self.checkout / "libexec").mkdir()
        (self.checkout / "core" / "sage").mkdir(parents=True)
        (self.checkout / "fake-bin").mkdir()

        for relative in ("libexec/raptor-sage-setup", "libexec/raptor-sage-mcp"):
            source = REPO_ROOT / relative
            destination = self.checkout / relative
            shutil.copy2(source, destination)
            destination.chmod(0o755)

        (self.checkout / "core" / "sage" / "docker-compose.yml").write_text(
            "services:\n  sage:\n    image: ghcr.io/l33tdawg/sage:11.11.6\n",
            encoding="utf-8",
        )

        self.docker_log = self.checkout / "docker.log"
        self._write_executable(
            "docker",
            r"""
            #!/usr/bin/env bash
            set -u
            args="$*"
            printf '%s\n' "$args" >> "$DOCKER_LOG"

            if [[ "$args" == "compose version" ]] || [[ "$args" == "info" ]]; then
                exit 0
            fi
            if [[ "$args" == *" ps --status running -q"* ]]; then
                printf 'running-container\n'
                exit 0
            fi
            if [[ "$args" == *" ps -aq sage"* ]]; then
                printf 'old-sage-container\n'
                exit 0
            fi
            if [[ "$args" == inspect\ old-sage-container*"}}old{{"* ]]; then
                printf 'old\n'
                exit 0
            fi
            if [[ "$args" == inspect\ old-sage-container*".Name"* ]]; then
                if [[ "$FAIL_STAGE" != "volume" ]]; then
                    printf 'raptor_sage_data\n'
                fi
                exit 0
            fi
            if [[ "$args" == *" stop sage"* ]]; then
                exit 0
            fi
            if [[ "$args" == cp\ * ]]; then
                if [[ "$FAIL_STAGE" == "copy" ]]; then
                    exit 9
                fi
                destination=""
                for destination; do :; done
                mkdir -p "$destination/data/badger" "$destination/data/cometbft"
                printf 'agent-key\n' > "$destination/agent.key"
                printf 'memory-db\n' > "$destination/data/sage.db"
                exit 0
            fi
            if [[ "$args" == *" config --images"* ]]; then
                printf 'ghcr.io/l33tdawg/sage:11.11.6\n'
                exit 0
            fi
            if [[ "$args" == *"ls -A /dest"* ]]; then
                printf 'existing-state\n'
                exit 0
            fi
            if [[ "$args" == *'find /src -type f ! -name ".migration-complete"'* ]]; then
                [[ "$FAIL_STAGE" == "comparison" ]] && exit 9
                exit 0
            fi
            if [[ "$args" == *"find /src -mindepth 1"* ]]; then
                [[ "$FAIL_STAGE" == "merge" ]] && exit 9
                exit 0
            fi
            if [[ "$args" == *"for rel in agent.key vault.key config.yaml data/sage.db"* ]]; then
                if [[ "$FAIL_STAGE" == "validation" ]]; then
                    printf 'missing critical file: agent.key\n'
                    exit 9
                fi
                printf 'critical state: verified\n'
                exit 0
            fi
            if [[ "$args" == *"date -u > /dest/.migration-complete"* ]]; then
                [[ "$FAIL_STAGE" == "marker" ]] && exit 9
                printf 'MARKER_WRITTEN\n' >> "$DOCKER_LOG"
                exit 0
            fi
            if [[ "$args" == *" up -d "* ]]; then
                printf 'COMPOSE_UP\n' >> "$DOCKER_LOG"
                exit 0
            fi
            exit 0
            """,
        )
        for command in ("jq", "python3", "curl"):
            self._write_executable(command, "#!/bin/sh\nexit 0\n")

    def tearDown(self):
        self.tempdir.cleanup()

    def _write_executable(self, name, body):
        path = self.checkout / "fake-bin" / name
        path.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")
        path.chmod(0o755)

    def _run_setup(self, fail_stage):
        env = os.environ.copy()
        env.update(
            {
                "_RAPTOR_TRUSTED": "1",
                "DOCKER_LOG": str(self.docker_log),
                "FAIL_STAGE": fail_stage,
                "PATH": f"{self.checkout / 'fake-bin'}:{env['PATH']}",
            }
        )
        return subprocess.run(
            [str(self.checkout / "libexec" / "raptor-sage-setup"), "install"],
            cwd=self.checkout,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_migration_failures_stop_before_recreation_and_marker(self):
        cases = {
            "copy": "could not copy the legacy volume state",
            "comparison": "collision comparison failed",
            "merge": "missing-file merge failed",
            "validation": "post-migration validation failed",
            "marker": "could not write the migration completion marker",
        }
        for stage, expected_error in cases.items():
            with self.subTest(stage=stage):
                result = self._run_setup(stage)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected_error, result.stderr)

                log = self.docker_log.read_text(encoding="utf-8")
                self.assertNotIn("COMPOSE_UP", log)
                self.assertNotIn("MARKER_WRITTEN", log)

                backups = list(self.checkout.glob(".sage-migration.*"))
                self.assertEqual(len(backups), 1)
                self.assertTrue((backups[0] / "agent.key").is_file() or stage == "copy")

                for backup in backups:
                    shutil.rmtree(backup)
                self.docker_log.unlink(missing_ok=True)

    def test_missing_legacy_volume_stops_before_recreation(self):
        result = self._run_setup("volume")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no volume backs /root/.sage-gui", result.stderr)
        log = self.docker_log.read_text(encoding="utf-8")
        self.assertNotIn("COMPOSE_UP", log)
        self.assertNotIn("MARKER_WRITTEN", log)


if __name__ == "__main__":
    unittest.main()
