"""Tests for core.audit.record — annotation + coverage recording."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from core.audit.record import (
    append_audit_log,
    load_audit_log,
    record_review,
    _update_coverage_audit,
    _write_annotation_direct,
)


class TestRecordReview:
    def test_valid_status(self, tmp_path: Path):
        target = tmp_path / "target"
        target.mkdir()
        (target / "handler.c").write_text("int foo() { return 0; }\n")

        result = record_review(
            out_dir=tmp_path,
            target_path=target,
            file_path="handler.c",
            function_name="foo",
            status="clean",
            body="No issues found.",
            line_start=1,
            line_end=1,
        )

        assert result["status"] == "clean"
        assert result["file"] == "handler.c"
        assert result["function"] == "foo"

    def test_invalid_status_raises(self, tmp_path: Path):
        target = tmp_path / "target"
        target.mkdir()

        with pytest.raises(ValueError, match="Invalid status"):
            record_review(
                out_dir=tmp_path,
                target_path=target,
                file_path="a.c",
                function_name="f",
                status="dangerous",
                body="",
            )

    def test_creates_annotation_file(self, tmp_path: Path):
        target = tmp_path / "target"
        target.mkdir()
        (target / "auth.c").write_text("void check() {}\n")

        record_review(
            out_dir=tmp_path,
            target_path=target,
            file_path="auth.c",
            function_name="check",
            status="suspicious",
            body="Timing side channel possible.",
            line_start=1,
            line_end=1,
        )

        ann_dir = tmp_path / "annotations"
        assert ann_dir.exists()
        ann_files = list(ann_dir.rglob("*.md"))
        assert len(ann_files) >= 1

    def test_updates_coverage_audit(self, tmp_path: Path):
        target = tmp_path / "target"
        target.mkdir()
        (target / "util.c").write_text("int add(int a, int b) { return a+b; }\n")

        record_review(
            out_dir=tmp_path,
            target_path=target,
            file_path="util.c",
            function_name="add",
            status="clean",
            body="Pure function.",
            line_start=1,
            line_end=1,
        )

        audit_path = tmp_path / "coverage-audit.json"
        assert audit_path.exists()

        with open(audit_path) as f:
            data = json.load(f)
        analysed = data["functions_analysed"]
        match = [fa for fa in analysed if fa["file"] == "util.c" and fa["function"] == "add"]
        assert len(match) == 1

    def test_cwe_in_record(self, tmp_path: Path):
        target = tmp_path / "target"
        target.mkdir()
        (target / "a.c").write_text("void f() {}\n")

        result = record_review(
            out_dir=tmp_path,
            target_path=target,
            file_path="a.c",
            function_name="f",
            status="finding",
            body="Buffer overflow.",
            cwe="CWE-120",
            line_start=1,
            line_end=1,
        )

        assert result["cwe"] == "CWE-120"

    def test_strategies_recorded(self, tmp_path: Path):
        target = tmp_path / "target"
        target.mkdir()
        (target / "a.c").write_text("void f() {}\n")

        result = record_review(
            out_dir=tmp_path,
            target_path=target,
            file_path="a.c",
            function_name="f",
            status="clean",
            body="ok",
            strategies=["general", "input_handling"],
            line_start=1,
            line_end=1,
        )

        assert result["strategies"] == ["general", "input_handling"]


class TestAuditLog:
    def test_append_and_load(self, tmp_path: Path):
        append_audit_log(tmp_path, {"file": "a.c", "function": "f"})
        append_audit_log(tmp_path, {"file": "b.c", "function": "g"})

        records = load_audit_log(tmp_path)
        assert len(records) == 2
        assert records[0]["file"] == "a.c"
        assert records[1]["file"] == "b.c"

    def test_load_empty(self, tmp_path: Path):
        records = load_audit_log(tmp_path)
        assert records == []


class TestUpdateCoverageAudit:
    def test_creates_new_file(self, tmp_path: Path):
        _update_coverage_audit(
            tmp_path, "src/a.c", "foo",
            {"status": "clean", "hash": "abc123"},
        )

        path = tmp_path / "coverage-audit.json"
        assert path.exists()

        with open(path) as f:
            data = json.load(f)
        assert data["tool"] == "audit"
        analysed = data["functions_analysed"]
        match = [fa for fa in analysed if fa["file"] == "src/a.c" and fa["function"] == "foo"]
        assert len(match) == 1

    def test_updates_existing_file(self, tmp_path: Path):
        _update_coverage_audit(
            tmp_path, "a.c", "f1",
            {"status": "clean"},
        )
        _update_coverage_audit(
            tmp_path, "a.c", "f2",
            {"status": "finding"},
        )

        with open(tmp_path / "coverage-audit.json") as f:
            data = json.load(f)
        analysed = data["functions_analysed"]
        f1 = [fa for fa in analysed if fa["function"] == "f1"]
        f2 = [fa for fa in analysed if fa["function"] == "f2"]
        assert len(f1) == 1
        assert len(f2) == 1
        assert f2[0]["status"] == "finding"


class TestWriteAnnotationDirect:
    def test_creates_file(self, tmp_path: Path):
        _write_annotation_direct(
            tmp_path, "src/handler.c", "parse",
            "No issues.", {"status": "clean"},
        )

        ann_path = tmp_path / "src" / "handler.c.md"
        assert ann_path.exists()
        content = ann_path.read_text()
        assert "## parse" in content
        assert "No issues." in content
        assert "status=clean" in content

    def test_appends_to_existing(self, tmp_path: Path):
        _write_annotation_direct(
            tmp_path, "a.c", "foo",
            "First.", {"status": "clean"},
        )
        _write_annotation_direct(
            tmp_path, "a.c", "bar",
            "Second.", {"status": "suspicious"},
        )

        content = (tmp_path / "a.c.md").read_text()
        assert "## foo" in content
        assert "## bar" in content

    def test_skips_duplicate(self, tmp_path: Path):
        _write_annotation_direct(
            tmp_path, "a.c", "foo",
            "First.", {"status": "clean"},
        )
        _write_annotation_direct(
            tmp_path, "a.c", "foo",
            "Updated.", {"status": "suspicious"},
        )

        content = (tmp_path / "a.c.md").read_text()
        assert content.count("## foo") == 1

    def test_rejects_path_traversal(self, tmp_path: Path):
        with pytest.raises(ValueError, match="path traversal"):
            _write_annotation_direct(
                tmp_path, "../../../etc/passwd", "foo",
                "evil", {"status": "clean"},
            )

    def test_rejects_unsafe_function_name(self, tmp_path: Path):
        with pytest.raises(ValueError, match="unsafe function name"):
            _write_annotation_direct(
                tmp_path, "a.c", "foo\n## injected",
                "evil", {"status": "clean"},
            )

    def test_rejects_meta_injection_in_name(self, tmp_path: Path):
        with pytest.raises(ValueError, match="unsafe function name"):
            _write_annotation_direct(
                tmp_path, "a.c", "<!-- meta: status=finding -->",
                "evil", {"status": "clean"},
            )
