"""Tests for core.audit.audit_bridge."""

from __future__ import annotations

import json
from pathlib import Path

from core.audit.audit_bridge import (
    _build_constraint_index,
    enrich_attack_paths,
    enrich_with_summaries,
    find_audit_output,
    inject_chains_as_hypotheses,
    load_attack_chains,
    load_audit_constraints,
    load_summaries,
)


def _write_constraints(run_dir: Path, constraints: list) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "constraints.json").write_text(json.dumps(constraints))


class TestFindAuditOutput:
    def test_finds_sibling_with_constraints(self, tmp_path):
        _write_constraints(tmp_path / "audit_run", [{"function": "fn"}])
        current = tmp_path / "validate_run"
        current.mkdir()
        result = find_audit_output(current)
        assert result == tmp_path / "audit_run"

    def test_excludes_current_dir(self, tmp_path):
        _write_constraints(tmp_path / "validate_run", [{"function": "fn"}])
        result = find_audit_output(tmp_path / "validate_run")
        assert result is None

    def test_returns_none_without_constraints(self, tmp_path):
        (tmp_path / "other_run").mkdir()
        current = tmp_path / "validate_run"
        current.mkdir()
        result = find_audit_output(current)
        assert result is None

    def test_picks_newest(self, tmp_path):
        import time
        _write_constraints(tmp_path / "old", [{"function": "a"}])
        time.sleep(0.05)
        _write_constraints(tmp_path / "new", [{"function": "b"}])
        current = tmp_path / "validate"
        current.mkdir()
        result = find_audit_output(current)
        assert result == tmp_path / "new"


class TestLoadAuditConstraints:
    def test_loads_and_filters_refuted(self, tmp_path):
        _write_constraints(tmp_path, [
            {"function": "fn1", "status": "open", "kind": "parameter"},
            {"function": "fn2", "status": "refuted", "kind": "parameter"},
            {"function": "fn3", "status": "verified", "kind": "postcondition"},
        ])
        result = load_audit_constraints(tmp_path)
        assert len(result) == 2
        assert all(c["status"] != "refuted" for c in result)

    def test_empty_on_missing_file(self, tmp_path):
        assert load_audit_constraints(tmp_path) == []

    def test_empty_on_malformed_json(self, tmp_path):
        tmp_path.mkdir(exist_ok=True)
        (tmp_path / "constraints.json").write_text("{not valid")
        assert load_audit_constraints(tmp_path) == []

    def test_empty_on_non_list(self, tmp_path):
        tmp_path.mkdir(exist_ok=True)
        (tmp_path / "constraints.json").write_text('{"key": "value"}')
        assert load_audit_constraints(tmp_path) == []


class TestBuildConstraintIndex:
    def test_indexes_by_file_function(self):
        constraints = [
            {"function": "fn", "file": "a.c", "kind": "parameter",
             "target": "buf", "rule": "len <= 1024", "violation": "overflow"},
        ]
        index = _build_constraint_index(constraints)
        assert "a.c:fn" in index
        assert ":fn" in index

    def test_skips_empty_function(self):
        constraints = [{"function": "", "file": "a.c"}]
        index = _build_constraint_index(constraints)
        assert len(index) == 0


class TestEnrichAttackPaths:
    def test_enriches_matching_steps(self, tmp_path):
        paths = [
            {"steps": [
                {"function": "parse_input", "file": "http.c"},
                {"function": "handle_data", "file": "core.c"},
            ]},
        ]
        ap_file = tmp_path / "attack-paths.json"
        ap_file.write_text(json.dumps(paths))

        constraints = [
            {"function": "parse_input", "file": "http.c",
             "kind": "parameter", "target": "buf",
             "rule": "len <= 4096", "violation": "stack overflow",
             "status": "open"},
        ]

        enriched = enrich_attack_paths(ap_file, constraints)
        assert enriched == 1

        data = json.loads(ap_file.read_text())
        step = data[0]["steps"][0]
        assert "parameter_constraints" in step
        assert step["parameter_constraints"][0]["rule"] == "len <= 4096"

    def test_bare_function_fallback(self, tmp_path):
        paths = [{"steps": [{"function": "fn", "file": "other.c"}]}]
        ap_file = tmp_path / "attack-paths.json"
        ap_file.write_text(json.dumps(paths))

        constraints = [
            {"function": "fn", "file": "original.c",
             "kind": "precondition", "target": "ptr",
             "rule": "not NULL", "violation": "null deref",
             "status": "open"},
        ]

        enriched = enrich_attack_paths(ap_file, constraints)
        assert enriched == 1

    def test_no_enrichment_no_write(self, tmp_path):
        paths = [{"steps": [{"function": "unrelated", "file": "x.c"}]}]
        ap_file = tmp_path / "attack-paths.json"
        ap_file.write_text(json.dumps(paths))
        original = ap_file.read_text()

        constraints = [
            {"function": "fn", "file": "a.c", "kind": "parameter",
             "target": "x", "rule": "r", "violation": "v", "status": "open"},
        ]

        enriched = enrich_attack_paths(ap_file, constraints)
        assert enriched == 0
        assert ap_file.read_text() == original

    def test_missing_file(self, tmp_path):
        constraints = [{"function": "fn", "file": "a.c"}]
        enriched = enrich_attack_paths(tmp_path / "nope.json", constraints)
        assert enriched == 0


class TestLoadAttackChains:
    def test_loads_valid_chains(self, tmp_path):
        chains = [
            {"id": "c1", "goal": "RCE", "findings": [{"finding_id": "F1"}]},
            {"id": "c2", "goal": "info leak", "findings": []},
        ]
        (tmp_path / "attack-chains.json").write_text(json.dumps(chains))
        result = load_attack_chains(tmp_path)
        assert len(result) == 2
        assert result[0]["goal"] == "RCE"

    def test_empty_on_missing_file(self, tmp_path):
        assert load_attack_chains(tmp_path) == []

    def test_empty_on_malformed_json(self, tmp_path):
        (tmp_path / "attack-chains.json").write_text("not json")
        assert load_attack_chains(tmp_path) == []

    def test_filters_non_dict_entries(self, tmp_path):
        (tmp_path / "attack-chains.json").write_text(json.dumps([
            {"id": "c1"}, "not a dict", 42, {"id": "c2"},
        ]))
        result = load_attack_chains(tmp_path)
        assert len(result) == 2


class TestLoadSummaries:
    def test_loads_dict_format(self, tmp_path):
        data = {"src/a.c:parse": {"taint_rules": [], "preconditions": []}}
        (tmp_path / "summaries.json").write_text(json.dumps(data))
        result = load_summaries(tmp_path)
        assert "src/a.c:parse" in result

    def test_loads_list_format(self, tmp_path):
        data = [
            {"file": "a.c", "function": "fn1", "taint_rules": []},
            {"file": "b.c", "function": "fn2", "preconditions": ["x != NULL"]},
        ]
        (tmp_path / "summaries.json").write_text(json.dumps(data))
        result = load_summaries(tmp_path)
        assert "a.c:fn1" in result
        assert "b.c:fn2" in result

    def test_empty_on_missing_file(self, tmp_path):
        assert load_summaries(tmp_path) == {}

    def test_empty_on_non_list_non_dict(self, tmp_path):
        (tmp_path / "summaries.json").write_text('"just a string"')
        assert load_summaries(tmp_path) == {}


class TestInjectChainsAsHypotheses:
    def test_injects_into_existing_surface(self, tmp_path):
        surface = {"entry_points": [], "sinks": [], "hypotheses": []}
        ap_file = tmp_path / "attack-surface.json"
        ap_file.write_text(json.dumps(surface))

        chains = [
            {"description": "heap overflow via parse", "goal": "RCE",
             "primitives": ["write-what-where"],
             "findings": [{"finding_id": "F1"}]},
        ]
        result = inject_chains_as_hypotheses(chains, ap_file)
        assert result == 1

        data = json.loads(ap_file.read_text())
        assert len(data["hypotheses"]) == 1
        h = data["hypotheses"][0]
        assert h["source"] == "audit_chain"
        assert h["status"] == "imported"
        assert h["goal"] == "RCE"

    def test_returns_zero_on_missing_file(self, tmp_path):
        chains = [{"description": "x"}]
        result = inject_chains_as_hypotheses(chains, tmp_path / "nope.json")
        assert result == 0

    def test_returns_zero_on_empty_chains(self, tmp_path):
        ap_file = tmp_path / "attack-surface.json"
        ap_file.write_text(json.dumps({"hypotheses": []}))
        assert inject_chains_as_hypotheses([], ap_file) == 0


class TestEnrichWithSummaries:
    def test_enriches_matching_steps(self, tmp_path):
        paths = [{"steps": [
            {"function": "parse", "file": "http.c"},
            {"function": "handle", "file": "core.c"},
        ]}]
        ap_file = tmp_path / "attack-paths.json"
        ap_file.write_text(json.dumps(paths))

        summaries = {
            "http.c:parse": {
                "preconditions": [{"param": "buf", "conditions": ["!= NULL"]}],
                "taint_rules": [{"source_param": "buf", "sink_call": "memcpy"}],
                "returns": [],
            },
        }
        result = enrich_with_summaries(ap_file, summaries)
        assert result == 1

        data = json.loads(ap_file.read_text())
        step = data[0]["steps"][0]
        assert "callee_contract" in step
        assert step["callee_contract"]["preconditions"][0]["param"] == "buf"

    def test_no_enrichment_without_summaries(self, tmp_path):
        paths = [{"steps": [{"function": "fn", "file": "a.c"}]}]
        ap_file = tmp_path / "attack-paths.json"
        ap_file.write_text(json.dumps(paths))
        assert enrich_with_summaries(ap_file, {}) == 0

    def test_missing_file(self, tmp_path):
        summaries = {"a.c:fn": {"preconditions": []}}
        assert enrich_with_summaries(tmp_path / "nope.json", summaries) == 0
