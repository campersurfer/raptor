"""Tests for core.audit.reachability — mechanical reachability gate."""

from __future__ import annotations

from core.audit.reachability import ReachabilityVerdict, check_reachability


def _inventory_with_no_callers(file_path, function_name):
    """Minimal inventory where the function exists but has zero callers."""
    return {"files": [{"path": file_path, "functions": [
        {"name": function_name, "line_start": 1, "callers": []},
    ]}]}


class TestEntryPointBypass:
    def test_entry_point_always_reachable(self):
        context_map = {
            "entry_points": [
                {"file": "a.c", "name": "main"},
            ],
        }
        v = check_reachability(
            file_path="a.c",
            function_name="main",
            context_map=context_map,
        )
        assert v.reachable is True
        assert v.is_entry_point is True

    def test_non_entry_point_with_inventory(self):
        context_map = {
            "entry_points": [
                {"file": "a.c", "name": "main"},
            ],
        }
        v = check_reachability(
            file_path="a.c",
            function_name="helper",
            context_map=context_map,
            inventory=_inventory_with_no_callers("a.c", "helper"),
        )
        assert v.reachable is False
        assert v.is_entry_point is False

    def test_non_entry_point_no_inventory_assumes_reachable(self):
        context_map = {
            "entry_points": [
                {"file": "a.c", "name": "main"},
            ],
        }
        v = check_reachability(
            file_path="a.c",
            function_name="helper",
            context_map=context_map,
        )
        assert v.reachable is True
        assert "unavailable" in v.reason


class TestBinaryOracleSignal:
    def _checklist_with_oracle(self, classification, tier="full"):
        return {
            "files": [
                {
                    "path": "a.c",
                    "items": [
                        {
                            "name": "dead_fn",
                            "kind": "function",
                            "metadata": {
                                "binary_oracle": {
                                    "classification": classification,
                                    "binaries": [
                                        {
                                            "path": "/bin/test",
                                            "classification": classification,
                                            "tier": tier,
                                        },
                                    ],
                                },
                            },
                        },
                    ],
                },
            ],
        }

    def test_absent_full_dwarf_is_dead(self):
        v = check_reachability(
            file_path="a.c",
            function_name="dead_fn",
            checklist=self._checklist_with_oracle("absent", "full"),
            inventory=_inventory_with_no_callers("a.c", "dead_fn"),
        )
        assert v.reachable is False
        assert v.binary_absent is True
        assert v.binary_tier == "full"
        assert "absent" in v.reason

    def test_absent_symbol_only_not_suppression_grade(self):
        """symbol_only tier is not suppression-grade per policy — reachable=True."""
        v = check_reachability(
            file_path="a.c",
            function_name="dead_fn",
            checklist=self._checklist_with_oracle("absent", "symbol_only"),
            inventory=_inventory_with_no_callers("a.c", "dead_fn"),
        )
        assert v.reachable is True
        assert v.binary_absent is True
        assert v.binary_tier == "symbol_only"

    def test_present_overrides_zero_callers(self):
        v = check_reachability(
            file_path="a.c",
            function_name="dead_fn",
            checklist=self._checklist_with_oracle("symbol_present"),
            inventory=_inventory_with_no_callers("a.c", "dead_fn"),
        )
        assert v.reachable is True
        assert v.has_callers is False

    def test_no_binary_data_with_inventory(self):
        checklist = {
            "files": [
                {
                    "path": "a.c",
                    "items": [
                        {
                            "name": "fn",
                            "kind": "function",
                            "metadata": {},
                        },
                    ],
                },
            ],
        }
        v = check_reachability(
            file_path="a.c",
            function_name="fn",
            checklist=checklist,
            inventory=_inventory_with_no_callers("a.c", "fn"),
        )
        assert v.reachable is False
        assert v.binary_tier is None

    def test_no_binary_data_no_inventory_assumes_reachable(self):
        checklist = {
            "files": [
                {
                    "path": "a.c",
                    "items": [
                        {
                            "name": "fn",
                            "kind": "function",
                            "metadata": {},
                        },
                    ],
                },
            ],
        }
        v = check_reachability(
            file_path="a.c",
            function_name="fn",
            checklist=checklist,
        )
        assert v.reachable is True
        assert "unavailable" in v.reason


class TestNoData:
    def test_no_data_assumes_reachable(self):
        v = check_reachability(
            file_path="a.c",
            function_name="fn",
        )
        assert v.reachable is True
        assert "unavailable" in v.reason

    def test_with_inventory_zero_callers(self):
        v = check_reachability(
            file_path="a.c",
            function_name="fn",
            inventory=_inventory_with_no_callers("a.c", "fn"),
        )
        assert v.reachable is False
        assert v.has_callers is False

    def test_function_not_in_checklist_with_inventory(self):
        checklist = {
            "files": [
                {
                    "path": "b.c",
                    "items": [
                        {"name": "other", "kind": "function", "metadata": {}},
                    ],
                },
            ],
        }
        v = check_reachability(
            file_path="a.c",
            function_name="fn",
            checklist=checklist,
            inventory=_inventory_with_no_callers("a.c", "fn"),
        )
        assert v.reachable is False
        assert v.binary_tier is None


class TestVerdictFields:
    def test_verdict_dataclass(self):
        v = ReachabilityVerdict(
            reachable=True,
            reason="test",
            is_entry_point=True,
            has_callers=True,
        )
        assert v.reachable is True
        assert v.reason == "test"
        assert v.binary_absent is False
        assert v.binary_tier is None
