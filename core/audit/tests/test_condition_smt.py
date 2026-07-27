"""Tests for condition_smt — constant-to-SMT bridge for guard sufficiency."""


from core.audit.condition_smt import (
    check_all_sufficiency,
    check_guard_sufficiency,
    extract_bounds_constraints,
)
from core.audit.condition_extraction import GuardCondition, SinkGuard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _guard(text: str, category: str = "bounds", **kw) -> GuardCondition:
    return GuardCondition(
        text=text, category=category, polarity="required", line=1, **kw
    )


def _sg(sink_api: str, guards: list, line: int = 10) -> SinkGuard:
    return SinkGuard(
        sink_file="test.c",
        sink_line=line,
        sink_function="process",
        sink_api=sink_api,
        guards=guards,
        unconditional=(len(guards) == 0),
    )


# ---------------------------------------------------------------------------
# Bounds constraint extraction tests
# ---------------------------------------------------------------------------


class TestExtractBoundsConstraints:
    def test_simple_less_than(self):
        g = _guard("len < 1024", resolvable=True, concrete_values={"MAX_SIZE": "1024"})
        constraints = extract_bounds_constraints(g)
        assert len(constraints) >= 1
        found = [c for c in constraints if c.variable == "len"]
        assert found
        assert found[0].operator == "<"
        assert found[0].bound_value == 1024

    def test_less_equal(self):
        g = _guard("n <= 256", resolvable=True, concrete_values={})
        constraints = extract_bounds_constraints(g)
        found = [c for c in constraints if c.variable == "n"]
        assert found
        assert found[0].operator == "<="
        assert found[0].bound_value == 256

    def test_greater_than(self):
        g = _guard("size > 0", resolvable=True, concrete_values={})
        constraints = extract_bounds_constraints(g)
        found = [c for c in constraints if c.variable == "size"]
        assert found
        assert found[0].operator == ">"
        assert found[0].bound_value == 0

    def test_hex_value(self):
        g = _guard("offset < 0x1000", resolvable=True, concrete_values={})
        constraints = extract_bounds_constraints(g)
        found = [c for c in constraints if c.variable == "offset"]
        assert found
        assert found[0].bound_value == 4096

    def test_constant_resolution(self):
        g = _guard(
            "len < MAX_BUF",
            resolvable=True,
            concrete_values={"MAX_BUF": "4096"},
        )
        constraints = extract_bounds_constraints(g)
        found = [c for c in constraints if c.variable == "len"]
        assert found
        assert found[0].bound_value == 4096

    def test_reversed_comparison(self):
        g = _guard("256 >= count", resolvable=True, concrete_values={})
        constraints = extract_bounds_constraints(g)
        found = [c for c in constraints if c.variable == "count"]
        assert found
        assert found[0].operator == "<="
        assert found[0].bound_value == 256

    def test_no_numeric_value(self):
        g = _guard("len < other_var", resolvable=False, concrete_values={})
        constraints = extract_bounds_constraints(g)
        # other_var is not numeric, so no constraint extracted
        assert len(constraints) == 0

    def test_not_resolvable_still_extracts_literals(self):
        g = _guard("x < 100", resolvable=True, concrete_values={})
        constraints = extract_bounds_constraints(g)
        assert len(constraints) >= 1


# ---------------------------------------------------------------------------
# SMT sufficiency tests (arithmetic fallback — Z3 may not be available)
# ---------------------------------------------------------------------------


class TestCheckGuardSufficiency:
    def test_guard_insufficient_for_buffer(self):
        g = _guard(
            "len < MAX_SIZE",
            resolvable=True,
            concrete_values={"MAX_SIZE": "4096"},
        )
        sg = _sg("memcpy", [g])
        results = check_guard_sufficiency(sg, buffer_size=256)
        assert len(results) >= 1
        # Guard allows up to 4095, buffer is 256 → insufficient
        r = results[0]
        assert r.feasible is True  # overflow IS feasible
        assert r.guard_sufficient is False

    def test_guard_sufficient_for_buffer(self):
        g = _guard(
            "len < BUF_SIZE",
            resolvable=True,
            concrete_values={"BUF_SIZE": "128"},
        )
        sg = _sg("memcpy", [g])
        results = check_guard_sufficiency(sg, buffer_size=256)
        assert len(results) >= 1
        r = results[0]
        assert r.feasible is False  # overflow NOT feasible
        assert r.guard_sufficient is True

    def test_no_resolvable_guards(self):
        g = _guard("len < limit", resolvable=False, concrete_values={})
        sg = _sg("memcpy", [g])
        results = check_guard_sufficiency(sg, buffer_size=256)
        assert results == []

    def test_no_buffer_size(self):
        g = _guard(
            "len < MAX",
            resolvable=True,
            concrete_values={"MAX": "1024"},
        )
        sg = _sg("memcpy", [g])
        results = check_guard_sufficiency(sg)  # no buffer_size
        assert len(results) >= 1
        # Without buffer_size, can't do arithmetic check
        r = results[0]
        # Should still extract the constraint even if verdict is None
        assert r.concrete_values

    def test_multiple_constraints_tightest_wins(self):
        """Guard with two upper bounds — tightest determines sufficiency."""
        # "len < 4096 && len < 256" with buffer=256
        # Tightest: max_allowed=255 <= 256 → sufficient
        g = _guard(
            "len < MAX_SIZE && len < LIMIT",
            resolvable=True,
            concrete_values={"MAX_SIZE": "4096", "LIMIT": "256"},
        )
        sg = _sg("memcpy", [g])
        results = check_guard_sufficiency(sg, buffer_size=256)
        assert len(results) >= 1
        r = results[0]
        assert r.feasible is False  # tightest constraint fits
        assert r.guard_sufficient is True

    def test_multiple_constraints_all_too_loose(self):
        """Guard with two upper bounds, both exceed buffer."""
        # "len < 4096 && len < 1024" with buffer=256
        # Tightest: max_allowed=1023 > 256 → insufficient
        g = _guard(
            "len < MAX_SIZE && len < MID",
            resolvable=True,
            concrete_values={"MAX_SIZE": "4096", "MID": "1024"},
        )
        sg = _sg("memcpy", [g])
        results = check_guard_sufficiency(sg, buffer_size=256)
        assert len(results) >= 1
        r = results[0]
        assert r.feasible is True  # even tightest is too loose
        assert r.guard_sufficient is False

    def test_to_dict(self):
        g = _guard(
            "n < 100",
            resolvable=True,
            concrete_values={"X": "100"},
        )
        sg = _sg("memcpy", [g])
        results = check_guard_sufficiency(sg, buffer_size=256)
        if results:
            d = results[0].to_dict()
            assert "guard_text" in d
            assert "reasoning" in d


# ---------------------------------------------------------------------------
# Batch API
# ---------------------------------------------------------------------------


class TestCheckAllSufficiency:
    def test_batch_with_sizes(self):
        g1 = _guard("len < MAX", resolvable=True, concrete_values={"MAX": "4096"})
        g2 = _guard("n <= SIZE", resolvable=True, concrete_values={"SIZE": "128"})
        guards = [_sg("memcpy", [g1], line=10), _sg("memcpy", [g2], line=20)]
        results = check_all_sufficiency(
            guards,
            buffer_sizes={10: 256, 20: 256},
        )
        assert len(results) == 2
        # First: 4096 > 256 → insufficient
        assert results[0][0].feasible is True
        # Second: 128 <= 256 → sufficient
        assert results[1][0].feasible is False


# ── Safety contract compliance ──────────────────────────────────────────


class TestContractCompliance:
    def test_assert_boost_only_succeeds(self):
        from core.audit.safety_contract import assert_boost_only
        assert_boost_only("condition_smt")

    def test_suppress_evidence_raises(self):
        import pytest
        from core.audit.safety_contract import ContractViolation, SuppressEvidence
        with pytest.raises(ContractViolation):
            SuppressEvidence(
                source="condition_smt",
                verdict="guard_sufficient",
                reason="test",
            )

    def test_boost_evidence_allowed(self):
        from core.audit.safety_contract import BoostEvidence
        ev = BoostEvidence(
            source="condition_smt",
            action="add_finding",
            description="guard insufficient — overflow still possible",
        )
        assert ev.source == "condition_smt"
        assert ev.action == "add_finding"


# ── Safety contract blind spots (adversarial) ──────────────────────────


class TestBlindSpotSufficientGuardIsBoost:
    """A 'guard sufficient' SMT result must be context injection (boost),
    not suppression.

    When SMT proves a guard IS sufficient, the orchestrator injects that
    as context for the LLM ('this guard looks adequate — focus elsewhere').
    It must NEVER suppress or remove the function from review.
    """

    def test_sufficient_guard_is_not_feasible(self):
        g = _guard(
            "n <= LIMIT",
            resolvable=True,
            concrete_values={"LIMIT": "128"},
        )
        sg = _sg("memcpy", [g], line=10)
        results = check_guard_sufficiency(sg, buffer_size=256)
        assert len(results) >= 1
        assert results[0].feasible is False
        assert results[0].guard_sufficient is True
