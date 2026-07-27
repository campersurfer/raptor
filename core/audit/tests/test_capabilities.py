"""Tests for core.audit.capabilities."""

from unittest.mock import patch

from core.audit.capabilities import (
    AuditCapabilities,
    format_capability_report,
    probe_capabilities,
)


class TestProbeCapabilities:
    def test_returns_dataclass(self):
        caps = probe_capabilities()
        assert isinstance(caps, AuditCapabilities)

    def test_no_binary(self):
        caps = probe_capabilities(binary_path=None)
        assert not caps.binary_available
        assert not caps.dwarf_available

    def test_nonexistent_binary(self, tmp_path):
        caps = probe_capabilities(binary_path=tmp_path / "nope")
        assert not caps.binary_available

    @patch("shutil.which", return_value=None)
    def test_all_tools_missing(self, _mock_which):
        caps = probe_capabilities()
        assert not caps.joern
        assert not caps.r2
        assert not caps.semgrep
        assert not caps.coccinelle
        assert not caps.codeql

    @patch("shutil.which", return_value="/usr/bin/joern")
    def test_tool_found(self, _mock_which):
        caps = probe_capabilities()
        assert caps.joern


class TestFormatCapabilityReport:
    def test_all_present(self):
        caps = AuditCapabilities(
            joern=True, r2=True, frida=True,
            semgrep=True, coccinelle=True, codeql=True, ghidra=True,
            binary_available=True, dwarf_available=True,
        )
        report = format_capability_report(caps)
        assert "Joern" in report
        assert "Frida" in report
        assert "unavailable" not in report
        assert "DWARF" in report

    def test_none_present(self):
        caps = AuditCapabilities()
        report = format_capability_report(caps)
        assert "unavailable" in report
        assert "No binary" in report

    def test_partial(self):
        caps = AuditCapabilities(joern=True, semgrep=True)
        report = format_capability_report(caps)
        assert "Joern" in report
        assert "Semgrep" in report
        assert "Frida" in report  # listed as unavailable
        assert "unavailable" in report

    def test_binary_no_dwarf(self):
        caps = AuditCapabilities(binary_available=True, dwarf_available=False)
        report = format_capability_report(caps)
        assert "no DWARF" in report
