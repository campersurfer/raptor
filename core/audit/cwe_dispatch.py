"""CWE-to-tool dispatch mapping for /audit.

Single source of truth for mapping CWE classes to tool-specific
dispatch rules (SMT verbs, Coccinelle rules, Joern taint, CodeQL
queries) and sink targets.

Consumed by:
  - Step 1e CWE fallback in _hypothesis_to_tool_chain
  - Step 1e' proactive tool validation
  - Any future CWE→tool consumer
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

CWE_TO_TOOL_DISPATCH: Dict[str, Dict[str, Any]] = {
    # Memory / bounds
    "CWE-120": {
        "smt": "check-oob",
        "cocci": None,
        "joern": True,
        "codeql": "cpp/overflow-buffer",
        "sinks": ["memcpy", "strcpy", "strncpy", "sprintf", "gets"],
    },
    "CWE-122": {
        "smt": "check-oob",
        "cocci": None,
        "joern": True,
        "codeql": "cpp/overflow-buffer",
        "sinks": ["memcpy", "strcpy", "strncpy", "sprintf", "gets"],
    },
    "CWE-125": {
        "smt": "check-oob",
        "cocci": None,
        "joern": True,
        "codeql": "cpp/out-of-bounds-read",
        "sinks": ["memcmp", "strlen", "strstr", "strcmp", "strncmp", "read", "fread"],
    },
    "CWE-787": {
        "smt": "check-oob",
        "cocci": None,
        "joern": True,
        "codeql": "cpp/overflow-buffer",
        "sinks": ["memcpy", "strcpy", "strncpy", "sprintf", "gets"],
    },
    # Integer
    "CWE-190": {
        "smt": "check-overflow",
        "cocci": None,
        "joern": False,
        "codeql": "cpp/integer-overflow",
        "sinks": [],
    },
    "CWE-191": {
        "smt": "check-overflow",
        "cocci": None,
        "joern": False,
        "codeql": "cpp/integer-overflow",
        "sinks": [],
    },
    # Injection
    "CWE-78": {
        "smt": None,
        "cocci": None,
        "joern": True,
        "codeql": "cpp/command-line-injection",
        "sinks": ["os.system", "popen", "execve", "subprocess.Popen"],
    },
    "CWE-89": {
        "smt": None,
        "cocci": None,
        "joern": True,
        "codeql": "py/sql-injection",
        "sinks": ["sqlite3_exec", "cursor.execute", "mysql_query"],
    },
    "CWE-79": {
        "smt": None,
        "cocci": None,
        "joern": True,
        "codeql": "js/xss",
        "sinks": ["document.write", "innerHTML", "eval"],
    },
    "CWE-90": {
        "smt": None,
        "cocci": None,
        "joern": True,
        "codeql": None,
        "sinks": ["ldap_search", "ldap_search_s"],
    },
    "CWE-94": {
        "smt": None,
        "cocci": None,
        "joern": True,
        "codeql": None,
        "sinks": ["eval", "exec", "loadstring", "dofile"],
    },
    # Error handling
    "CWE-252": {
        "smt": None,
        "cocci": "unchecked_return.cocci",
        "joern": False,
        "codeql": None,
        "sinks": [],
    },
    "CWE-476": {
        "smt": None,
        "cocci": "missing_null_check.cocci",
        "joern": False,
        "codeql": "cpp/null-dereference",
        "sinks": [],
    },
    # Format string
    "CWE-134": {
        "smt": None,
        "cocci": None,
        "joern": True,
        "codeql": "cpp/non-constant-format",
        "sinks": ["printf", "fprintf", "sprintf", "syslog"],
    },
    # Concurrency
    "CWE-362": {
        "smt": None,
        "cocci": "lock_imbalance.cocci",
        "joern": False,
        "codeql": None,
        "sinks": [],
    },
    "CWE-667": {
        "smt": None,
        "cocci": "lock_imbalance.cocci",
        "joern": False,
        "codeql": None,
        "sinks": [],
    },
    # Authentication / authorisation
    "CWE-287": {
        "smt": None,
        "cocci": None,
        "joern": True,
        "codeql": None,
        "sinks": [
            "authenticate", "verify_password", "check_credentials",
            "login", "verify_token", "jwt.decode", "check_auth",
        ],
        "dark_verify": True,
    },
    "CWE-862": {
        "smt": None,
        "cocci": None,
        "joern": True,
        "codeql": None,
        "sinks": [
            "authorize", "check_permission", "has_role", "is_admin",
            "check_access", "require_auth", "can_access",
        ],
        "dark_verify": True,
    },
    "CWE-863": {
        "smt": None,
        "cocci": None,
        "joern": True,
        "codeql": None,
        "sinks": [
            "authorize", "check_permission", "has_role", "is_admin",
            "check_access", "require_auth", "can_access",
        ],
        "dark_verify": True,
    },
}


def lookup(cwe: str) -> Optional[Dict[str, Any]]:
    """Look up dispatch rules for a CWE identifier.

    Accepts "CWE-78", "cwe-78", "78", etc.
    """
    normalized = cwe.upper().strip()
    if not normalized.startswith("CWE-"):
        normalized = f"CWE-{normalized}"
    return CWE_TO_TOOL_DISPATCH.get(normalized)


def sinks_for_cwe(cwe: str) -> List[str]:
    """Return sink targets for a CWE, or empty list."""
    entry = lookup(cwe)
    if entry is None:
        return []
    return entry.get("sinks", [])


def smt_verb_for_cwe(cwe: str) -> Optional[str]:
    """Return the SMT verb for a CWE, or None."""
    entry = lookup(cwe)
    if entry is None:
        return None
    return entry.get("smt")


def cocci_rule_for_cwe(cwe: str) -> Optional[str]:
    """Return the Coccinelle rule filename for a CWE, or None."""
    entry = lookup(cwe)
    if entry is None:
        return None
    return entry.get("cocci")


def codeql_query_for_cwe(cwe: str) -> Optional[str]:
    """Return the CodeQL query ID for a CWE, or None."""
    entry = lookup(cwe)
    if entry is None:
        return None
    return entry.get("codeql")


def joern_applicable(cwe: str) -> bool:
    """Return whether Joern taint tracking is useful for a CWE."""
    entry = lookup(cwe)
    if entry is None:
        return False
    return bool(entry.get("joern")) and bool(entry.get("sinks"))


def dark_verify_applicable(cwe: str) -> bool:
    """Return whether dark-verify witness execution is the primary grounding for a CWE."""
    entry = lookup(cwe)
    if entry is None:
        return False
    return bool(entry.get("dark_verify"))
