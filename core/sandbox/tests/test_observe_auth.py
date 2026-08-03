"""Authentication tests for target-writable observe JSONL."""

from __future__ import annotations

import json

import pytest

from core.sandbox.observe_auth import ObserveRecordSigner
from core.sandbox.observe_profile import OBSERVE_FILENAME, parse_observe_log


_KEY = "7a" * 32
_NONCE = "public-run-id"


def _open_record(path: str) -> dict:
    return {
        "ts": "2026-08-02T00:00:00Z",
        "cmd": f"<sandbox audit: openat {path}>",
        "returncode": 0,
        "type": "write",
        "observe": True,
        "syscall": "openat",
        "syscall_nr": 257,
        "target_pid": 1234,
        "args": [-100, 0, 0, 0, 0, 0],
        "path": path,
    }


def _write_jsonl(path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_hmac_rejects_forged_append_using_visible_nonce(tmp_path):
    """A target can read the nonce but cannot manufacture its HMAC."""
    signer = ObserveRecordSigner(_KEY, run_id=_NONCE)
    canonical = _open_record("/etc/hostname")
    signer.sign(canonical)
    observe_path = tmp_path / OBSERVE_FILENAME
    _write_jsonl(observe_path, [canonical])

    # Model the target: read one live canonical record, learn its public
    # nonce, then append a recognized record with invented content.
    learned_nonce = json.loads(observe_path.read_text(encoding="utf-8").splitlines()[0])["nonce"]
    forged = _open_record("/attacker-forged")
    forged["nonce"] = learned_nonce
    forged["observe_seq"] = 2
    with observe_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(forged) + "\n")

    profile = parse_observe_log(
        tmp_path,
        expected_nonce=_NONCE,
        expected_hmac_key=_KEY,
    )
    assert profile.paths_read == ["/etc/hostname"]
    assert "/attacker-forged" not in profile.paths_read


def test_hmac_sequence_rejects_replay_and_gap(tmp_path):
    signer = ObserveRecordSigner(_KEY, run_id=_NONCE)
    first = _open_record("/one")
    second = _open_record("/two")
    third = _open_record("/three")
    for record in (first, second, third):
        signer.sign(record)

    # A target replays sequence one and suppresses sequence two. The parser
    # retains only the authenticated contiguous prefix.
    _write_jsonl(tmp_path / OBSERVE_FILENAME, [first, first, third])
    profile = parse_observe_log(
        tmp_path,
        expected_nonce=_NONCE,
        expected_hmac_key=_KEY,
    )
    assert profile.paths_read == ["/one"]


def test_hmac_authenticates_budget_summary_and_markers(tmp_path):
    signer = ObserveRecordSigner(_KEY, run_id=_NONCE)
    marker = {"type": "audit_budget_marker", "dropped": 1}
    summary = {
        "type": "audit_summary",
        "dropped_by_category": {"write": 3},
    }
    for record in (marker, summary):
        signer.sign(record)

    _write_jsonl(tmp_path / OBSERVE_FILENAME, [marker, summary])
    profile = parse_observe_log(
        tmp_path,
        expected_nonce=_NONCE,
        expected_hmac_key=_KEY,
    )
    assert profile.budget_truncated is True
    assert profile.dropped_by_category == {"write": 3}


def test_invalid_hmac_key_fails_closed(tmp_path):
    with pytest.raises(ValueError, match="HMAC key"):
        parse_observe_log(tmp_path, expected_hmac_key="not-a-key")
