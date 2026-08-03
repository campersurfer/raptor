"""Authentication helpers for target-writable observe JSONL records.

Observe logs intentionally live in the sandbox output directory, which lets the
untrusted target write them. A per-run HMAC key stays in the parent and tracer,
while a monotonically increasing sequence number prevents replayed records from
being accepted by the parser.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from typing import Mapping, MutableMapping


AUTH_KEY_BYTES = 32
SEQUENCE_FIELD = "observe_seq"
HMAC_FIELD = "observe_hmac"
_DOMAIN_SEPARATOR = b"raptor-observe-record-v1\x00"


def generate_observe_hmac_key() -> str:
    """Return a fresh 256-bit HMAC key encoded for the audit config."""
    return secrets.token_hex(AUTH_KEY_BYTES)


def _decode_key(key: str) -> bytes:
    if not isinstance(key, str):
        raise ValueError("observe HMAC key must be a hex string")
    try:
        raw = bytes.fromhex(key)
    except ValueError as exc:
        raise ValueError("observe HMAC key must be valid hexadecimal") from exc
    if len(raw) != AUTH_KEY_BYTES:
        raise ValueError(
            f"observe HMAC key must contain {AUTH_KEY_BYTES} bytes"
        )
    return raw


def _canonical_record_bytes(record: Mapping) -> bytes:
    """Encode the signed record fields in a stable, versioned form."""
    unsigned = {
        key: value
        for key, value in record.items()
        if key != HMAC_FIELD
    }
    payload = json.dumps(
        unsigned,
        default=str,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return _DOMAIN_SEPARATOR + payload


class ObserveRecordSigner:
    """Sign one ordered observe-record stream with an in-memory key."""

    def __init__(self, key: str, *, run_id: str | None = None):
        self._key = _decode_key(key)
        self._run_id = run_id
        self._sequence = 0

    def sign(self, record: MutableMapping) -> None:
        """Add the run id, next sequence number, and HMAC in place."""
        self._sequence += 1
        if self._run_id is not None:
            record["nonce"] = self._run_id
        record[SEQUENCE_FIELD] = self._sequence
        record.pop(HMAC_FIELD, None)
        record[HMAC_FIELD] = hmac.new(
            self._key,
            _canonical_record_bytes(record),
            hashlib.sha256,
        ).hexdigest()

    def rollback(self, record: Mapping) -> None:
        """Reuse a sequence slot when its signed record was not written."""
        if record.get(SEQUENCE_FIELD) == self._sequence:
            self._sequence -= 1


def _matches(record: Mapping, key: bytes, expected_sequence: int) -> bool:
    if type(record.get(SEQUENCE_FIELD)) is not int:
        return False
    if record[SEQUENCE_FIELD] != expected_sequence:
        return False
    supplied = record.get(HMAC_FIELD)
    if not isinstance(supplied, str):
        return False
    expected = hmac.new(
        key,
        _canonical_record_bytes(record),
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(supplied, expected)


class ObserveRecordVerifier:
    """Verify one complete, ordered observe-record stream."""

    def __init__(self, key: str):
        self._key = _decode_key(key)
        self._expected_sequence = 1

    def accepts(self, record: Mapping) -> bool:
        """Accept only the next authentic record in this stream."""
        if not _matches(record, self._key, self._expected_sequence):
            return False
        self._expected_sequence += 1
        return True


def verify_record(record: Mapping, key: str, expected_sequence: int) -> bool:
    """Return whether ``record`` has the expected signed sequence slot."""
    return _matches(record, _decode_key(key), expected_sequence)
