"""Offline fixtures for Gemini native structured serialization."""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("google.genai")
from google.genai import types

from core.llm.config import ModelConfig
from core.llm.providers import GeminiProvider


def _terminal_map_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "context_map": {
                "type": "object",
                "properties": {
                    "sources": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"name": {"type": "string"}},
                            "required": ["name"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["sources"],
                "additionalProperties": False,
            },
        },
        "required": ["context_map"],
        "additionalProperties": False,
    }


class _Models:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            text=json.dumps({"context_map": {"sources": []}}),
            candidates=[],
            usage_metadata=None,
        )


def test_native_structured_terminal_map_uses_sdk_schema_serialization() -> None:
    provider = GeminiProvider(ModelConfig(provider="gemini", model_name="gemini-test"))
    models = _Models()
    provider._clients[threading.get_ident()] = SimpleNamespace(models=models)

    result = provider.generate_structured("map", _terminal_map_schema())

    assert result.result == {"context_map": {"sources": []}}
    config = models.calls[0]["config"]
    wire_config = types.GenerateContentConfig.model_validate(config).model_dump(
        by_alias=True,
        exclude_none=True,
    )
    assert "responseJsonSchema" not in wire_config
    assert wire_config["responseSchema"] == config["response_schema"]
    assert "response_json_schema" not in config
    assert config["response_schema"] == {
        "type": "OBJECT",
        "properties": {
            "context_map": {
                "type": "OBJECT",
                "properties": {
                    "sources": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {"name": {"type": "STRING"}},
                            "required": ["name"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["sources"],
                "additionalProperties": False,
            },
        },
        "required": ["context_map"],
        "additionalProperties": False,
    }
