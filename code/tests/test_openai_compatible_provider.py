from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping

import pytest

from paretoskill.models import canonical_json
from paretoskill.providers import (
    ExecutionRequest,
    MockProvider,
    ModelSpec,
    NetworkDisabledError,
    NetworkPolicy,
    OpenAICompatibleConfig,
    OpenAICompatibleProvider,
    ProviderError,
    ResponseSchemaError,
    TransportError,
    TransportResponse,
    build_provider,
)


def execution_request() -> ExecutionRequest:
    return ExecutionRequest(
        experiment_id="exp",
        candidate_id="candidate",
        content_hash="a" * 64,
        task_id="task",
        seed=7,
        target_id="target",
        model=ModelSpec(
            "test-model",
            "compat",
            "frozen-revision",
            {"temperature": 0.0, "max_tokens": 32},
        ),
        skill_files={"SKILL.md": "Use the supplied rubric."},
        task_payload={"question": "six times seven", "expected": 42},
    )


def chat_response(
    content: str = '{"answer":42,"correct":false}',
    *,
    prompt_tokens: Any = 12,
    completion_tokens: Any = 5,
) -> TransportResponse:
    body = {
        "id": "response-1",
        "choices": [
            {
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }
    return TransportResponse(200, json.dumps(body).encode("utf-8"))


@dataclass
class FakeTransport:
    outcomes: list[TransportResponse | Exception]
    calls: list[dict[str, Any]] = field(default_factory=list)

    def post_json(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> TransportResponse:
        self.calls.append(
            {
                "url": url,
                "headers": dict(headers),
                "payload": dict(payload),
                "timeout_seconds": timeout_seconds,
                "max_response_bytes": max_response_bytes,
            }
        )
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def enabled_policy(monkeypatch) -> NetworkPolicy:
    monkeypatch.setenv("PARETOSKILL_ENABLE_NETWORK", "1")
    return NetworkPolicy(config_allows_network=True, cli_allows_network=True)


def adapter(
    transport: FakeTransport,
    policy: NetworkPolicy,
    *,
    verifier=None,
    clock=None,
    sleeper=None,
    max_retries: int = 2,
) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        config=OpenAICompatibleConfig(
            provider_id="compat",
            base_url="https://example.invalid/v1",
            api_key_env="PARETOSKILL_TEST_API_KEY",
            timeout_seconds=3.0,
            max_retries=max_retries,
            retry_backoff_seconds=0.01,
            max_output_tokens=64,
        ),
        policy=policy,
        transport=transport,
        verifier=verifier,
        clock=clock or (lambda: 1.0),
        sleeper=sleeper or (lambda delay: None),
    )


def test_default_policy_blocks_before_key_lookup_or_transport(monkeypatch) -> None:
    monkeypatch.delenv("PARETOSKILL_ENABLE_NETWORK", raising=False)
    monkeypatch.delenv("PARETOSKILL_TEST_API_KEY", raising=False)
    transport = FakeTransport([chat_response()])
    provider = adapter(transport, NetworkPolicy())

    with pytest.raises(NetworkDisabledError, match="No network request"):
        provider.generate(execution_request())
    assert transport.calls == []


def test_generate_exposes_structured_text_without_trusting_correct(monkeypatch) -> None:
    monkeypatch.setenv("PARETOSKILL_TEST_API_KEY", "super-secret-value")
    transport = FakeTransport([chat_response()])
    provider = adapter(
        transport,
        enabled_policy(monkeypatch),
        clock=iter((10.0, 10.125)).__next__,
    )

    generated = provider.generate(execution_request())

    assert generated.response_text == '{"answer":42,"correct":false}'
    assert generated.parsed_output["answer"] == 42
    assert generated.latency_ms == 125.0
    assert generated.input_tokens == 12
    assert generated.output_tokens == 5
    call = transport.calls[0]
    assert call["url"] == "https://example.invalid/v1/chat/completions"
    assert call["timeout_seconds"] == 3.0
    user_payload = json.loads(call["payload"]["messages"][1]["content"])
    assert user_payload["task"]["question"] == "six times seven"
    assert user_payload["skill_files"] == {"SKILL.md": "Use the supplied rubric."}
    assert call["payload"]["max_tokens"] == 32
    assert call["payload"]["seed"] == 7
    assert "super-secret-value" not in canonical_json(generated.provider_metadata)
    assert "super-secret-value" not in execution_request().cache_key
    assert "super-secret-value" not in repr(provider)


def test_execute_requires_trusted_verifier_and_overrides_model_claim(monkeypatch) -> None:
    monkeypatch.setenv("PARETOSKILL_TEST_API_KEY", "fake-key")
    no_verifier_transport = FakeTransport([chat_response()])
    no_verifier = adapter(no_verifier_transport, enabled_policy(monkeypatch))
    with pytest.raises(ProviderError, match="trusted verifier"):
        no_verifier.execute(execution_request())
    assert no_verifier_transport.calls == []

    provider = adapter(
        FakeTransport([chat_response()]),
        enabled_policy(monkeypatch),
        verifier=lambda request, output: output["answer"] == request.task_payload["expected"],
        clock=iter((2.0, 2.01)).__next__,
    )
    result = provider.execute(execution_request())
    assert result.correct is True
    assert result.trace["response_text"] == '{"answer":42,"correct":false}'
    assert result.trace["model_output"]["correct"] is False


def test_retry_and_response_schema_are_bounded(monkeypatch) -> None:
    monkeypatch.setenv("PARETOSKILL_TEST_API_KEY", "fake-key")
    sleeps: list[float] = []
    transport = FakeTransport(
        [
            TransportResponse(429, b'{"error":"busy"}'),
            TransportError("temporary"),
            chat_response(),
        ]
    )
    provider = adapter(
        transport,
        enabled_policy(monkeypatch),
        clock=iter((1.0, 1.1)).__next__,
        sleeper=sleeps.append,
        max_retries=2,
    )
    generated = provider.generate(execution_request())
    assert generated.provider_metadata["attempts"] == 3
    assert sleeps == [0.01, 0.02]

    malformed = adapter(
        FakeTransport([chat_response(prompt_tokens=True)]),
        enabled_policy(monkeypatch),
    )
    with pytest.raises(ResponseSchemaError, match="prompt_tokens"):
        malformed.generate(execution_request())


def test_missing_key_and_factory_are_offline_safe(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("PARETOSKILL_TEST_API_KEY", raising=False)
    transport = FakeTransport([chat_response()])
    provider = build_provider(
        {
            "type": "openai_compatible",
            "provider_id": "compat",
            "base_url": "https://example.invalid/v1",
            "api_key_env": "PARETOSKILL_TEST_API_KEY",
        },
        policy=enabled_policy(monkeypatch),
        transport=transport,
    )
    with pytest.raises(ProviderError, match="environment variable"):
        provider.generate(execution_request())
    assert transport.calls == []

    assert isinstance(build_provider({"type": "mock"}), MockProvider)
    with pytest.raises(ProviderError, match="disabled by default"):
        build_provider(
            {"type": "dotted_path", "class_path": "paretoskill.providers.MockProvider"}
        )
    dynamic = build_provider(
        {
            "type": "dotted_path",
            "class_path": "paretoskill.providers.MockProvider",
            "kwargs": {"provider_id": "local"},
        },
        allow_dynamic_imports=True,
        allowed_dynamic_prefixes=("paretoskill",),
    )
    assert isinstance(dynamic, MockProvider)
    assert dynamic.provider_id == "local"

    external = build_provider(
        {
            "type": "dotted_path",
            "class_path": "paretoskill.providers.DisabledExternalProvider",
            "kwargs": {
                "provider_id": "dynamic-external",
                "policy": NetworkPolicy(
                    config_allows_network=True,
                    cli_allows_network=True,
                ),
            },
        },
        allow_dynamic_imports=True,
        allowed_dynamic_prefixes=("paretoskill",),
    )
    with pytest.raises(NetworkDisabledError):
        external.execute(execution_request())

    replay_path = tmp_path / "replay.jsonl"
    replay_path.write_text(
        json.dumps(
            {
                "cache_key": execution_request().cache_key,
                "result": {
                    "correct": True,
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "latency_ms": 0.0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    replay = build_provider({"type": "replay", "path": str(replay_path)})
    assert replay.execute(execution_request()).correct is True


def test_adapter_config_rejects_unsafe_or_unbounded_values() -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        OpenAICompatibleConfig("compat", "http://remote.example/v1", "KEY")
    with pytest.raises(ValueError, match="max_retries"):
        OpenAICompatibleConfig(
            "compat",
            "https://example.invalid/v1",
            "KEY",
            max_retries=11,
        )
    with pytest.raises(ProviderError, match="unknown provider config"):
        build_provider({"type": "mock", "api_key": "must-not-be-accepted"})
