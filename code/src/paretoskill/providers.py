"""Provider-neutral execution contracts with an offline-by-default safety boundary."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

from .models import canonical_json, freeze_json, freeze_mapping, thaw_json


class ProviderError(RuntimeError):
    pass


class NetworkDisabledError(ProviderError):
    pass


class ReplayMissError(ProviderError):
    pass


class TransportError(ProviderError):
    """A retryable failure before a valid HTTP response was received."""


class ResponseSchemaError(ProviderError):
    """The endpoint returned a response that cannot be evaluated safely."""


@dataclass(frozen=True, slots=True)
class NetworkPolicy:
    """External execution requires config, CLI, and environment consent."""

    offline_by_default: bool = True
    config_allows_network: bool = False
    cli_allows_network: bool = False
    required_env: str = "PARETOSKILL_ENABLE_NETWORK"
    required_value: str = "1"

    def __post_init__(self) -> None:
        for name in (
            "offline_by_default",
            "config_allows_network",
            "cli_allows_network",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        if not isinstance(self.required_env, str) or not self.required_env.strip():
            raise ValueError("required_env must be non-empty")
        if not isinstance(self.required_value, str) or not self.required_value:
            raise ValueError("required_value must be non-empty")

    @property
    def external_enabled(self) -> bool:
        configured = os.environ.get(self.required_env, "")
        env_enabled = configured == self.required_value
        return self.config_allows_network and self.cli_allows_network and env_enabled

    def require_external_enabled(self) -> None:
        if not self.external_enabled:
            raise NetworkDisabledError(
                "external providers are disabled; enabling one requires all three of: "
                "safety.allow_network=true in the manifest, CLI --allow-network, and "
                f"{self.required_env}={self.required_value!r}. No network request was made."
            )


@dataclass(frozen=True, slots=True)
class ModelSpec:
    model_id: str
    provider_id: str
    revision: str
    decoding: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value.strip()
            for value in (self.model_id, self.provider_id, self.revision)
        ):
            raise ValueError("model_id, provider_id, and revision must be frozen and non-empty")
        object.__setattr__(self, "decoding", freeze_mapping(self.decoding))


@dataclass(frozen=True, slots=True)
class ExecutionRequest:
    experiment_id: str
    candidate_id: str
    content_hash: str
    task_id: str
    seed: int
    target_id: str
    model: ModelSpec
    skill_files: Mapping[str, str]
    task_payload: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        identifiers = (self.experiment_id, self.candidate_id, self.task_id, self.target_id)
        if not all(isinstance(value, str) and value.strip() for value in identifiers):
            raise ValueError("execution request identifiers must be non-empty strings")
        if not isinstance(self.content_hash, str) or len(self.content_hash) != 64 or any(
            character not in "0123456789abcdef" for character in self.content_hash
        ):
            raise ValueError("execution content_hash must be lowercase SHA-256 hex")
        if not isinstance(self.model, ModelSpec):
            raise ValueError("execution request model must be a ModelSpec")
        if not isinstance(self.skill_files, Mapping) or any(
            not isinstance(path, str) or not isinstance(content, str)
            for path, content in self.skill_files.items()
        ):
            raise ValueError("execution skill_files must map string paths to string content")
        object.__setattr__(self, "skill_files", freeze_mapping(self.skill_files))
        object.__setattr__(self, "task_payload", freeze_mapping(self.task_payload))
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("execution seed must be an integer")

    @property
    def cache_key(self) -> str:
        payload = {
            "schema_version": 1,
            "experiment_id": self.experiment_id,
            "content_hash": self.content_hash,
            "task_id": self.task_id,
            "seed": self.seed,
            "target_id": self.target_id,
            "model_id": self.model.model_id,
            "model_revision": self.model.revision,
            "decoding": thaw_json(self.model.decoding),
            "task_payload": thaw_json(self.task_payload),
            "execution_metadata": thaw_json(self.metadata),
        }
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    correct: bool
    input_tokens: int
    output_tokens: int
    latency_ms: float
    trace: Mapping[str, Any] = field(default_factory=dict)
    provider_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.correct, bool):
            raise ValueError("correct must be a boolean")
        token_values = (self.input_tokens, self.output_tokens)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in token_values):
            raise ValueError("token counts must be integers")
        if isinstance(self.latency_ms, bool) or not isinstance(self.latency_ms, (int, float)):
            raise ValueError("latency_ms must be numeric")
        if not math.isfinite(float(self.latency_ms)):
            raise ValueError("latency_ms must be finite")
        if self.input_tokens < 0 or self.output_tokens < 0 or self.latency_ms < 0:
            raise ValueError("token counts and latency must be non-negative")
        object.__setattr__(self, "trace", freeze_mapping(self.trace))
        object.__setattr__(
            self,
            "provider_metadata",
            freeze_mapping(self.provider_metadata),
        )

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "correct": self.correct,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": self.latency_ms,
            "trace": thaw_json(self.trace),
            "provider_metadata": thaw_json(self.provider_metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ExecutionResult:
        correct = value["correct"]
        if not isinstance(correct, bool):
            raise ValueError("correct must be a JSON boolean")
        return cls(
            correct=correct,
            input_tokens=value["input_tokens"],
            output_tokens=value["output_tokens"],
            latency_ms=value["latency_ms"],
            trace=value.get("trace", {}),
            provider_metadata=value.get("provider_metadata", {}),
        )


@runtime_checkable
class Provider(Protocol):
    provider_id: str
    is_external: bool

    def execute(self, request: ExecutionRequest) -> ExecutionResult: ...


@dataclass(frozen=True, slots=True)
class TransportResponse:
    """Small HTTP response value used by injectable and stdlib transports."""

    status_code: int
    body: bytes
    headers: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            isinstance(self.status_code, bool)
            or not isinstance(self.status_code, int)
            or not 100 <= self.status_code <= 599
        ):
            raise ValueError("transport status_code must be an integer from 100 through 599")
        if not isinstance(self.body, bytes):
            raise ValueError("transport body must be bytes")
        if not isinstance(self.headers, Mapping) or any(
            not isinstance(name, str) or not isinstance(value, str)
            for name, value in self.headers.items()
        ):
            raise ValueError("transport headers must map strings to strings")
        object.__setattr__(self, "headers", MappingProxyType(dict(self.headers)))


class JSONTransport(Protocol):
    def post_json(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> TransportResponse: ...


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        del request, file_pointer, code, message, headers, new_url
        return None


@dataclass(frozen=True, slots=True)
class StdlibJSONTransport:
    """Minimal synchronous JSON transport with bounded response reads."""

    user_agent: str = "ParetoSkill/0.1"

    def post_json(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> TransportResponse:
        request_headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": self.user_agent,
            **dict(headers),
        }
        body = canonical_json(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers=request_headers,
            method="POST",
        )
        try:
            opener = urllib.request.build_opener(_NoRedirectHandler())
            with opener.open(request, timeout=timeout_seconds) as response:
                response_body = response.read(max_response_bytes + 1)
                if len(response_body) > max_response_bytes:
                    raise ResponseSchemaError("provider response exceeds configured byte limit")
                return TransportResponse(
                    status_code=response.status,
                    body=response_body,
                    headers={name: value for name, value in response.headers.items()},
                )
        except urllib.error.HTTPError as exc:
            response_body = exc.read(max_response_bytes + 1)
            if len(response_body) > max_response_bytes:
                raise ResponseSchemaError("provider error response exceeds byte limit") from exc
            return TransportResponse(
                status_code=exc.code,
                body=response_body,
                headers=(
                    {}
                    if exc.headers is None
                    else {name: value for name, value in exc.headers.items()}
                ),
            )
        except (TimeoutError, urllib.error.URLError, OSError) as exc:
            raise TransportError(
                f"provider transport failed ({type(exc).__name__}); response details suppressed"
            ) from exc


OutputVerifier = Callable[[ExecutionRequest, Any], bool]


@dataclass(frozen=True, slots=True)
class GeneratedResponse:
    """Unscored model generation safe to consume by a proposer or trusted verifier."""

    response_text: str
    parsed_output: Any
    input_tokens: int
    output_tokens: int
    latency_ms: float
    finish_reason: str | None = None
    provider_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.response_text, str):
            raise ValueError("response_text must be a string")
        for name in ("input_tokens", "output_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if (
            isinstance(self.latency_ms, bool)
            or not isinstance(self.latency_ms, (int, float))
            or not math.isfinite(float(self.latency_ms))
            or self.latency_ms < 0
        ):
            raise ValueError("latency_ms must be a finite non-negative number")
        if self.finish_reason is not None and not isinstance(self.finish_reason, str):
            raise ValueError("finish_reason must be a string when present")
        object.__setattr__(self, "parsed_output", freeze_json(self.parsed_output))
        object.__setattr__(self, "provider_metadata", freeze_mapping(self.provider_metadata))


@dataclass(frozen=True, slots=True)
class OpenAICompatibleConfig:
    """Non-secret configuration for a Chat Completions compatible endpoint."""

    provider_id: str
    base_url: str
    api_key_env: str
    timeout_seconds: float = 60.0
    max_retries: int = 2
    retry_backoff_seconds: float = 0.25
    max_input_tokens: int = 1_000_000
    max_output_tokens: int = 4096
    max_response_bytes: int = 2_000_000
    allow_insecure_http: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.provider_id, str) or not self.provider_id.strip():
            raise ValueError("provider_id must be non-empty")
        if (
            not isinstance(self.api_key_env, str)
            or not self.api_key_env.isascii()
            or not self.api_key_env.isidentifier()
        ):
            raise ValueError("api_key_env must name an environment variable")
        numeric_limits = (
            ("timeout_seconds", self.timeout_seconds, 3600.0),
            ("retry_backoff_seconds", self.retry_backoff_seconds, 60.0),
        )
        for name, value, upper_bound in numeric_limits:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0 < value <= upper_bound
            ):
                raise ValueError(f"{name} must be finite, positive, and bounded")
        if (
            isinstance(self.max_retries, bool)
            or not isinstance(self.max_retries, int)
            or not 0 <= self.max_retries <= 10
        ):
            raise ValueError("max_retries must be an integer from 0 through 10")
        integer_limits = (
            ("max_input_tokens", self.max_input_tokens, 10_000_000),
            ("max_output_tokens", self.max_output_tokens, 10_000_000),
            ("max_response_bytes", self.max_response_bytes, 100_000_000),
        )
        for name, value, upper_bound in integer_limits:
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 < value <= upper_bound
            ):
                raise ValueError(f"{name} must be a positive bounded integer")
        if not isinstance(self.allow_insecure_http, bool):
            raise ValueError("allow_insecure_http must be a boolean")
        if not isinstance(self.base_url, str) or not self.base_url.strip():
            raise ValueError("base_url must be a non-empty string")
        parsed = urllib.parse.urlsplit(self.base_url)
        valid_scheme = parsed.scheme == "https" or (
            parsed.scheme == "http" and self.allow_insecure_http
        )
        if (
            not valid_scheme
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "base_url must be an HTTPS URL without credentials, query, or fragment"
            )

    @property
    def endpoint_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"


_ALLOWED_DECODING_KEYS = frozenset(
    {
        "frequency_penalty",
        "max_completion_tokens",
        "max_output_tokens",
        "max_tokens",
        "presence_penalty",
        "response_format",
        "seed",
        "stop",
        "temperature",
        "top_p",
    }
)
_RETRYABLE_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


@dataclass(slots=True)
class OpenAICompatibleProvider:
    """Strict Chat Completions adapter; every call is policy-gated."""

    config: OpenAICompatibleConfig
    policy: NetworkPolicy = field(default_factory=NetworkPolicy)
    transport: JSONTransport = field(default_factory=StdlibJSONTransport, repr=False)
    verifier: OutputVerifier | None = field(default=None, repr=False)
    sleeper: Callable[[float], None] = field(default=time.sleep, repr=False)
    clock: Callable[[], float] = field(default=time.perf_counter, repr=False)
    is_external: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.config, OpenAICompatibleConfig):
            raise ValueError("config must be an OpenAICompatibleConfig")
        if not isinstance(self.policy, NetworkPolicy):
            raise ValueError("policy must be a NetworkPolicy")
        if not callable(getattr(self.transport, "post_json", None)):
            raise ValueError("transport must implement post_json")
        if self.verifier is not None and not callable(self.verifier):
            raise ValueError("verifier must be callable when present")
        if not callable(self.sleeper) or not callable(self.clock):
            raise ValueError("sleeper and clock must be callable")

    @property
    def provider_id(self) -> str:
        return self.config.provider_id

    @staticmethod
    def _validate_decoding(decoding: Mapping[str, Any]) -> None:
        numeric_ranges = {
            "frequency_penalty": (-2.0, 2.0),
            "presence_penalty": (-2.0, 2.0),
            "temperature": (0.0, 2.0),
            "top_p": (0.0, 1.0),
        }
        for name, (lower, upper) in numeric_ranges.items():
            if name not in decoding:
                continue
            value = decoding[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not lower <= value <= upper
            ):
                raise ProviderError(f"decoding.{name} is outside its supported range")
        if "seed" in decoding and (
            isinstance(decoding["seed"], bool) or not isinstance(decoding["seed"], int)
        ):
            raise ProviderError("decoding.seed must be an integer")
        if "stop" in decoding:
            stop = decoding["stop"]
            valid_stop = isinstance(stop, str) or (
                isinstance(stop, list)
                and 0 < len(stop) <= 4
                and all(isinstance(item, str) for item in stop)
            )
            if not valid_stop:
                raise ProviderError("decoding.stop must be a string or up to four strings")
        if "response_format" in decoding and not isinstance(
            decoding["response_format"], Mapping
        ):
            raise ProviderError("decoding.response_format must be an object")

    def _payload(self, request: ExecutionRequest) -> dict[str, Any]:
        if request.model.provider_id != self.provider_id:
            raise ProviderError("request model provider_id does not match provider")
        decoding = thaw_json(request.model.decoding)
        unknown = sorted(set(decoding) - _ALLOWED_DECODING_KEYS)
        if unknown:
            raise ProviderError(f"unsupported decoding settings: {unknown}")
        self._validate_decoding(decoding)
        configured_output_limit = decoding.pop("max_output_tokens", None)
        if configured_output_limit is not None:
            if "max_tokens" in decoding or "max_completion_tokens" in decoding:
                raise ProviderError(
                    "max_output_tokens cannot be combined with provider token-limit fields"
                )
            decoding["max_tokens"] = configured_output_limit
        if "max_tokens" in decoding and "max_completion_tokens" in decoding:
            raise ProviderError("choose only one output-token parameter")
        token_key = (
            "max_completion_tokens"
            if "max_completion_tokens" in decoding
            else "max_tokens"
        )
        requested_tokens = decoding.get(token_key, self.config.max_output_tokens)
        if (
            isinstance(requested_tokens, bool)
            or not isinstance(requested_tokens, int)
            or not 0 < requested_tokens <= self.config.max_output_tokens
        ):
            raise ProviderError(
                "requested output-token limit must be a positive integer within provider cap"
            )
        decoding[token_key] = requested_tokens
        if "seed" not in decoding:
            decoding["seed"] = request.seed
        task_message = {
            "task": thaw_json(request.task_payload),
            "skill_files": thaw_json(request.skill_files),
        }
        return {
            "model": request.model.model_id,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Complete the structured task using the supplied skill files. "
                        "Return JSON when the task supports it. Do not claim correctness "
                        "unless the task itself asks for a boolean correctness judgment."
                    ),
                },
                {"role": "user", "content": canonical_json(task_message)},
            ],
            **decoding,
        }

    @staticmethod
    def _json_response(response: TransportResponse) -> Mapping[str, Any]:
        try:
            decoded = response.body.decode("utf-8")
            value = json.loads(decoded, parse_constant=_reject_json_constant)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ResponseSchemaError("provider response is not valid UTF-8 JSON") from exc
        if not isinstance(value, Mapping):
            raise ResponseSchemaError("provider response root must be a JSON object")
        return value

    @staticmethod
    def _model_output(content: str) -> Any:
        stripped = content.strip()
        if stripped.startswith("```") and stripped.endswith("```"):
            lines = stripped.splitlines()
            if len(lines) >= 3 and lines[-1].strip() == "```":
                stripped = "\n".join(lines[1:-1])
                if stripped.lstrip().lower().startswith("json\n"):
                    stripped = stripped.lstrip()[5:]
        try:
            return json.loads(stripped, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, ValueError):
            return content

    def _generation(
        self,
        response: TransportResponse,
        *,
        latency_ms: float,
        attempts: int,
        output_token_limit: int,
    ) -> GeneratedResponse:
        value = self._json_response(response)
        choices = value.get("choices")
        usage = value.get("usage")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ResponseSchemaError("provider response must contain exactly one choice")
        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise ResponseSchemaError("provider choice must be an object")
        message = choice.get("message")
        if not isinstance(message, Mapping):
            raise ResponseSchemaError("provider choice.message must be an object")
        if message.get("refusal") not in (None, ""):
            raise ResponseSchemaError("provider refused the request")
        content = message.get("content")
        if not isinstance(content, str):
            raise ResponseSchemaError("provider message.content must be a string")
        if not isinstance(usage, Mapping):
            raise ResponseSchemaError("provider response usage must be an object")
        input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")
        for name, token_count in (
            ("prompt_tokens", input_tokens),
            ("completion_tokens", output_tokens),
        ):
            if (
                isinstance(token_count, bool)
                or not isinstance(token_count, int)
                or token_count < 0
            ):
                raise ResponseSchemaError(f"usage.{name} must be a non-negative integer")
        if input_tokens > self.config.max_input_tokens:
            raise ResponseSchemaError("reported prompt tokens exceed provider cap")
        if output_tokens > output_token_limit:
            raise ResponseSchemaError("reported completion tokens exceed request limit")
        total_tokens = usage.get("total_tokens")
        if total_tokens is not None and (
            isinstance(total_tokens, bool)
            or not isinstance(total_tokens, int)
            or total_tokens != input_tokens + output_tokens
        ):
            raise ResponseSchemaError("usage.total_tokens must equal prompt plus completion")
        output = self._model_output(content)
        response_id = value.get("id")
        if response_id is not None and not isinstance(response_id, str):
            raise ResponseSchemaError("provider response id must be a string when present")
        finish_reason = choice.get("finish_reason")
        if finish_reason is not None and not isinstance(finish_reason, str):
            raise ResponseSchemaError("choice finish_reason must be a string when present")
        return GeneratedResponse(
            response_text=content,
            parsed_output=output,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            finish_reason=finish_reason,
            provider_metadata={
                "provider": self.provider_id,
                "external": True,
                "response_id": response_id,
                "attempts": attempts,
            },
        )

    def generate(self, request: ExecutionRequest) -> GeneratedResponse:
        """Generate an unscored response; this method never trusts model self-evaluation."""

        self.policy.require_external_enabled()
        api_key = os.environ.get(self.config.api_key_env)
        if api_key is None or not api_key.strip():
            raise ProviderError(
                f"API credential environment variable {self.config.api_key_env!r} is not set"
            )
        api_key = api_key.strip()
        if any(character in api_key for character in ("\r", "\n", "\x00")):
            raise ProviderError("API credential contains forbidden control characters")
        payload = self._payload(request)
        headers = {"Authorization": f"Bearer {api_key}"}
        started = self.clock()
        attempts = 0
        while True:
            attempts += 1
            try:
                response = self.transport.post_json(
                    url=self.config.endpoint_url,
                    headers=headers,
                    payload=payload,
                    timeout_seconds=float(self.config.timeout_seconds),
                    max_response_bytes=self.config.max_response_bytes,
                )
                if not isinstance(response, TransportResponse):
                    raise ResponseSchemaError("transport must return a TransportResponse")
                if len(response.body) > self.config.max_response_bytes:
                    raise ResponseSchemaError("provider response exceeds configured byte limit")
            except TransportError:
                if attempts > self.config.max_retries:
                    raise
            else:
                if 200 <= response.status_code < 300:
                    elapsed_ms = (self.clock() - started) * 1000.0
                    if not math.isfinite(elapsed_ms) or elapsed_ms < 0:
                        raise ProviderError("provider clock returned invalid latency")
                    return self._generation(
                        response,
                        latency_ms=elapsed_ms,
                        attempts=attempts,
                        output_token_limit=int(
                            payload.get(
                                "max_completion_tokens",
                                payload.get("max_tokens", self.config.max_output_tokens),
                            )
                        ),
                    )
                if (
                    response.status_code not in _RETRYABLE_STATUS_CODES
                    or attempts > self.config.max_retries
                ):
                    raise ProviderError(
                        f"provider returned HTTP {response.status_code}; body suppressed"
                    )
            self.sleeper(self.config.retry_backoff_seconds * (2 ** (attempts - 1)))

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        if self.verifier is None:
            raise ProviderError(
                "execute() requires a trusted verifier; use generate() for unscored text"
            )
        generation = self.generate(request)
        correct = self.verifier(request, thaw_json(generation.parsed_output))
        if not isinstance(correct, bool):
            raise ResponseSchemaError("harness verifier must return a boolean")
        return ExecutionResult(
            correct=correct,
            input_tokens=generation.input_tokens,
            output_tokens=generation.output_tokens,
            latency_ms=generation.latency_ms,
            trace={
                "response_text": generation.response_text,
                "model_output": thaw_json(generation.parsed_output),
                "finish_reason": generation.finish_reason,
            },
            provider_metadata=thaw_json(generation.provider_metadata),
        )


@dataclass(slots=True)
class SafetyGatedProvider:
    """Execution gateway that applies policy even if an adapter forgets to."""

    inner: Provider
    policy: NetworkPolicy

    @property
    def provider_id(self) -> str:
        return self.inner.provider_id

    @property
    def is_external(self) -> bool:
        return self.inner.is_external

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        if self.inner.is_external:
            self.policy.require_external_enabled()
        return self.inner.execute(request)

    def generate(self, request: ExecutionRequest) -> GeneratedResponse:
        """Policy-gated unscored generation for verifier/proposer harnesses."""

        if self.inner.is_external:
            self.policy.require_external_enabled()
        generate = getattr(self.inner, "generate", None)
        if not callable(generate):
            raise ProviderError(
                f"provider {self.provider_id!r} does not expose unscored generation"
            )
        response = generate(request)
        if not isinstance(response, GeneratedResponse):
            raise ResponseSchemaError("provider generate() must return GeneratedResponse")
        return response


@dataclass(slots=True)
class MockProvider:
    """A deterministic synthetic provider suitable only for tests and dry-runs."""

    provider_id: str = "mock"
    is_external: bool = False
    scripted: Mapping[str, ExecutionResult] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.scripted = MappingProxyType(dict(self.scripted))

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        if request.cache_key in self.scripted:
            return self.scripted[request.cache_key]
        digest = hashlib.sha256(request.cache_key.encode("ascii")).digest()
        skill_tokens = sum(max(1, len(text.split())) for text in request.skill_files.values())
        # Stable synthetic variation; it is deliberately unrelated to any real model quality.
        correct = digest[0] < int(request.metadata.get("mock_success_threshold", 166))
        return ExecutionResult(
            correct=correct,
            input_tokens=24 + skill_tokens + digest[1] % 9,
            output_tokens=8 + digest[2] % 13,
            latency_ms=float(1 + digest[3] % 5),
            trace={"mode": "synthetic", "digest_prefix": digest[:4].hex()},
            provider_metadata={"provider": self.provider_id, "offline": True},
        )


@dataclass(slots=True)
class ReplayProvider:
    """Strictly replay frozen request results; a miss never falls through to network."""

    records: Mapping[str, ExecutionResult]
    provider_id: str = "replay"
    is_external: bool = False

    def __post_init__(self) -> None:
        copied = dict(self.records)
        for cache_key, result in copied.items():
            invalid = (
                not isinstance(cache_key, str)
                or len(cache_key) != 64
                or any(character not in "0123456789abcdef" for character in cache_key)
            )
            if invalid or not isinstance(result, ExecutionResult):
                raise ValueError("replay records require SHA-256 keys and ExecutionResult values")
        self.records = MappingProxyType(copied)

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        try:
            return self.records[request.cache_key]
        except KeyError as exc:
            raise ReplayMissError(
                f"no replay record for cache key {request.cache_key}; network fallback is forbidden"
            ) from exc

    def generate(self, request: ExecutionRequest) -> GeneratedResponse:
        """Replay an unscored generation when raw model output was preserved."""

        result = self.execute(request)
        trace = thaw_json(result.trace)
        if "model_output" in trace:
            parsed_output = trace["model_output"]
        elif "raw_response" in trace:
            parsed_output = trace["raw_response"]
        elif "response_text" in trace:
            parsed_output = OpenAICompatibleProvider._model_output(trace["response_text"])
        else:
            raise ReplayMissError(
                f"replay record {request.cache_key} has no preserved model output"
            )
        response_text = trace.get("response_text")
        if not isinstance(response_text, str):
            response_text = canonical_json(parsed_output)
        finish_reason = trace.get("finish_reason")
        if finish_reason is not None and not isinstance(finish_reason, str):
            raise ProviderError("replay finish_reason must be a string or null")
        return GeneratedResponse(
            response_text=response_text,
            parsed_output=parsed_output,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_ms=result.latency_ms,
            finish_reason=finish_reason,
            provider_metadata={
                **thaw_json(result.provider_metadata),
                "replay": True,
            },
        )

    @classmethod
    def from_jsonl(cls, path: str | Path, *, provider_id: str = "replay") -> ReplayProvider:
        records: dict[str, ExecutionResult] = {}
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                cache_key = row["cache_key"]
                if not isinstance(cache_key, str):
                    raise ProviderError(f"replay key on line {line_number} must be a string")
                if cache_key in records:
                    raise ProviderError(f"duplicate replay key on line {line_number}: {cache_key}")
                records[cache_key] = ExecutionResult.from_dict(row["result"])
        return cls(records=records, provider_id=provider_id)


@dataclass(slots=True)
class DisabledExternalProvider:
    """Protocol boundary for future adapters; this class contains no API client."""

    provider_id: str
    policy: NetworkPolicy
    is_external: bool = True

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        self.policy.require_external_enabled()
        raise ProviderError(
            f"external provider {self.provider_id!r} has no installed implementation. "
            "Add a separately reviewed adapter before a real run; no request was made."
        )


@dataclass(slots=True)
class ProviderRegistry:
    _providers: dict[str, Provider] = field(default_factory=dict)

    def register(self, provider: Provider) -> None:
        if provider.provider_id in self._providers:
            raise ProviderError(f"duplicate provider id: {provider.provider_id}")
        self._providers[provider.provider_id] = provider

    def get(self, provider_id: str) -> Provider:
        try:
            return self._providers[provider_id]
        except KeyError as exc:
            raise ProviderError(f"unknown provider: {provider_id}") from exc


def _factory_fields(
    config: Mapping[str, Any],
    *,
    allowed: frozenset[str],
) -> dict[str, Any]:
    if not isinstance(config, Mapping) or any(not isinstance(key, str) for key in config):
        raise ProviderError("provider config must be a string-keyed mapping")
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ProviderError(f"unknown provider config fields: {unknown}")
    return dict(config)


def _dynamic_provider(
    config: Mapping[str, Any],
    *,
    policy: NetworkPolicy,
    allow_dynamic_imports: bool,
    allowed_dynamic_prefixes: Sequence[str],
) -> Provider:
    values = _factory_fields(
        config,
        allowed=frozenset({"type", "class_path", "kwargs"}),
    )
    if not allow_dynamic_imports:
        raise ProviderError("dynamic provider imports are disabled by default")
    if isinstance(allowed_dynamic_prefixes, str) or not all(
        isinstance(prefix, str) and prefix.strip() for prefix in allowed_dynamic_prefixes
    ):
        raise ProviderError("allowed_dynamic_prefixes must contain non-empty strings")
    class_path = values.get("class_path")
    if not isinstance(class_path, str) or not class_path.strip() or "." not in class_path:
        raise ProviderError("dynamic provider class_path must be a dotted path")
    module_name, _, attribute = class_path.rpartition(".")
    if not any(
        module_name == prefix or module_name.startswith(f"{prefix}.")
        for prefix in allowed_dynamic_prefixes
    ):
        raise ProviderError("dynamic provider module is outside the explicit allowlist")
    kwargs = values.get("kwargs", {})
    if not isinstance(kwargs, Mapping) or any(not isinstance(key, str) for key in kwargs):
        raise ProviderError("dynamic provider kwargs must be a string-keyed mapping")
    secret_markers = ("authorization", "password", "secret", "token", "api_key")
    if any(any(marker in key.lower() for marker in secret_markers) for key in kwargs):
        raise ProviderError("dynamic provider secrets must not be embedded in config kwargs")
    try:
        module = importlib.import_module(module_name)
        constructor = getattr(module, attribute)
        provider = constructor(**dict(kwargs))
    except Exception:
        raise ProviderError("could not construct allowlisted dynamic provider") from None
    if not isinstance(provider, Provider):
        raise ProviderError("dynamic object does not implement the Provider protocol")
    if provider.is_external:
        return SafetyGatedProvider(provider, policy)
    return provider


def build_provider(
    config: Mapping[str, Any],
    *,
    policy: NetworkPolicy | None = None,
    transport: JSONTransport | None = None,
    verifier: OutputVerifier | None = None,
    allow_dynamic_imports: bool = False,
    allowed_dynamic_prefixes: Sequence[str] = (),
) -> Provider:
    """Construct a provider from strict config without performing external I/O."""

    if not isinstance(config, Mapping):
        raise ProviderError("provider config must be a mapping")
    provider_type = config.get("type")
    if not isinstance(provider_type, str) or not provider_type.strip():
        raise ProviderError("provider config requires a non-empty type")
    network_policy = NetworkPolicy() if policy is None else policy
    if not isinstance(network_policy, NetworkPolicy):
        raise ProviderError("policy must be a NetworkPolicy")

    if provider_type == "mock":
        values = _factory_fields(
            config,
            allowed=frozenset({"type", "provider_id"}),
        )
        provider_id = values.get("provider_id", "mock")
        if not isinstance(provider_id, str) or not provider_id.strip():
            raise ProviderError("mock provider_id must be non-empty")
        return MockProvider(provider_id=provider_id)

    if provider_type == "replay":
        values = _factory_fields(
            config,
            allowed=frozenset({"type", "provider_id", "path"}),
        )
        path = values.get("path")
        provider_id = values.get("provider_id", "replay")
        if not isinstance(path, str) or not path.strip():
            raise ProviderError("replay provider requires a non-empty path")
        if not isinstance(provider_id, str) or not provider_id.strip():
            raise ProviderError("replay provider_id must be non-empty")
        return ReplayProvider.from_jsonl(path, provider_id=provider_id)

    if provider_type == "openai_compatible":
        allowed = frozenset(
            {
                "type",
                "provider_id",
                "base_url",
                "api_key_env",
                "timeout_seconds",
                "max_retries",
                "retry_backoff_seconds",
                "max_input_tokens",
                "max_output_tokens",
                "max_response_bytes",
                "allow_insecure_http",
            }
        )
        values = _factory_fields(config, allowed=allowed)
        required = ("provider_id", "base_url", "api_key_env")
        missing = [name for name in required if name not in values]
        if missing:
            raise ProviderError(f"openai_compatible config missing fields: {missing}")
        values.pop("type")
        try:
            adapter_config = OpenAICompatibleConfig(**values)
        except (TypeError, ValueError) as exc:
            raise ProviderError("invalid openai_compatible provider config") from exc
        return OpenAICompatibleProvider(
            config=adapter_config,
            policy=network_policy,
            transport=StdlibJSONTransport() if transport is None else transport,
            verifier=verifier,
        )

    if provider_type == "dotted_path":
        return _dynamic_provider(
            config,
            policy=network_policy,
            allow_dynamic_imports=allow_dynamic_imports,
            allowed_dynamic_prefixes=allowed_dynamic_prefixes,
        )

    raise ProviderError(f"unsupported provider type: {provider_type!r}")
