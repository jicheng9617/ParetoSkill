"""Strict local runtime data loading and response-verifying harness adapters.

Task files are deliberately local-only. JSON files use this envelope::

    {"schema_version": 1, "tasks": [TASK, ...]}

JSONL files contain one ``TASK`` object per non-empty line. A task has exactly
``schema_version``, ``task_id``, ``split_id``, ``domain_id``, ``group_id``,
``objective_roles``, ``payload``, and ``verifier``. ``objective_roles`` permits
one logical task to be evaluated as both ID and transfer without duplicating its
canonical task id. Expected answers live only in the verifier registry and are
never placed in provider-visible task payloads.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from .evaluation import (
    EvaluationCache,
    EvaluationCandidate,
    EvaluationRecord,
    Harness,
    TaskSeedBlock,
    TaskSpec,
    TargetSpec,
)
from .models import canonical_json, freeze_mapping, stable_hash, thaw_json
from .providers import ExecutionRequest, ExecutionResult, GeneratedResponse, ModelSpec, Provider
from .runtime_adapters import RuntimeAdapterError, load_runtime_adapter_registry


ObjectiveRole = Literal["id", "transfer", "final", "diagnostic"]
VerifierKind = Literal[
    "exact_match", "json_boolean", "json_field", "builtin", "external_command"
]
PathPart = str | int
_OBJECTIVE_ROLES = frozenset({"id", "transfer", "final", "diagnostic"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PLACEHOLDER = re.compile(r"\$\{[^}]+\}")
_REMOTE_PATH = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_MISSING = object()
_EXTERNAL_PROFILES = frozenset({"real", "replay"})


class RuntimeDataError(ValueError):
    """A local runtime input or verification contract is malformed."""


class _DuplicateJsonKey(ValueError):
    pass


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _loads_json(payload: str, *, location: str) -> Any:
    try:
        return json.loads(
            payload,
            object_pairs_hook=_object_pairs,
            parse_constant=_reject_json_constant,
        )
    except ValueError as exc:
        raise RuntimeDataError(f"invalid JSON at {location}: {exc}") from exc


def _strict_keys(
    value: Any,
    *,
    required: set[str],
    optional: set[str] | None = None,
    location: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeDataError(f"{location} must be a JSON object")
    keys = set(value)
    if any(not isinstance(key, str) for key in keys):
        raise RuntimeDataError(f"{location} keys must be strings")
    missing = required - keys
    extra = keys - required - (optional or set())
    if missing or extra:
        raise RuntimeDataError(
            f"{location} schema mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return value


def _identifier(value: Any, *, location: str, allow_star: bool = False) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeDataError(f"{location} must be a non-empty string")
    if value != value.strip():
        raise RuntimeDataError(f"{location} cannot contain surrounding whitespace")
    if not allow_star and value == "*":
        raise RuntimeDataError(f"{location} cannot be '*'")
    if _PLACEHOLDER.search(value):
        raise RuntimeDataError(f"{location} contains an unresolved placeholder")
    return value


def _sha256(value: Any, *, location: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise RuntimeDataError(f"{location} must be a lowercase SHA-256 digest")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _field_path(value: Any, *, location: str, required: bool) -> tuple[PathPart, ...]:
    if value is None and not required:
        return ()
    if not isinstance(value, (list, tuple)) or (required and not value):
        qualification = "a non-empty" if required else "an"
        raise RuntimeDataError(f"{location} must be {qualification} array")
    result: list[PathPart] = []
    for index, part in enumerate(value):
        valid = (
            isinstance(part, str)
            and bool(part)
            or isinstance(part, int)
            and not isinstance(part, bool)
            and part >= 0
        )
        if not valid:
            raise RuntimeDataError(
                f"{location}[{index}] must be a non-empty key or non-negative index"
            )
        result.append(part)
    return tuple(result)


def _lookup(value: Any, path: Sequence[PathPart]) -> Any:
    current = value
    for part in path:
        if isinstance(part, str) and isinstance(current, Mapping) and part in current:
            current = current[part]
        elif (
            isinstance(part, int)
            and isinstance(current, (list, tuple))
            and part < len(current)
        ):
            current = current[part]
        else:
            return _MISSING
    return current


def _json_value(value: Any, *, location: str) -> Any:
    try:
        # Round-tripping also rejects NaN, infinity, and non-JSON objects.
        return json.loads(canonical_json(value))
    except (TypeError, ValueError) as exc:
        raise RuntimeDataError(f"{location} must be a finite JSON value") from exc


def _json_equal(left: Any, right: Any) -> bool:
    try:
        return canonical_json(left) == canonical_json(right)
    except (TypeError, ValueError):
        return False


def _within(path: Path, roots: Sequence[Path]) -> bool:
    resolved = path.resolve()
    return any(resolved == root or root in resolved.parents for root in roots)


def _external_command_options(
    base: Mapping[str, Any], *, location: str
) -> dict[str, Any]:
    required = {
        "kind",
        "argv",
        "timeout_seconds",
        "enabled_profiles",
        "allowed_path_roots",
        "path_sha256",
    }
    allowed = required | {"cwd", "verifier_input"}
    keys = set(base)
    if required - keys or keys - allowed:
        raise RuntimeDataError(
            f"{location} schema mismatch for 'external_command': "
            f"missing={sorted(required - keys)}, extra={sorted(keys - allowed)}"
        )
    argv = base["argv"]
    if not isinstance(argv, list) or not argv or any(
        not isinstance(value, str) or not value or "\0" in value for value in argv
    ):
        raise RuntimeDataError(f"{location}.argv must be a non-empty string array")
    executable = Path(argv[0])
    if not executable.is_absolute() or not executable.is_file():
        raise RuntimeDataError(
            f"{location}.argv[0] must be an existing absolute executable path"
        )
    raw_roots = base["allowed_path_roots"]
    if not isinstance(raw_roots, list) or not raw_roots:
        raise RuntimeDataError(
            f"{location}.allowed_path_roots must be a non-empty array"
        )
    roots: list[Path] = []
    for index, raw_root in enumerate(raw_roots):
        if not isinstance(raw_root, str) or not Path(raw_root).is_absolute():
            raise RuntimeDataError(
                f"{location}.allowed_path_roots[{index}] must be an absolute path"
            )
        root = Path(raw_root).resolve()
        if not root.is_dir():
            raise RuntimeDataError(
                f"{location}.allowed_path_roots[{index}] is not a directory"
            )
        roots.append(root)
    if not _within(executable, roots):
        raise RuntimeDataError(f"{location}.argv[0] is outside allowed_path_roots")
    cwd = executable.resolve().parent
    if "cwd" in base:
        raw_cwd = base["cwd"]
        if not isinstance(raw_cwd, str) or not Path(raw_cwd).is_absolute():
            raise RuntimeDataError(f"{location}.cwd must be an absolute path")
        cwd = Path(raw_cwd).resolve()
        if not cwd.is_dir() or not _within(cwd, roots):
            raise RuntimeDataError(
                f"{location}.cwd must be an allowed existing directory"
            )
    for index, argument in enumerate(argv[1:], start=1):
        candidate = Path(argument)
        if candidate.is_absolute() and not _within(candidate, roots):
            raise RuntimeDataError(
                f"{location}.argv[{index}] is outside allowed_path_roots"
            )
        if (
            not candidate.is_absolute()
            and ("/" in argument or "\\" in argument or argument.endswith(".py"))
        ):
            raise RuntimeDataError(
                f"{location}.argv[{index}] looks like a relative path; use an allowed "
                "absolute path"
            )
    raw_pins = base["path_sha256"]
    if not isinstance(raw_pins, Mapping) or any(
        not isinstance(key, str) for key in raw_pins
    ):
        raise RuntimeDataError(f"{location}.path_sha256 must be a path-to-digest object")
    pinned: dict[str, str] = {}
    for raw_path, raw_digest in raw_pins.items():
        pinned_path = Path(raw_path)
        if not pinned_path.is_absolute() or not pinned_path.is_file():
            raise RuntimeDataError(
                f"{location}.path_sha256 keys must be existing absolute files"
            )
        pinned_path = pinned_path.resolve()
        if not _within(pinned_path, roots):
            raise RuntimeDataError(
                f"{location}.path_sha256 contains a path outside allowed roots"
            )
        digest = _sha256(raw_digest, location=f"{location}.path_sha256[{raw_path!r}]")
        observed = _file_sha256(pinned_path)
        if observed != digest:
            raise RuntimeDataError(
                f"{location}.path_sha256 digest mismatch for {pinned_path}"
            )
        pinned[str(pinned_path)] = digest
    command_files = {
        str(candidate.resolve())
        for candidate in (Path(argument) for argument in argv)
        if candidate.is_absolute() and candidate.is_file()
    }
    if set(pinned) != command_files:
        raise RuntimeDataError(
            f"{location}.path_sha256 must pin every executable/file argv exactly"
        )
    timeout = base["timeout_seconds"]
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(float(timeout))
        or not 0.0 < float(timeout) <= 300.0
    ):
        raise RuntimeDataError(
            f"{location}.timeout_seconds must be finite in (0, 300]"
        )
    profiles = base["enabled_profiles"]
    if (
        not isinstance(profiles, list)
        or not profiles
        or any(profile not in _EXTERNAL_PROFILES for profile in profiles)
        or len(set(profiles)) != len(profiles)
    ):
        raise RuntimeDataError(
            f"{location}.enabled_profiles must be a unique non-empty subset of "
            "['real', 'replay']"
        )
    verifier_input = _json_value(
        base.get("verifier_input", {}), location=f"{location}.verifier_input"
    )
    return {
        "argv": list(argv),
        "timeout_seconds": float(timeout),
        "enabled_profiles": sorted(profiles),
        "allowed_path_roots": [str(root) for root in roots],
        "path_sha256": pinned,
        "cwd": str(cwd),
        "verifier_input": verifier_input,
    }


@dataclass(frozen=True, slots=True)
class VerifierSpec:
    """A safe built-in verifier configuration; no dynamic imports or code execution."""

    kind: VerifierKind
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in {
            "exact_match",
            "json_boolean",
            "json_field",
            "builtin",
            "external_command",
        }:
            raise RuntimeDataError(f"unsupported verifier kind: {self.kind!r}")
        object.__setattr__(self, "options", freeze_mapping(self.options))

    @property
    def digest(self) -> str:
        return stable_hash(
            {"schema_version": 1, "kind": self.kind, "options": thaw_json(self.options)}
        )

    @classmethod
    def from_mapping(cls, value: Any, *, location: str = "verifier") -> VerifierSpec:
        base = _strict_keys(
            value,
            required={"kind"},
            optional={
                "expected",
                "strip",
                "case_sensitive",
                "field_path",
                "rule",
                "pattern",
                "absolute_tolerance",
                "argv",
                "timeout_seconds",
                "enabled_profiles",
                "allowed_path_roots",
                "path_sha256",
                "cwd",
                "verifier_input",
            },
            location=location,
        )
        kind = base["kind"]
        if kind == "exact_match":
            allowed = {"kind", "expected", "strip", "case_sensitive"}
            required = {"kind", "expected"}
        elif kind == "json_boolean":
            allowed = {"kind", "expected", "field_path"}
            required = {"kind"}
        elif kind == "json_field":
            allowed = {"kind", "expected", "field_path"}
            required = {"kind", "expected", "field_path"}
        elif kind == "builtin":
            rule = base.get("rule")
            if rule == "nonempty":
                allowed = {"kind", "rule", "strip"}
                required = {"kind", "rule"}
            elif rule == "contains":
                allowed = {"kind", "rule", "expected", "case_sensitive"}
                required = {"kind", "rule", "expected"}
            elif rule == "regex_fullmatch":
                allowed = {"kind", "rule", "pattern", "case_sensitive"}
                required = {"kind", "rule", "pattern"}
            elif rule == "numeric_tolerance":
                allowed = {"kind", "rule", "expected", "absolute_tolerance"}
                required = {"kind", "rule", "expected", "absolute_tolerance"}
            else:
                raise RuntimeDataError(f"{location}.rule is not a supported built-in rule")
        elif kind == "external_command":
            return cls(
                kind="external_command",
                options=_external_command_options(base, location=location),
            )
        else:
            raise RuntimeDataError(f"{location}.kind is not supported")
        keys = set(base)
        if required - keys or keys - allowed:
            raise RuntimeDataError(
                f"{location} schema mismatch for {kind!r}: "
                f"missing={sorted(required - keys)}, extra={sorted(keys - allowed)}"
            )

        options = {key: value for key, value in base.items() if key != "kind"}
        if "expected" in options:
            options["expected"] = _json_value(
                options["expected"], location=f"{location}.expected"
            )
        if kind == "json_boolean":
            expected = options.get("expected", True)
            if not isinstance(expected, bool):
                raise RuntimeDataError(f"{location}.expected must be a JSON boolean")
            options["expected"] = expected
            options["field_path"] = _field_path(
                options.get("field_path"),
                location=f"{location}.field_path",
                required=False,
            )
        elif kind == "json_field":
            options["field_path"] = _field_path(
                options["field_path"],
                location=f"{location}.field_path",
                required=True,
            )
        if kind == "exact_match":
            options.setdefault("strip", True)
            options.setdefault("case_sensitive", True)
        if kind == "builtin" and options["rule"] == "nonempty":
            options.setdefault("strip", True)
        if kind == "builtin" and options["rule"] in {"contains", "regex_fullmatch"}:
            options.setdefault("case_sensitive", True)
        if "strip" in options and not isinstance(options["strip"], bool):
            raise RuntimeDataError(f"{location}.strip must be boolean")
        if "case_sensitive" in options and not isinstance(
            options["case_sensitive"], bool
        ):
            raise RuntimeDataError(f"{location}.case_sensitive must be boolean")
        if kind == "builtin":
            rule = options["rule"]
            if rule in {"contains", "numeric_tolerance"} and isinstance(
                options.get("expected"), bool
            ):
                raise RuntimeDataError(f"{location}.expected has the wrong type")
            if rule == "contains" and not isinstance(options["expected"], str):
                raise RuntimeDataError(f"{location}.expected must be a string")
            if rule == "regex_fullmatch":
                pattern = options["pattern"]
                if not isinstance(pattern, str):
                    raise RuntimeDataError(f"{location}.pattern must be a string")
                try:
                    re.compile(pattern)
                except re.error as exc:
                    raise RuntimeDataError(f"{location}.pattern is invalid: {exc}") from exc
            if rule == "numeric_tolerance":
                expected = options["expected"]
                tolerance = options["absolute_tolerance"]
                if (
                    isinstance(expected, bool)
                    or not isinstance(expected, (int, float))
                    or not math.isfinite(float(expected))
                ):
                    raise RuntimeDataError(f"{location}.expected must be finite numeric")
                if (
                    isinstance(tolerance, bool)
                    or not isinstance(tolerance, (int, float))
                    or not math.isfinite(float(tolerance))
                    or float(tolerance) < 0.0
                ):
                    raise RuntimeDataError(
                        f"{location}.absolute_tolerance must be finite and non-negative"
                    )
        return cls(kind=kind, options=options)  # type: ignore[arg-type]

    def verify(
        self,
        response: Any,
        *,
        runtime_profile: str | None = None,
        allow_external_commands: bool = False,
    ) -> bool:
        """Return correctness; malformed model output is a normal false result."""

        options = self.options
        if self.kind == "external_command":
            enabled_profiles = set(options["enabled_profiles"])
            if (
                not allow_external_commands
                or runtime_profile not in _EXTERNAL_PROFILES
                or runtime_profile not in enabled_profiles
            ):
                raise RuntimeDataError(
                    "external verifier commands require an explicitly enabled real/replay "
                    "runtime profile"
                )
            request = {
                "schema_version": 1,
                "response": _json_value(response, location="external verifier response"),
                "verifier_input": thaw_json(options["verifier_input"]),
            }
            for raw_path, expected_digest in options["path_sha256"].items():
                command_path = Path(raw_path)
                if not command_path.is_file():
                    raise RuntimeDataError(
                        f"pinned external verifier file disappeared: {command_path}"
                    )
                observed_digest = _file_sha256(command_path)
                if observed_digest != expected_digest:
                    raise RuntimeDataError(
                        f"pinned external verifier file changed: {command_path}"
                    )
            environment = {
                key: os.environ[key]
                for key in (
                    "PATH",
                    "SYSTEMROOT",
                    "WINDIR",
                    "TEMP",
                    "TMP",
                    "LANG",
                    "LC_ALL",
                )
                if key in os.environ
            }
            environment["PYTHONIOENCODING"] = "utf-8"
            try:
                completed = subprocess.run(
                    tuple(options["argv"]),
                    input=canonical_json(request),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="strict",
                    timeout=float(options["timeout_seconds"]),
                    cwd=options["cwd"],
                    env=environment,
                    shell=False,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeDataError("external verifier command timed out") from exc
            except (OSError, UnicodeError) as exc:
                raise RuntimeDataError(
                    f"external verifier command could not run: {exc}"
                ) from exc
            if completed.returncode != 0:
                stderr_digest = hashlib.sha256(
                    completed.stderr.encode("utf-8")
                ).hexdigest()[:16]
                raise RuntimeDataError(
                    f"external verifier command failed with exit code "
                    f"{completed.returncode}; stderr suppressed "
                    f"(sha256-prefix={stderr_digest})"
                )
            output = _strict_keys(
                _loads_json(completed.stdout, location="external verifier stdout"),
                required={"schema_version", "correct"},
                optional={"details"},
                location="external verifier stdout",
            )
            if output["schema_version"] != 1 or not isinstance(output["correct"], bool):
                raise RuntimeDataError(
                    "external verifier stdout requires schema_version=1 and boolean correct"
                )
            if "details" in output and not isinstance(output["details"], Mapping):
                raise RuntimeDataError("external verifier stdout details must be an object")
            return output["correct"]

        if self.kind == "exact_match":
            expected = options["expected"]
            if isinstance(expected, str) and isinstance(response, str):
                actual_text = response.strip() if options.get("strip", True) else response
                expected_text = expected.strip() if options.get("strip", True) else expected
                if not options.get("case_sensitive", True):
                    actual_text = actual_text.casefold()
                    expected_text = expected_text.casefold()
                return actual_text == expected_text
            return _json_equal(response, expected)

        if self.kind in {"json_boolean", "json_field"}:
            parsed = response
            if isinstance(response, str):
                try:
                    parsed = json.loads(
                        response,
                        object_pairs_hook=_object_pairs,
                        parse_constant=_reject_json_constant,
                    )
                except ValueError:
                    return False
            selected = _lookup(parsed, options["field_path"])
            if selected is _MISSING:
                return False
            if self.kind == "json_boolean":
                return isinstance(selected, bool) and selected is options["expected"]
            return _json_equal(selected, options["expected"])

        rule = options["rule"]
        if rule == "nonempty":
            if not isinstance(response, str):
                return False
            return bool(response.strip() if options.get("strip", True) else response)
        if rule == "contains":
            if not isinstance(response, str):
                return False
            expected = options["expected"]
            if not options.get("case_sensitive", True):
                response = response.casefold()
                expected = expected.casefold()
            return expected in response
        if rule == "regex_fullmatch":
            if not isinstance(response, str):
                return False
            flags = 0 if options.get("case_sensitive", True) else re.IGNORECASE
            return re.fullmatch(options["pattern"], response, flags=flags) is not None
        if rule == "numeric_tolerance":
            if isinstance(response, bool):
                return False
            try:
                actual = float(response)
            except (TypeError, ValueError):
                return False
            return math.isfinite(actual) and math.isclose(
                actual,
                float(options["expected"]),
                rel_tol=0.0,
                abs_tol=float(options["absolute_tolerance"]),
            )
        raise RuntimeDataError(f"unsupported verifier rule: {rule!r}")


@dataclass(frozen=True, slots=True)
class LoadedTaskManifest:
    source_path: Path
    content_sha256: str
    source_task_ids: tuple[str, ...]
    tasks: tuple[TaskSpec, ...]
    verifiers: Mapping[str, VerifierSpec]

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_path", self.source_path.resolve())
        _sha256(self.content_sha256, location="content_sha256")
        source_task_ids = tuple(self.source_task_ids)
        tasks = tuple(self.tasks)
        verifiers = dict(self.verifiers)
        if not source_task_ids or any(
            not isinstance(task_id, str) or not task_id for task_id in source_task_ids
        ):
            raise RuntimeDataError("loaded manifest requires non-empty source task ids")
        if len(set(source_task_ids)) != len(source_task_ids):
            raise RuntimeDataError("loaded manifest source task ids contain duplicates")
        if not tasks or any(not isinstance(task, TaskSpec) for task in tasks):
            raise RuntimeDataError("loaded manifest requires TaskSpec values")
        if {task.task_id for task in tasks} != set(source_task_ids):
            raise RuntimeDataError("loaded manifest tasks differ from source task ids")
        if set(verifiers) != set(source_task_ids) or any(
            not isinstance(verifier, VerifierSpec) for verifier in verifiers.values()
        ):
            raise RuntimeDataError("loaded manifest verifier registry is incomplete")
        object.__setattr__(self, "source_task_ids", source_task_ids)
        object.__setattr__(self, "tasks", tasks)
        object.__setattr__(self, "verifiers", MappingProxyType(verifiers))

    @property
    def logical_task_count(self) -> int:
        return len(self.source_task_ids)


def _task_rows(source: Path, payload: str) -> list[Any]:
    if source.suffix.lower() == ".jsonl":
        rows: list[Any] = []
        for line_number, line in enumerate(payload.splitlines(), start=1):
            if line.strip():
                rows.append(_loads_json(line, location=f"{source}:{line_number}"))
        return rows
    if source.suffix.lower() == ".json":
        root = _strict_keys(
            _loads_json(payload, location=str(source)),
            required={"schema_version", "tasks"},
            location=str(source),
        )
        if root["schema_version"] != 1:
            raise RuntimeDataError(f"unsupported task manifest schema in {source}")
        if not isinstance(root["tasks"], list):
            raise RuntimeDataError(f"{source}.tasks must be an array")
        return list(root["tasks"])
    raise RuntimeDataError(
        f"unsupported local task manifest extension {source.suffix!r}; use .json or .jsonl"
    )


def load_local_task_manifest(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_count: int,
) -> LoadedTaskManifest:
    """Load a hash-pinned local task manifest without any network fallback."""

    expected_digest = _sha256(expected_sha256, location="expected_sha256")
    if isinstance(path, str) and _REMOTE_PATH.match(path):
        raise RuntimeDataError("task manifests must be local paths; URLs are forbidden")
    source = Path(path).resolve()
    if not source.is_file():
        raise RuntimeDataError(f"local task manifest does not exist: {source}")
    if isinstance(expected_count, bool) or not isinstance(expected_count, int):
        raise RuntimeDataError("expected_count must be an integer")
    if expected_count < 1:
        raise RuntimeDataError("expected_count must be positive for an executable matrix")
    raw = source.read_bytes()
    observed_digest = hashlib.sha256(raw).hexdigest()
    if observed_digest != expected_digest:
        raise RuntimeDataError(
            f"task manifest digest mismatch: expected {expected_digest}, "
            f"observed {observed_digest}"
        )
    try:
        payload = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeDataError(f"task manifest is not UTF-8: {source}") from exc
    rows = _task_rows(source, payload)
    if len(rows) != expected_count:
        raise RuntimeDataError(
            f"task manifest has {len(rows)} tasks, expected {expected_count}: {source}"
        )

    identifiers: list[str] = []
    tasks: list[TaskSpec] = []
    verifiers: dict[str, VerifierSpec] = {}
    required_keys = {
        "schema_version",
        "task_id",
        "split_id",
        "domain_id",
        "group_id",
        "objective_roles",
        "payload",
        "verifier",
    }
    for index, raw_row in enumerate(rows):
        location = f"{source}:task[{index}]"
        row = _strict_keys(raw_row, required=required_keys, location=location)
        if row["schema_version"] != 1:
            raise RuntimeDataError(f"{location}.schema_version must be 1")
        task_id = _identifier(row["task_id"], location=f"{location}.task_id")
        split_id = _identifier(row["split_id"], location=f"{location}.split_id")
        domain_id = _identifier(row["domain_id"], location=f"{location}.domain_id")
        group_id = _identifier(
            row["group_id"], location=f"{location}.group_id", allow_star=True
        )
        raw_roles = row["objective_roles"]
        if not isinstance(raw_roles, list) or not raw_roles:
            raise RuntimeDataError(f"{location}.objective_roles must be a non-empty array")
        if any(not isinstance(role, str) or role not in _OBJECTIVE_ROLES for role in raw_roles):
            raise RuntimeDataError(f"{location}.objective_roles contains an invalid role")
        if len(set(raw_roles)) != len(raw_roles):
            raise RuntimeDataError(f"{location}.objective_roles contains duplicates")
        payload_value = row["payload"]
        if not isinstance(payload_value, Mapping):
            raise RuntimeDataError(f"{location}.payload must be a JSON object")
        try:
            safe_payload = freeze_mapping(payload_value)
        except ValueError as exc:
            raise RuntimeDataError(f"{location}.payload is not finite JSON") from exc
        verifier = VerifierSpec.from_mapping(
            row["verifier"], location=f"{location}.verifier"
        )
        identifiers.append(task_id)
        verifiers[task_id] = verifier
        for role in sorted(raw_roles):
            tasks.append(
                TaskSpec(
                    task_id=task_id,
                    split=role,
                    domain_id=domain_id,
                    group_id=group_id,
                    payload=safe_payload,
                    split_id=split_id,
                    objective_role=role,
                )
            )
    duplicates = sorted(
        task_id for task_id, count in Counter(identifiers).items() if count > 1
    )
    if duplicates:
        raise RuntimeDataError(f"duplicate task ids in {source}: {duplicates[:5]}")
    return LoadedTaskManifest(
        source_path=source,
        content_sha256=observed_digest,
        source_task_ids=tuple(sorted(identifiers)),
        tasks=tuple(
            sorted(tasks, key=lambda task: (task.task_id, task.objective_role or ""))
        ),
        verifiers=verifiers,
    )


@dataclass(frozen=True, slots=True)
class LocalDomainAdapter:
    """In-memory view over already-validated local tasks for one domain."""

    domain_id: str
    task_specs: tuple[TaskSpec, ...]

    def __post_init__(self) -> None:
        _identifier(self.domain_id, location="domain_id")
        tasks = tuple(self.task_specs)
        if not tasks:
            raise RuntimeDataError(f"domain {self.domain_id!r} has no runtime tasks")
        if any(task.domain_id != self.domain_id for task in tasks):
            raise RuntimeDataError("domain adapter contains a task from another domain")
        keys = [
            (task.split_id, task.task_id, task.objective_role, task.group_id)
            for task in tasks
        ]
        if len(set(keys)) != len(keys):
            raise RuntimeDataError("domain adapter contains duplicate task specifications")
        object.__setattr__(self, "task_specs", tasks)

    def tasks(self, split: str) -> Iterable[TaskSpec]:
        return tuple(task for task in self.task_specs if task.split_id == split)


def _compatible(target: TargetSpec, task: TaskSpec) -> bool:
    return (
        target.domain_id == task.domain_id
        and (target.split_id is None or target.split_id == task.split_id)
        and (
            target.objective_role is None
            or target.objective_role == task.objective_role
        )
        and (target.task_group == "*" or target.task_group == task.group_id)
    )


@dataclass(slots=True)
class VerifiedResponseHarness:
    """Score a provider's raw response with a task-local verifier.

    The provider's ``ExecutionResult.correct`` is retained only as audit metadata;
    it never determines the returned correctness value.
    """

    harness_id: str
    verifiers: Mapping[str, VerifierSpec]
    response_path: tuple[PathPart, ...] = ("trace", "raw_response")
    harness_revision: str = "local-response-verifier-v1"
    cache: EvaluationCache | None = None
    runtime_profile: str | None = None
    allow_external_commands: bool = False

    def __post_init__(self) -> None:
        _identifier(self.harness_id, location="harness_id")
        _identifier(self.harness_revision, location="harness_revision")
        if self.runtime_profile is not None and not isinstance(self.runtime_profile, str):
            raise RuntimeDataError("runtime_profile must be a string or null")
        if not isinstance(self.allow_external_commands, bool):
            raise RuntimeDataError("allow_external_commands must be boolean")
        self.response_path = _field_path(
            self.response_path, location="response_path", required=True
        )
        copied = dict(self.verifiers)
        if not copied or any(
            not isinstance(task_id, str) or not isinstance(spec, VerifierSpec)
            for task_id, spec in copied.items()
        ):
            raise RuntimeDataError("harness verifiers must map task ids to VerifierSpec values")
        self.verifiers = MappingProxyType(copied)

    def evaluate(
        self,
        *,
        provider: Provider,
        experiment_id: str,
        candidate: EvaluationCandidate,
        target: TargetSpec,
        block: TaskSeedBlock,
        is_base: bool,
    ) -> EvaluationRecord:
        if target.harness_id != self.harness_id:
            raise RuntimeDataError(
                f"target requests harness {target.harness_id!r}, got {self.harness_id!r}"
            )
        if provider.provider_id != target.provider_id:
            raise RuntimeDataError("provider_id does not match the target model/provider")
        if not _compatible(target, block.task):
            raise RuntimeDataError("task block is incompatible with target domain/split/group")
        try:
            verifier = self.verifiers[block.task.task_id]
        except KeyError as exc:
            raise RuntimeDataError(
                f"task {block.task.task_id!r} has no configured verifier"
            ) from exc
        request = ExecutionRequest(
            experiment_id=experiment_id,
            candidate_id=candidate.candidate_id,
            content_hash=candidate.content_hash,
            task_id=block.task.task_id,
            seed=block.seed,
            target_id=target.target_id,
            model=target.model,
            skill_files=candidate.skill_files,
            task_payload=block.task.payload,
            metadata={
                "group_id": block.task.group_id,
                "split": target.objective_role or block.task.objective_role,
                "split_id": target.split_id or block.task.split_id,
                "objective_role": target.objective_role or block.task.objective_role,
                "transfer_group": target.transfer_group,
                "domain_id": target.domain_id,
                "harness_id": self.harness_id,
                "harness_revision": self.harness_revision,
                "verifier_digest": verifier.digest,
                "response_path": list(self.response_path),
                "injection_mode": candidate.injection_mode,
            },
        )
        cached = self.cache.get(request.cache_key) if self.cache is not None else None
        if cached is not None:
            if (
                cached.cache_key != request.cache_key
                or cached.content_hash != request.content_hash
                or cached.target_id != request.target_id
                or cached.task_id != request.task_id
                or cached.seed != request.seed
            ):
                raise RuntimeDataError("evaluation cache returned a mismatched record")
            raw_result = cached.result
        else:
            underlying = getattr(provider, "inner", provider)
            generate = getattr(provider, "generate", None)
            if callable(generate) and callable(getattr(underlying, "generate", None)):
                generation = generate(request)
                if not isinstance(generation, GeneratedResponse):
                    raise RuntimeDataError("provider generate() returned an invalid response")
                raw_result = ExecutionResult(
                    correct=False,
                    input_tokens=generation.input_tokens,
                    output_tokens=generation.output_tokens,
                    latency_ms=generation.latency_ms,
                    trace={
                        "raw_response": thaw_json(generation.parsed_output),
                        "response_text": generation.response_text,
                        "finish_reason": generation.finish_reason,
                    },
                    provider_metadata={
                        **thaw_json(generation.provider_metadata),
                        "provider_reported_correct": None,
                    },
                )
            else:
                raw_result = provider.execute(request)
        root = {
            "trace": thaw_json(raw_result.trace),
            "provider_metadata": thaw_json(raw_result.provider_metadata),
        }
        response = _lookup(root, self.response_path)
        if response is _MISSING:
            raise RuntimeDataError(
                "provider result has no raw response at "
                + ".".join(str(part) for part in self.response_path)
            )
        correct = verifier.verify(
            response,
            runtime_profile=self.runtime_profile,
            allow_external_commands=self.allow_external_commands,
        )
        trace = thaw_json(raw_result.trace)
        assert isinstance(trace, dict)
        trace["local_verification"] = {
            "correct": correct,
            "kind": verifier.kind,
            "verifier_digest": verifier.digest,
            "response_path": list(self.response_path),
        }
        provider_metadata = thaw_json(raw_result.provider_metadata)
        assert isinstance(provider_metadata, dict)
        provider_reported = provider_metadata.get(
            "provider_reported_correct", raw_result.correct
        )
        provider_metadata.update(
            {
                "provider_reported_correct": provider_reported,
                "correctness_source": "local_verifier",
                "injection_mode": candidate.injection_mode,
            }
        )
        result = ExecutionResult(
            correct=correct,
            input_tokens=raw_result.input_tokens,
            output_tokens=raw_result.output_tokens,
            latency_ms=raw_result.latency_ms,
            trace=trace,
            provider_metadata=provider_metadata,
        )
        record = EvaluationRecord(
            experiment_id=experiment_id,
            candidate_id=candidate.candidate_id,
            content_hash=candidate.content_hash,
            target_id=target.target_id,
            task_id=block.task.task_id,
            group_id=block.task.group_id,
            split=target.objective_role or block.task.objective_role or block.task.split,
            seed=block.seed,
            result=result,
            cache_key=request.cache_key,
            is_base=is_base,
            split_id=target.split_id or block.task.split_id,
            transfer_group=target.transfer_group,
        )
        if cached is None and self.cache is not None:
            self.cache.put(record)
        return record


def _target_specs(
    configuration: Mapping[str, Any],
    *,
    phase: str | None,
    target_ids: Iterable[str] | None,
) -> tuple[TargetSpec, ...]:
    for name in ("providers", "models", "harnesses", "domains", "targets", "objectives"):
        if not isinstance(configuration.get(name), Mapping):
            raise RuntimeDataError(f"configuration.{name} must be a mapping")
    providers = configuration["providers"]
    models = configuration["models"]
    harnesses = configuration["harnesses"]
    domains = configuration["domains"]
    raw_targets = configuration["targets"]
    objectives = configuration["objectives"]
    assert all(
        isinstance(value, Mapping)
        for value in (providers, models, harnesses, domains, raw_targets, objectives)
    )

    requested: set[str] | None = None
    if target_ids is not None:
        raw_requested = tuple(target_ids)
        if any(not isinstance(value, str) or not value for value in raw_requested):
            raise RuntimeDataError("target_ids must contain non-empty strings")
        if len(set(raw_requested)) != len(raw_requested):
            raise RuntimeDataError("target_ids contains duplicates")
        requested = set(raw_requested)
        missing = requested - set(raw_targets)
        if missing:
            raise RuntimeDataError(f"unknown requested targets: {sorted(missing)}")

    role_sets: dict[str, set[str]] = {"id": set(), "transfer": set()}
    for role, objective_name in (
        ("id", "id_accuracy"),
        ("transfer", "worst_target_transfer"),
    ):
        objective = objectives.get(objective_name)
        if not isinstance(objective, Mapping) or not isinstance(
            objective.get("target_ids"), list
        ):
            raise RuntimeDataError(f"objective {objective_name!r} has no target_ids array")
        values = objective["target_ids"]
        if any(not isinstance(value, str) for value in values):
            raise RuntimeDataError(f"objective {objective_name!r} target ids must be strings")
        role_sets[role] = set(values)
    overlap = role_sets["id"] & role_sets["transfer"]
    if overlap:
        raise RuntimeDataError(f"targets cannot be both ID and transfer: {sorted(overlap)}")

    result: list[TargetSpec] = []
    for target_id, raw_target in sorted(raw_targets.items()):
        if requested is not None and target_id not in requested:
            continue
        if not isinstance(target_id, str) or not isinstance(raw_target, Mapping):
            raise RuntimeDataError("target entries must map string ids to objects")
        _identifier(target_id, location="targets target_id")
        target_phase = raw_target.get("phase")
        if target_phase is not None:
            _identifier(target_phase, location=f"targets.{target_id}.phase")
        if phase is not None and target_phase != phase:
            continue
        model_key = _identifier(
            raw_target.get("model"), location=f"targets.{target_id}.model"
        )
        harness_id = _identifier(
            raw_target.get("harness"), location=f"targets.{target_id}.harness"
        )
        domain_id = _identifier(
            raw_target.get("domain"), location=f"targets.{target_id}.domain"
        )
        split_id = _identifier(
            raw_target.get("split"), location=f"targets.{target_id}.split"
        )
        if model_key not in models:
            raise RuntimeDataError(f"target {target_id!r} references unknown model {model_key!r}")
        if harness_id not in harnesses:
            raise RuntimeDataError(
                f"target {target_id!r} references unknown harness {harness_id!r}"
            )
        if domain_id not in domains:
            raise RuntimeDataError(
                f"target {target_id!r} references unknown domain {domain_id!r}"
            )
        raw_model = models[model_key]
        if not isinstance(raw_model, Mapping):
            raise RuntimeDataError(f"models.{model_key} must be a mapping")
        provider_id = _identifier(
            raw_model.get("provider"), location=f"models.{model_key}.provider"
        )
        if provider_id not in providers:
            raise RuntimeDataError(
                f"model {model_key!r} references unknown provider {provider_id!r}"
            )
        model_id = _identifier(
            raw_model.get("model_id"), location=f"models.{model_key}.model_id"
        )
        revision = _identifier(
            raw_model.get("revision"), location=f"models.{model_key}.revision"
        )
        decoding = raw_model.get("decoding", {})
        if not isinstance(decoding, Mapping):
            raise RuntimeDataError(f"models.{model_key}.decoding must be a mapping")
        if target_id in role_sets["id"]:
            inferred_role: ObjectiveRole = "id"
        elif target_id in role_sets["transfer"]:
            inferred_role = "transfer"
        elif target_phase == "final_only":
            inferred_role = "final"
        else:
            inferred_role = "diagnostic"
        explicit_role = raw_target.get("objective_role")
        if explicit_role is not None and explicit_role != inferred_role:
            raise RuntimeDataError(
                f"target {target_id!r} objective_role conflicts with objective membership"
            )
        task_group = raw_target.get("task_group", "*")
        task_group = _identifier(
            task_group, location=f"targets.{target_id}.task_group", allow_star=True
        )
        transfer_group = raw_target.get("transfer_group")
        if transfer_group is not None:
            transfer_group = _identifier(
                transfer_group, location=f"targets.{target_id}.transfer_group"
            )
        result.append(
            TargetSpec(
                target_id=target_id,
                provider_id=provider_id,
                model=ModelSpec(model_id, provider_id, revision, decoding),
                harness_id=harness_id,
                domain_id=domain_id,
                task_group=task_group,
                split_id=split_id,
                transfer_group=transfer_group,
                objective_role=inferred_role,
            )
        )
    if not result:
        raise RuntimeDataError("no targets match the requested runtime matrix")
    if requested is not None:
        selected = {target.target_id for target in result}
        excluded = requested - selected
        if excluded:
            raise RuntimeDataError(
                f"requested targets do not match phase {phase!r}: {sorted(excluded)}"
            )
    return tuple(result)


def _execution_seeds(
    configuration: Mapping[str, Any], values: Iterable[int] | None
) -> tuple[int, ...]:
    if values is None:
        blocks = configuration.get("task_seed_blocks")
        if not isinstance(blocks, Mapping):
            raise RuntimeDataError("configuration.task_seed_blocks must be a mapping")
        raw_values = blocks.get("execution_seeds")
        if not isinstance(raw_values, list):
            raise RuntimeDataError("task_seed_blocks.execution_seeds must be an array")
        seeds = tuple(raw_values)
    else:
        seeds = tuple(values)
    if not seeds or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise RuntimeDataError("execution seeds must be a non-empty integer sequence")
    if len(set(seeds)) != len(seeds):
        raise RuntimeDataError("execution seeds contain duplicates")
    return tuple(sorted(seeds))


def _response_path_for(
    harness_id: str,
    response_paths: Mapping[str, Sequence[PathPart] | str] | None,
) -> tuple[PathPart, ...]:
    if response_paths is None or harness_id not in response_paths:
        return ("trace", "raw_response")
    raw = response_paths[harness_id]
    if isinstance(raw, str):
        if not raw:
            raise RuntimeDataError(f"response path for {harness_id!r} is empty")
        raw = tuple(raw.split("."))
    return _field_path(raw, location=f"response_paths.{harness_id}", required=True)


@dataclass(frozen=True, slots=True)
class RuntimeMatrix:
    """A closed local target-task-seed matrix plus ready-to-use adapters."""

    targets: tuple[TargetSpec, ...]
    blocks: tuple[TaskSeedBlock, ...]
    harnesses: Mapping[str, Harness]
    domains: Mapping[str, LocalDomainAdapter]
    manifests: Mapping[str, LoadedTaskManifest]

    def __post_init__(self) -> None:
        object.__setattr__(self, "targets", tuple(self.targets))
        object.__setattr__(self, "blocks", tuple(self.blocks))
        object.__setattr__(self, "harnesses", MappingProxyType(dict(self.harnesses)))
        object.__setattr__(self, "domains", MappingProxyType(dict(self.domains)))
        object.__setattr__(self, "manifests", MappingProxyType(dict(self.manifests)))

    def __iter__(
        self,
    ) -> Iterator[
        tuple[TargetSpec, ...]
        | tuple[TaskSeedBlock, ...]
        | Mapping[str, VerifiedResponseHarness]
    ]:
        # Convenience for ``targets, blocks, harnesses = load_runtime_matrix(...)``.
        yield self.targets
        yield self.blocks
        yield self.harnesses


def load_runtime_matrix(
    configuration: Mapping[str, Any],
    *,
    phase: str | None = None,
    base_directory: str | Path | None = None,
    target_ids: Iterable[str] | None = None,
    execution_seeds: Iterable[int] | None = None,
    response_paths: Mapping[str, Sequence[PathPart] | str] | None = None,
) -> RuntimeMatrix:
    """Resolve a complete local execution matrix from a resolved configuration.

    This function performs filesystem reads only. A URL, unresolved hash/path,
    missing task, duplicate identity, or incomplete target-task-seed Cartesian
    product raises :class:`RuntimeDataError` before any provider can run.
    """

    targets = _target_specs(configuration, phase=phase, target_ids=target_ids)
    seeds = _execution_seeds(configuration, execution_seeds)
    raw_splits = configuration.get("splits")
    if not isinstance(raw_splits, Mapping):
        raise RuntimeDataError("configuration.splits must be a mapping")
    root = Path.cwd() if base_directory is None else Path(base_directory).resolve()
    try:
        adapter_registry = load_runtime_adapter_registry(
            configuration,
            domain_ids=(target.domain_id for target in targets),
            harness_ids=(target.harness_id for target in targets),
            base_directory=root,
        )
    except RuntimeAdapterError as exc:
        raise RuntimeDataError(f"cannot load runtime adapters: {exc}") from exc
    split_ids = sorted(
        {target.split_id for target in targets if target.split_id is not None}
    )
    manifests: dict[str, LoadedTaskManifest] = {}
    global_task_split: dict[str, str] = {}
    verifiers: dict[str, VerifierSpec] = {}
    all_tasks: list[TaskSpec] = []
    for split_id in split_ids:
        declaration = raw_splits.get(split_id)
        if not isinstance(declaration, Mapping):
            raise RuntimeDataError(f"split {split_id!r} is missing or malformed")
        raw_path = declaration.get("manifest")
        if not isinstance(raw_path, str) or not raw_path:
            raise RuntimeDataError(f"split {split_id!r} manifest path is missing")
        if _PLACEHOLDER.search(raw_path):
            raise RuntimeDataError(f"split {split_id!r} manifest path is unresolved")
        source = Path(raw_path)
        if not source.is_absolute():
            source = root / source
        expected_count = declaration.get("expected_count")
        if isinstance(expected_count, bool) or not isinstance(expected_count, int):
            raise RuntimeDataError(
                f"split {split_id!r} expected_count must be frozen before execution"
            )
        loaded = load_local_task_manifest(
            source,
            expected_sha256=_sha256(
                declaration.get("manifest_sha256"),
                location=f"splits.{split_id}.manifest_sha256",
            ),
            expected_count=expected_count,
        )
        for task_id in loaded.source_task_ids:
            if task_id in global_task_split:
                raise RuntimeDataError(
                    f"duplicate task id {task_id!r} across splits "
                    f"{global_task_split[task_id]!r} and {split_id!r}"
                )
            global_task_split[task_id] = split_id
        if any(task.split_id != split_id for task in loaded.tasks):
            raise RuntimeDataError(
                f"task manifest for split {split_id!r} contains another split_id"
            )
        manifests[split_id] = loaded
        verifiers.update(loaded.verifiers)
        all_tasks.extend(loaded.tasks)

    adapted_tasks: list[TaskSpec] = []
    for domain_id in sorted({target.domain_id for target in targets}):
        source_tasks = tuple(
            task
            for task in all_tasks
            if task.domain_id == domain_id
            and any(
                target.domain_id == domain_id and _compatible(target, task)
                for target in targets
            )
        )
        try:
            adapted_tasks.extend(
                adapter_registry.adapt_domain(
                    domain_id=domain_id,
                    tasks=source_tasks,
                    phase=phase,
                )
            )
        except RuntimeAdapterError as exc:
            raise RuntimeDataError(f"cannot apply runtime domain adapter: {exc}") from exc
    relevant_tasks = tuple(
        sorted(
            (
                task
                for task in adapted_tasks
                if any(_compatible(target, task) for target in targets)
            ),
            key=lambda task: (
                task.split_id or "",
                task.domain_id,
                task.task_id,
                task.objective_role or "",
                task.group_id,
            ),
        )
    )
    blocks = tuple(
        TaskSeedBlock(
            block_id="block-"
            + stable_hash(
                {
                    "schema_version": 1,
                    "split_id": task.split_id,
                    "task_id": task.task_id,
                    "domain_id": task.domain_id,
                    "group_id": task.group_id,
                    "objective_role": task.objective_role,
                    "seed": seed,
                }
            )[:24],
            task=task,
            seed=seed,
        )
        for task in relevant_tasks
        for seed in seeds
    )
    block_ids = [block.block_id for block in blocks]
    if len(set(block_ids)) != len(block_ids):
        raise RuntimeDataError("task-seed block id collision")

    for target in targets:
        assert target.split_id is not None
        source_ids = set(manifests[target.split_id].source_task_ids)
        compatible = [block for block in blocks if _compatible(target, block.task)]
        identities = [(block.task.task_id, block.seed) for block in compatible]
        if len(set(identities)) != len(identities):
            raise RuntimeDataError(
                f"target {target.target_id!r} has duplicate task-seed identities"
            )
        expected = {(task_id, seed) for task_id in source_ids for seed in seeds}
        actual = set(identities)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise RuntimeDataError(
                f"target {target.target_id!r} matrix is not closed: "
                f"missing={missing[:3]}, extra={extra[:3]}"
            )

    domain_adapters = {
        domain_id: LocalDomainAdapter(
            domain_id,
            tuple(task for task in relevant_tasks if task.domain_id == domain_id),
        )
        for domain_id in sorted({target.domain_id for target in targets})
    }
    runtime_profile = configuration.get("active_runtime_profile")
    if runtime_profile is not None and not isinstance(runtime_profile, str):
        raise RuntimeDataError("configuration.active_runtime_profile must be a string")
    safety = configuration.get("safety", {})
    if not isinstance(safety, Mapping):
        raise RuntimeDataError("configuration.safety must be a mapping")
    allow_external_commands = safety.get("allow_external_verifier_commands") is True
    external_verifiers = [
        task_id for task_id, verifier in verifiers.items() if verifier.kind == "external_command"
    ]
    if external_verifiers and (
        runtime_profile not in _EXTERNAL_PROFILES or not allow_external_commands
    ):
        raise RuntimeDataError(
            "external verifier tasks require active_runtime_profile real/replay and "
            "safety.allow_external_verifier_commands=true: "
            f"{sorted(external_verifiers)[:3]}"
        )
    harness_adapters: dict[str, Harness] = {}
    for harness_id in sorted({target.harness_id for target in targets}):
        default_harness = VerifiedResponseHarness(
            harness_id=harness_id,
            verifiers=verifiers,
            response_path=_response_path_for(harness_id, response_paths),
            runtime_profile=runtime_profile,
            allow_external_commands=allow_external_commands,
        )
        try:
            harness_adapters[harness_id] = adapter_registry.build_harness(
                harness_id=harness_id,
                default_harness=default_harness,
                verifiers=verifiers,
                phase=phase,
            )
        except RuntimeAdapterError as exc:
            raise RuntimeDataError(f"cannot build runtime harness adapter: {exc}") from exc
    return RuntimeMatrix(
        targets=targets,
        blocks=blocks,
        harnesses=harness_adapters,
        domains=domain_adapters,
        manifests=manifests,
    )


__all__ = [
    "LoadedTaskManifest",
    "LocalDomainAdapter",
    "RuntimeDataError",
    "RuntimeMatrix",
    "VerifiedResponseHarness",
    "VerifierSpec",
    "load_local_task_manifest",
    "load_runtime_matrix",
]
