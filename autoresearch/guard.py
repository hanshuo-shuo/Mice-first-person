"""Trusted source, runtime, hash, path, and artifact guards.

The editable candidate is intentionally much less trusted than the evaluator.
Static checks make the permitted language surface small; runtime patches are a
second line of defense for accidental file, environment, network, or process
access.  They are not a general-purpose Python security sandbox, so real runs
must still execute in the isolated experiment worktree/worker described by the
handoff.
"""

from __future__ import annotations

import ast
import builtins
import contextlib
import ctypes
import hashlib
import io
import json
import multiprocessing
import os
from pathlib import Path, PurePosixPath
import pathlib
import re
import socket
import subprocess
import threading
import types
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
import urllib.request
from unittest import mock


DEFAULT_MUTABLE_PATHS = frozenset({"autoresearch/candidate.py"})
DEFAULT_CANDIDATE_IMPORTS = frozenset({"math"})


class GuardError(RuntimeError):
    """Base class for a failed autoresearch trust-boundary check."""


class CandidateSourceError(GuardError):
    """Candidate source exceeds the static allowlist."""


class CandidateRuntimeError(GuardError):
    """Candidate attempted a forbidden runtime operation."""


class ChangedPathError(GuardError):
    """An experiment changed source outside its one-file whitelist."""


class HashManifestError(GuardError):
    """A content manifest is invalid or no longer matches disk."""


class LeakError(GuardError):
    """Forbidden or secret content was found in an artifact."""


@dataclass(frozen=True)
class HashMismatch:
    path: str
    expected_sha256: str | None
    actual_sha256: str | None
    reason: str


@dataclass(frozen=True)
class LeakFinding:
    source: str
    rule_id: str
    line: int


_DEFAULT_LEAK_RULES = (
    ("provider-secret-name", b"OPENROUTER_API_KEY"),
    ("authorization-header", b"Authorization: Bearer"),
    ("bearer-credential", b"Bearer "),
    ("privileged-state-name", b"privileged_state"),
    ("exact-state-name", b"exact_state"),
    ("state-dictionary-access", b"get_state_dict"),
    ("predator-location-name", b"predator_location"),
    ("predator-position-name", b"predator_position"),
    ("predator-coordinate-name", b"predator_coordinates"),
    ("geometric-los-name", b"predator_geometric_los"),
)


def sha256_bytes(content: bytes | bytearray | memoryview | str) -> str:
    if isinstance(content, str):
        payload = content.encode("utf-8")
    else:
        payload = bytes(content)
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_repo_path(path: str | os.PathLike[str]) -> str:
    raw = os.fspath(path).replace("\\", "/")
    if not raw or "\x00" in raw or raw.startswith("/"):
        raise GuardError("repository paths must be non-empty and relative")
    raw_parts = raw.split("/")
    if any(part in ("", ".", "..") for part in raw_parts):
        raise GuardError("repository paths may not contain empty, dot, or parent segments")
    if raw_parts[0].endswith(":"):
        raise GuardError("repository paths may not use drive prefixes")
    normalized = PurePosixPath(*raw_parts).as_posix()
    if normalized == ".":
        raise GuardError("repository paths must identify a file")
    return normalized


def validate_changed_paths(
    changed_paths: Iterable[str | os.PathLike[str]],
    *,
    allowed_paths: Iterable[str | os.PathLike[str]] = DEFAULT_MUTABLE_PATHS,
) -> tuple[str, ...]:
    """Return normalized paths or fail if any path is not explicitly mutable."""

    try:
        allowed = {_normalized_repo_path(path) for path in allowed_paths}
        normalized = tuple(_normalized_repo_path(path) for path in changed_paths)
    except GuardError as exc:
        raise ChangedPathError(str(exc)) from exc
    disallowed = sorted(set(normalized).difference(allowed))
    if disallowed:
        raise ChangedPathError(
            "experiment changed paths outside the whitelist: " + ", ".join(disallowed),
        )
    return normalized


assert_changed_paths = validate_changed_paths


def _manifest_file(repo_root: Path, relative_path: str) -> Path:
    root = repo_root.resolve()
    candidate = root.joinpath(*PurePosixPath(relative_path).parts)
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise HashManifestError("manifest path escapes repository root") from exc
    if candidate.is_symlink():
        raise HashManifestError(f"manifest path may not be a symlink: {relative_path}")
    return candidate


def build_hash_manifest(
    repo_root: str | os.PathLike[str],
    paths: Iterable[str | os.PathLike[str]],
) -> dict[str, str]:
    """Hash explicitly named source files into a canonical, sorted manifest."""

    root = Path(repo_root)
    normalized = sorted({_normalized_repo_path(path) for path in paths})
    manifest: dict[str, str] = {}
    for relative_path in normalized:
        source_path = _manifest_file(root, relative_path)
        if not source_path.is_file():
            raise HashManifestError(f"manifest source is not a file: {relative_path}")
        manifest[relative_path] = sha256_file(source_path)
    return manifest


create_hash_manifest = build_hash_manifest


def manifest_sha256(manifest: Mapping[str, str]) -> str:
    canonical = json.dumps(
        dict(sorted(manifest.items())),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return sha256_bytes(canonical)


def verify_hash_manifest(
    repo_root: str | os.PathLike[str],
    manifest: Mapping[str, str],
) -> tuple[HashMismatch, ...]:
    """Report every missing, malformed, or changed manifest entry."""

    root = Path(repo_root)
    mismatches: list[HashMismatch] = []
    for raw_path, expected in sorted(manifest.items()):
        try:
            relative_path = _normalized_repo_path(raw_path)
        except GuardError:
            mismatches.append(HashMismatch(str(raw_path), str(expected), None, "invalid_path"))
            continue
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            mismatches.append(HashMismatch(relative_path, str(expected), None, "invalid_digest"))
            continue
        try:
            source_path = _manifest_file(root, relative_path)
        except HashManifestError:
            mismatches.append(HashMismatch(relative_path, expected, None, "unsafe_path"))
            continue
        if not source_path.is_file():
            mismatches.append(HashMismatch(relative_path, expected, None, "missing"))
            continue
        actual = sha256_file(source_path)
        if actual != expected:
            mismatches.append(HashMismatch(relative_path, expected, actual, "changed"))
    return tuple(mismatches)


def assert_hash_manifest(
    repo_root: str | os.PathLike[str],
    manifest: Mapping[str, str],
) -> None:
    mismatches = verify_hash_manifest(repo_root, manifest)
    if mismatches:
        paths = ", ".join(item.path for item in mismatches)
        raise HashManifestError("content hash manifest mismatch: " + paths)


def _content_bytes(content: str | bytes | bytearray | memoryview) -> bytes:
    if isinstance(content, str):
        return content.encode("utf-8")
    return bytes(content)


def scan_for_leaks(
    content: str | bytes | bytearray | memoryview,
    *,
    source: str = "<memory>",
    forbidden_tokens: Sequence[str | bytes] | None = None,
    secret_values: Sequence[str | bytes] = (),
) -> tuple[LeakFinding, ...]:
    """Scan content without ever putting a matched secret in the result."""

    payload = _content_bytes(content)
    rules: list[tuple[str, bytes]] = list(_DEFAULT_LEAK_RULES)
    if forbidden_tokens is not None:
        rules = [
            (f"forbidden-token-{index}", _content_bytes(token))
            for index, token in enumerate(forbidden_tokens)
        ]
    rules.extend(
        (f"secret-value-{index}", _content_bytes(secret))
        for index, secret in enumerate(secret_values)
        if _content_bytes(secret)
    )

    findings: list[LeakFinding] = []
    lowered = payload.lower()
    for rule_id, token in rules:
        if not token:
            continue
        needle = token.lower()
        start = 0
        while True:
            offset = lowered.find(needle, start)
            if offset < 0:
                break
            findings.append(
                LeakFinding(
                    source=str(source),
                    rule_id=rule_id,
                    line=payload.count(b"\n", 0, offset) + 1,
                ),
            )
            start = offset + max(1, len(needle))
    return tuple(findings)


def scan_files_for_leaks(
    paths: Iterable[str | os.PathLike[str]],
    *,
    repo_root: str | os.PathLike[str] | None = None,
    forbidden_tokens: Sequence[str | bytes] | None = None,
    secret_values: Sequence[str | bytes] = (),
) -> tuple[LeakFinding, ...]:
    root = Path(repo_root).resolve() if repo_root is not None else None
    findings: list[LeakFinding] = []
    for raw_path in paths:
        path = Path(raw_path)
        if root is not None:
            relative_path = _normalized_repo_path(raw_path)
            path = _manifest_file(root, relative_path)
            source = relative_path
        else:
            source = str(path)
        findings.extend(
            scan_for_leaks(
                path.read_bytes(),
                source=source,
                forbidden_tokens=forbidden_tokens,
                secret_values=secret_values,
            ),
        )
    return tuple(findings)


def assert_no_leaks(
    content: str | bytes | bytearray | memoryview,
    *,
    source: str = "<memory>",
    forbidden_tokens: Sequence[str | bytes] | None = None,
    secret_values: Sequence[str | bytes] = (),
) -> None:
    findings = scan_for_leaks(
        content,
        source=source,
        forbidden_tokens=forbidden_tokens,
        secret_values=secret_values,
    )
    if findings:
        rules = ", ".join(sorted({finding.rule_id for finding in findings}))
        raise LeakError(f"artifact failed leak scan ({rules})")


_FORBIDDEN_IMPORT_ROOTS = frozenset(
    {
        "aiohttp",
        "asyncio",
        "builtins",
        "ctypes",
        "glob",
        "http",
        "importlib",
        "inspect",
        "io",
        "marshal",
        "multiprocessing",
        "os",
        "pathlib",
        "pickle",
        "requests",
        "resource",
        "secrets",
        "shutil",
        "signal",
        "socket",
        "subprocess",
        "sys",
        "tempfile",
        "threading",
        "time",
        "urllib",
    },
)
_FORBIDDEN_NAMES = frozenset(
    {
        "__builtins__",
        "__import__",
        "breakpoint",
        "compile",
        "delattr",
        "dir",
        "environment",
        "eval",
        "exec",
        "exit",
        "getattr",
        "globals",
        "help",
        "info",
        "input",
        "locals",
        "memoryview",
        "open",
        "quit",
        "reward",
        "setattr",
        "simulator",
        "snapshot",
        "state_dict",
        "SystemExit",
        "terminated",
        "truncated",
        "vars",
    },
)
_FORBIDDEN_ATTRIBUTES = frozenset(
    {
        "BitGenerator",
        "DataSource",
        "Generator",
        "MT19937",
        "Popen",
        "PCG64",
        "PCG64DXSM",
        "Philox",
        "RandomState",
        "SFC64",
        "SeedSequence",
        "chdir",
        "connect",
        "create_connection",
        "ctypes",
        "array",
        "asarray",
        "astype",
        "busday_count",
        "busday_offset",
        "datetime_as_string",
        "datetime_data",
        "datetime64",
        "dtype",
        "environ",
        "environb",
        "execv",
        "fork",
        "fromfile",
        "fromstring",
        "genfromtxt",
        "getcwd",
        "getenv",
        "getpid",
        "kill",
        "is_busday",
        "listdir",
        "lstat",
        "load",
        "load_library",
        "loadtxt",
        "memmap",
        "open",
        "popen",
        "putenv",
        "read_bytes",
        "readlink",
        "read_text",
        "remove",
        "rename",
        "replace",
        "rmdir",
        "run",
        "save",
        "savetxt",
        "savez",
        "savez_compressed",
        "scandir",
        "socket",
        "spawn",
        "stat",
        "system",
        "timedelta64",
        "tofile",
        "unlink",
        "unsetenv",
        "urlopen",
        "urandom",
        "view",
        "walk",
        "write_bytes",
        "write_text",
        "_datasource",
        "dump",
        "dumps",
    },
)


def _safe_constant_expression(node: ast.AST | None) -> bool:
    if node is None or isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.Tuple):
        return all(_safe_constant_expression(item) for item in node.elts)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _safe_constant_expression(node.operand)
    if isinstance(node, ast.BinOp) and isinstance(
        node.op,
        (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow),
    ):
        return _safe_constant_expression(node.left) and _safe_constant_expression(node.right)
    return False


def _attribute_chain(node: ast.AST) -> tuple[str, ...]:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return tuple(reversed(parts))
    return ()


class _CandidateAstGuard(ast.NodeVisitor):
    def __init__(self, allowed_imports: frozenset[str]) -> None:
        self.allowed_imports = allowed_imports
        self.errors: list[str] = []
        self.function_stack: list[str] = []
        self.direct_rng_nodes: set[int] = set()

    def error(self, node: ast.AST, message: str) -> None:
        self.errors.append(f"line {getattr(node, 'lineno', '?')}: {message}")

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".", 1)[0]
            if root in _FORBIDDEN_IMPORT_ROOTS or root not in self.allowed_imports:
                self.error(node, f"import is not allowlisted: {root}")
            if root == "numpy" and alias.asname not in (None, "np"):
                self.error(node, "numpy may only be imported as np")
            if root == "math" and alias.asname not in (None, "math"):
                self.error(node, "math may not be aliased")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        root = (node.module or "").split(".", 1)[0]
        if node.level or root in _FORBIDDEN_IMPORT_ROOTS or root not in self.allowed_imports:
            self.error(node, f"import is not allowlisted: {root or '<relative>'}")
        if root == "numpy":
            self.error(node, "from-imports from numpy are not permitted")
        if any(alias.name == "*" for alias in node.names):
            self.error(node, "wildcard imports are not permitted")

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in _FORBIDDEN_NAMES or node.id.startswith("__"):
            self.error(node, f"forbidden name: {node.id}")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        chain = _attribute_chain(node)
        if (
            len(chain) >= 2
            and chain[:2] == ("np", "random")
            and id(node) not in self.direct_rng_nodes
        ):
            self.error(
                node,
                "np.random is available only through a direct seeded default_rng call",
            )
        if node.attr.startswith("__") or node.attr in _FORBIDDEN_ATTRIBUTES:
            self.error(node, f"forbidden attribute: {node.attr}")
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            lowered = node.value.strip().lower()
            compact = lowered.replace(" ", "")
            if lowered in {"now", "today"} or (
                "datetime64" in compact
                or "timedelta64" in compact
                or re.fullmatch(r"[<>=|]?[mM]8(?:\[[^]]*\])?", node.value.strip())
            ):
                self.error(node, "clock-derived NumPy datetime values are not permitted")

    def visit_Call(self, node: ast.Call) -> None:
        chain = _attribute_chain(node.func)
        if len(chain) >= 2 and chain[:2] == ("np", "random") and chain[-1] != "default_rng":
            self.error(node, "only np.random.default_rng is permitted")
        if (
            len(chain) >= 3
            and chain[0] == "self"
            and chain[-1]
            in {"add", "append", "clear", "extend", "insert", "pop", "remove", "setdefault", "update"}
        ):
            self.error(node, "candidate instance collections may not grow or mutate")
        for keyword in node.keywords:
            if keyword.arg != "dtype":
                continue
            allowed_dtype = False
            if isinstance(keyword.value, ast.Attribute):
                dtype_chain = _attribute_chain(keyword.value)
                allowed_dtype = dtype_chain in {
                    ("np", "bool_"),
                    ("np", "float32"),
                    ("np", "float64"),
                    ("np", "int8"),
                    ("np", "int16"),
                    ("np", "int32"),
                    ("np", "int64"),
                    ("np", "uint8"),
                    ("np", "uint16"),
                    ("np", "uint32"),
                    ("np", "uint64"),
                }
            elif isinstance(keyword.value, ast.Constant) and isinstance(
                keyword.value.value,
                str,
            ):
                allowed_dtype = keyword.value.value.lower() in {
                    "bool",
                    "float32",
                    "float64",
                    "int8",
                    "int16",
                    "int32",
                    "int64",
                    "uint8",
                    "uint16",
                    "uint32",
                    "uint64",
                }
            if not allowed_dtype:
                self.error(keyword.value, "NumPy dtype is not in the numeric allowlist")
        if isinstance(node.func, ast.Attribute) and node.func.attr == "default_rng":
            current: ast.AST = node.func
            while isinstance(current, ast.Attribute):
                self.direct_rng_nodes.add(id(current))
                current = current.value
            seed_keywords = [
                keyword.value
                for keyword in node.keywords
                if keyword.arg in {"seed", "seed_sequence"}
            ]
            has_unpacking = any(
                isinstance(argument, ast.Starred) for argument in node.args
            ) or any(keyword.arg is None for keyword in node.keywords)
            seed_nodes = list(node.args) + seed_keywords
            if (
                has_unpacking
                or len(seed_nodes) != 1
                or len(seed_keywords) != len(node.keywords)
            ):
                self.error(
                    node,
                    "private RNG construction requires exactly one explicit seed",
                )
            if any(
                isinstance(seed, ast.Constant) and seed.value is None
                for seed in seed_nodes
            ):
                self.error(node, "private RNG seed may not be None")
        self.generic_visit(node)

    def _check_assignment_target(self, target: ast.AST) -> None:
        if isinstance(target, ast.Attribute):
            chain = _attribute_chain(target)
            if not chain or chain[0] != "self":
                self.error(target, "attribute assignment is permitted only on self")
            elif (
                self.function_stack
                and self.function_stack[-1] == "head_action"
                and chain[-1] not in {"_head_yaw_degrees", "_rng_state"}
            ):
                self.error(
                    target,
                    "head_action may persist only the bounded physical head-yaw state",
                )
        elif isinstance(target, ast.Subscript):
            self.error(target, "subscript assignment is not permitted")
        elif isinstance(target, (ast.Tuple, ast.List)):
            for item in target.elts:
                self._check_assignment_target(item)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._check_assignment_target(target)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._check_assignment_target(node.target)
        self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> None:
        if node.annotation is not None:
            self.error(node, "annotations are not permitted")
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.error(node, "annotations are not permitted")
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.returns is not None:
            self.error(node, "annotations are not permitted")
        self.function_stack.append(node.name)
        try:
            self.generic_visit(node)
        finally:
            self.function_stack.pop()

    def visit_Global(self, node: ast.Global) -> None:
        self.error(node, "global statements are not permitted")

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.error(node, "nonlocal statements are not permitted")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.error(node, "async functions are not permitted")

    def visit_Await(self, node: ast.Await) -> None:
        self.error(node, "await is not permitted")

    def visit_Yield(self, node: ast.Yield) -> None:
        self.error(node, "generators are not permitted")

    def visit_YieldFrom(self, node: ast.YieldFrom) -> None:
        self.error(node, "generators are not permitted")


def _check_definition_safety(tree: ast.Module, errors: list[str]) -> None:
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(
            node.value.value,
            str,
        ):
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
            pass
        elif isinstance(node, ast.Assign):
            if not _safe_constant_expression(node.value):
                errors.append(f"line {node.lineno}: module assignments must be constant")
        elif isinstance(node, ast.AnnAssign):
            if not _safe_constant_expression(node.value):
                errors.append(f"line {node.lineno}: module assignments must be constant")
        else:
            errors.append(
                f"line {getattr(node, 'lineno', '?')}: executable module statement is forbidden",
            )

        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.decorator_list:
            errors.append(f"line {node.lineno}: decorators are not permitted")
        if isinstance(node, ast.FunctionDef):
            defaults = [*node.args.defaults, *node.args.kw_defaults]
            if any(not _safe_constant_expression(default) for default in defaults):
                errors.append(f"line {node.lineno}: function defaults must be constant")
        if isinstance(node, ast.ClassDef):
            if node.bases or node.keywords:
                errors.append(f"line {node.lineno}: candidate classes may not use bases/metaclasses")
            for item in node.body:
                if isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant) and isinstance(
                    item.value.value,
                    str,
                ):
                    continue
                if isinstance(item, ast.FunctionDef):
                    if item.decorator_list:
                        errors.append(f"line {item.lineno}: decorators are not permitted")
                    defaults = [*item.args.defaults, *item.args.kw_defaults]
                    if any(not _safe_constant_expression(default) for default in defaults):
                        errors.append(f"line {item.lineno}: method defaults must be constant")
                    continue
                if isinstance(item, ast.Assign) and _safe_constant_expression(item.value):
                    continue
                if isinstance(item, ast.AnnAssign) and _safe_constant_expression(item.value):
                    continue
                errors.append(
                    f"line {getattr(item, 'lineno', '?')}: executable class statement is forbidden",
                )


def _check_candidate_interface(tree: ast.Module, errors: list[str]) -> None:
    candidates = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "CandidateGazeController"
    ]
    if len(candidates) != 1:
        errors.append("source must define exactly one CandidateGazeController class")
        return
    methods = {
        node.name: node for node in candidates[0].body if isinstance(node, ast.FunctionDef)
    }
    required = {
        "reset": ("episode_seed",),
        "head_action": (
            "observation",
            "public_history",
            "base_head_action",
            "step_index",
        ),
    }
    for method_name, keyword_names in required.items():
        method = methods.get(method_name)
        if method is None:
            errors.append(f"CandidateGazeController must define {method_name}()")
            continue
        positional = [argument.arg for argument in method.args.posonlyargs + method.args.args]
        keyword_only = [argument.arg for argument in method.args.kwonlyargs]
        if positional != ["self"] or tuple(keyword_only) != keyword_names:
            errors.append(
                f"{method_name}() must accept self plus keyword-only " + ", ".join(keyword_names),
            )
        if method.args.vararg is not None or method.args.kwarg is not None:
            errors.append(f"{method_name}() may not accept variadic arguments")


def validate_candidate_source_text(
    source: str,
    *,
    allowed_imports: Iterable[str] = DEFAULT_CANDIDATE_IMPORTS,
    require_interface: bool = True,
) -> ast.Module:
    """Parse candidate source and enforce the narrow phase-1 AST policy."""

    try:
        tree = ast.parse(source, filename="autoresearch/candidate.py")
    except SyntaxError as exc:
        raise CandidateSourceError(f"candidate syntax error at line {exc.lineno}") from exc
    guard = _CandidateAstGuard(frozenset(allowed_imports))
    guard.visit(tree)
    _check_definition_safety(tree, guard.errors)
    if require_interface:
        _check_candidate_interface(tree, guard.errors)
    if guard.errors:
        raise CandidateSourceError("; ".join(guard.errors))
    return tree


def validate_candidate_source(
    path: str | os.PathLike[str],
    *,
    allowed_imports: Iterable[str] = DEFAULT_CANDIDATE_IMPORTS,
) -> ast.Module:
    candidate_path = Path(path)
    return validate_candidate_source_text(
        candidate_path.read_text(encoding="utf-8"),
        allowed_imports=allowed_imports,
    )


def _restricted_import(
    name: str,
    globals_: Mapping[str, Any] | None = None,
    locals_: Mapping[str, Any] | None = None,
    fromlist: Sequence[str] = (),
    level: int = 0,
) -> Any:
    del globals_, locals_
    root = name.split(".", 1)[0]
    # Static validation rejects candidate ``import numpy`` statements.  Array
    # methods on the defensive-copy inputs may lazily import NumPy internals,
    # so those internal absolute imports remain available at runtime.
    runtime_imports = DEFAULT_CANDIDATE_IMPORTS | {"numpy"}
    if level or root not in runtime_imports:
        raise CandidateRuntimeError("candidate import was denied")
    return builtins.__import__(name, {}, {}, fromlist, 0)


_SAFE_BUILTINS = {
    "__build_class__": builtins.__build_class__,
    "__import__": _restricted_import,
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "Exception": Exception,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "round": round,
    "set": set,
    "sum": sum,
    "tuple": tuple,
    "ValueError": ValueError,
    "zip": zip,
}


def load_candidate_controller_from_source(
    source: str | bytes,
    *,
    filename: str = "autoresearch/candidate.py",
    allowed_imports: Iterable[str] = DEFAULT_CANDIDATE_IMPORTS,
) -> Any:
    """Validate and instantiate one immutable in-memory source snapshot."""

    if isinstance(source, bytes):
        try:
            source_text = source.decode("utf-8")
        except UnicodeError as exc:
            raise CandidateSourceError("candidate source is not UTF-8") from exc
    elif isinstance(source, str):
        source_text = source
    else:
        raise CandidateSourceError("candidate source must be str or bytes")
    tree = validate_candidate_source_text(
        source_text,
        allowed_imports=allowed_imports,
    )
    code = compile(tree, str(filename), "exec", dont_inherit=True)
    module = types.ModuleType("autoresearch_candidate")
    module.__dict__.update(
        {
            "__builtins__": dict(_SAFE_BUILTINS),
            "__file__": str(filename),
            "__name__": "autoresearch_candidate",
            "__package__": "",
        },
    )
    with restricted_candidate_runtime():
        # Function annotations, defaults, decorators, bases, and executable
        # module statements are statically rejected.  Keep module execution
        # inside the runtime guard as a second line of defense.
        exec(code, module.__dict__, module.__dict__)
        controller_type = module.__dict__.get("CandidateGazeController")
        if not isinstance(controller_type, type):
            raise CandidateSourceError("CandidateGazeController is not a class")
        controller = controller_type()
    if not callable(getattr(controller, "reset", None)) or not callable(
        getattr(controller, "head_action", None),
    ):
        raise CandidateSourceError("candidate does not implement the public interface")
    return controller


def load_candidate_controller(
    path: str | os.PathLike[str],
    *,
    allowed_imports: Iterable[str] = DEFAULT_CANDIDATE_IMPORTS,
) -> Any:
    """Load exactly the bytes read once from a guarded candidate path."""

    candidate_path = Path(path)
    try:
        source = candidate_path.read_bytes()
    except OSError as exc:
        raise CandidateSourceError(f"cannot read candidate source: {candidate_path}") from exc
    return load_candidate_controller_from_source(
        source,
        filename=str(candidate_path),
        allowed_imports=allowed_imports,
    )


class _DeniedEnvironment(Mapping[str, str]):
    def _deny(self) -> None:
        raise CandidateRuntimeError("candidate environment access was denied")

    def __getitem__(self, key: str) -> str:
        del key
        self._deny()

    def __iter__(self):
        self._deny()

    def __len__(self) -> int:
        self._deny()

    def get(self, key: str, default: Any = None) -> Any:
        del key, default
        self._deny()

    def copy(self) -> dict[str, str]:
        self._deny()


def _deny_runtime(*args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    raise CandidateRuntimeError("candidate I/O or process access was denied")


_RUNTIME_GUARD_LOCK = threading.RLock()


@contextlib.contextmanager
def restricted_candidate_runtime():
    """Temporarily deny common filesystem, env, network, and process APIs.

    The patches are process-global, so the lock serializes candidate calls.
    Production runners should additionally use their dedicated worker process
    for timeout and resource isolation.
    """

    patch_targets: tuple[tuple[Any, str], ...] = (
        (builtins, "open"),
        (builtins, "input"),
        (io, "open"),
        (pathlib.Path, "open"),
        (pathlib.Path, "read_bytes"),
        (pathlib.Path, "read_text"),
        (pathlib.Path, "write_bytes"),
        (pathlib.Path, "write_text"),
        (os, "chdir"),
        (os, "execv"),
        (os, "execve"),
        (os, "fork"),
        (os, "forkpty"),
        (os, "getcwd"),
        (os, "getenv"),
        (os, "getpid"),
        (os, "kill"),
        (os, "killpg"),
        (os, "listdir"),
        (os, "lstat"),
        (os, "popen"),
        (os, "putenv"),
        (os, "readlink"),
        (os, "scandir"),
        (os, "spawnv"),
        (os, "stat"),
        (os, "system"),
        (os, "unsetenv"),
        (os, "urandom"),
        (os, "walk"),
        (os, "_exit"),
        (socket, "create_connection"),
        (socket, "getaddrinfo"),
        (socket, "socket"),
        (subprocess, "Popen"),
        (subprocess, "call"),
        (subprocess, "check_call"),
        (subprocess, "check_output"),
        (subprocess, "run"),
        (urllib.request, "urlopen"),
        (multiprocessing, "Process"),
        (ctypes, "CDLL"),
        (ctypes, "PyDLL"),
    )
    with _RUNTIME_GUARD_LOCK:
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(os, "environ", _DeniedEnvironment()))
            if hasattr(os, "environb"):
                stack.enter_context(
                    mock.patch.object(os, "environb", _DeniedEnvironment()),
                )
            for owner, attribute in patch_targets:
                if hasattr(owner, attribute):
                    stack.enter_context(mock.patch.object(owner, attribute, _deny_runtime))
            yield
