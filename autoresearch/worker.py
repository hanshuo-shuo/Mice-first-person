"""Killable process boundary for editable autoresearch candidates.

Candidate source is validated in the parent, then loaded exactly once in a
``spawn`` child.  Keeping the controller in that child preserves episode
state while ensuring a hung or memory-hungry candidate can be killed without
terminating the evaluator (or constraining the evaluator's SAC process).

The proxy intentionally implements the ordinary candidate ``reset`` and
``head_action`` signatures.  It can therefore be passed directly to
``CandidateBoundary`` and the frozen evaluator.  The child still calls
``reset_candidate`` and ``call_candidate`` so the existing runtime and public
input guards remain authoritative inside the isolation boundary.
"""

from __future__ import annotations

import math
import multiprocessing
import os
from pathlib import Path
import resource
import signal
import sys
import threading
import time
import weakref
from collections.abc import Mapping, Sequence
from multiprocessing.connection import Connection
from typing import Any

import numpy as np

from .guard import (
    CandidateRuntimeError,
    CandidateSourceError,
    validate_candidate_source_text,
)


DEFAULT_REQUEST_TIMEOUT_SECONDS = 0.75
DEFAULT_STARTUP_TIMEOUT_SECONDS = 10.0
DEFAULT_MEMORY_LIMIT_BYTES = 512 * 1024 * 1024
DEFAULT_CPU_LIMIT_SECONDS = 300

_READY = "ready"
_OK = "ok"
_FAILED = "failed"
_RESET = "reset"
_HEAD_ACTION = "head_action"
_CLOSE = "close"
_CLOSED = "closed"

# ``CandidateBoundary`` invokes proxy methods while the process-global runtime
# guard has replaced os.kill/os.getpid.  Capture the trusted primitives before
# that guard is entered so timeout cleanup cannot itself be blocked.
_ORIGINAL_OS_KILL = os.kill
_ORIGINAL_OS_GETPID = os.getpid


def _positive_finite_seconds(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive finite number") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be a positive finite number")
    return result


def _positive_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return result


def _source_snapshot(
    source: str | bytes | bytearray | memoryview | os.PathLike[str],
    *,
    filename: str | os.PathLike[str] | None,
) -> tuple[bytes, str]:
    """Resolve source once in the trusted parent without retaining a path."""

    source_filename = str(filename or "autoresearch/candidate.py")
    if isinstance(source, os.PathLike):
        source_path = Path(source)
        try:
            payload = source_path.read_bytes()
        except OSError as exc:
            raise CandidateSourceError("cannot read candidate source") from exc
        source_filename = str(filename or source_path)
    elif isinstance(source, str):
        # A string is normally source text, matching guard.py.  A single-line
        # string naming an existing regular file is also accepted for CLI
        # callers; pathlib objects remain the unambiguous path form.
        source_path = Path(source)
        try:
            is_path = "\n" not in source and "\r" not in source and source_path.is_file()
        except (OSError, ValueError):
            is_path = False
        if is_path:
            try:
                payload = source_path.read_bytes()
            except OSError as exc:
                raise CandidateSourceError("cannot read candidate source") from exc
            source_filename = str(filename or source_path)
        else:
            payload = source.encode("utf-8")
    elif isinstance(source, (bytes, bytearray, memoryview)):
        payload = bytes(source)
    else:
        raise CandidateSourceError("candidate source must be text, bytes, or a path")

    try:
        decoded = payload.decode("utf-8")
    except UnicodeError as exc:
        raise CandidateSourceError("candidate source is not UTF-8") from exc
    validate_candidate_source_text(decoded)
    return payload, source_filename


def _maximum_rss_bytes() -> int:
    """Return this process's peak resident set in bytes on supported Unix."""

    maximum = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Darwin reports bytes; Linux and the BSDs used by CI report KiB.
    return maximum if sys.platform == "darwin" else maximum * 1024


def _linux_virtual_memory_bytes() -> int | None:
    if not sys.platform.startswith("linux"):
        return None
    try:
        with open("/proc/self/statm", "rt", encoding="ascii") as stream:
            pages = int(stream.read().split()[0])
        return pages * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, IndexError):
        return None


def _set_child_resource_limits(
    *,
    memory_limit_bytes: int,
    cpu_limit_seconds: int,
) -> None:
    """Apply best-effort kernel limits in the child only.

    NumPy/OpenBLAS can reserve a very large virtual address range, especially
    on macOS.  Linux therefore gets an address-space ceiling relative to the
    already-loaded runtime, while every platform also uses the RSS watchdog
    below.  A finite cumulative CPU ceiling complements the per-request wall
    deadline enforced by the parent.
    """

    virtual_bytes = _linux_virtual_memory_bytes()
    if virtual_bytes is not None and hasattr(resource, "RLIMIT_AS"):
        address_limit = virtual_bytes + int(memory_limit_bytes)
        try:
            resource.setrlimit(resource.RLIMIT_AS, (address_limit, address_limit))
        except (OSError, ValueError):
            # RSS monitoring remains the portable enforcement path.
            pass

    if hasattr(resource, "RLIMIT_CPU"):
        soft_cpu = max(1, int(cpu_limit_seconds))
        hard_cpu = soft_cpu + 1
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (soft_cpu, hard_cpu))
        except (OSError, ValueError):
            # The parent wall deadline is authoritative for each request.
            pass


def _rss_watchdog(*, pid: int, limit_bytes: int, stop: threading.Event) -> None:
    """Kill only the candidate child after it exceeds its peak-RSS budget."""

    while not stop.wait(0.005):
        try:
            exceeded = _maximum_rss_bytes() > limit_bytes
        except Exception:
            # Failure to obtain an advisory sample must not kill valid work;
            # the kernel limit and request timeout remain in force.
            return
        if exceeded:
            try:
                _ORIGINAL_OS_KILL(pid, signal.SIGKILL)
            except OSError:
                pass
            return


def _safe_child_send(connection: Connection, message: tuple[Any, ...]) -> bool:
    try:
        connection.send(message)
    except (BrokenPipeError, EOFError, OSError):
        return False
    return True


def _process_is_alive(process: multiprocessing.process.BaseProcess) -> bool:
    """Check the child without multiprocessing's patched ``os.getpid`` call."""

    try:
        return process.exitcode is None
    except (AssertionError, ValueError):
        return False


def _wait_process(
    process: multiprocessing.process.BaseProcess,
    timeout: float,
) -> None:
    """Reap through the existing popen object while a runtime guard is active."""

    popen = getattr(process, "_popen", None)
    if popen is None:
        return
    try:
        popen.wait(timeout=timeout)
    except (AssertionError, OSError, ValueError):
        pass


def _candidate_worker_main(
    connection: Connection,
    source: bytes,
    filename: str,
    memory_limit_bytes: int,
    cpu_limit_seconds: int,
) -> None:
    """Load and serve one controller; never send candidate exception text."""

    # Import the contract before taking the RSS baseline.  This makes a small
    # configured budget compatible with the repository's normal NumPy runtime
    # rather than charging the candidate for importing trusted dependencies.
    from .contract import call_candidate, reset_candidate
    from .guard import load_candidate_controller_from_source

    pid = _ORIGINAL_OS_GETPID()
    baseline_rss = _maximum_rss_bytes()
    rss_ceiling = baseline_rss + int(memory_limit_bytes)
    stop_watchdog = threading.Event()
    watchdog = threading.Thread(
        target=_rss_watchdog,
        kwargs={"pid": pid, "limit_bytes": rss_ceiling, "stop": stop_watchdog},
        name="candidate-rss-watchdog",
        daemon=True,
    )
    watchdog.start()
    _set_child_resource_limits(
        memory_limit_bytes=int(memory_limit_bytes),
        cpu_limit_seconds=int(cpu_limit_seconds),
    )

    try:
        try:
            controller = load_candidate_controller_from_source(source, filename=filename)
        except BaseException:
            _safe_child_send(connection, (_FAILED,))
            return
        if not _safe_child_send(connection, (_READY,)):
            return

        while True:
            try:
                request = connection.recv()
            except (EOFError, OSError):
                return
            if not isinstance(request, tuple) or len(request) != 3:
                _safe_child_send(connection, (_FAILED,))
                return
            request_id, operation, payload = request
            if operation == _CLOSE:
                _safe_child_send(connection, (_CLOSED, request_id))
                return

            try:
                if operation == _RESET:
                    reset_candidate(controller, episode_seed=payload)
                    value: float | None = None
                elif operation == _HEAD_ACTION:
                    if not isinstance(payload, tuple) or len(payload) != 4:
                        raise CandidateRuntimeError("invalid worker request")
                    observation, public_history, base_head_action, step_index = payload
                    value = call_candidate(
                        controller,
                        observation=observation,
                        public_history=public_history,
                        base_head_action=base_head_action,
                        step_index=step_index,
                    )
                else:
                    raise CandidateRuntimeError("invalid worker operation")
            except BaseException:
                # Candidate-controlled exception strings and tracebacks never
                # cross the IPC boundary or enter parent-side artifacts.
                _safe_child_send(connection, (_FAILED, request_id))
                return

            if _maximum_rss_bytes() > rss_ceiling:
                _safe_child_send(connection, (_FAILED, request_id))
                return
            if not _safe_child_send(connection, (_OK, request_id, value)):
                return
    finally:
        stop_watchdog.set()
        try:
            connection.close()
        except OSError:
            pass


def _stop_process(process: multiprocessing.process.BaseProcess, connection: Connection) -> None:
    """Bounded, idempotent child cleanup usable under the runtime guard."""

    try:
        connection.close()
    except OSError:
        pass

    alive = _process_is_alive(process)
    if alive and process.pid is not None:
        try:
            _ORIGINAL_OS_KILL(int(process.pid), signal.SIGTERM)
        except (OSError, ValueError):
            pass
        _wait_process(process, 0.15)
    alive = _process_is_alive(process)
    if alive and process.pid is not None:
        try:
            _ORIGINAL_OS_KILL(int(process.pid), signal.SIGKILL)
        except (OSError, ValueError):
            pass
        _wait_process(process, 0.25)
    else:
        _wait_process(process, 0.0)


class IsolatedCandidateController:
    """Persistent, killable proxy implementing the candidate controller API."""

    def __init__(
        self,
        source: str | bytes | bytearray | memoryview | os.PathLike[str],
        *,
        timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        startup_timeout_seconds: float = DEFAULT_STARTUP_TIMEOUT_SECONDS,
        memory_limit_bytes: int = DEFAULT_MEMORY_LIMIT_BYTES,
        cpu_limit_seconds: int = DEFAULT_CPU_LIMIT_SECONDS,
        filename: str | os.PathLike[str] | None = None,
    ) -> None:
        payload, source_filename = _source_snapshot(source, filename=filename)
        timeout = _positive_finite_seconds(
            timeout_seconds,
            label="timeout_seconds",
        )
        startup_timeout = _positive_finite_seconds(
            startup_timeout_seconds,
            label="startup_timeout_seconds",
        )
        memory_limit = _positive_integer(
            memory_limit_bytes,
            label="memory_limit_bytes",
        )
        cpu_limit = _positive_integer(
            cpu_limit_seconds,
            label="cpu_limit_seconds",
        )

        # ``spawn`` does not inherit evaluator threads, CUDA/SAC state, open
        # descriptors, or an active runtime-guard patch.  This is safer than
        # fork for both macOS and Linux and is deliberately non-configurable.
        context = multiprocessing.get_context("spawn")
        parent_connection, child_connection = context.Pipe(duplex=True)
        process = context.Process(
            target=_candidate_worker_main,
            args=(
                child_connection,
                payload,
                source_filename,
                memory_limit,
                cpu_limit,
            ),
            name="autoresearch-candidate",
            daemon=True,
        )
        try:
            process.start()
        except BaseException as exc:
            parent_connection.close()
            child_connection.close()
            raise CandidateRuntimeError("candidate worker could not start") from exc
        child_connection.close()

        self._connection = parent_connection
        self._process = process
        self._pid = int(process.pid) if process.pid is not None else None
        self._timeout_seconds = timeout
        self._request_lock = threading.Lock()
        self._next_request_id = 0
        self._closed = False
        self._finalizer = weakref.finalize(
            self,
            _stop_process,
            process,
            parent_connection,
        )

        try:
            if not parent_connection.poll(startup_timeout):
                raise CandidateRuntimeError("candidate worker startup timed out")
            response = parent_connection.recv()
            if response != (_READY,):
                raise CandidateRuntimeError("candidate worker failed during startup")
        except (EOFError, OSError) as exc:
            self._abort()
            raise CandidateRuntimeError("candidate worker exited during startup") from exc
        except CandidateRuntimeError:
            self._abort()
            raise

    @classmethod
    def from_source(
        cls,
        source: str | bytes | bytearray | memoryview | os.PathLike[str],
        *,
        timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        startup_timeout_seconds: float = DEFAULT_STARTUP_TIMEOUT_SECONDS,
        memory_limit_bytes: int = DEFAULT_MEMORY_LIMIT_BYTES,
        cpu_limit_seconds: int = DEFAULT_CPU_LIMIT_SECONDS,
        filename: str | os.PathLike[str] | None = None,
    ) -> "IsolatedCandidateController":
        """Create a spawned worker from an immutable source snapshot or path."""

        return cls(
            source,
            timeout_seconds=timeout_seconds,
            startup_timeout_seconds=startup_timeout_seconds,
            memory_limit_bytes=memory_limit_bytes,
            cpu_limit_seconds=cpu_limit_seconds,
            filename=filename,
        )

    @property
    def pid(self) -> int | None:
        return self._pid

    @property
    def is_alive(self) -> bool:
        if self._closed:
            return False
        return _process_is_alive(self._process)

    @property
    def closed(self) -> bool:
        return self._closed

    def _abort(self) -> None:
        if not self._closed:
            self._closed = True
            if self._finalizer.alive:
                self._finalizer()

    def _request(self, operation: str, payload: Any) -> Any:
        deadline = time.monotonic() + self._timeout_seconds
        if not self._request_lock.acquire(timeout=self._timeout_seconds):
            self._abort()
            raise CandidateRuntimeError("candidate worker request timed out")
        try:
            if self._closed:
                raise CandidateRuntimeError("candidate worker is closed")
            if not self.is_alive:
                self._abort()
                raise CandidateRuntimeError("candidate worker exited unexpectedly")
            request_id = self._next_request_id
            self._next_request_id += 1
            try:
                self._connection.send((request_id, operation, payload))
                remaining = deadline - time.monotonic()
                if remaining <= 0.0 or not self._connection.poll(remaining):
                    self._abort()
                    raise CandidateRuntimeError("candidate worker request timed out")
                response = self._connection.recv()
            except CandidateRuntimeError:
                raise
            except (BrokenPipeError, EOFError, OSError) as exc:
                self._abort()
                raise CandidateRuntimeError("candidate worker exited unexpectedly") from exc

            if (
                not isinstance(response, tuple)
                or len(response) != 3
                or response[0] != _OK
                or response[1] != request_id
            ):
                self._abort()
                raise CandidateRuntimeError("candidate worker rejected the request")
            return response[2]
        finally:
            self._request_lock.release()

    def reset(self, *, episode_seed: int) -> None:
        self._request(_RESET, episode_seed)

    def head_action(
        self,
        *,
        observation: Mapping[str, np.ndarray],
        public_history: Sequence[Mapping[str, np.ndarray]],
        base_head_action: float,
        step_index: int,
    ) -> float:
        return float(
            self._request(
                _HEAD_ACTION,
                (
                    observation,
                    public_history,
                    base_head_action,
                    step_index,
                ),
            ),
        )

    def close(self) -> None:
        """Close gracefully when possible, then forcibly reap; safe to repeat."""

        with self._request_lock:
            if self._closed:
                return
            was_alive = _process_is_alive(self._process)
            self._closed = True
            request_id = self._next_request_id
            self._next_request_id += 1
            if was_alive:
                try:
                    self._connection.send((request_id, _CLOSE, None))
                    wait_for = min(0.2, self._timeout_seconds)
                    if self._connection.poll(wait_for):
                        self._connection.recv()
                except (BrokenPipeError, EOFError, OSError):
                    pass
            if self._finalizer.alive:
                self._finalizer()

    def __enter__(self) -> "IsolatedCandidateController":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        self.close()


__all__ = [
    "DEFAULT_CPU_LIMIT_SECONDS",
    "DEFAULT_MEMORY_LIMIT_BYTES",
    "DEFAULT_REQUEST_TIMEOUT_SECONDS",
    "DEFAULT_STARTUP_TIMEOUT_SECONDS",
    "IsolatedCandidateController",
]
