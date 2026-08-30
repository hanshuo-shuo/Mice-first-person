"""Bounded, reproducible phase-1 autoresearch for legal gaze schedules."""

from autoresearch.runner import (
    AutoresearchRunner,
    ConfirmationError,
    RunContractError,
    Runner,
    RunnerError,
    SetupError,
    SourceArtifactError,
)


__version__ = "0.1.0"

__all__ = [
    "AutoresearchRunner",
    "ConfirmationError",
    "RunContractError",
    "Runner",
    "RunnerError",
    "SetupError",
    "SourceArtifactError",
    "__version__",
]
