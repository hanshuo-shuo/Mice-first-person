"""OpenRouter VLM backend with strict output validation and safe mock fallback."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from PIL import Image
from jsonschema.exceptions import ValidationError

from .base import (
    DECISION_JSON_SCHEMA,
    MockVisionPolicy,
    PolicyDecision,
    PolicyInput,
    PolicyResult,
    PolicyTelemetry,
    VisionPolicy,
)


SYSTEM_PROMPT = """You are a vision-only safety policy for a simulated mouse.
Use only the supplied left/right RGB images, optional prior RGB images,
normalized proprioception, and previous action. Do not infer access to world
coordinates, simulator state, visibility labels, rewards, or future outcomes.
Return exactly one JSON object matching the supplied strict schema. The
recommended actions are semantic suggestions only; you cannot modify the
environment."""


@dataclass(frozen=True)
class OpenRouterConfig:
    model: str
    provider: Mapping[str, Any]
    timeout_seconds: float = 30.0
    max_retries: int = 2
    retry_backoff_seconds: float = 0.5
    max_history_frames: int = 4
    risk_horizon_seconds: float | None = None
    macro_duration_seconds: float | None = None
    base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    cache_dir: Path = Path(".cache/openrouter")
    log_path: Path = Path("policy_calls.jsonl")

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        experiment_dir: Path,
    ) -> "OpenRouterConfig":
        provider = dict(value.get("provider", {}))
        return cls(
            model=str(value.get("model", "openai/gpt-4.1-mini")),
            provider=provider,
            timeout_seconds=float(value.get("timeout_seconds", 30.0)),
            max_retries=int(value.get("max_retries", 2)),
            retry_backoff_seconds=float(value.get("retry_backoff_seconds", 0.5)),
            max_history_frames=int(value.get("max_history_frames", 4)),
            risk_horizon_seconds=(
                float(value["risk_horizon_seconds"])
                if value.get("risk_horizon_seconds") is not None
                else None
            ),
            macro_duration_seconds=(
                float(value["macro_duration_seconds"])
                if value.get("macro_duration_seconds") is not None
                else None
            ),
            base_url=str(
                value.get(
                    "base_url",
                    "https://openrouter.ai/api/v1/chat/completions",
                ),
            ),
            cache_dir=experiment_dir / "response_cache",
            log_path=experiment_dir / "policy_calls.jsonl",
        )

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("OpenRouter model must be an exact non-empty identifier")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must be non-negative")
        if self.max_history_frames < 0:
            raise ValueError("max_history_frames must be non-negative")
        if self.risk_horizon_seconds is not None and self.risk_horizon_seconds <= 0:
            raise ValueError("risk_horizon_seconds must be positive when supplied")
        if self.macro_duration_seconds is not None and self.macro_duration_seconds <= 0:
            raise ValueError("macro_duration_seconds must be positive when supplied")


class OpenRouterVLMPolicy(VisionPolicy):
    """Call OpenRouter when keyed; otherwise use a deterministic local mock."""

    def __init__(
        self,
        config: OpenRouterConfig,
        *,
        api_key: str | None = None,
        transport: Callable[[Mapping[str, Any], float], Mapping[str, Any]] | None = None,
    ) -> None:
        self.config = config
        self._api_key = api_key if api_key is not None else os.environ.get("OPENROUTER_API_KEY")
        self._transport = transport
        self._mock = MockVisionPolicy()
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        self.config.log_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def backend(self) -> str:
        return "openrouter" if self._api_key else "mock"

    @staticmethod
    def _image_data_url(image) -> str:
        buffer = io.BytesIO()
        Image.fromarray(image, mode="RGB").save(buffer, format="PNG", optimize=False)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def _public_text(self, policy_input: PolicyInput) -> str:
        values = policy_input.public_sensor_values()
        text = (
            "Public normalized sensors:\n"
            + json.dumps(values, sort_keys=True, separators=(",", ":"))
            + "\nReturn the semantic threat/risk/look/motion decision."
        )
        if (
            self.config.risk_horizon_seconds is not None
            and self.config.macro_duration_seconds is not None
        ):
            text += (
                "\nRegistered EXP-01 semantics: threat_visible describes only "
                "the current eye images; risk_next_horizon may use public image "
                "history and estimates capture or unsafe proximity within "
                f"{self.config.risk_horizon_seconds:g} seconds. The recommended "
                "motion/look macro may run for up to "
                f"{self.config.macro_duration_seconds:g} seconds before the next "
                "decision. Look targets are relative head yaw: far_left=+60, "
                "left=+30, center=0, right=-30, far_right=-60 degrees; hold keeps "
                "the current head direction."
            )
        return text

    def _content(self, policy_input: PolicyInput) -> list[Mapping[str, Any]]:
        content: list[Mapping[str, Any]] = [{"type": "text", "text": self._public_text(policy_input)}]
        history = (
            policy_input.history[-self.config.max_history_frames :]
            if self.config.max_history_frames
            else ()
        )
        for index, frame in enumerate(history):
            content.extend(
                [
                    {"type": "text", "text": f"History frame {index} left eye"},
                    {
                        "type": "image_url",
                        "image_url": {"url": self._image_data_url(frame.image_left)},
                    },
                    {"type": "text", "text": f"History frame {index} right eye"},
                    {
                        "type": "image_url",
                        "image_url": {"url": self._image_data_url(frame.image_right)},
                    },
                ],
            )
        content.extend(
            [
                {"type": "text", "text": "Current left eye"},
                {
                    "type": "image_url",
                    "image_url": {"url": self._image_data_url(policy_input.image_left)},
                },
                {"type": "text", "text": "Current right eye"},
                {
                    "type": "image_url",
                    "image_url": {"url": self._image_data_url(policy_input.image_right)},
                },
            ],
        )
        return content

    def build_request(self, policy_input: PolicyInput) -> Mapping[str, Any]:
        """Build a provider request from public fields only.

        This method is intentionally public for security-boundary tests.  It
        does not accept a snapshot, ``info`` dictionary, or privileged label.
        """

        request: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": self._content(policy_input)},
            ],
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "self_preservation_decision",
                    "strict": True,
                    "schema": DECISION_JSON_SCHEMA,
                },
            },
        }
        if self.config.provider:
            request["provider"] = dict(self.config.provider)
        return request

    def _prompt_hash(
        self,
        policy_input: PolicyInput,
        image_hashes: Mapping[str, Any],
    ) -> str:
        material = {
            "system_prompt": SYSTEM_PROMPT,
            "public_sensors": policy_input.public_sensor_values(),
            "image_hashes": image_hashes,
            "model": self.config.model,
            "provider": dict(self.config.provider),
            "base_url": self.config.base_url,
            "schema": DECISION_JSON_SCHEMA,
            "risk_horizon_seconds": self.config.risk_horizon_seconds,
            "macro_duration_seconds": self.config.macro_duration_seconds,
        }
        return hashlib.sha256(
            json.dumps(material, sort_keys=True, separators=(",", ":")).encode(
                "utf-8",
            ),
        ).hexdigest()

    def _cache_path(self, prompt_hash: str) -> Path:
        return self.config.cache_dir / f"{prompt_hash}.json"

    def _sanitize(self, value: Any) -> Any:
        """Remove the credential if a provider unexpectedly echoes it."""

        if not self._api_key:
            return value
        serialized = json.dumps(value, ensure_ascii=False)
        serialized = serialized.replace(self._api_key, "[REDACTED]")
        return json.loads(serialized)

    def _default_transport(
        self,
        request_payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        assert self._api_key is not None
        encoded = json.dumps(request_payload).encode("utf-8")
        request = urllib.request.Request(
            self.config.base_url,
            data=encoded,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _message_content(response: Mapping[str, Any]) -> str:
        content = response["choices"][0]["message"]["content"]
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, Mapping)
            )
        raise TypeError("OpenRouter message content must be a string or text list")

    @classmethod
    def _parse_decision(cls, response: Mapping[str, Any]) -> PolicyDecision:
        content = cls._message_content(response).strip()
        value = json.loads(content)
        if not isinstance(value, Mapping):
            raise TypeError("Model output must be one JSON object")
        return PolicyDecision.from_mapping(value)

    @staticmethod
    def _usage(response: Mapping[str, Any]) -> Mapping[str, Any]:
        usage = response.get("usage", {})
        if not isinstance(usage, Mapping):
            return {}
        return {
            key: usage[key]
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            if key in usage
        }

    @staticmethod
    def _cost(response: Mapping[str, Any]) -> float | None:
        usage = response.get("usage", {})
        candidates = []
        if isinstance(usage, Mapping):
            candidates.extend((usage.get("cost"), usage.get("total_cost")))
        candidates.extend((response.get("cost"), response.get("total_cost")))
        for value in candidates:
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
        return None

    def _append_log(self, telemetry: PolicyTelemetry, error: str | None = None) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **telemetry.to_dict(),
            "error": error,
        }
        with self.config.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def _mock_decide(self, policy_input: PolicyInput) -> PolicyResult:
        result = self._mock.decide(policy_input)
        telemetry = PolicyTelemetry(
            backend="mock",
            model=result.telemetry.model,
            provider=result.telemetry.provider,
            prompt_hash=result.telemetry.prompt_hash,
            image_hashes=result.telemetry.image_hashes,
            latency_ms=result.telemetry.latency_ms,
            token_usage=result.telemetry.token_usage,
            cost=result.telemetry.cost,
            parse_success=True,
            cache_hit=False,
            raw_response=result.telemetry.raw_response,
        )
        wrapped = PolicyResult(decision=result.decision, telemetry=telemetry)
        self._append_log(telemetry)
        return wrapped

    def decide(self, policy_input: PolicyInput) -> PolicyResult:
        if not self._api_key:
            return self._mock_decide(policy_input)

        started = time.perf_counter()
        image_hashes = policy_input.image_hashes()
        prompt_hash = self._prompt_hash(policy_input, image_hashes)
        cache_path = self._cache_path(prompt_hash)
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            response = cached["raw_response"]
            decision = self._parse_decision(response)
            telemetry = PolicyTelemetry(
                backend="openrouter",
                model=self.config.model,
                provider=dict(self.config.provider),
                prompt_hash=prompt_hash,
                image_hashes=image_hashes,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                token_usage=self._usage(response),
                cost=self._cost(response),
                parse_success=True,
                cache_hit=True,
                raw_response=self._sanitize(response),
            )
            self._append_log(telemetry)
            return PolicyResult(decision=decision, telemetry=telemetry)

        request_payload = self.build_request(policy_input)
        transport = self._transport or self._default_transport
        last_error: Exception | None = None
        last_response: Mapping[str, Any] | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response = transport(request_payload, self.config.timeout_seconds)
                last_response = response
                decision = self._parse_decision(response)
                sanitized = self._sanitize(response)
                temporary = cache_path.with_name(f".{cache_path.name}.tmp")
                temporary.write_text(
                    json.dumps(
                        {"raw_response": sanitized},
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                os.replace(temporary, cache_path)
                telemetry = PolicyTelemetry(
                    backend="openrouter",
                    model=self.config.model,
                    provider=dict(self.config.provider),
                    prompt_hash=prompt_hash,
                    image_hashes=image_hashes,
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                    token_usage=self._usage(response),
                    cost=self._cost(response),
                    parse_success=True,
                    cache_hit=False,
                    raw_response=sanitized,
                )
                self._append_log(telemetry)
                return PolicyResult(decision=decision, telemetry=telemetry)
            except (
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
                ValidationError,
                urllib.error.HTTPError,
                urllib.error.URLError,
                TimeoutError,
            ) as error:
                last_error = error
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_backoff_seconds * (2**attempt))

        telemetry = PolicyTelemetry(
            backend="openrouter",
            model=self.config.model,
            provider=dict(self.config.provider),
            prompt_hash=prompt_hash,
            image_hashes=image_hashes,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            token_usage=self._usage(last_response or {}),
            cost=self._cost(last_response or {}),
            parse_success=False,
            cache_hit=False,
            raw_response=self._sanitize(last_response),
        )
        self._append_log(telemetry, error=type(last_error).__name__ if last_error else "unknown")
        raise RuntimeError("OpenRouter response failed strict validation") from last_error


def build_policy(
    policy_config: Mapping[str, Any],
    *,
    experiment_dir: Path,
) -> OpenRouterVLMPolicy:
    """Build the configured policy; missing API key automatically means mock."""

    config = OpenRouterConfig.from_mapping(
        policy_config,
        experiment_dir=experiment_dir,
    )
    return OpenRouterVLMPolicy(config)
