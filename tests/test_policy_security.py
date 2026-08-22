import json
import urllib.error

import numpy as np
import pytest
from jsonschema.exceptions import ValidationError

from policies.base import PolicyInput, validate_decision
from policies.openrouter_vlm import OpenRouterConfig, OpenRouterVLMPolicy


def public_observation():
    return {
        "image_left": np.zeros((24, 32, 3), dtype=np.uint8),
        "image_right": np.zeros((24, 32, 3), dtype=np.uint8),
        "proprio": np.asarray((0.1, -0.2, 0.3), dtype=np.float32),
        "previous_action": np.asarray((0.0, 0.0, 0.0), dtype=np.float32),
    }


def valid_decision():
    return {
        "threat_visible": False,
        "threat_bearing": "unknown",
        "risk_next_horizon": 0.2,
        "uncertainty": 0.6,
        "recommended_look": "left",
        "recommended_motion": "forward",
    }


def test_strict_json_schema_validation():
    validate_decision(valid_decision())

    missing = valid_decision()
    missing.pop("uncertainty")
    with pytest.raises(ValidationError):
        validate_decision(missing)

    extra = {**valid_decision(), "predator_x": 0.42}
    with pytest.raises(ValidationError):
        validate_decision(extra)

    out_of_range = {**valid_decision(), "risk_next_horizon": 1.1}
    with pytest.raises(ValidationError):
        validate_decision(out_of_range)


def test_privileged_fields_never_enter_model_request(tmp_path):
    sentinel = "PRIVILEGED_SENTINEL_8d94f"
    privileged_record = {
        "privileged_state": sentinel,
        "predator_location": [0.2, 0.3],
        "predator_geometric_los": True,
    }
    assert privileged_record  # Stored by evaluator, never passed to PolicyInput.
    config = OpenRouterConfig(
        model="test/exact-model",
        provider={"order": ["ExactProvider"], "allow_fallbacks": False},
        cache_dir=tmp_path / "cache",
        log_path=tmp_path / "calls.jsonl",
    )
    policy = OpenRouterVLMPolicy(config, api_key=None)
    request = policy.build_request(PolicyInput.from_observation(public_observation()))
    serialized = json.dumps(request, sort_keys=True)
    for forbidden in (
        sentinel,
        "privileged_state",
        "predator_location",
        "predator_geometric_los",
        "OPENROUTER_API_KEY",
        "Bearer ",
    ):
        assert forbidden not in serialized


def test_no_key_uses_mock_and_logs_no_secret(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    config = OpenRouterConfig(
        model="test/exact-model",
        provider={"order": ["ExactProvider"], "allow_fallbacks": False},
        cache_dir=tmp_path / "cache",
        log_path=tmp_path / "calls.jsonl",
    )
    policy = OpenRouterVLMPolicy(config)
    assert policy.backend == "mock"
    result = policy.decide(PolicyInput.from_observation(public_observation()))
    validate_decision(result.decision.to_dict())
    assert result.telemetry.backend == "mock"
    log_text = config.log_path.read_text(encoding="utf-8")
    for forbidden in (
        "OPENROUTER_API_KEY",
        "Bearer ",
        "privileged_state",
        "predator_location",
        "predator_geometric_los",
    ):
        assert forbidden not in log_text


def test_exp01_public_timing_semantics_enter_request(tmp_path):
    config = OpenRouterConfig(
        model="test/exact-model",
        provider={"order": ["ExactProvider"], "allow_fallbacks": False},
        risk_horizon_seconds=4.0,
        macro_duration_seconds=0.8,
        cache_dir=tmp_path / "cache",
        log_path=tmp_path / "calls.jsonl",
    )
    policy = OpenRouterVLMPolicy(config, api_key=None)
    request = policy.build_request(
        PolicyInput.from_observation(public_observation()),
    )
    serialized = json.dumps(request, sort_keys=True)
    assert "within 4 seconds" in serialized
    assert "up to 0.8 seconds" in serialized
    assert "threat_visible describes only the current eye images" in serialized


def test_keyed_backend_retries_caches_and_records_usage(tmp_path):
    calls = []

    def transport(request, timeout):
        calls.append((request, timeout))
        if len(calls) == 1:
            raise urllib.error.URLError("transient")
        return {
            "model": "test/exact-model",
            "choices": [
                {"message": {"content": json.dumps(valid_decision())}},
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "cost": 0.001,
            },
        }

    secret = "test-secret-must-not-be-logged"
    config = OpenRouterConfig(
        model="test/exact-model",
        provider={
            "order": ["exact-provider"],
            "allow_fallbacks": False,
            "require_parameters": True,
        },
        max_retries=1,
        retry_backoff_seconds=0.0,
        cache_dir=tmp_path / "cache",
        log_path=tmp_path / "calls.jsonl",
    )
    policy = OpenRouterVLMPolicy(
        config,
        api_key=secret,
        transport=transport,
    )
    policy_input = PolicyInput.from_observation(public_observation())
    first = policy.decide(policy_input)
    second = policy.decide(policy_input)

    assert len(calls) == 2  # one failed attempt, one successful; cache avoids a third
    assert first.telemetry.cache_hit is False
    assert second.telemetry.cache_hit is True
    assert first.telemetry.token_usage["total_tokens"] == 15
    assert first.telemetry.cost == pytest.approx(0.001)
    assert calls[-1][0]["model"] == "test/exact-model"
    assert calls[-1][0]["provider"] == config.provider
    assert calls[-1][0]["response_format"]["json_schema"]["strict"] is True
    assert secret not in config.log_path.read_text(encoding="utf-8")
