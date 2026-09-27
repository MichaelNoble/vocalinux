"""Decode-profile tests: preserve explicit resets and prevent mode leakage."""

import copy
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vocalinux.speech_recognition import whispercpp_config as config
from vocalinux.speech_recognition.recognition_manager import SpeechRecognitionManager

GLOBAL = {
    "no_timestamps": False,
    "no_context": True,
    "initial_prompt": "Saved vocabulary",
    "temperature": 0.7,
    "temperature_inc": -1.0,
    "entropy_thold": 2.4,
    "logprob_thold": -1.0,
    "no_speech_thold": 0.6,
}
SUPPORTED = set(GLOBAL) | {"single_segment", "suppress_blank"}


@pytest.mark.parametrize(
    "mode,temperature,prompt,blank,context",
    [
        ("dictation", 0.1, None, False, False),
        ("clean", 0.0, None, True, True),
        ("direct", 0.0, "", False, False),
    ],
)
def test_resolve_mode_overrides_without_mutating_defaults(
    mode, temperature, prompt, blank, context
):
    original = copy.deepcopy(GLOBAL)
    profiles = copy.deepcopy(config.WHISPER_MODE_PARAMS)
    result = config.resolve_decode_params(GLOBAL, mode)
    assert result["temperature"] == temperature
    assert result["suppress_blank"] is blank
    assert result["no_context"] is context
    assert result["entropy_thold"] == 2.4
    assert result["no_timestamps"] is False
    if prompt is not None:
        assert result["initial_prompt"] == prompt
    assert GLOBAL == original
    assert config.WHISPER_MODE_PARAMS == profiles


def test_unknown_mode_rejected():
    with pytest.raises(ValueError, match="mode"):
        config.resolve_decode_params(GLOBAL, "typo")


class RecordingDecoder:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, **params):
        # This binding requires a string, even when clearing an earlier prompt.
        if not isinstance(params["initial_prompt"], str):
            raise TypeError("initial_prompt must be str")
        self.calls.append(params)
        return [SimpleNamespace(text="hello")]


@pytest.fixture
def manager():
    with patch.object(SpeechRecognitionManager, "_init_whispercpp"):
        mgr = SpeechRecognitionManager(
            engine="whisper_cpp",
            model_size="distil-large",
            language="en-us",
            defer_download=True,
            **{f"whispercpp_{key}": value for key, value in GLOBAL.items()},
        )
    mgr.model = RecordingDecoder()
    return mgr


def test_mode_roundtrip_uses_one_model_with_complete_resets(manager):
    model = manager.model
    audio = [b"\0\0" * 1600]
    with patch.object(manager, "_get_supported_whispercpp_params", return_value=SUPPORTED):
        for mode in ["dictation", "coding", "direct", "dictation"]:
            assert manager._transcribe_with_whispercpp(audio, mode=mode) == "hello"
    assert manager.model is model
    assert len(model.calls) == 4
    first, coding, direct, last = model.calls
    assert first == last
    assert first["temperature"] == 0.1
    assert first["suppress_blank"] is False
    assert coding["no_context"] is True
    assert direct["initial_prompt"] == ""
    assert direct["temperature"] == 0.0
    assert direct["no_context"] is False
    assert direct["no_timestamps"] is False
    assert direct["entropy_thold"] == 2.4
    assert direct["temperature_inc"] == -1.0
    assert "strategy" not in direct


def test_legacy_call_honors_suppress_blank_and_clears_prompt(manager):
    with patch.object(manager, "_get_supported_whispercpp_params", return_value=SUPPORTED):
        assert (
            manager._transcribe_with_whispercpp([b"\0\0" * 1600], suppress_blank=False) == "hello"
        )
    assert manager.model.calls[0]["suppress_blank"] is False
    assert manager.model.calls[0]["initial_prompt"] == ""


def test_unsupported_mode_parameter_stops_before_inference(manager, caplog):
    with patch.object(
        manager, "_get_supported_whispercpp_params", return_value=SUPPORTED - {"single_segment"}
    ):
        assert manager._transcribe_with_whispercpp([b"\0\0" * 1600], mode="dictation") == ""
    assert manager.model.calls == []
    assert "single_segment" in caplog.text
    assert "support" in caplog.text


def test_unknown_native_capabilities_do_not_risk_partial_updates(manager, caplog):
    with patch.object(manager, "_get_supported_whispercpp_params", return_value=None):
        assert manager._transcribe_with_whispercpp([b"\0\0" * 1600], mode="dictation") == ""
    assert manager.model.calls == []
    assert "binding" in caplog.text


def test_invalid_numeric_setting_stops_before_inference(manager, caplog):
    manager.whispercpp_entropy_thold = float("nan")
    with patch.object(manager, "_get_supported_whispercpp_params", return_value=SUPPORTED):
        assert manager._transcribe_with_whispercpp([b"\0\0" * 1600], mode="dictation") == ""
    assert manager.model.calls == []
    assert "entropy_thold" in caplog.text


@pytest.mark.parametrize("setting", [True, False])
def test_optional_timestamps_missing_from_older_binding_still_transcribes(manager, caplog, setting):
    manager.whispercpp_no_timestamps = setting
    with patch.object(
        manager, "_get_supported_whispercpp_params", return_value=SUPPORTED - {"no_timestamps"}
    ):
        for _ in range(2):
            assert (
                manager._transcribe_with_whispercpp([b"\0\0" * 1600], mode="dictation") == "hello"
            )
    assert len(manager.model.calls) == 2
    assert "no_timestamps" not in manager.model.calls[0]
    assert manager.model.calls[0]["suppress_blank"] is False
    assert (
        sum("optional" in r.message and "no_timestamps" in r.message for r in caplog.records) == 1
    )
