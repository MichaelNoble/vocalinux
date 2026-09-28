"""Use real configuration files while replacing desktop/audio boundaries."""

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

from vocalinux.ui.config_manager import ConfigManager
from vocalinux.utils.whispercpp_model_info import WHISPERCPP_MODEL_INFO, get_model_path


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    monkeypatch.setattr("vocalinux.ui.config_manager.CONFIG_DIR", str(tmp_path))
    path = tmp_path / "config.json"
    monkeypatch.setattr("vocalinux.ui.config_manager.CONFIG_FILE", str(path))
    return path


def write_config(path, speech):
    path.write_text(
        json.dumps(
            {
                "speech_recognition": speech,
                "general": {"first_run": False},
                "custom": {"keep": "value"},
            }
        )
    )


@pytest.mark.parametrize("cli,expected", [([], "distil-large"), (["--model", "small"], "small")])
def test_startup_uses_per_engine_selection_unless_cli_overrides(config_file, cli, expected):
    from vocalinux.main import main

    write_config(
        config_file,
        {"engine": "whisper_cpp", "model_size": "tiny", "whisper_cpp_model_size": "distil-large"},
    )
    received = []

    class StopStartup(BaseException):
        pass

    def capture_engine(**kwargs):
        received.append(kwargs["model_size"])
        raise StopStartup

    with (
        patch("vocalinux.single_instance.acquire_lock", return_value=True),
        patch("atexit.register"),
        patch("vocalinux.main.check_dependencies", return_value=True),
        patch("vocalinux.main.check_display_available", return_value=True),
        patch("vocalinux.main.check_appindicator_support", return_value=True),
        patch("vocalinux.ui.logging_manager.initialize_logging"),
        patch("vocalinux.text_injection.start_ibus_daemon", return_value=False),
        patch(
            "vocalinux.speech_recognition.recognition_manager.SpeechRecognitionManager",
            side_effect=capture_engine,
        ),
        patch.object(sys, "argv", ["vocalinux", *cli]),
    ):
        with pytest.raises(StopStartup):
            main()
    assert received == [expected]
    assert ConfigManager().get_model_size_for_engine("whisper_cpp") == "distil-large"


def test_legacy_whispercpp_selection_migrates(config_file):
    write_config(
        config_file,
        {
            "engine": "whisper_cpp",
            "model_size": "distil-large",
            "vosk_model_size": "small",
            "whisper_model_size": "tiny",
        },
    )
    manager = ConfigManager()
    assert manager.get_model_size_for_engine("whisper_cpp") == "distil-large"
    assert ConfigManager().get_model_size_for_engine("whisper_cpp") == "distil-large"


def test_unrelated_save_preserves_selection_and_unknown_keys(config_file):
    write_config(
        config_file,
        {"engine": "whisper_cpp", "model_size": "tiny", "whisper_cpp_model_size": "distil-large"},
    )
    manager = ConfigManager()
    manager.set("advanced", "whispercpp_temperature", 0.2)
    assert manager.save_config()
    loaded = ConfigManager()
    assert loaded.get_model_size_for_engine("whisper_cpp") == "distil-large"
    assert loaded.get_settings()["custom"] == {"keep": "value"}


def test_distil_model_registry_and_missing_file_preserve_preference(config_file):
    write_config(
        config_file,
        {
            "engine": "whisper_cpp",
            "model_size": "distil-large",
            "whisper_cpp_model_size": "distil-large",
        },
    )
    manager = ConfigManager()
    assert get_model_path("distil-large").endswith("/distil-large-v3.5.bin")
    assert "/distil-large-v3.5-ggml/" in WHISPERCPP_MODEL_INFO["distil-large"]["url"]
    from vocalinux.speech_recognition.recognition_manager import SpeechRecognitionManager

    with (
        patch(
            "vocalinux.speech_recognition.recognition_manager.is_model_downloaded",
            return_value=False,
        ),
        patch.object(SpeechRecognitionManager, "_init_whispercpp"),
    ):
        recognition = SpeechRecognitionManager(
            engine="whisper_cpp",
            model_size=manager.get_model_size_for_engine("whisper_cpp"),
            defer_download=True,
        )
    assert recognition.model_size == "distil-large"
    assert ConfigManager().get_model_size_for_engine("whisper_cpp") == "distil-large"
