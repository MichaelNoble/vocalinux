"""Regression tests for mode processing, without a live audio device or IBus."""

from unittest.mock import patch

import pytest

from vocalinux.speech_recognition.command_processor import CommandProcessor
from vocalinux.speech_recognition.mode_controller import ModeController
from vocalinux.speech_recognition.recognition_manager import SpeechRecognitionManager
from vocalinux.speech_recognition.text_post_processor import TextPostProcessor


@pytest.fixture
def context():
    with patch(
        "vocalinux.speech_recognition.text_post_processor.get_context",
        return_value={
            "cursor_pos": 0,
            "surrounding_text": "",
            "context_changed": True,
            "last_surrounding_text": "",
        },
    ) as query:
        yield query


def manager_for(text, commands=True):
    manager = SpeechRecognitionManager.__new__(SpeechRecognitionManager)
    manager.engine = "whisper"
    manager._transcribe_with_whisper = lambda audio: text
    manager.mode_controller = ModeController()
    manager.command_processor = CommandProcessor()
    manager.text_post_processor = TextPostProcessor()
    manager._voice_commands_enabled = commands
    texts, actions = [], []
    manager.text_callbacks = [texts.append]
    manager.action_callbacks = [actions.append]
    return manager, texts, actions


def test_empty_transcript_emits_nothing(context):
    manager, texts, actions = manager_for("")
    manager._process_audio_buffer([b"audio"])
    assert (texts, actions) == ([], [])
    context.assert_not_called()


def test_action_only_dispatches_without_text(context):
    manager, texts, actions = manager_for("delete that")
    manager._process_audio_buffer([b"audio"])
    assert texts == []
    assert actions == ["delete_last"]


@pytest.mark.parametrize("mode", ["direct", "raw"])
def test_direct_preserves_identifier_and_whitespace(context, mode):
    assert TextPostProcessor().process("  my_var()  ", mode=mode) == "  my_var()  "
    context.assert_not_called()


def test_strict_preserves_existing_clean_behavior(context):
    assert TextPostProcessor().process("hello", mode="strict") == TextPostProcessor().process(
        "hello", mode="clean"
    )


def test_mode_switch_consumed(context):
    manager, texts, actions = manager_for("dictation mode")
    manager._process_audio_buffer([b"audio"])
    assert manager.mode_controller.mode == "dictation"
    assert (texts, actions) == ([], [])


def test_commands_disabled_preserves_command_words(context):
    manager, texts, actions = manager_for("delete that", commands=False)
    manager.mode_controller.mode = "direct"
    manager._process_audio_buffer([b"audio"])
    assert texts == ["delete that"]
    assert actions == []


@pytest.mark.parametrize("mode", ["clean", "strict", "dictation", "coding", "terminal", "direct"])
@pytest.mark.parametrize(
    "spoken,expected",
    [
        ("new line", "\n"),
        ("New line.", "\n"),
        ("New line?", "\n"),
        ("new paragraph", "\n\n"),
        ("tab", "\t"),
        ("space", " "),
        ("tab tab hello", "\t\thello"),
        ("hello new line, world", "hello\nworld"),
        ("hello space space world", "hello  world"),
        ("hello new line period", "hello\n."),
    ],
)
def test_whitespace_commands_survive_processing(context, mode, spoken, expected):
    manager, texts, actions = manager_for(spoken)
    manager.mode_controller.mode = mode
    manager._process_audio_buffer([b"audio"])
    assert texts == [expected]
    assert actions == []
