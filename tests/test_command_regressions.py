"""User-visible command regressions discovered during stable-branch review."""

from unittest.mock import patch

import pytest

from vocalinux.speech_recognition.command_definitions import ACTION_COMMANDS
from vocalinux.speech_recognition.command_processor import CommandProcessor
from vocalinux.speech_recognition.text_post_processor import TextPostProcessor
from vocalinux.ui.action_handler import ActionHandler


@pytest.mark.parametrize(
    "spoken,expected",
    [
        ("period", "."),
        ("comma", ","),
        ("semicolon", ";"),
        ("not equals", "!="),
        ("double colon", "::"),
    ],
)
def test_generated_punctuation_survives(spoken, expected):
    assert CommandProcessor().process_text(spoken) == (expected, [])


@pytest.mark.parametrize(
    "before,after,spoken,expected",
    [
        ("hello", "", "world period", " world."),
        ("hello", "", "period", "."),
        ("hello", "again", "world period", " world. "),
        ("hello ", " again", "new line", "\n"),
    ],
)
def test_transformed_prose_keeps_boundaries(before, after, spoken, expected):
    context = {
        "surrounding_text": before + after,
        "cursor_pos": len(before),
        "context_changed": False,
        "last_surrounding_text": "",
    }
    transformed, actions = CommandProcessor().process_text(spoken)
    with patch(
        "vocalinux.speech_recognition.text_post_processor.get_context", return_value=context
    ):
        assert (
            TextPostProcessor().process(transformed, mode="dictation", was_transformed=True)
            == expected
        )
    assert actions == []


@pytest.mark.parametrize("mode", ["coding", "terminal"])
def test_identifier_modes_keep_exact_command_spacing(mode):
    with patch(
        "vocalinux.speech_recognition.text_post_processor.get_context",
        return_value={
            "surrounding_text": "prefix_",
            "cursor_pos": 7,
            "context_changed": False,
        },
    ):
        assert TextPostProcessor().process("my_var", mode=mode, was_transformed=True) == "my_var"


@pytest.mark.parametrize(
    "phrase", ["delete last word", "delete last sentence", "save", "save file", "find", "open find"]
)
def test_unimplemented_actions_remain_text(phrase):
    assert CommandProcessor().process_text(phrase) == (phrase, [])


def test_all_recognized_actions_have_a_handler():
    handler = ActionHandler(None)
    assert set(ACTION_COMMANDS.values()) <= handler.action_handlers.keys()


def test_action_only_recognition_punctuation_does_not_insert_text():
    assert CommandProcessor().process_text("Delete that.") == ("", ["delete_last"])


@pytest.mark.parametrize(
    "spoken,expected", [("space space", "  "), ("tab tab", "\t\t"), ("new paragraph.", "\n\n")]
)
def test_explicit_whitespace_survives_existing_editor_spaces(spoken, expected):
    with patch(
        "vocalinux.speech_recognition.text_post_processor.get_context",
        return_value={
            "surrounding_text": "before  after",
            "cursor_pos": 7,
            "context_changed": False,
        },
    ):
        text, _ = CommandProcessor().process_text(spoken)
        assert TextPostProcessor().process(text, mode="dictation", was_transformed=True) == expected


def test_literal_marker_like_text_survives_command_processing():
    assert CommandProcessor().process_text("\ue0000\ue000 new line hello") == (
        "\ue0000\ue000\nhello",
        [],
    )


@pytest.mark.parametrize(
    "spoken,expected",
    [
        ("new line question mark", "\n?"),
        ("new line full stop", "\n."),
        ("hello space exclamation point world", "hello ! world"),
    ],
)
def test_explicit_punctuation_after_whitespace_command_is_not_discarded(spoken, expected):
    assert CommandProcessor().process_text(spoken) == (expected, [])


@pytest.mark.parametrize("spoken", ["hello comma.", "hello period,"])
@pytest.mark.parametrize("mode", ["clean", "strict", "coding", "terminal"])
def test_transformed_punctuation_pairs_still_cleaned(spoken, mode):
    with patch(
        "vocalinux.speech_recognition.text_post_processor.get_context",
        return_value={
            "surrounding_text": "",
            "cursor_pos": 0,
            "context_changed": True,
        },
    ):
        text, _ = CommandProcessor().process_text(spoken)
        assert TextPostProcessor().process(text, mode=mode, was_transformed=True) == "hello."
