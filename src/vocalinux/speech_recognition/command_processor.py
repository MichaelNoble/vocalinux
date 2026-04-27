"""
Command processor for Vocalinux.

Orchestrates the processing pipeline: loads definitions, compiles regex
patterns once at init, and delegates each processing step to the pure
functions in command_actions.py.

The public interface is a single method:

    processed_text, actions = processor.process_text(text)

This contract is unchanged from the previous monolithic version, so
_process_audio_buffer in the speech recognition module needs no edits.
"""

import logging
import re

from .command_actions import (
    clean_whitespace,
    process_action_commands,
    process_format_modifiers,
    process_multiword_format_commands,
    process_text_commands,
)
from .command_definitions import (
    ACTION_COMMANDS,
    FORMAT_MODIFIERS,
    MULTIWORD_FORMAT_COMMANDS,
    PUNCTUATION_COMMANDS,
    TEXT_COMMANDS,
)

logger = logging.getLogger(__name__)


class CommandProcessor:
    """
    Processes text commands in speech recognition results.

    Handles punctuation, formatting (camelCase, snake_case, PascalCase),
    case modifiers, and action commands like delete/undo/copy.

    Definitions live in command_definitions.py.
    Step implementations live in command_actions.py.
    This class owns pattern compilation and pipeline sequencing only.
    """

    def __init__(self):
        """Initialize the command processor and compile regex patterns."""
        self._compile_patterns()

    # ------------------------------------------------------------------
    # Pattern compilation
    # ------------------------------------------------------------------

    def _compile_patterns(self) -> None:
        """
        Pre-compile regex patterns from all command dictionaries.

        Called once at init. Patterns are used by the action functions
        on each process_text() call — we build them here to avoid
        recompiling on every utterance.
        """
        def _make_pattern(keys: list[str]) -> re.Pattern:
            """Word-boundary pattern, longest key first to avoid prefix shadowing."""
            sorted_keys = sorted(keys, key=len, reverse=True)
            return re.compile(
                r"\b(" + "|".join(re.escape(k) for k in sorted_keys) + r")\b",
                re.IGNORECASE,
            )

        # These compiled patterns are available for any caller that wants
        # to pre-screen text cheaply before calling process_text().
        self.text_cmd_pattern      = _make_pattern(list(TEXT_COMMANDS.keys()))
        self.action_cmd_pattern    = _make_pattern(list(ACTION_COMMANDS.keys()))
        self.format_mod_pattern    = _make_pattern(list(FORMAT_MODIFIERS.keys()))
        self.multiword_fmt_pattern = _make_pattern(list(MULTIWORD_FORMAT_COMMANDS.keys()))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def process_text(self, text: str) -> tuple[str, list[str]]:
        """
        Process text commands in the recognized speech.

        Pipeline (each step feeds into the next):
            1. Extract action commands  -> removes them from text, collects action ids
            2. Multi-word format cmds   -> camelCase, snake_case, etc.
            3. Single-word modifiers    -> capitalize, uppercase, lowercase
            4. Text/punctuation cmds    -> period -> ".", new line -> "\\n", etc.
            5. Whitespace cleanup

        Args:
            text: Raw transcription text from the speech engine.

        Returns:
            Tuple of:
              - processed_text: Text with commands applied/replaced.
              - actions: List of action identifier strings for the caller
                         to dispatch via action callbacks. May be empty.

        Example:
            >>> cp = CommandProcessor()
            >>> cp.process_text("camel case this is great")
            ("thisIsGreat", [])
            >>> cp.process_text("delete that hello period")
            ("hello.", ["delete_last"])
        """
        if not text:
            return "", []

        logger.debug("Processing commands in: %r", text)

        # Step 1 — action commands (consume from text, accumulate action ids)
        text, actions = process_action_commands(text)

        # Step 2 — multi-word format commands (camelCase, snake_case, …)
        text = process_multiword_format_commands(text)

        # Step 3 — single-word format modifiers (capitalize, uppercase, …)
        text = process_format_modifiers(text)

        # Step 4 — punctuation and symbol replacements
        text = process_text_commands(text)

        # Step 5 — whitespace normalisation
        text = clean_whitespace(text)

        logger.debug("Result: text=%r  actions=%r", text, actions)
        return text, actions

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def has_any_command(self, text: str) -> bool:
        """
        Quick pre-screen: does this text contain any known command phrase?

        Cheaper than a full process_text() call. Useful if the caller wants
        to skip processing for utterances that are pure dictation.

        Args:
            text: Transcription text to check.

        Returns:
            True if any command pattern matches.
        """
        for pattern in (
            self.text_cmd_pattern,
            self.action_cmd_pattern,
            self.format_mod_pattern,
            self.multiword_fmt_pattern,
        ):
            if pattern.search(text):
                return True
        return False