"""
Command action implementations for Vocalinux.

Pure functions — no class state, no imports from the processor.
Each function takes text (and sometimes the definitions it needs)
and returns transformed text or a list of action strings.

Adding a new transform: add a callable to command_definitions.py
and add entries to MULTIWORD_FORMAT_COMMANDS or FORMAT_MODIFIERS.
Nothing in this file needs to change for new case styles.
"""

import logging
import re
from typing import Callable

from .command_definitions import (
    ACTION_COMMANDS,
    FORMAT_MODIFIERS,
    MULTIWORD_FORMAT_COMMANDS,
    PUNCTUATION_COMMANDS,
    TEXT_COMMANDS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Case / format transforms
# ---------------------------------------------------------------------------


def apply_case(words: list[str], fn: Callable[[list[str]], str]) -> str:
    """
    Clean words and apply a case transform function.

    Args:
        words: Raw words from transcription, may contain punctuation artifacts.
        fn:    Callable from MULTIWORD_FORMAT_COMMANDS, e.g. _camel, _snake.

    Returns:
        Single formatted string, or empty string if no valid words remain.
    """
    # Strip non-word characters after stripping — "." becomes "" and is dropped.
    # Filter AFTER substitution, not before, so punctuation-only tokens are removed.
    words = [cleaned for w in words if (cleaned := re.sub(r"\W+", "", w))]
    if not words:
        return ""
    return fn(words)


# ---------------------------------------------------------------------------
# Processing steps (called in order by CommandProcessor.process_text)
# ---------------------------------------------------------------------------


def process_action_commands(
    text: str,
    action_commands: dict[str, str] | None = None,
) -> tuple[str, list[str]]:
    """
    Extract action commands from text.

    Removes matched command phrases from the text and returns the
    corresponding action identifiers so the caller can fire them
    via action callbacks.

    Args:
        text: Raw transcription text.
        action_commands: Override mapping (defaults to ACTION_COMMANDS).

    Returns:
        (remaining_text, [action_id, ...])
    """
    cmds = action_commands if action_commands is not None else ACTION_COMMANDS
    actions: list[str] = []

    for cmd in sorted(cmds.keys(), key=len, reverse=True):
        pattern = r"(?i)\b" + re.escape(cmd) + r"\b"
        if re.search(pattern, text):
            actions.append(cmds[cmd])
            text = re.sub(pattern, "", text)

    if actions and re.fullmatch(r"[\s.,!?;:]*", text):
        text = ""
    return text, actions


def process_multiword_format_commands(
    text: str,
    multiword_commands: dict[str, Callable] | None = None,
) -> str:
    """
    Handle multi-word case formatting: camelCase, snake_case, PascalCase, etc.

    Scans left-to-right for format triggers, consuming all following words
    up to the next trigger (or end of string).

    e.g. "camel case this is my name"  -> "thisIsMyName"
         "snake case get user name"    -> "get_user_name"
         "camel case foo snake case bar" -> "foo bar"  (two triggers)

    Args:
        text: Text after action commands have been stripped.
        multiword_commands: Override mapping (defaults to MULTIWORD_FORMAT_COMMANDS).

    Returns:
        Text with format triggers and their target words replaced.
    """
    cmds = multiword_commands if multiword_commands is not None else MULTIWORD_FORMAT_COMMANDS

    sorted_triggers = sorted(cmds.keys(), key=len, reverse=True)
    trigger_regex = re.compile(
        r"(?i)\b(" + "|".join(re.escape(k) for k in sorted_triggers) + r")\b"
    )

    result_parts: list[str] = []
    remaining = text

    while True:
        match = trigger_regex.search(remaining)
        if not match:
            result_parts.append(remaining)
            break

        before = remaining[: match.start()].strip()
        if before:
            result_parts.append(before)

        trigger = match.group(1).lower()
        fn = cmds[trigger]

        after = remaining[match.end() :].strip()
        next_trigger = trigger_regex.search(after)

        if next_trigger:
            words_to_format = after[: next_trigger.start()].strip().split()
            remaining = after[next_trigger.start() :]
        else:
            words_to_format = after.split()
            remaining = ""

        logger.debug("format trigger=%r  words=%r", trigger, words_to_format)
        result_parts.append(apply_case(words_to_format, fn))

        if not remaining:
            break

    return " ".join(p for p in result_parts if p)


def process_format_modifiers(
    text: str,
    format_modifiers: dict[str, Callable] | None = None,
) -> str:
    """
    Handle single-word format modifiers: capitalize, uppercase, lowercase.

    Each modifier applies to the immediately following word only.

    e.g. "capitalize name"  -> "Name"
         "uppercase warning" -> "WARNING"
         "lowercase CONST"   -> "const"

    Args:
        text: Text after multi-word format commands have been applied.
        format_modifiers: Override mapping (defaults to FORMAT_MODIFIERS).

    Returns:
        Text with modifiers and their target words replaced.
    """
    mods = format_modifiers if format_modifiers is not None else FORMAT_MODIFIERS

    for cmd in sorted(mods.keys(), key=len, reverse=True):
        fn = mods[cmd]
        pattern = r"(?i)\b" + re.escape(cmd) + r"\s+(\w+)"

        text = re.sub(pattern, lambda m, f=fn: f(m.group(1)), text, flags=re.IGNORECASE)
        # Remove a dangling modifier at end of string with no following word
        text = re.sub(r"(?i)\b" + re.escape(cmd) + r"\s*$", "", text)

    return text


def process_text_commands(
    text: str,
    text_commands: dict[str, str] | None = None,
    punctuation_commands: frozenset[str] | None = None,
) -> str:
    """
    Replace spoken punctuation and symbol commands with their characters.

    Punctuation commands (period, comma, etc.) eat the space before them
    so "end of sentence period" -> "end of sentence."

    Args:
        text: Text after format modifiers have been applied.
        text_commands: Override mapping (defaults to TEXT_COMMANDS).
        punctuation_commands: Override set (defaults to PUNCTUATION_COMMANDS).

    Returns:
        Text with command words replaced by their character equivalents.
    """
    cmds = text_commands if text_commands is not None else TEXT_COMMANDS
    punct = punctuation_commands if punctuation_commands is not None else PUNCTUATION_COMMANDS

    for cmd in sorted(cmds.keys(), key=len, reverse=True):
        replacement = cmds[cmd]
        repl_fn = lambda m, r=replacement: r  # noqa: E731

        if cmd in punct:
            # Eat preceding space when there is actual text before the command
            pattern = r"(?i)(?<=\S) *\b" + re.escape(cmd) + r"\b"
            if re.search(pattern, text):
                text = re.sub(pattern, repl_fn, text)
            else:
                # Standalone command at start of string — just replace the word
                text = re.sub(r"(?i)\b" + re.escape(cmd) + r"\b", repl_fn, text)
        else:
            text = re.sub(r"(?i)\b" + re.escape(cmd) + r"\b", repl_fn, text)

    return text


def clean_whitespace(text: str) -> str:
    """
    Normalize whitespace while preserving intentional newlines.

    - Collapses multiple spaces/tabs to a single space.
    - Removes stray space before sentence punctuation.
    - Strips leading/trailing spaces and tabs (not newlines).

    Args:
        text: Text after all command substitutions.

    Returns:
        Cleaned text.
    """
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ([.,!?;:)\]}])", r"\1", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    # strip() would eat intentional leading/trailing newlines (e.g. "new line"
    # as a standalone command). Only strip spaces and tabs, not newlines.
    return text.strip(" \t")
