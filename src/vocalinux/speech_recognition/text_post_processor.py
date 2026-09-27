"""
Text post-processor for Vocalinux.

This module cleans up raw speech recognition output to make it
look natural and readable.
"""

import logging
import re

from .context_manager import get_context

logger = logging.getLogger(__name__)


class TextPostProcessor:
    def __init__(self):
        self.last_text = ""
        self.chars_before = ""
        self.chars_after = ""
        self.context_changed = True
        self.preserve_command_spacing = False
        self.surrounding = ""
        self.cursor_pos = 0

    _MODE_PIPELINE = {
        "raw": [],
        "direct": [],
        "strict": [
            "_normalize_whitespace",
            "_cleanup_punctuation",
            "_capitalize_sentences",
            "_handle_boundaries",
        ],
        "dictation": ["_normalize_whitespace", "_soft_capitalize", "_handle_boundaries"],
        "clean": [
            "_normalize_whitespace",
            "_cleanup_punctuation",
            "_capitalize_sentences",
            "_handle_boundaries",
        ],
        "terminal": ["_normalize_whitespace", "_cleanup_punctuation", "_handle_boundaries"],
        "coding": ["_normalize_whitespace", "_cleanup_punctuation", "_handle_boundaries"],
    }

    def process(self, text, mode="dictation", was_transformed=False):
        if mode in {"direct", "raw"}:
            return text

        logger.info(f"Initial: {text} | mode={mode}")

        context = get_context()
        self.cursor_pos = context["cursor_pos"]
        self.surrounding = context["surrounding_text"]
        self.context_changed = context["context_changed"]
        self.was_transformed = was_transformed
        self.preserve_command_spacing = was_transformed and mode in {"coding", "terminal"}

        if self.surrounding:
            self.chars_before = self.surrounding[max(0, self.cursor_pos - 3) : self.cursor_pos]
            self.chars_after = self.surrounding[self.cursor_pos : self.cursor_pos + 3]
        else:
            self.chars_before = ""
            self.chars_after = ""

        logger.debug(
            f"context: before='{self.chars_before}' | after='{self.chars_after}' "
            f"| cursor={self.cursor_pos} | changed={self.context_changed}"
        )

        for step_name in self._MODE_PIPELINE.get(mode, self._MODE_PIPELINE["clean"]):
            text = text.lstrip(" \t")  # not \n
            text = getattr(self, step_name)(text)

        self.last_text = text
        logger.debug(f"Final output: '{text}'")
        return text

    # -------------------------
    # Processing steps
    # -------------------------

    def _normalize_whitespace(self, text: str) -> str:
        # Collapse multiple spaces
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip(" \t")  # not newlines

    def _cleanup_punctuation(self, text: str) -> str:
        # Remove space before punctuation
        text = re.sub(r"\s+([.,!?])", r"\1", text)

        # Collapse duplicate punctuation
        # text = re.sub(r"([.,!?])\s*\1+", r"\1", text)

        # Fix ", ." → "."
        text = re.sub(r",\s*\.", ".", text)

        # Fix ". ," → "."
        text = re.sub(r"\.\s*,", ".", text)

        # If chunk starts with punctuation, avoid leading space later
        if text and text[0] in ".,!?":
            text = text.lstrip(" ")

        return text

    def _capitalize_sentences(self, text: str) -> str:
        def capitalize(match):
            return match.group(1) + match.group(2).upper()

        if self.was_transformed:
            return text

        # If inserting mid-sentence, lowercase the first character
        if self.chars_before and self.chars_after:
            source = self.chars_before.rstrip()
            last_char = source[-1] if source else ""
            if last_char not in ".!?\n" and last_char != "":
                text = text[0].lower() + text[1:]
                # Fix lowercase "i" but skip sentence capitalization
                text = re.sub(r"\bi\b", "I", text)
                return text

        # Appending to end or fresh context — full capitalization applies
        text = re.sub(r"(^|(?<=[.!?])\s+)([a-z])", capitalize, text)
        text = re.sub(r"\bi\b", "I", text)
        return text

    def _soft_capitalize(self, text: str) -> str:
        if not text:
            return text

        if self.was_transformed:
            return text

        logger.debug(
            f"_soft_capitalize: last_text='{self.last_text}' | "
            f"chars_before='{self.chars_before}' | incoming='{text}'"
        )

        terminal_punctuation = {".", "!", "?"}

        # Prefer real editor context over last_text
        # source = self.chars_before or self.last_text.rstrip()
        source = self.chars_before.rstrip() if self.chars_before else ""

        last_char = source[-1] if source else ""
        last_chars = source[-3:] if len(source) >= 3 else source

        ellipsis_ending = last_char == "…" or last_chars == "..."

        if ellipsis_ending:
            return text[0].lower() + text[1:]

        # Only capitalize on a period when we're at the end of the document.
        if last_char in terminal_punctuation and self.chars_after:
            return text[0].lower() + text[1:]

        should_capitalize = not source or last_char in terminal_punctuation or last_char == "\n"

        if should_capitalize:
            return text[0].upper() + text[1:]
        else:
            return text[0].lower() + text[1:]

    def _handle_internal_and_right_spacing(self, text: str) -> str:
        if not text or not self.chars_after:
            return text

        last_char = text[-1]
        next_char = self.chars_after[0]

        # If inserting mid-word or mid-sentence, we need a trailing space
        # unless the existing text already has one
        if not self.chars_after.startswith((" ", "\n")):
            # Don't add space if text already ends with space or punctuation
            # that would naturally be followed by a space
            if last_char not in (" ", "\n"):
                # Check if this is mid-sentence (next char is lowercase)
                # or before a new sentence (next char is uppercase after period)
                if last_char in ".!?" and next_char.islower():
                    # Mid-sentence — remove the punctuation and add space
                    logger.debug("Right boundary: mid-sentence insertion, removing punctuation")
                    text = text[:-1] + " "
                else:
                    logger.debug("Right boundary: adding trailing space")
                    text += " "

        return text

    def _handle_left_boundary(self, text: str) -> str:
        if not text:
            return ""

        text = text.lstrip()  # avoid double spaces

        # Context has changed — we're in a new field or app.
        # Don't prepend a space; we have no reliable information
        # about what precedes the cursor.
        # Only block on context_changed if we have no cursor information.
        # If surrounding text arrived, use it regardless of context_changed.

        # If previous char is punctuation and no space before next word → force space
        if self.chars_before and self.chars_before[-1] in ".!?":
            if not self.chars_after or not self.chars_after.startswith((" ", "\n")):
                logger.debug("_handle_left_boundary: fixing missing space after punctuation")
                return " " + text

        if self.context_changed and not self.chars_before:
            logger.debug("_handle_left_boundary: context changed — no space")
            return text

        # Nothing before the cursor — start of field or empty field
        if not self.chars_before:
            logger.debug("_handle_left_boundary: nothing before cursor — no space")
            return text

        # Cursor is right after a newline — no space
        if self.chars_before[-1] == "\n":
            logger.debug("_handle_left_boundary: newline before cursor — no space")
            return text

        # New chunk starts with a newline — no space
        if text.startswith("\n"):
            logger.debug("_handle_left_boundary: chunk starts with newline — no space")
            return text

        # Cursor is already preceded by a space — no double space
        if self.chars_before[-1] == " ":
            logger.debug("_handle_left_boundary: space already before cursor — no space")
            return text

        # Chars_before stripped of trailing spaces to correctly
        # identify the last meaningful character for the space decision.
        # If nothing meaningful remains, no space needed.
        chars_before_stripped = self.chars_before.rstrip()
        if not chars_before_stripped:
            logger.debug("_handle_left_boundary: nothing meaningful before cursor — no space")
            return text

        # Otherwise insert a space between existing text and new chunk
        logger.debug(
            f"_handle_left_boundary: prepending space (chars_before='{self.chars_before}')"
        )
        return " " + text

    def _handle_boundaries(self, text: str) -> str:
        if not text:
            return ""
        if self.preserve_command_spacing:
            return text  # identifier/shell transformations retain exact spacing

        text = text.lstrip(" \t")

        # --------------------------------
        # Establish effective context
        # --------------------------------
        before = self.chars_before
        after = self.chars_after

        # Fallback when context is missing but we're continuing
        if not before and not self.context_changed and self.last_text:
            before = self.last_text[-3:]

        prev_char = before[-1] if before else ""
        next_char = after[0] if after else ""

        # --------------------------------
        # LEFT SIDE DECISION
        # --------------------------------
        prepend_space = False

        # If already clean boundary (space on both sides), do nothing
        if before.endswith(" ") and after.startswith(" "):
            prepend_space = False

        elif text.startswith(tuple(".,!?;:)]}")):
            prepend_space = False

        elif not before:
            prepend_space = False

        elif prev_char in ("\n", " "):
            prepend_space = False

        elif text.startswith((" ", "\n")):
            prepend_space = False

        elif prev_char in ".!?":
            # FIX: a space in `after` is to the RIGHT of the cursor — it doesn't
            # sit between prev_char and the new text. Always prepend here,
            # unless there's genuinely nothing before the cursor.
            prepend_space = bool(before)

        else:
            prepend_space = True

        # --------------------------------
        # RIGHT SIDE DECISION
        # --------------------------------
        append_space = False

        if after:
            if not after.startswith((" ", "\n")) and not text.endswith(  # no space ahead
                (" ", "\n")
            ):  # no space already
                last_char = text[-1]

                is_word_boundary = prev_char == " " or (after and after[0] == " ")

                is_mid_sentence = (
                    last_char in ".!?"
                    and next_char.islower()
                    and not is_word_boundary
                    and not self.was_transformed
                )

                if is_mid_sentence:
                    logger.debug("Boundaries: mid-sentence insertion, removing punctuation")
                    # Protect ellipsis
                    if not text.endswith("..."):
                        text = text[:-1]
                    append_space = True
                else:
                    append_space = True

        # --------------------------------
        # APPLY (in correct order)
        # --------------------------------
        if prepend_space:
            text = " " + text

        if append_space:
            text = text + " "

        # --------------------------------
        # FINAL NORMALIZATION (critical)
        # --------------------------------
        # Collapse multiple spaces
        text = re.sub(r" {2,}", " ", text)

        # Respect existing surrounding spaces (prevents accumulation)
        if after.startswith(" "):
            text = text.rstrip(" ")

        if before.endswith(" "):
            text = text.lstrip(" ")

        logger.debug(
            f"_handle_boundaries: prepend={prepend_space} append={append_space} "
            f"before='{before}' after='{after}' result='{text}'"
        )

        return text
