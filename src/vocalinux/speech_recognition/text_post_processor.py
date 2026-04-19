"""
Text post-processor for Vocalinux.

This module cleans up raw speech recognition output to make it
look natural and readable.
"""

import logging
import re

logger = logging.getLogger(__name__)


class TextPostProcessor:

    def __init__(self):
        self.last_text = ""  # for future context-aware improvements

    _MODE_PIPELINE = {
        "raw": ["_prepend_space_if_needed"],
        "dictation": ["_normalize_whitespace", "_fix_spacing", "_soft_capitalize", "_prepend_space_if_needed"],
        "clean": ["_normalize_whitespace", "_cleanup_punctuation", "_fix_spacing", "_capitalize_sentences",
                  "_prepend_space_if_needed"],
        "terminal": ["_normalize_whitespace", "_cleanup_punctuation", "_fix_spacing", "_prepend_space_if_needed"],
        "coding": ["_normalize_whitespace", "_cleanup_punctuation", "_fix_spacing", "_prepend_space_if_needed"],
    }

    def process(self, text, mode="clean"):
        logger.info(f"Initial: {text} | mode={mode}")

        for step_name in self._MODE_PIPELINE.get(mode, self._MODE_PIPELINE["clean"]):
            text = text.strip()
            text = getattr(self, step_name)(text)

        self.last_text = text
        
        return text

    # -------------------------
    # Processing steps
    # -------------------------

    def _normalize_whitespace(self, text: str) -> str:
        # Collapse multiple spaces
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _cleanup_punctuation(self, text: str) -> str:
        # Remove space before punctuation
        text = re.sub(r"\s+([.,!?])", r"\1", text)

        # Collapse duplicate punctuation
        text = re.sub(r"([.,!?])\s*\1+", r"\1", text)

        # Fix ", ." → "."
        text = re.sub(r",\s*\.", ".", text)

        # Fix ". ," → "."
        text = re.sub(r"\.\s*,", ".", text)

        return text

    def _fix_spacing(self, text: str) -> str:
        # Ensure space after punctuation
        text = re.sub(r"([.!?])([A-Za-z])", r"\1 \2", text)

        return text

    def _capitalize_sentences(self, text: str) -> str:
        def capitalize(match):
            return match.group(1) + match.group(2).upper()

        # Capitalize first letter and after punctuation
        text = re.sub(r"(^|(?<=[.!?])\s+)([a-z])", capitalize, text)

        # Fix lowercase "i"
        text = re.sub(r"\bi\b", "I", text)

        return text

    def _soft_capitalize(self, text: str) -> str:
        if not text:
            return text

        logger.debug(f"_soft_capitalize: last_text='{self.last_text}' | incoming='{text}'")

        terminal_punctuation = {".", "!", "?"}
        stripped = self.last_text.rstrip()
        last_char = stripped[-1] if stripped else ""
        last_chars = stripped[-3:] if stripped else ""

        ellipsis_ending = (last_char == "…" or last_chars == "...")

        if ellipsis_ending:
            return text[0].lower() + text[1:]

        should_capitalize = (
                not self.last_text
                or last_char in terminal_punctuation
        )

        if should_capitalize:
            return text[0].upper() + text[1:]
        else:
            return text[0].lower() + text[1:]


    def _prepend_space_if_needed(self, text: str) -> str:
        if not text:
            return ""

        if not self.last_text:
            return text

        text = text.lstrip()  # avoid double spaces

        last_char = self.last_text[-1] if self.last_text else ""
        logger.debug(f"last character: {last_char}")

        # If previous chunk ended with newline → no space
        if last_char == "\n":
            return text

        # If new chunk starts with newline → no space
        if text.startswith("\n"):
            return text

        # Otherwise, insert a space between chunks
        return " " + text