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

    def process(self, text: str, mode: str = "clean") -> str:
        logger.info(f"Initial: {text} | mode={mode}")

        if not text:
            return ""

        original = text

        # RAW mode = minimal interference
        if mode == "raw":
            text = text.strip()
            text = self._merge_with_previous(text)
            self.last_text = text
            return text

        # --- shared baseline ---
        text = text.strip()
        text = self._normalize_whitespace(text)

        # CLEAN / STRICT / CODING share some behavior
        if mode in ("clean", "strict", "coding", "terminal"):
            text = self._cleanup_punctuation(text)
            text = self._fix_spacing(text)

        # STRICT = more aggressive corrections
        # if mode == "strict":
        #     text = self._aggressive_cleanup(text)  # (you can add later)
        #
        # # CODING = avoid messing with symbols too much
        # if mode == "coding":
        #     text = self._light_spacing(text)  # optional future tweak

        # TERMINAL = minimal formatting, no capitalization
        if mode != "terminal":
            text = self._capitalize_sentences(text)

        # Merge chunks LAST
        text = self._merge_with_previous(text)

        logger.debug(f"RAW:   {original}")
        logger.debug(f"CLEAN: {text}")

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
        text = re.sub(r"(^|[.!?]\s+)([a-z])", capitalize, text)

        # Fix lowercase "i"
        text = re.sub(r"\bi\b", "I", text)

        return text

    def _merge_with_previous(self, text: str) -> str:
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