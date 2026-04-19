import re

class ModeController:
    def __init__(self):
        self.mode = "clean"

        self.valid_modes = {
            "clean",
            "strict",
            "coding",
            "terminal",
            "direct",  # preferred over "raw"
            "dictation",
        }

        self.aliases = {
            "code": "coding",
            "term": "terminal",
            "raw": "direct",
            "literal": "direct",
        }

    def _normalize(self, word: str) -> str | None:
        word = word.lower()

        if word in self.valid_modes:
            return word

        if word in self.aliases:
            return self.aliases[word]

        return None

    def handle(self, text: str) -> tuple[str | None, bool]:
        if not text:
            return None, False

        # Normalize input (strip punctuation + lowercase)
        t = re.sub(r"[^\w\s]", "", text).strip().lower()

        words = t.split()

        # --- Only allow EXACTLY two words
        if len(words) == 2:
            w1, w2 = words

            # Case 1: "mode X"
            if w1 == "mode":
                mode = self._normalize(w2)
                if mode:
                    self.mode = mode
                    return mode, True

            # Case 2: "X mode"
            if w2 == "mode":
                mode = self._normalize(w1)
                if mode:
                    self.mode = mode
                    return mode, True

        return None, False