"""
Command definitions for Vocalinux.

Pure data — no logic. All dictionaries defining what commands exist,
their trigger phrases, aliases, and what they map to.

To add a new command: add an entry to the appropriate dictionary.
The processor picks it up automatically.
"""

from typing import Callable

# ---------------------------------------------------------------------------
# Text replacement commands
# ---------------------------------------------------------------------------
# Replaced inline in the transcription. Punctuation commands additionally
# consume the space before them (see PUNCTUATION_COMMANDS set below).

TEXT_COMMANDS: dict[str, str] = {
    # Line / whitespace
    "new line":             "\n",
    "new paragraph":        "\n\n",
    "tab":                  "\t",
    "space":                " ",

    # Sentence punctuation
    "period":               ".",
    "full stop":            ".",
    "comma":                ",",
    "question mark":        "?",
    "exclamation mark":     "!",
    "exclamation point":    "!",
    "semicolon":            ";",
    "colon":                ":",
    "ellipsis":             "...",
    "dot dot dot":          "...",

    # Paired delimiters
    "open parenthesis":     "(",
    "close parenthesis":    ")",
    "open paren":           "(",
    "close paren":          ")",
    "open bracket":         "[",
    "close bracket":        "]",
    "open brace":           "{",
    "close brace":          "}",
    "open angle bracket":   "<",
    "close angle bracket":  ">",

    # Quotes
    "quote":                '"',
    "double quote":         '"',
    "open quote":           '"',
    "close quote":          '"',
    "single quote":         "'",
    "apostrophe":           "'",
    "backtick":             "`",

    # Operators & symbols
    "dash":                 "-",
    "hyphen":               "-",
    "underscore":           "_",
    "equals":               "=",
    "double equals":        "==",
    "triple equals":        "===",
    "not equals":           "!=",
    "plus":                 "+",
    "minus":                "-",
    "asterisk":             "*",
    "slash":                "/",
    "backslash":            "\\",
    "percent":              "%",
    "ampersand":            "&",
    "pipe":                 "|",
    "caret":                "^",
    "tilde":                "~",
    "at sign":              "@",
    "hash":                 "#",
    "dollar sign":          "$",
    "dollar":               "$",
    "less than":            "<",
    "greater than":         ">",

    # Composite operators
    "arrow":                "->",
    "fat arrow":            "=>",
    "double colon":         "::",
}

# Subset of TEXT_COMMANDS that eat the space before them so
# "end of sentence period" -> "end of sentence."
PUNCTUATION_COMMANDS: frozenset[str] = frozenset({
    "period",
    "full stop",
    "comma",
    "question mark",
    "exclamation mark",
    "exclamation point",
    "semicolon",
    "colon",
    "ellipsis",
    "dot dot dot",
})

# ---------------------------------------------------------------------------
# Action commands
# ---------------------------------------------------------------------------
# These trigger editor/keyboard actions rather than producing text output.
# Values are action identifiers consumed by action callbacks upstream.

ACTION_COMMANDS: dict[str, str] = {
    "delete that":          "delete_last",
    "scratch that":         "delete_last",
    "delete last word":     "delete_last_word",
    "delete last sentence": "delete_last_sentence",
    "undo":                 "undo",
    "undo that":            "undo",
    "redo":                 "redo",
    "redo that":            "redo",
    "select all":           "select_all",
    "select line":          "select_line",
    "select word":          "select_word",
    "select paragraph":     "select_paragraph",
    "cut":                  "cut",
    "cut that":             "cut",
    "copy":                 "copy",
    "copy that":            "copy",
    "paste":                "paste",
    "paste that":           "paste",
    "save":                 "save",
    "save file":            "save",
    "find":                 "find",
    "open find":            "find",
}

# ---------------------------------------------------------------------------
# Multi-word format commands
# ---------------------------------------------------------------------------
# Consume ALL words following the trigger to end of phrase (or next trigger).
# Values are callables: fn(words: list[str]) -> str
#
# e.g. "camel case this is my name" -> "thisIsMyName"
# e.g. "snake case get user name"   -> "get_user_name"
#
# To add a new style: define a _fn below and add entries to the dict.
# Nothing else needs to change.

def _camel(words: list[str]) -> str:
    return words[0].lower() + "".join(w.capitalize() for w in words[1:])

def _pascal(words: list[str]) -> str:
    return "".join(w.capitalize() for w in words)

def _snake(words: list[str]) -> str:
    return "_".join(w.lower() for w in words)

def _kebab(words: list[str]) -> str:
    return "-".join(w.lower() for w in words)

def _upper_snake(words: list[str]) -> str:
    return "_".join(w.upper() for w in words)

def _uppercase(words: list[str]) -> str:
    return "".join(w.upper() for w in words)

def _lowercase(words: list[str]) -> str:
    return "".join(w.lower() for w in words)


MULTIWORD_FORMAT_COMMANDS: dict[str, Callable[[list[str]], str]] = {
    "camel case":           _camel,
    "camelcase":            _camel,
    "pascal case":          _pascal,
    "pascalcase":           _pascal,
    "title case":           _pascal,
    "snake case":           _snake,
    "snakecase":            _snake,
    "kebab case":           _kebab,
    "kebabcase":            _kebab,
    "upper snake case":     _upper_snake,
    "screaming snake case": _upper_snake,
    "constant case":        _upper_snake,
    "upper case":           _uppercase,
    "lower case":           _lowercase,
}

# ---------------------------------------------------------------------------
# Single-word format modifiers
# ---------------------------------------------------------------------------
# Apply to the immediately following word only.
# Values are callables: fn(word: str) -> str
#
# e.g. "capitalize name" -> "Name"
# e.g. "uppercase warning" -> "WARNING"

def _cap_next(word: str) -> str:   return word.capitalize()
def _upper_next(word: str) -> str: return word.upper()
def _lower_next(word: str) -> str: return word.lower()

FORMAT_MODIFIERS: dict[str, Callable[[str], str]] = {
    "capitalize":   _cap_next,
    "uppercase":    _upper_next,
    "all caps":     _upper_next,
    "lowercase":    _lower_next,
    "no caps":      _lower_next,
}