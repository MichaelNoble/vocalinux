"""Per-utterance whisper.cpp decoder profiles.

Saved advanced settings supply defaults; mode profiles override the five fields
listed below. Decoder overrides persist in pywhispercpp, so callers must send
all managed fields each time. These profiles do not select GPU backends or
change the sampling strategy. Each utterance discards past decoder output to
prevent history feedback across independent recordings. Recognition quality still requires an audio trial.
"""

import math

WHISPER_MODE_PARAMS: dict[str, dict] = {
    "clean": {
        "initial_prompt": (
            "camel case snake case pascal case new line "
            "delete that open brace close paren period comma."
        ),
        "single_segment": True,
        "suppress_blank": True,
        "temperature": 0.0,
        "no_context": True,
    },
    "dictation": {
        "initial_prompt": (
            "Natural speech, complete sentences with punctuation. "
            "Period, comma, question mark, exclamation point."
        ),
        "single_segment": False,
        "suppress_blank": False,
        "temperature": 0.1,
        "no_context": True,
    },
    "strict": {
        # Like clean but unambiguous command-only input — no prose expected.
        "initial_prompt": (
            "camel case snake case pascal case new line "
            "delete that open brace close paren period comma."
        ),
        "single_segment": True,
        "suppress_blank": True,
        "temperature": 0.0,
        "no_context": True,
    },
    "direct": {
        # Minimal intervention — pass audio through with as little
        # decoder bias as possible. No prompt or previous-transcript history.
        "initial_prompt": "",
        "single_segment": False,
        "suppress_blank": False,
        "temperature": 0.0,
        "no_context": True,
    },
    "coding": {
        "initial_prompt": (
            # Case formatting — most common voice commands
            "camel case snake case pascal case upper snake case kebab case "
            # JavaScript / React keywords and hooks
            "const let var function return async await import export default "
            "useState useEffect useRef useCallback useMemo useContext "
            "props state component render jsx fragment "
            "arrow function fat arrow double equals triple equals not equals "
            # PHP keywords and operators
            "echo foreach foreach match namespace use trait interface "
            "public private protected static abstract extends implements "
            "dollar sign this arrow double colon null false true "
            "spaceship operator null coalescing Elvis operator "
            # Shared / structural
            "class interface return if else elif switch case break continue "
            "open brace close brace open bracket close bracket "
            "open paren close paren semicolon double colon fat arrow "
            "def function void string integer boolean array object."
        ),
        "single_segment": True,
        "suppress_blank": True,
        "temperature": 0.0,
        "no_context": True,
    },
    "terminal": {
        "initial_prompt": (
            # npm / Node
            "npm run start npm run build npm run dev npm install "
            "npm run test npx create react app node modules "
            "package dot json dot env "
            # Composer / PHP tooling
            "composer install composer update composer require "
            "artisan migrate artisan serve php artisan "
            # Git
            "git commit git push git pull git status git diff "
            "git checkout git branch git merge git stash git log "
            "git add dot git commit dash m "
            # PHPStorm / general dev CLI
            "phpunit pest run dash dash filter dash dash coverage "
            "ssh sudo chmod chown mkdir rm dash rf ls dash la "
            "grep dash r pipe cat dot slash tilde backslash "
            "slash var slash www slash html "
            # Symbols spoken as words
            "dash dash flag dot slash at sign percent ampersand."
        ),
        "single_segment": True,
        "suppress_blank": True,
        "temperature": 0.0,
        "no_context": True,
    },
}

DECODE_DEFAULTS = {
    "no_timestamps": True,
    "no_context": True,
    "initial_prompt": "",
    "single_segment": True,
    "suppress_blank": True,
    "temperature": 0.0,
    "temperature_inc": -1.0,
    "entropy_thold": 2.4,
    "logprob_thold": -1.0,
    "no_speech_thold": 0.6,
}


def validate_decode_params(params: dict) -> None:
    """Reject malformed managed values before the binding mutates its params."""
    unknown = params.keys() - DECODE_DEFAULTS.keys()
    if unknown:
        raise ValueError(f"Unknown whisper.cpp decoding parameters: {sorted(unknown)}")
    for key, value in params.items():
        expected = DECODE_DEFAULTS[key]
        if isinstance(expected, bool):
            valid = isinstance(value, bool)
        elif isinstance(expected, str):
            valid = isinstance(value, str)
        else:
            valid = (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
            )
        if not valid:
            raise ValueError(f"Invalid whisper.cpp setting '{key}'; check Advanced settings")


def resolve_decode_params(global_params: dict, mode: str) -> dict:
    """Return complete validated defaults with explicit per-mode overrides."""
    if mode not in WHISPER_MODE_PARAMS:
        raise ValueError(f"Unknown dictation mode: {mode}")
    validate_decode_params(global_params)
    params = {**DECODE_DEFAULTS, **global_params, **WHISPER_MODE_PARAMS[mode]}
    validate_decode_params(params)
    return params
