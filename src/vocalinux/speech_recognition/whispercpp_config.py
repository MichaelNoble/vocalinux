"""
Whisper.cpp per-mode inference parameters for Vocalinux.

Each mode gets its own parameter set passed directly to model.transcribe()
on every utterance — no model reload required, parameters are decoder-level.

PARAMETER REFERENCE
───────────────────

initial_prompt : str
    Text prepended to the decoder's context before transcription begins.
    Whisper treats this as prior conversation history, which biases the
    vocabulary and phrasing of the output toward words that appear in the
    prompt. For command recognition, listing the exact trigger phrases here
    (e.g. "camel case", "snake case") significantly improves the chance
    Whisper produces those exact words rather than phonetic near-misses like
    "a scalp case" or "Pascal K". For dictation, a natural-language prompt
    biases toward sentence structure and punctuation.

single_segment : bool
    When True, forces the entire audio clip into one output segment.
    Whisper normally splits longer audio into multiple segments, adding its
    own punctuation and capitalization at each boundary — which interferes
    with command parsing. For short discrete utterances (commands, code),
    True gives cleaner, flatter output. For dictation over longer pauses,
    False allows natural segmentation.

suppress_non_speech_tokens : bool
    When True, suppresses special tokens Whisper uses to inject filler
    words, laughter, music markers, etc. ([MUSIC], [LAUGHTER], (applause)).
    Useful for command and code modes where these would corrupt output.
    In dictation mode, leaving this False lets Whisper handle ambient
    speech more naturally without aggressively filtering.

temperature : float
    Controls decoder randomness. 0.0 is fully deterministic — the highest
    probability token is always chosen, giving consistent repeatable output.
    Ideal for commands where you want exact phrase recognition every time.
    A small value like 0.1 introduces slight variation which can help
    dictation avoid repetition loops on longer audio. Do not go above 0.2
    for voice input — higher values produce hallucinations.

no_context : bool
    When True, the decoder ignores any cached context from previous
    segments. For commands this is correct — each utterance is independent
    and prior context can actively hurt recognition by biasing the decoder
    toward continuing a sentence rather than recognizing a fresh command.
    For dictation, False allows Whisper to use the tail of the previous
    segment as context, improving coherence across utterances.

beam_size: int
    default is 1 (greedy). Beam search considers multiple candidate sequences and picks the best.
    Costs more compute but improves accuracy on
    ambiguous phrases like "pascal case". On your
    GPU the latency hit should be small.


PARAMETERS AVAILABLE BUT NOT CURRENTLY USED
────────────────────────────────────────────

n_threads : int  (default: 4)
    CPU threads for the encoder. You are running on CUDA so this hasA couple others that only do with the model, and forgetting the names of these parameters.
    minimal impact — GPU handles the heavy lifting.

max_context : int  (default: -1, meaning use model default of 224 tokens)
    Maximum tokens of prior context carried into the decoder. Only
    relevant when no_context=False. Could be tuned for dictation mode
    to limit how far back Whisper looks, preventing old context from
    corrupting new segments.

word_thold : float  (default: 0.01)
    Confidence threshold below which a word-level timestamp is suppressed.
    Only relevant if you use word-level timestamps — not currently used.

max_len : int  (default: 0, no limit)
    Maximum segment length in characters. Could enforce short output in
    command mode as a safety net, but single_segment=True already
    handles this adequately.

token_timestamps : bool  (default: False)
    Enables per-token timestamp output. Not needed unless you want to
    implement cursor-position-aware editing in the future.

dtw : bool  (default: False — confirmed off in your model load output)
    Dynamic time warping for improved timestamp alignment. Disabled in
    your build, not relevant to transcription quality.

split_on_word : bool  (default: False)
    Splits segments on word boundaries rather than token boundaries.
    Only meaningful if max_len is set.
"""

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
        "no_context": False,
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
        # decoder bias as possible. No prompt, context allowed.
        "initial_prompt": "",
        "single_segment": False,
        "suppress_blank": False,
        "temperature": 0.0,
        "no_context": False,
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