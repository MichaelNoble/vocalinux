"""
Context manager module for Vocalinux.

Provides get_context() — a lightweight call that queries the Vocalinux IBus
engine via Unix socket and returns the current input context as a dict.

Any processor (text, command, effects, actions) can import and call
get_context() independently without knowing about each other.

Returned dict fields:
    surrounding_text  (str)  — text in the focused field around the cursor
    last_surrounding_text  (str)  — text from the previously focused field around the cursor
    cursor_pos        (int)  — character offset of the cursor within surrounding_text
    context_changed   (bool) — True if focus moved to a new field since last call
"""

import json
import logging
import socket
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Must match the path defined in ibus_engine.py
_SOCKET_PATH = Path.home() / ".local" / "share" / "vocalinux-ibus" / "inject.sock"

# Returned when the engine is unreachable or returns unexpected data
_FALLBACK_CONTEXT = {
    "surrounding_text": "",
    "cursor_pos": 0,
    "context_changed": True,
    "last_surrounding_text": "",
}

# How long to wait for surrounding text to arrive after a context change
_SURROUNDING_TEXT_POLL_TIMEOUT = 0.3  # seconds
_SURROUNDING_TEXT_POLL_INTERVAL = 0.05  # seconds

"""
    Connects to the engine's Unix socket, sends GET_CONTEXT, and parses
    the JSON response. The engine resets its context_changed flag after
    each successful call, so each caller gets a consistent snapshot.

    On any failure (engine not running, timeout, malformed response) a
    fallback dict is returned with context_changed=True so that callers
    fail safely — i.e. they will not prepend a space or assume continuity.
"""


def _raw_get_context(timeout: float = 1.0) -> dict:
    """
    Single socket call to GET_CONTEXT. Returns fallback on any failure.
    Internal use only — callers should use get_context().
    """
    if not _SOCKET_PATH.exists():
        logger.debug("IBus engine socket not found — returning fallback context")
        return dict(_FALLBACK_CONTEXT)

    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect(str(_SOCKET_PATH))
            sock.sendall(b"GET_CONTEXT")

            # Read until the connection closes — response may be larger than
            # a single recv() depending on surrounding text length
            chunks = []
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                chunks.append(chunk)

            raw = b"".join(chunks).decode("utf-8")
            context = json.loads(raw)

            # Ensure all expected keys are present even if engine is older
            for key, default in _FALLBACK_CONTEXT.items():
                context.setdefault(key, default)

            logger.debug(
                f"Context: changed={context['context_changed']} "
                f"cursor={context['cursor_pos']} "
                f"previous_text='{context['last_surrounding_text'][:30]}'"
                f"text='{context['surrounding_text'][:30]}'"
            )
            return context

    except socket.timeout:
        logger.warning("Timeout querying IBus engine context")
        return dict(_FALLBACK_CONTEXT)
    except json.JSONDecodeError as e:
        logger.warning(f"Malformed context response from IBus engine: {e}")
        return dict(_FALLBACK_CONTEXT)
    except OSError as e:
        logger.warning(f"Could not connect to IBus engine socket: {e}")
        return dict(_FALLBACK_CONTEXT)


def _context_looks_stale(context: dict) -> bool:
    text = context["surrounding_text"]
    cursor = context["cursor_pos"]

    if not text:
        return False  # already handled by your existing poll

    # Cursor out of bounds → definitely stale
    if cursor > len(text):
        return True

    # Suspicious: cursor at start but text is long
    if cursor <= 1 and len(text) > 20:
        return True

    return False


def get_context(timeout: float = 1.0) -> dict:
    """
    Query the Vocalinux IBus engine for the current input context.

    Connects to the engine's Unix socket, sends GET_CONTEXT, and parses
    the JSON response. The engine resets its context_changed flag after
    each successful call, so each caller gets a consistent snapshot.

    If context has just changed (focus moved to a new field or cursor
    repositioned), polls briefly for surrounding text to arrive from the
    application via do_set_surrounding_text. This gives processors enough
    information to make correct space, capitalization, and punctuation
    decisions even at the start of a new context.

    On any failure (engine not running, timeout, malformed response) a
    fallback dict is returned with context_changed=True so that callers
    fail safely — i.e. they will not prepend a space or assume continuity.

    Args:
        timeout: Socket timeout in seconds (default 1.0)

    Returns:
        dict with keys: surrounding_text, last_surrounding_text, cursor_pos, context_changed,

    """
    context = _raw_get_context(timeout)

    # If context just changed and surrounding text hasn't arrived yet,
    # poll briefly — do_set_surrounding_text fires asynchronously after
    # do_focus_in, usually within a few tens of milliseconds.
    # Once surrounding_text is populated we can make accurate decisions.
    # If nothing arrives within the timeout we return what we have and
    # let the caller handle empty surrounding text gracefully.

    should_poll = (
        context["context_changed"] and not context["surrounding_text"]
    ) or _context_looks_stale(context)

    if not context["surrounding_text"]:
        logger.debug("Empty context — polling for recovery")
        # logger.debug("Context may be stale — polling for update")

        deadline = time.monotonic() + _SURROUNDING_TEXT_POLL_TIMEOUT
        while time.monotonic() < deadline:
            time.sleep(_SURROUNDING_TEXT_POLL_INTERVAL)
            context = _raw_get_context(timeout)
            if context["surrounding_text"]:
                elapsed = _SURROUNDING_TEXT_POLL_TIMEOUT - (deadline - time.monotonic())
                logger.debug(f"Surrounding text arrived after {elapsed:.2f}s")
                break
        else:
            logger.debug(
                f"No surrounding text after {_SURROUNDING_TEXT_POLL_TIMEOUT}s — "
                "returning empty context, caller will handle gracefully"
            )

    logger.debug(
        f"Context: changed={context['context_changed']} "
        f"cursor={context['cursor_pos']} "
        f"previous_text='{context['last_surrounding_text'][:30]}'"
        f"text='{context['surrounding_text']}'"
    )

    return context
