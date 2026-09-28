"""Whitespace command payloads must reach the chosen backend unchanged."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.parametrize("text", ["\n", "\n\n", "\t", " ", "\t\tvalue\n"])
def test_text_injector_forwards_whitespace_to_ibus(text):
    from vocalinux.text_injection.text_injector import DesktopEnvironment, TextInjector

    injector = TextInjector.__new__(TextInjector)
    injector.environment = DesktopEnvironment.WAYLAND_IBUS
    injector._ibus_injector = MagicMock()
    injector._log_current_window_info = MagicMock()
    injector._should_copy_to_clipboard = lambda: False
    assert injector.inject_text(text)
    injector._ibus_injector.inject_text.assert_called_once_with(text)


@pytest.mark.parametrize("text", ["\n", "\n\n", "\t", " ", "\t\tvalue\n"])
def test_ibus_client_sends_whitespace_bytes(text):
    from vocalinux.text_injection import ibus_engine

    injector = ibus_engine.IBusTextInjector.__new__(ibus_engine.IBusTextInjector)
    with (
        patch.object(ibus_engine, "is_engine_active", return_value=True),
        patch.object(ibus_engine, "SOCKET_PATH"),
        patch.object(ibus_engine.socket, "socket") as socket_factory,
    ):
        sock = socket_factory.return_value.__enter__.return_value
        sock.recv.return_value = b"OK"
        assert injector.inject_text(text)
        sock.sendall.assert_called_once_with(text.encode("utf-8"))
