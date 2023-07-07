"""Networking utility functions."""

import socket


def get_unused_port(default: int | None = None) -> int:
    """Returns an unused port number on the local machine.

    Args:
        default: A default port to try before trying other ports.

    Returns:
        A port number which is currently unused
    """
    if default is not None:
        sock = socket.socket()
        try:
            sock.bind(("", default))
            return default
        except OSError:
            pass
        finally:
            sock.close()

    sock = socket.socket()
    sock.bind(("", 0))
    return sock.getsockname()[1]
