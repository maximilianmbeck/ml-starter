import socket


def get_unused_port() -> int:
    """Returns an unused port number on the local machine.

    Returns:
        A port number which is currently unused
    """

    sock = socket.socket()
    sock.bind(("", 0))
    return sock.getsockname()[1]
