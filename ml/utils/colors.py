"""Defines helper functions for printing colored text to the terminal.

Colors can be disabled by setting ``DISABLE_COLORS=1`` in the environment.
"""

from typing import Literal

from ml.core.env import are_colors_disabled

RESET_SEQ = "\033[0m"
REG_COLOR_SEQ = "\033[%dm"
BOLD_COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"


Color = Literal[
    "black",
    "red",
    "green",
    "yellow",
    "blue",
    "magenta",
    "cyan",
    "white",
    "grey",
    "light-red",
    "light-green",
    "light-yellow",
    "light-blue",
    "light-magenta",
    "light-cyan",
]

COLOR_INDEX: dict[Color, int] = {
    "black": 30,
    "red": 31,
    "green": 32,
    "yellow": 33,
    "blue": 34,
    "magenta": 35,
    "cyan": 36,
    "white": 37,
    "grey": 90,
    "light-red": 91,
    "light-green": 92,
    "light-yellow": 93,
    "light-blue": 94,
    "light-magenta": 95,
    "light-cyan": 96,
}


def get_colorize_parts(color: Color, bold: bool = False) -> tuple[str, str]:
    if bold:
        return BOLD_COLOR_SEQ % COLOR_INDEX[color], RESET_SEQ
    return REG_COLOR_SEQ % COLOR_INDEX[color], RESET_SEQ


def colorize(s: str, color: Color, bold: bool = False) -> str:
    if are_colors_disabled():
        return s
    start, end = get_colorize_parts(color, bold=bold)
    return start + s + end


def maybe_colorize(s: str, color: Color | None, bold: bool = False) -> str:
    if color is None:
        return s
    return colorize(s, color, bold=bold)


def make_bold(strs: list[str], inner: Color | None = None, side: Color | None = None) -> str:
    strs = [s.strip() for s in strs]
    max_len = max(len(s) for s in strs)
    strs = [maybe_colorize(s, inner, bold=True) for s in strs]
    strs = [f"{s}{' ' * (max_len - len(s))}" for s in strs]
    strs_with_sides = [f"{maybe_colorize('│', side)} {s} {maybe_colorize('│', side)}" for s in strs]
    top = maybe_colorize("┌─" + "─" * max_len + "─┐", side)
    bottom = maybe_colorize("└─" + "─" * max_len + "─┘", side)
    return "\n".join([top] + strs_with_sides + [bottom])
