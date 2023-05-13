"""Utilities for working with OmegaConf configs."""

from typing import Any

from omegaconf import MISSING, Container, OmegaConf


def is_missing(cfg: Any, key: str) -> bool:
    """Utility function for checking if a config key is missing.

    This is for cases when you are using a raw dataclass rather than an
    OmegaConf container but want to treat them the same way.

    Args:
        cfg: The config to check
        key: The key to check

    Returns:
        Whether or not the key is missing a value in the config
    """
    if isinstance(cfg, Container) and OmegaConf.is_missing(cfg, key):
        return True
    if getattr(cfg, key) is MISSING:
        return True
    return False
