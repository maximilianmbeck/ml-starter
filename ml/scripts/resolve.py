"""Resolves the command line to a config and prints it."""

from omegaconf import DictConfig, OmegaConf


def resolve_main(config: DictConfig) -> None:
    """Resolves the command line to a config and prints it.

    Args:
        config: The configuration object.
    """
    print(OmegaConf.to_yaml(config))
