import logging

from omegaconf import DictConfig, OmegaConf

logger: logging.Logger = logging.getLogger(__name__)


def resolve_main(config: DictConfig) -> None:
    """Resolves the command line to a config and prints it.

    Args:
        config: The configuration object.
    """

    print(OmegaConf.to_yaml(config))
