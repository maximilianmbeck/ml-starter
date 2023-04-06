import os
from pathlib import Path

import ml.api as ml
import pytest
from ml.core.registry import Objects
from ml.utils.cli import parse_cli
from omegaconf import DictConfig

# Be mindful that these configs are supposed to run regularly in CI
# and therefore should not be too expensive.
CONFIGS_TO_CHECK: list[list[str]] = [
    ["configs/image_demo.yaml", "model.pretrained=False"],
]


@pytest.mark.parametrize("config_paths", CONFIGS_TO_CHECK)
@pytest.mark.slow()
def test_instantiate_configs(config_paths: list[str], tmpdir: Path) -> None:
    # Resolves config paths.
    config_parent_dir = Path(ml.__path__[0]).parent
    config_paths_absolute = [
        str((config_parent_dir / config_path).resolve()) if (config_parent_dir / config_path).exists() else config_path
        for config_path in config_paths
    ]

    def create_and_set(key: str) -> None:
        env_dir = Path(tmpdir / key)
        env_dir.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(env_dir)

    # Resolves various environment variables.
    create_and_set("RUN_DIR")
    create_and_set("DATA_DIR")
    create_and_set("MODEL_DIR")

    # Constructs the objects from the config.
    config = parse_cli(config_paths_absolute)
    Objects.resolve_config(config)
    assert isinstance(config, DictConfig)
    objs = Objects.parse_raw_config(config)
    assert objs.model is not None
    assert objs.task is not None
