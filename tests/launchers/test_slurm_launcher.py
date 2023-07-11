"""Tests that the slurm launcher can write a config file.

This test doesn't actually launch a job because that wouldn't be possible on
CI, just runs through the process of staging a directory and writing an
sbatch file.
"""

import sys
import types
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from ml.core.env import set_stage_dir
from ml.core.registry import add_project_dir
from ml.launchers.slurm import SlurmLauncher
from ml.utils.checkpoint import instantiate_config


@pytest.mark.slow
def test_slurm_launcher(tmpdir: Path) -> None:
    config = OmegaConf.create(
        {
            "trainer": {
                "name": "sl",
                "exp_name": "test",
                "log_dir_name": "test",
                "base_run_dir": str(tmpdir),
            },
            "launcher": {
                "name": "slurm",
                "num_nodes": 1,
                "gpus_per_node": 1,
            },
        }
    )

    (stage_dir := Path(tmpdir / "staging")).mkdir()
    set_stage_dir(stage_dir)

    # Mocking a project directory, for staging purposes.
    (project_dir := Path(tmpdir / "project")).mkdir()
    fpath = project_dir / "a.py"
    with open(fpath, "w", encoding="utf-8") as f:
        f.write("print('hello world!')")
    module = types.ModuleType(fpath.name)
    module.__file__ = str(fpath)
    sys.modules[fpath.name] = module
    add_project_dir(project_dir)

    objects = instantiate_config(config)
    assert (trainer := objects.trainer) is not None
    assert isinstance(launcher := objects.launcher, SlurmLauncher)

    sbatch_file_path = launcher.write_sbatch_file(trainer)
    assert Path(sbatch_file_path).exists()
