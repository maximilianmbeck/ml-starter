"""Defines a launcher which uses `torchrun` to launch a job.

This is a light-weight werapper around PyTorch's `torch.distributed.launch`
script. It is used to launch a job on a single node with multiple processes,
each with multiple devices.
"""

import logging
import shutil
import subprocess
from dataclasses import dataclass

import torch
from omegaconf import MISSING, OmegaConf

from ml.core.config import conf_field
from ml.core.registry import project_dirs, register_launcher, register_trainer
from ml.launchers.base import BaseLauncher, BaseLauncherConfig
from ml.utils.networking import get_unused_port

logger: logging.Logger = logging.getLogger(__name__)

TORCHRUN_TEMPLATE: str = """
#!/usr/bin/env python

from pathlib import Path

from omegaconf import OmegaConf

from ml.core.registry import project_dirs as registry_project_dirs
from ml.scripts.train import train_main

PROJECT_DIRS = {project_root}
CONFIG_PATH = '{config_path}'


def main() -> None:
    for p in PROJECT_DIRS:
        registry_project_dirs.add(Path(p))
    config = OmegaConf.load(CONFIG_PATH)
    train_main(config)


if __name__ == "__main__":
    main()
"""


@dataclass
class TorchRunLauncherConfig(BaseLauncherConfig):
    nproc_per_node: int = conf_field(MISSING, help="The number of processes per node")
    master_addr: str = conf_field("127.0.0.1", help="The address of the master")
    master_port: int = conf_field(MISSING, help="The port of the master")
    backend: str = conf_field("nccl", help="The backend to use for distributed training")
    start_method: str = conf_field("spawn", help="The method to use to start processes")
    torchrun_path: str = conf_field(MISSING, help="The path to the TorchRun script")

    @classmethod
    def resolve(cls: type["TorchRunLauncherConfig"], config: "TorchRunLauncherConfig") -> None:
        super().resolve(config)

        if OmegaConf.is_missing(config, "nproc_per_node"):
            config.nproc_per_node = torch.cuda.device_count()
        if OmegaConf.is_missing(config, "master_port"):
            config.master_port = get_unused_port()
        if OmegaConf.is_missing(config, "torchrun_path"):
            torchrun_path = shutil.which("torchrun")
            if torchrun_path is None:
                raise ValueError("Could not find torchrun in PATH")
            config.torchrun_path = torchrun_path


@register_launcher("torchrun", TorchRunLauncherConfig)
class TorchRunLauncher(BaseLauncher[TorchRunLauncherConfig]):
    def launch(self) -> None:
        """Launches the job by calling the TorchRun CLI in a subprocess."""
        trainer = register_trainer.build_entry_non_null(self.raw_config)
        trainer.save_config()

        # Builds the run file.
        torchrun_file = TORCHRUN_TEMPLATE.format(
            project_root=[str(p) for p in project_dirs.paths],
            config_path=trainer.config_path,
        ).strip()

        torchrun_fpath = trainer.exp_dir / "torchrun.py"
        with open(torchrun_fpath, "w", encoding="utf-8") as f:
            f.write(torchrun_file)
        logger.info("Wrote torchrun file to %s", torchrun_fpath)

        # Makes a specific log directory for TorchRun logs.
        (log_dir := trainer.log_dir / "torchrun").mkdir(parents=True, exist_ok=True)

        # This launcher expects to run on only one node. A multi-node TorchRun
        # launcher would require a way to launch TorchRun processes across
        # multiple target nodes.
        node_rank, num_nodes = 0, 1

        cmd = [
            self.config.torchrun_path,
            "--nproc-per-node",
            str(self.config.nproc_per_node),
            "--node-rank",
            str(node_rank),
            "--nnodes",
            str(num_nodes),
            "--master-addr",
            self.config.master_addr,
            "--master-port",
            str(self.config.master_port),
            "--start-method",
            self.config.start_method,
            "--log-dir",
            str(log_dir),
            "--run-path",
            str(torchrun_fpath),
        ]

        # Launch the job
        logger.info("Launching job with command: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)
