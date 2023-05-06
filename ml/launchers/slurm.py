"""Defines a Distributed Data Parallel trainer that launches Slurm jobs.

This is a light-weight wrapper around the PyTorch's built-in Distributed Data
Parallel class. Note that this only supports one GPU per task, and one task
per GPU.

Steps
-----

1. Stages the environment to a new working directory
2. Writes an `sbatch.sh` file
3. Schedules `sbatch.sh` file

This allows for repeatability by just scheduling the same `sbatch.sh` file.
"""

import logging
import os
import re
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import TypeVar

from omegaconf import II, MISSING, OmegaConf

from ml.core.config import conf_field
from ml.core.env import get_stage_dir
from ml.core.registry import Objects, project_dirs, register_launcher, register_trainer
from ml.launchers.base import BaseLauncher, BaseLauncherConfig
from ml.scripts.train import train_main_with_objects
from ml.trainers.base import BaseTrainer
from ml.utils.distributed import (
    get_master_addr,
    get_master_port,
    get_random_port,
    is_master,
    set_init_method,
    set_master_addr,
    set_rank,
    set_world_size,
)
from ml.utils.logging import configure_logging
from ml.utils.staging import stage_environment
from ml.utils.torch_distributed import get_distributed_backend, init_process_group

logger: logging.Logger = logging.getLogger(__name__)

T = TypeVar("T", bound="SlurmLauncher")

SBATCH_TEMPLATE: str = """
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --requeue
#SBATCH --signal=USR1@90
#SBATCH --time={time_limit}
#SBATCH --comment='{comment}'
#SBATCH --nodes={num_nodes}
#SBATCH --ntasks-per-node={tasks_per_node}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --gres={gres}
#SBATCH --gpu-bind={gpu_bind}
#SBATCH --output={output_path}
#SBATCH --error={error_path}
#SBATCH --open-mode=append
{extra_sbatch_lines}

# Sets the environment variables.
export STAGE_DIR={stage_dir}
export PYTHONPATH={pythonpath}
export MASTER_PORT={master_port}

# Set some debugging flags.
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1
export NCCL_DEBUG=1

# Disables Tensorboard subprocess.
export DISABLE_TENSORBOARD=1

echo "***"
echo "Job ID: ${{SLURM_JOBID}}"
echo "***"
echo ""

# Runs the training command.
srun \\
    --nodes={num_nodes} \\
    --ntasks-per-node={tasks_per_node} \\
    --cpus-per-task={cpus_per_task} \\
    --gres={gres} \\
    --gpu-bind={gpu_bind} \\
    python -m ml.trainers.slurm {config_path}

echo ""
""".strip()


def set_slurm_rank_and_world_size() -> tuple[int, int]:
    node_id = int(os.environ["SLURM_NODEID"])
    local_id = int(os.environ["SLURM_LOCALID"])
    tasks_per_node = int(os.environ["SLURM_NTASKS_PER_NODE"])
    num_nodes = int(os.environ["SLURM_NNODES"])
    rank = node_id * tasks_per_node + local_id
    world_size = num_nodes * tasks_per_node
    set_rank(rank)
    set_world_size(world_size)
    return rank, world_size


def set_slurm_master_addr() -> str:
    node_list = os.environ.get("SLURM_STEP_NODELIST")
    if node_list is None:
        node_list = os.environ.get("SLURM_JOB_NODELIST")
    assert node_list is not None, "`SLURM_JOB_NODELIST` environment variable not set"
    hostnames = subprocess.check_output(["scontrol", "show", "hostnames", node_list])
    host = hostnames.split()[0].decode("utf-8")
    set_master_addr(host)
    return host


def requeue_job() -> None:
    if is_master():
        if "SLURM_JOB_ID" in os.environ:
            cmd = ["scontrol", "requeue", os.environ["SLURM_JOB_ID"]]
            logger.info("Running %s", " ".join(cmd))
            subprocess.check_call(cmd)
        else:
            logger.info("SLURM_JOB_ID environment variable not found; not requeueing")


@dataclass
class SlurmLauncherConfig(BaseLauncherConfig):
    partition: str = conf_field(II("oc.env:SLURM_PARTITION,none"), help="Which partition to launch")
    time_limit: str = conf_field(II("oc.env:SLURM_TIME_LIMIT,3-00:00:00"), help="Time limit string")
    num_nodes: int = conf_field(MISSING, help="Total number of nodes to use")
    gpus_per_node: int = conf_field(II("oc.env:SLURM_GPUS_PER_NODE,8"), help="Number of GPUs per node")
    cpus_per_gpu: int = conf_field(II("oc.env:SLURM_CPUS_PER_GPU,1"), help="Number of CPUs per task")
    gpu_type: str | None = conf_field(None, help="Specific GPU type to pass to gres")
    num_jobs: int = conf_field(1, help="Number of redundant jobs to launch")
    comment: str | None = conf_field(None, help="An optional comment to add to the experiment")
    master_port: int = conf_field(get_random_port, help="The master port to use")


def ignore_signal(signum: int, _: FrameType | None) -> None:
    sig = signal.Signals(signum)
    logger.info("Ignoring signal %s", sig.name)


@register_launcher("slurm", SlurmLauncherConfig)
class SlurmLauncher(BaseLauncher[SlurmLauncherConfig]):
    def write_sbatch_file(self, trainer: BaseTrainer) -> Path:
        # Gets some configuration options.
        gpus_per_node = self.config.gpus_per_node
        gpu_type = self.config.gpu_type
        tasks_per_node = gpus_per_node
        cpus_per_task = self.config.cpus_per_gpu

        # GRES and GPU Bind SBatch options.
        gres = f"gpu:{gpus_per_node}" if gpu_type is None else f"gpu:{gpu_type}:{gpus_per_node}"
        gpu_bind = f"map_gpu:{','.join(str(i) for i in range(gpus_per_node))}"

        # Gets extra SBatch options.
        sbatch_lines: list[str] = []
        if "EMAIL" in os.environ:
            sbatch_lines += [f"--mail-user={os.environ['EMAIL']}", "--mail-type=ALL"]

        # Writes all Slurm stuff (including logs) to this folder.
        slurm_log_dir = trainer.exp_dir / "logs"
        slurm_log_dir.mkdir(exist_ok=True, parents=True)
        sbatch_path = trainer.exp_dir / "sbatch.sh"

        # Stages all files to a new directory.
        stage_dir = stage_environment(project_dirs.paths[1:], get_stage_dir())

        # Gets the python path with the new output directory.
        python_path_parts = [str(stage_dir)] + os.environ.get("PYTHONPATH", "").split(":")
        python_path = ":".join(p for p in python_path_parts if p)

        # Comment miscellaneous stuff here.
        comments: list[str] = []
        if self.config.comment is not None:
            comments += [self.config.comment]
        comments += [f"Log directory: {trainer.exp_dir}"]
        comments += [f"Code location: {stage_dir}"]

        # Saves the config that is used to launch the Slurm job.
        trainer.save_config()

        # Builds the SBatch file.
        sbatch_file = SBATCH_TEMPLATE.format(
            job_name=trainer.exp_name,
            partition=self.config.partition,
            time_limit=self.config.time_limit,
            comment="; ".join(comments),
            num_nodes=self.config.num_nodes,
            tasks_per_node=tasks_per_node,
            cpus_per_task=cpus_per_task,
            gres=gres,
            gpu_bind=gpu_bind,
            output_path=slurm_log_dir / "slurm_out.txt",
            error_path=slurm_log_dir / "slurm_err.%j.txt",
            extra_sbatch_lines="\n".join(f"#SBATCH {line}" for line in sbatch_lines),
            stage_dir=stage_dir,
            pythonpath=python_path,
            master_port=self.config.master_port,
            config_path=trainer.exp_dir / "config.yaml",
            lock_file_path=trainer.exp_dir / ".lock_running",
        )

        with open(sbatch_path, "w", encoding="utf-8") as f:
            f.write(sbatch_file)
        logger.info("Wrote sbatch file to %s", sbatch_path)

        return sbatch_path

    def launch(self) -> None:
        trainer = register_trainer.build_entry_non_null(self.raw_config)

        sbatch_path = self.write_sbatch_file(trainer)

        # Call `sbatch` on the given file.
        all_run_ids: list[str] = []
        for _ in range(self.config.num_jobs):
            command = ["sbatch", str(sbatch_path)]
            if all_run_ids:
                command += ["--dependency", all_run_ids[-1]]
            proc = subprocess.Popen(  # pylint: disable=consider-using-with
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            assert proc is not None and proc.stdout is not None
            proc.wait()
            log_line = proc.stdout.read().decode("utf-8").strip()
            run_ids = re.findall(r"Submitted batch job (\d+)", log_line)
            assert len(run_ids) == 1, f"Unexpected log line: {log_line}"
            all_run_ids += [run_ids[0]]

        run_ids_str = "".join(f"\n - {run_id}" for run_id in all_run_ids)
        logger.info("Launched %d job(s):%s", len(all_run_ids), run_ids_str)

        trainer.add_lock_file("scheduled", exists_ok=False)


def slurm_main() -> None:
    args = sys.argv[1:]
    assert len(args) == 1, f"Unexpected arguments to `slurm_main`: {sys.argv}"

    # Adds the stage directories as project directories.
    stage_dir = Path(os.environ["STAGE_DIR"])
    for sub_dir in stage_dir.iterdir():
        if sub_dir.is_dir():
            project_dirs.add(sub_dir)

    # Loads the raw config.
    raw_config = OmegaConf.load(args[0])

    # Sets environment variables from Clurm environment variables.
    set_slurm_master_addr()
    rank, world_size = set_slurm_rank_and_world_size()

    # Sets the initialization method and configures per-rank logging.
    set_init_method(f"tcp://{get_master_addr()}:{get_master_port()}")
    configure_logging(rank=rank, world_size=world_size)
    init_process_group(backend=get_distributed_backend())

    objs = Objects.parse_raw_config(raw_config)  # type: ignore

    assert (trainer := objs.trainer) is not None
    trainer.add_lock_file("running", exists_ok=True)
    trainer.remove_lock_file("scheduled", missing_ok=True)

    signal.signal(signal.SIGTERM, ignore_signal)
    trainer.add_signal_handler(signal.SIGUSR1, requeue_job)

    train_main_with_objects(objs)


if __name__ == "__main__":
    slurm_main()
