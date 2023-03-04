import itertools
import logging
from dataclasses import dataclass
from typing import Any, cast

import torch
import torchvision
from torch import Tensor
from torch.utils.data.dataset import Dataset

import ml.api as M

logger = logging.getLogger(__name__)


@dataclass
class KineticsDemoTaskConfig(M.BaseTaskConfig):
    frames_per_clip: int = M.conf_field(16, help="Number of frames per clip")
    key: str = M.conf_field("400", help="Kinetics dataset key")
    step_between_clips: int = M.conf_field(1, help="Number of frames between clips")
    num_download_workers: int = M.conf_field(10, help="Number of workers to download videos")
    num_workers: int = M.conf_field(10, help="Number of workers to process videos")


@M.register_task("hmdb_demo", KineticsDemoTaskConfig)
class KineticsDemoTask(M.BaseTask[KineticsDemoTaskConfig]):
    def __init__(self, config: KineticsDemoTaskConfig) -> None:
        super().__init__(config)

        # Gets the class names for each index.
        self.idx_to_classes = cast(torchvision.datasets.HMDB51, self.get_dataset("test")).classes
        self.classes_to_idx = {class_name: idx for idx, class_name in enumerate(self.idx_to_classes)}

    def run_model(self, model: M.BaseModel, batch: tuple[Tensor, Tensor], state: M.State) -> Tensor:
        raise NotImplementedError

    def compute_loss(self, model: M.BaseModel, batch: tuple[Tensor, Tensor], state: M.State, output: Tensor) -> Tensor:
        raise NotImplementedError

    def get_dataset(self, phase: M.Phase) -> Dataset:
        root_dir = M.get_data_dir() / "Kinetics"

        # Cache the metadata for the dataset.
        metadata_file = root_dir / "metadata.pt"
        metadata: dict[str, Any] | None = None
        if metadata_file.exists():
            metadata = torch.load(metadata_file)

        dataset = torchvision.datasets.Kinetics(
            root=root_dir,
            frames_per_clip=self.config.frames_per_clip,
            num_classes=self.config.key,
            split=phase,
            download=not (root_dir / "test").exists(),
            step_between_clips=self.config.step_between_clips,
            num_download_workers=self.config.num_download_workers,
            num_workers=self.config.num_workers,
            _precomputed_metadata=metadata,
        )

        # Saves the metadata when done processing
        if metadata is None:
            torch.save(dataset.metadata, metadata_file)

        return dataset


def run_hmdb_demo_adhoc_test() -> None:
    """Runs a quick test to make sure the task runs.

    Usage:
        python -m ml.tasks.kinetics_demo
    """

    M.configure_logging()

    task = KineticsDemoTask(KineticsDemoTaskConfig())
    dataset = task.get_dataset("train")

    # Just prints some samples.
    for i, sample in enumerate(itertools.islice(dataset, 0, 10)):  # type: ignore
        logger.info("Sample %d: %s", i, sample)


if __name__ == "__main__":
    run_hmdb_demo_adhoc_test()
