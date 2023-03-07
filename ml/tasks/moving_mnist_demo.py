import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor
from torch.utils.data.dataset import Dataset
from torchvision.datasets.moving_mnist import MovingMNIST

import ml.api as M

logger = logging.getLogger(__name__)


@dataclass
class MovingMNISTDemoTaskConfig(M.BaseTaskConfig):
    frames_per_clip: int = M.conf_field(16, help="Number of frames per clip")
    key: int = M.conf_field(400, help="Kinetics dataset key")
    step_between_clips: int = M.conf_field(1, help="Number of frames between clips")
    num_download_workers: int = M.conf_field(32, help="Number of workers to download videos")
    num_extract_workers: int = M.conf_field(32, help="Number of workers to extract frames")
    extensions: list[str] = M.conf_field(["avi", "mp4"], help="Video file extensions to consider")


@M.register_task("moving_mnist_demo", MovingMNISTDemoTaskConfig)
class MovingMNISTDemoTask(M.BaseTask[MovingMNISTDemoTaskConfig]):
    def __init__(self, config: MovingMNISTDemoTaskConfig) -> None:
        super().__init__(config)

    def run_model(self, model: M.BaseModel, batch: Tensor, state: M.State) -> tuple[Tensor, Tensor]:
        batch = batch.repeat(1, 1, 3, 1, 1).transpose(1, 2)  # (B, C, T, H, W)
        batch_left, batch_right = batch.chunk(2, dim=2)
        return model(batch_left), model(batch_right)

    def compute_loss(self, model: M.BaseModel, batch: Tensor, state: M.State, output: tuple[Tensor, Tensor]) -> Tensor:
        output_left, output_right = output
        dot_prod = output_left @ output_right.transpose(0, 1)
        loss = F.cross_entropy(dot_prod, torch.arange(len(dot_prod), device=dot_prod.device), reduction="none")
        return loss

    def get_dataset(self, phase: M.Phase) -> Dataset:
        return MovingMNIST(
            root=M.get_data_dir() / "MovingMNIST",
            split="train" if phase == "train" else "test",
            split_ratio=10,
            download=True,
            transform=torchvision.transforms.ConvertImageDtype(torch.float32),
        )
