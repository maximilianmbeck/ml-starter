import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor
from torchvision.datasets.moving_mnist import MovingMNIST

import ml.api as M
from ml.models.video_resnet import VideoResNetModel

logger = logging.getLogger(__name__)


@dataclass
class VideoDemoTaskConfig(M.BaseTaskConfig):
    split_ratio: int = M.conf_field(10, help="Dataset split ratio")


@M.register_task("video_demo", VideoDemoTaskConfig)
class VideoDemoTask(M.BaseTask[VideoDemoTaskConfig, VideoResNetModel, Tensor, tuple[Tensor, Tensor], Tensor]):
    def __init__(self, config: VideoDemoTaskConfig) -> None:
        super().__init__(config)

    def run_model(self, model: VideoResNetModel, batch: Tensor, state: M.State) -> tuple[Tensor, Tensor]:
        batch = batch.repeat(1, 1, 3, 1, 1).transpose(1, 2)  # (B, C, T, H, W)
        batch_left, batch_right = batch.chunk(2, dim=2)
        return model(batch_left), model(batch_right)

    def compute_loss(
        self,
        model: VideoResNetModel,
        batch: Tensor,
        state: M.State,
        output: tuple[Tensor, Tensor],
    ) -> Tensor:
        output_left, output_right = output
        dot_prod = output_left @ output_right.transpose(0, 1)
        loss = F.cross_entropy(dot_prod, torch.arange(len(dot_prod), device=dot_prod.device), reduction="none")
        return loss

    def get_dataset(self, phase: M.Phase) -> MovingMNIST:
        return MovingMNIST(
            root=M.get_data_dir() / "MovingMNIST",
            split="train" if phase == "train" else "test",
            split_ratio=self.config.split_ratio,
            download=True,
            transform=torchvision.transforms.ConvertImageDtype(torch.float32),
        )
