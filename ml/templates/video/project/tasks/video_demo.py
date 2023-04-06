import logging
from dataclasses import dataclass

import ml.api as ml
import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor
from torchvision.datasets.moving_mnist import MovingMNIST

from project.models.video_resnet import VideoResNetModel

logger = logging.getLogger(__name__)


@dataclass
class VideoDemoTaskConfig(ml.SupervisedLearningTaskConfig):
    split_ratio: int = ml.conf_field(10, help="Dataset split ratio")


@ml.register_task("video_demo", VideoDemoTaskConfig)
class VideoDemoTask(
    ml.SupervisedLearningTask[
        VideoDemoTaskConfig,
        VideoResNetModel,
        Tensor,
        tuple[Tensor, Tensor],
        Tensor,
    ],
):
    def __init__(self, config: VideoDemoTaskConfig) -> None:
        super().__init__(config)

    def run_model(self, model: VideoResNetModel, batch: Tensor, state: ml.State) -> tuple[Tensor, Tensor]:
        batch = batch.repeat(1, 1, 3, 1, 1).transpose(1, 2)  # (B, C, T, H, W)
        batch_left, batch_right = batch.chunk(2, dim=2)
        return model(batch_left), model(batch_right)

    def compute_loss(
        self,
        model: VideoResNetModel,
        batch: Tensor,
        state: ml.State,
        output: tuple[Tensor, Tensor],
    ) -> Tensor:
        output_left, output_right = output
        dot_prod = output_left @ output_right.transpose(0, 1)
        loss = F.cross_entropy(dot_prod, torch.arange(len(dot_prod), device=dot_prod.device), reduction="none")
        if state.phase == "valid":
            self.logger.log_videos("sample", batch)
        return loss

    def get_dataset(self, phase: ml.Phase) -> MovingMNIST:
        return MovingMNIST(
            root=ml.get_data_dir() / "MovingMNIST",
            split="train" if phase == "train" else "test",
            split_ratio=self.config.split_ratio,
            download=True,
            transform=torchvision.transforms.ConvertImageDtype(torch.float32),
        )
