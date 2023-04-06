import logging
from dataclasses import dataclass
from typing import Callable

import ml.api as ml
from omegaconf import MISSING
from torch import Tensor
from torchvision.models.video.resnet import VideoResNet, mc3_18, r2plus1d_18, r3d_18

logger = logging.getLogger(__name__)

MODELS: dict[str, Callable[[bool], VideoResNet]] = {
    "r3d_18": r3d_18,
    "mc3_18": mc3_18,
    "r2plus1d_18": r2plus1d_18,
}


@dataclass
class VideoResNetModelConfig(ml.BaseModelConfig):
    size: str = ml.conf_field(MISSING, help="Video ResNet size to use")
    pretrained: bool = ml.conf_field(True, help="Load pretrained model")

    @classmethod
    def get_defaults(cls) -> dict[str, "VideoResNetModelConfig"]:
        return {key: VideoResNetModelConfig(size=key) for key in MODELS}


@ml.register_model("video_resnet", VideoResNetModelConfig)
class VideoResNetModel(ml.BaseModel[VideoResNetModelConfig]):
    def __init__(self, config: VideoResNetModelConfig) -> None:
        super().__init__(config)

        if config.size not in MODELS:
            raise KeyError(f"Invalid model size: {config.size} Choices are: {sorted(MODELS.keys())}")

        self.model = MODELS[config.size](config.pretrained)

    def forward(self, image: Tensor) -> Tensor:
        return self.model(image)


def run_video_resnet_adhoc_test() -> None:
    """Runs a quick test of the VideoResNet model.

    Usage:
        python -m ml.models.video_resnet
    """

    ml.configure_logging()
    config = VideoResNetModelConfig(size="r3d_18")
    model = VideoResNetModel(config)
    logger.info("Model: %s", model)


if __name__ == "__main__":
    run_video_resnet_adhoc_test()
