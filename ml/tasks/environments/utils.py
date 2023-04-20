import logging
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import tqdm
from torch import Tensor

from ml.tasks.environments.base import Environment
from ml.utils.logging import configure_logging
from ml.utils.video import Writer, write_video

logger: logging.Logger = logging.getLogger(__name__)


def test_environment(
    env: Environment,
    max_steps: int = 100,
    save_path: str | Path | None = None,
    writer: Writer = "ffmpeg",
) -> None:
    """Samples a clip from the environment using a random policy.

    Args:
        env: The environment to test
        max_steps: Maximum number of steps to loop through
        save_path: Where to save the recorded clip
        writer: The video writer to use
    """

    configure_logging(use_tqdm=True)
    time.time()

    def iter_environment() -> Iterator[np.ndarray | Tensor]:
        state = env.reset()
        if env.terminated(state):
            raise RuntimeError("Initial state is terminated")

        for i in tqdm.trange(max_steps):
            if env.terminated(state):
                logger.info("Terminating environment early, after %d / %d steps", i, max_steps)
                break
            action = env.sample_action()
            state = env.step(action)

            if save_path is not None:
                yield env.render(state)

    # Save the video if a path is provided, otherwise just iterate through the
    # samples from the environment.
    if save_path is None:
        iter_environment()
    else:
        write_video(writer, iter_environment(), save_path)
