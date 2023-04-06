import functools
import logging
import multiprocessing as mp
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Iterator, TypeVar, overload

import numpy as np
import torch
import tqdm
from omegaconf import MISSING
from torch import Tensor
from torch.utils.data.dataset import Dataset

from ml.core.common_types import Loss, Output, RLAction, RLState
from ml.core.config import conf_field
from ml.core.state import State
from ml.tasks.base import BaseTask, BaseTaskConfig, ModelT
from ml.tasks.environments.base import Environment
from ml.tasks.environments.worker import (
    AsyncEnvironmentWorker,
    BaseEnvironmentWorker,
    SpecialState,
    SyncEnvironmentWorker,
    WorkerPool,
    cast_worker_mode,
)
from ml.tasks.rl.replay import MultiReplaySamples, ReplayDataset, ReplaySamples
from ml.utils.video import WRITERS as VIDEO_WRITERS, Writer, standardize_image

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentConfig:
    num_env_workers: int = conf_field(1, help="Number of environment workers (0 to run synchronously)")
    env_worker_mode: str = conf_field("process", help="Mode for running environment worker")
    env_seed: int = conf_field(1337, help="Environment seed")
    env_cleanup_time: float = conf_field(5.0, help="Cleanup time for the async environment worker")
    max_steps: int = conf_field(1_000, help="Maximum number of steps in a clip")


@dataclass
class DatasetConfig:
    num_samples: int = conf_field(1, help="Number of training samples in replay")
    num_update_steps: int = conf_field(MISSING, help="How often to interact with the environment")
    stride: int = conf_field(1, help="Replay stride to use")
    replay_buffer_sample_size: int = conf_field(10000, help="Number of epochs of experience to keep in replay buffer")


@dataclass
class ReinforcementLearningTaskConfig(BaseTaskConfig):
    environment: EnvironmentConfig = conf_field(EnvironmentConfig())
    dataset: DatasetConfig = conf_field(DatasetConfig())


ReinforcementLearningTaskConfigT = TypeVar("ReinforcementLearningTaskConfigT", bound=ReinforcementLearningTaskConfig)


class ReinforcementLearningTask(
    BaseTask[ReinforcementLearningTaskConfigT, ModelT, tuple[RLState, RLAction], Output, Loss],
    Generic[ReinforcementLearningTaskConfigT, ModelT, RLState, RLAction, Output, Loss],
    ABC,
):
    @abstractmethod
    def get_actions(self, model: ModelT, states: list[RLState], optimal: bool) -> list[RLAction]:
        """Samples an action from the policy, given the previous state.

        Args:
            model: The model to sample from.
            states: The previous states.
            optimal: Whether to get the optimal action or to sample from the policy.

        Returns:
            The next actions to take for each state.
        """

    @abstractmethod
    def get_environment(self) -> Environment[RLState, RLAction]:
        """Returns the environment for the task.

        Returns:
            The environment for the task
        """

    def build_rl_dataset(
        self, samples: MultiReplaySamples[tuple[RLState, RLAction]]
    ) -> Dataset[tuple[RLState, RLAction]]:
        return ReplayDataset(
            samples,
            clip_size=self.config.dataset.num_samples,
            stride=self.config.dataset.stride,
        )

    @functools.lru_cache
    def get_environment_cached(self) -> Environment[RLState, RLAction]:
        return self.get_environment()

    def get_environment_workers(self, force_sync: bool = False) -> list[BaseEnvironmentWorker[RLState, RLAction]]:
        env_cfg = self.config.environment

        if env_cfg.num_env_workers <= 0 or force_sync:
            return [
                SyncEnvironmentWorker(self.get_environment(), seed=env_cfg.env_seed)
                for _ in range(max(1, env_cfg.num_env_workers))
            ]

        manager = mp.Manager()

        return [
            AsyncEnvironmentWorker(
                self.get_environment(),
                manager,
                rank=rank,
                world_size=env_cfg.num_env_workers,
                seed=env_cfg.env_seed,
                cleanup_time=env_cfg.env_cleanup_time,
                mode=cast_worker_mode(env_cfg.env_worker_mode),
            )
            for rank in range(env_cfg.num_env_workers)
        ]

    def postprocess_trajectory(self, samples: list[tuple[RLState, RLAction]]) -> list[tuple[RLState, RLAction]]:
        """Performs any global postprocessing on the trajectory.

        Args:
            samples: The trajectory to postprocess.

        Returns:
            The postprocessed trajectory.
        """

        return samples

    def postprocess_trajectories(
        self,
        trajectories: list[list[tuple[RLState, RLAction]]],
    ) -> list[list[tuple[RLState, RLAction]]]:
        """Performs any global postprocessing on all of the trajectories.

        Args:
            trajectories: The trajectories to postprocess.

        Returns:
            The postprocessed trajectories.
        """

        return trajectories

    def collect_samples(
        self,
        model: ModelT,
        worker_pool: WorkerPool[RLState, RLAction],
        total_samples: int,
        min_trajectory_length: int = 1,
        max_trajectory_length: int | None = None,
        min_batch_size: int = 1,
        max_batch_size: int | None = None,
        max_wait_time: float | None = None,
        use_tqdm: bool = True,
    ) -> MultiReplaySamples[tuple[RLState, RLAction]]:
        """Collects samples from the environment.

        Args:
            model: The model to sample from.
            worker_pool: The pool of workers for the environment
            total_samples: The total number of samples to collect.
            min_trajectory_length: Minimum sequence length to consider a
                sequence as having contributed to `total_samples`
            max_trajectory_length: Maximum sequence length to consider a
                sequence as having contributed to `total_samples`
            min_batch_size: Minimum batch size for doing inference on model
            max_batch_size: Maximum batch size for doing inference on model
            max_wait_time: Maximum amount of time to wait to build batch
            use_tqdm: Whether to use tqdm to display progress

        Returns:
            A list of lists of samples, where each list is a trajectory.

        Raises:
            ValueError: If `min_batch_size` is greater than `max_batch_size`.
        """

        min_trajectory_length = max(min_trajectory_length, self.config.dataset.num_samples, 1)
        num_samples = 0

        worker_pool.reset()
        trajectories: list[list[tuple[RLState, RLAction]]] = [[] for _ in range(len(worker_pool))]
        all_trajectories: list[list[tuple[RLState, RLAction]]] = []
        max_batch_size = len(worker_pool) if max_batch_size is None else min(max_batch_size, len(worker_pool))

        if min_trajectory_length > total_samples:
            raise ValueError(f"{min_trajectory_length=} > {total_samples=}")

        if min_batch_size > max_batch_size:
            raise ValueError(f"{min_batch_size=} > {max_batch_size=}")

        with tqdm.tqdm(total=total_samples, disable=not use_tqdm, desc="Sampling") as pbar, torch.no_grad():
            while num_samples < total_samples:
                start_time = time.time()

                # Wait for new samples to be ready.
                batch: list[tuple[RLState, int]] = []
                batch_special: list[tuple[SpecialState, int]] = []
                while len(batch) + len(batch_special) < max_batch_size:
                    elapsed_time = time.time() - start_time
                    if max_wait_time is not None and elapsed_time > max_wait_time and len(batch) >= min_batch_size:
                        break
                    state, worker_id = worker_pool.get_state()
                    pbar.update()  # Update every time we get a new state.
                    if state == "terminated":
                        if len(trajectories[worker_id]) >= min_trajectory_length:
                            all_trajectories.append(self.postprocess_trajectory(trajectories[worker_id]))
                            num_samples += len(trajectories[worker_id])
                        else:
                            logger.warning(
                                "Discarding trajectory of length %d because it is less than %d",
                                len(trajectories[worker_id]),
                                min_trajectory_length,
                            )
                        trajectories[worker_id] = []
                        batch_special.append((state, worker_id))
                    else:
                        batch.append((state, worker_id))

                # Sample actions for the new samples
                states, worker_ids = [state for state, _ in batch], [worker_id for _, worker_id in batch]
                actions = self.get_actions(model, states, False) if states else []

                # Send the actions to the workers
                trajectory_lengths = 0
                for state, action, worker_id in zip(states, actions, worker_ids):
                    if max_trajectory_length is not None and len(trajectories[worker_id]) >= max_trajectory_length:
                        all_trajectories.append(self.postprocess_trajectory(trajectories[worker_id]))
                        num_samples += len(trajectories[worker_id])
                        trajectories[worker_id] = []
                        worker_pool.send_action("reset", worker_id)
                    else:
                        trajectories[worker_id].append((state, action))
                        trajectory_len = len(trajectories[worker_id])
                        if trajectory_len >= min_trajectory_length:
                            trajectory_lengths += trajectory_len
                        worker_pool.send_action(action, worker_id)
                for state, worker_id in batch_special:
                    if state == "terminated":
                        worker_pool.send_action("reset", worker_id)
                    else:
                        raise ValueError(f"Unknown special state {state}")

                # If the current trajectories would finish the episode, then
                # add them to the list of all trajectories.
                if num_samples + trajectory_lengths >= total_samples:
                    all_trajectories.extend(
                        self.postprocess_trajectory(t) for t in trajectories if len(t) >= min_trajectory_length
                    )
                    num_samples += trajectory_lengths
                    pbar.update(trajectory_lengths)
                    break

        all_trajectories = self.postprocess_trajectories(all_trajectories)
        logger.info("Collected %d total samples and %d trajectories", num_samples, len(all_trajectories))

        return MultiReplaySamples([ReplaySamples(t) for t in all_trajectories])

    def epoch_is_over(self, state: State) -> bool:
        return state.num_epoch_steps >= self.config.dataset.num_update_steps

    @overload
    def sample_clip(
        self,
        *,
        save_path: str | Path,
        model: ModelT | None = None,
        writer: Writer = "ffmpeg",
        use_tqdm: bool = True,
    ) -> None:
        ...

    @overload
    def sample_clip(
        self,
        *,
        model: ModelT | None = None,
        use_tqdm: bool = True,
    ) -> Tensor:
        ...

    def sample_clip(
        self,
        *,
        save_path: str | Path | None = None,
        model: ModelT | None = None,
        writer: Writer = "ffmpeg",
        use_tqdm: bool = True,
    ) -> Tensor | None:
        """Samples a clip for a given model.

        Args:
            save_path: Where to save the sampled clips
            model: The model to sample from; if not provided, samples actions
                randomly from the model
            writer: The writer to use to save the clip
            use_tqdm: Whether to use tqdm to display the progress

        Returns:
            The sampled clip, if `save_path` is not provided, otherwise `None`
            (the clip is saved to `save_path`).
        """

        env_cfg = self.config.environment

        def iter_environment() -> Iterator[np.ndarray | Tensor]:
            environment = self.get_environment_cached()
            state = environment.reset()
            if environment.terminated(state):
                raise RuntimeError("Initial state is terminated")
            iterator = tqdm.trange(env_cfg.max_steps) if use_tqdm else range(env_cfg.max_steps)
            for i in iterator:
                if environment.terminated(state):
                    logger.info("Terminating environment early, after %d / %d steps", i, env_cfg.max_steps)
                    break
                if model is None:
                    action = environment.sample_action()
                else:
                    (action,) = self.get_actions(model, [state], True)
                state = environment.step(action)
                yield environment.render(state)

        if save_path is None:
            images_np = [standardize_image(image) for image in iter_environment()]
            return torch.from_numpy(np.stack(images_np))

        VIDEO_WRITERS[writer](iter_environment(), save_path)
        return None