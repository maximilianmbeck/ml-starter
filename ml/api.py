"""Defines an API for the most commonly used components.

This lets you just have a single import like `from ml import api` then use,
for example, `api.conf_field(...)` to access various components.

There wasn't an explicit dictating which functions to include here, it was just
whichever functions seemed like they would be generally useful to have on
hand quickly.
"""

__all__ = [
    "ActivationType",
    "assert_no_nans",
    "AsyncEnvironmentWorker",
    "AsyncIterableDataset",
    "AsyncWorkerPool",
    "atomic_save",
    "AutoDevice",
    "BaseEnvironmentWorker",
    "BaseLRScheduler",
    "BaseLRSchedulerConfig",
    "BaseModel",
    "BaseModelConfig",
    "BaseOptimizer",
    "BaseOptimizerConfig",
    "BaseTask",
    "BaseTaskConfig",
    "BaseTrainer",
    "BaseTrainerConfig",
    "Batch",
    "cached_object",
    "cast_activation_type",
    "cast_init_type",
    "cast_norm_type",
    "cast_reduce_type",
    "ChunkSampler",
    "ClippifyDataset",
    "collate_non_null",
    "collate",
    "colorize",
    "conf_field",
    "configure_logging",
    "Environment",
    "format_timedelta",
    "from_args",
    "get_activation",
    "get_args",
    "get_cache_dir",
    "get_data_dir",
    "get_eval_dir",
    "get_exp_name",
    "get_image_mask",
    "get_log_dir",
    "get_master_addr",
    "get_master_port",
    "get_model_dir",
    "get_norm_1d",
    "get_norm_2d",
    "get_norm_3d",
    "get_norm_linear",
    "get_rank_optional",
    "get_rank",
    "get_type_from_string",
    "get_worker_info",
    "get_world_size_optional",
    "get_world_size",
    "init_",
    "init_empty_weights",
    "InitializationType",
    "is_distributed",
    "is_master",
    "load_model_and_task",
    "Loss",
    "meta_to_empty_func",
    "MultiIterDataset",
    "NormType",
    "open_atomic",
    "Output",
    "Phase",
    "reduce",
    "register_logger",
    "register_lr_scheduler",
    "register_model",
    "register_optimizer",
    "register_task",
    "register_trainer",
    "ReinforcementLearningTask",
    "ReinforcementLearningTaskConfig",
    "State",
    "StreamingDataset",
    "SupervisedLearningTask",
    "SupervisedLearningTaskConfig",
    "SyncEnvironmentWorker",
    "SyncWorkerPool",
    "test_dataset",
    "timeout",
    "Timer",
    "transforms",
    "VIDEO_READERS",
    "VIDEO_WRITERS",
    "VideoFileDataset",
    "WorkerPool",
]

from ml.core.common_types import Batch, Loss, Output
from ml.core.config import conf_field
from ml.core.env import (
    get_cache_dir,
    get_data_dir,
    get_eval_dir,
    get_exp_name,
    get_log_dir,
    get_model_dir,
)
from ml.core.registry import (
    register_logger,
    register_lr_scheduler,
    register_model,
    register_optimizer,
    register_task,
    register_trainer,
)
from ml.core.state import Phase, State
from ml.lr_schedulers.base import BaseLRScheduler, BaseLRSchedulerConfig
from ml.models.activations import ActivationType, cast_activation_type, get_activation
from ml.models.base import BaseModel, BaseModelConfig
from ml.models.init import InitializationType, cast_init_type, init_
from ml.models.norms import (
    NormType,
    cast_norm_type,
    get_norm_1d,
    get_norm_2d,
    get_norm_3d,
    get_norm_linear,
)
from ml.optimizers.base import BaseOptimizer, BaseOptimizerConfig
from ml.tasks.base import BaseTask, BaseTaskConfig
from ml.tasks.datasets import transforms
from ml.tasks.datasets.async_iterable import AsyncIterableDataset
from ml.tasks.datasets.clippify import ClippifyDataset
from ml.tasks.datasets.collate import collate, collate_non_null
from ml.tasks.datasets.multi_iter import MultiIterDataset
from ml.tasks.datasets.samplers import ChunkSampler
from ml.tasks.datasets.streaming import StreamingDataset
from ml.tasks.datasets.utils import test_dataset
from ml.tasks.datasets.video_file import VideoFileDataset
from ml.tasks.environments.base import Environment
from ml.tasks.environments.worker import (
    AsyncEnvironmentWorker,
    AsyncWorkerPool,
    BaseEnvironmentWorker,
    SyncEnvironmentWorker,
    SyncWorkerPool,
    WorkerPool,
)
from ml.tasks.losses.reduce import cast_reduce_type, reduce
from ml.tasks.rl.base import ReinforcementLearningTask, ReinforcementLearningTaskConfig
from ml.tasks.sl.base import SupervisedLearningTask, SupervisedLearningTaskConfig
from ml.trainers.base import BaseTrainer, BaseTrainerConfig
from ml.trainers.mixins.device.auto import AutoDevice
from ml.utils.argparse import from_args, get_args, get_type_from_string
from ml.utils.atomic import atomic_save, open_atomic
from ml.utils.augmentation import get_image_mask
from ml.utils.caching import cached_object
from ml.utils.checks import assert_no_nans
from ml.utils.colors import colorize
from ml.utils.data import get_worker_info
from ml.utils.datetime import format_timedelta
from ml.utils.distributed import (
    get_master_addr,
    get_master_port,
    get_rank,
    get_rank_optional,
    get_world_size,
    get_world_size_optional,
    is_distributed,
    is_master,
)
from ml.utils.io import load_model_and_task
from ml.utils.large_models import init_empty_weights, meta_to_empty_func
from ml.utils.logging import configure_logging
from ml.utils.timer import Timer, timeout
from ml.utils.video import READERS as VIDEO_READERS, WRITERS as VIDEO_WRITERS
