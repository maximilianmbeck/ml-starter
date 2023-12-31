"""Defines any core environment variables used in the ML repository.

In order to keep all environment variables in one place, so that they can be
easily referenced, don't use `os.environ` or `os.getenv` outside of this file.
Instead, add a new accessor function to this file.
"""


import os
from pathlib import Path


class _StrEnvVar:
    def __init__(self, key: str, *, default: str | None = None) -> None:
        self.key = key
        self.default = default

    def get(self) -> str:
        value = self.maybe_get()
        assert value is not None, f"Value for {self.key} environment variable is not set"
        return value

    def maybe_get(self) -> str | None:
        return os.environ.get(self.key, self.default)

    def set(self, value: str) -> None:
        os.environ[self.key] = value


class _StrSetEnvVar:
    def __init__(self, key: str, *, sep: str = ",") -> None:
        self.key = key
        self.sep = sep

    def get(self) -> set[str]:
        return {v for v in os.environ.get(self.key, "").split(self.key) if v}

    def set(self, values: set[str]) -> None:
        os.environ[self.key] = self.sep.join(v for v in sorted(values) if v)

    def add(self, value: str) -> None:
        self.set(self.get() | {value})


class _BoolEnvVar:
    def __init__(self, key: str, default: bool = False) -> None:
        self.key = key
        self.default = default

    def get(self) -> bool:
        return bool(int(os.environ.get(self.key, "1" if self.default else "0")))

    def set(self, val: bool) -> None:
        os.environ[self.key] = "1" if val else "0"


class _IntEnvVar:
    def __init__(self, key: str, *, default: int | None = None) -> None:
        self.key = key
        self.default = default

    def get(self) -> int:
        value = self.maybe_get()
        assert value is not None, f"Value for {self.key} environment variable is not set"
        return value

    def maybe_get(self) -> int | None:
        return int(os.environ[self.key]) if self.key in os.environ else self.default

    def set(self, value: int) -> None:
        os.environ[self.key] = str(value)


class _PathEnvVar:
    def __init__(self, key: str, *, default: Path | None = None) -> None:
        self.key = key
        self.default = default

    def get(self) -> Path:
        value = self.maybe_get()
        assert value is not None, f"Value for {self.key} environment variable is not set"
        return value

    def maybe_get(self) -> Path | None:
        value = Path(os.environ[self.key]).resolve() if self.key in os.environ else self.default
        return value

    def set(self, value: Path) -> None:
        os.environ[self.key] = str(value.resolve())


# Option to toggle debug mode (turns off dataloader multiprocessing, improves logging).
Debugging = _BoolEnvVar("DEBUG")
is_debugging = Debugging.get

# Where to store miscellaneous cache artifacts.
CacheDir = _PathEnvVar("CACHE_DIR", default=Path.home() / ".cache" / "ml-starter" / "model-artifacts")
get_cache_dir = CacheDir.get

# Root directory for training runs.
RunDir = _PathEnvVar("RUN_DIR", default=Path.cwd() / "runs")
get_run_dir = RunDir.get
set_run_dir = RunDir.set

# Root directory for evaluation runs.
EvalRunDir = _PathEnvVar("EVAL_RUN_DIR", default=Path.cwd() / "evals")
get_eval_run_dir = EvalRunDir.get
set_eval_run_dir = EvalRunDir.set

# The name of the experiment (set by the training script).
ExpName = _StrEnvVar("EXPERIMENT_NAME", default="Experiment")
get_exp_name = ExpName.get
set_exp_name = ExpName.set

# Base directory where various datasets are stored.
DataDir = _PathEnvVar("DATA_DIR", default=Path.home() / ".cache" / "ml-starter" / "datasets")
get_data_dir = DataDir.get
set_data_dir = DataDir.set

# Base directory where various pretrained models are stored.
ModelDir = _PathEnvVar("MODEL_DIR", default=Path.home() / ".cache" / "ml-starter" / "models")
get_model_dir = ModelDir.get
set_model_dir = ModelDir.set

# The global random seed.
RandomSeed = _IntEnvVar("RANDOM_SEED", default=1337)
get_env_random_seed = RandomSeed.get
set_env_random_seed = RandomSeed.set

# Directory where code is staged before running large-scale experiments.
StageDir = _PathEnvVar("STAGE_DIR", default=Path.home() / ".cache" / "ml-starter" / "staging")
get_stage_dir = StageDir.get
set_stage_dir = StageDir.set

# Global experiment tags (used for the experiment name, among other things).
GlobalTags = _StrSetEnvVar("GLOBAL_MODEL_TAGS")
get_global_tags = GlobalTags.get
set_global_tags = GlobalTags.set
add_global_tag = GlobalTags.add

# Disables using accelerator on Mac.
DisableMetal = _BoolEnvVar("DISABLE_METAL", default=False)
is_metal_disabled = DisableMetal.get

# Disables using the GPU.
DisableGPU = _BoolEnvVar("DISABLE_GPU", default=False)
is_gpu_disabled = DisableGPU.get

# Disables colors in various parts.
DisableColors = _BoolEnvVar("DISABLE_COLORS", default=False)
are_colors_disabled = DisableColors.get

# Disables Tensorboard subprocess.
DisableTensorboard = _BoolEnvVar("DISABLE_TENSORBOARD", default=False)
is_tensorboard_disabled = DisableTensorboard.get

# Show full error message when trying to import a file.
ShowFullImportError = _BoolEnvVar("SHOW_FULL_IMPORT_ERROR", default=False)
should_show_full_import_error = ShowFullImportError.get

# The path to the resolved config.
MLConfigPath = _PathEnvVar("ML_CONFIG_PATH")
get_ml_config_path = MLConfigPath.maybe_get
set_ml_config_path = MLConfigPath.set
