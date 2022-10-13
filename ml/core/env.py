"""Defines any core environment variables used in the models repository.

In order to keep all environment variables in one place, so that they can be
easily referenced, don't use `os.environ` or `os.getenv` outside of this file.
Instead, add a new accessor function to this file.
"""

import os
from pathlib import Path
from typing import List

import torch.distributed as dist


def is_debugging() -> bool:
    return "DEBUG" in os.environ and bool(int(os.environ["DEBUG"]))


def get_cache_dir() -> Path:
    cache_dir = Path(os.environ["CACHE_DIR"]) if "CACHE_DIR" in os.environ else Path.home() / ".cache"
    return cache_dir / "model-artifacts"


def get_run_dir() -> Path:
    return Path(os.environ["RUN_DIR"])


def set_run_dir(run_dir: str | Path) -> None:
    os.environ["RUN_DIR"] = str(Path(run_dir).resolve())


def get_eval_dir() -> Path:
    return Path(os.environ["EVAL_DIR"])


def set_eval_dir(eval_dir: str | Path) -> None:
    os.environ["EVAL_DIR"] = str(Path(eval_dir).resolve())


def get_exp_name() -> str:
    return os.environ.get("EXPERIMENT_NAME", "Experiment")


def set_exp_name(exp_name: str) -> None:
    os.environ["EXPERIMENT_NAME"] = exp_name


def get_data_dir() -> Path:
    return Path(os.environ["DATA_DIR"])


def set_data_dir(data_dir: str | Path) -> None:
    os.environ["DATA_DIR"] = str(Path(data_dir).resolve())


def get_data_cache_dir() -> Path | None:
    if "DATA_CACHE_DIR" in os.environ:
        return Path(os.environ["DATA_CACHE_DIR"])
    return None


def set_data_cache_dir(data_cache_dir: str | Path) -> None:
    os.environ["DATA_CACHE_DIR"] = str(Path(data_cache_dir).resolve())


def get_model_dir() -> Path:
    return Path(os.environ["MODEL_DIR"])


def set_model_dir(model_dir: str | Path) -> None:
    os.environ["MODEL_DIR"] = str(Path(model_dir).resolve())


def get_random_seed() -> int:
    return int(os.getenv("RANDOM_SEED", "1337"))


def set_random_seed(seed: int) -> None:
    os.environ["RANDOM_SEED"] = str(seed)


def get_stage_dir() -> Path:
    if "STAGE_DIR" not in os.environ:
        raise KeyError("Set STAGE_DIR to point at the environment staging directory")
    return Path(os.environ["STAGE_DIR"])


def set_stage_dir(stage_dir: str | Path) -> None:
    os.environ["STAGE_DIR"] = str(Path(stage_dir).resolve())


def get_global_tags() -> List[str]:
    return [tag.strip() for tag in os.environ.get("GLOBAL_MODEL_TAGS", "").split(",")]


def set_global_tags(tags: List[str]) -> None:
    os.environ["GLOBAL_MODEL_TAGS"] = ",".join(sorted(set(tags)))


def add_global_tag(tag: str) -> None:
    set_global_tags(get_global_tags() + [tag])


def get_distributed_backend() -> dist.Backend:
    return dist.Backend(os.environ["TORCH_DISTRIBUTED_BACKEND"])


def set_distributed_backend(backend: dist.Backend) -> None:
    os.environ["TORCH_DISTRIBUTED_BACKEND"] = str(backend)
