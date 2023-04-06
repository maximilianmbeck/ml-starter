import multiprocessing as mp
import os
from dataclasses import dataclass
from typing import Type, cast

import numpy as np
import psutil
import pytest
import tqdm
from torch import Tensor

from ml.tasks.environments.base import Environment
from ml.tasks.environments.worker import (
    AsyncEnvironmentWorker,
    AsyncWorkerPool,
    BaseEnvironmentWorker,
    SyncEnvironmentWorker,
    SyncWorkerPool,
    WorkerPool,
)


@dataclass
class DummyState:
    obs: int
    reward: float
    terminated: bool


@dataclass
class DummyAction:
    choice: int


class DummyEnvironment(Environment[DummyState, DummyAction]):
    def __init__(self, num_actions: int) -> None:
        super().__init__()

        self.num_actions = num_actions

    def reset(self, seed: int | None = None) -> DummyState:
        return DummyState(obs=0, reward=0.0, terminated=False)

    def render(self, state: DummyState) -> np.ndarray | Tensor:
        return np.array([state.obs])

    def sample_action(self) -> DummyAction:
        raise NotImplementedError

    def step(self, action: DummyAction) -> DummyState:
        return DummyState(
            obs=action.choice,
            reward=float(action.choice),
            terminated=action.choice >= self.num_actions - 1,
        )

    def terminated(self, state: DummyState) -> bool:
        return state.terminated


@pytest.mark.parametrize("worker_cls", [SyncEnvironmentWorker, AsyncEnvironmentWorker])
@pytest.mark.parametrize("pool_cls", [SyncWorkerPool, AsyncWorkerPool])
@pytest.mark.parametrize("num_workers", [3])
@pytest.mark.parametrize("num_actions", [10])
def test_worker_and_pool(
    worker_cls: Type[BaseEnvironmentWorker],
    pool_cls: Type[WorkerPool],
    num_workers: int,
    num_actions: int,
) -> None:
    """Test that the worker and pool classes work as expected.

    Args:
        worker_cls: The worker class to test.
        pool_cls: The pool class to test.
        num_workers: The number of workers to use.
        num_actions: The number of actions to take.
    """

    env = DummyEnvironment(num_actions=num_actions)
    workers = cast(list[BaseEnvironmentWorker[DummyState, DummyAction]], worker_cls.from_environment(env, num_workers))
    pool = cast(WorkerPool[DummyState, DummyAction], pool_cls.from_workers(workers))

    # Tests sending and receiving actions.
    pool.reset()
    states: list[list[DummyState]] = [[] for _ in range(num_workers)]
    not_terminated = set(range(num_workers))
    while not_terminated:
        state, worker_id = pool.get_state()
        if state == "terminated":
            not_terminated.remove(worker_id)
            continue
        states[worker_id].append(state)
        action = DummyAction(choice=state.obs + 1)
        pool.send_action(action, worker_id)

    assert all(len(s) == num_actions - 1 for s in states)
    assert all(all(s.obs == i for i, s in enumerate(s_)) for s_ in states)


@pytest.mark.skip(reason="This test is too slow to run on CI, run using CLI instead.")
@pytest.mark.parametrize("num_workers", [3])
def test_for_memory_leaks(num_workers: int) -> None:
    """Repeatedly creates and destroys a worker pool to check for memory leaks.

    This is only relevent for async worker pools.

    Args:
        num_workers: The number of workers to use.
    """

    process = psutil.Process(os.getpid())

    info = process.memory_info()
    init_rss = info.rss
    init_vms = info.vms
    init_shared = getattr(info, "shared", 0)

    env = DummyEnvironment(num_actions=10)
    for i in tqdm.trange(10):
        manager = mp.Manager()
        workers = [AsyncEnvironmentWorker(env, manager, mode="process") for _ in range(num_workers)]
        pool = AsyncWorkerPool.from_workers(workers)
        pool.reset()
        for worker_id in range(num_workers):
            pool.send_action(DummyAction(choice=0), worker_id)
        for _ in range(num_workers):
            state, worker_id = pool.get_state()
            assert state != "terminated"
        del workers
        del pool

        info = process.memory_info()
        mems = [
            (info.rss, init_rss, "rss"),
            (info.vms, init_vms, "vms"),
            (getattr(info, "shared", 0), init_shared, "shared"),
        ]
        tqdm.tqdm.write(f"----- Iteration {i} -----")
        for mem, init_mem, kind in mems:
            tqdm.tqdm.write(f"[{kind}] mem: {mem}, init_mem: {init_mem}, diff: {mem - init_mem}")


if __name__ == "__main__":
    # python -m tests.tasks.environments.test_workers
    test_for_memory_leaks(3)
