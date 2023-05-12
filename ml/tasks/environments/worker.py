"""Defines workers for reinforcement learning environments."""

import collections
import logging
import queue
import threading
from abc import ABC, abstractmethod
from multiprocessing.managers import SyncManager
from queue import Queue
from typing import TYPE_CHECKING, Deque, Generic, Literal, Sequence, cast, get_args, overload

import torch.multiprocessing as mp

from ml.core.common_types import RLAction, RLState
from ml.utils.logging import configure_logging

if TYPE_CHECKING:
    from ml.tasks.environments.base import Environment

logger = logging.getLogger(__name__)

Mode = Literal["thread", "process"]
SpecialAction = Literal["reset", "close"]
SpecialState = Literal["terminated"]


def clear_queue(q: Queue) -> None:
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            break


def cast_worker_mode(m: str) -> Mode:
    choices = get_args(Mode)
    if m not in choices:
        raise ValueError(f"`{m}` is not a valid mode; choices are {choices}")
    return cast(Mode, m)


class BaseEnvironmentWorker(ABC, Generic[RLState, RLAction]):
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup the worker."""

    def __del__(self) -> None:
        self.cleanup()

    @abstractmethod
    def get_state(self) -> RLState | SpecialState:
        """Returns the current environment states.

        Returns:
            The current environment state.
        """

    @abstractmethod
    def send_action(self, action: RLAction | SpecialAction) -> None:
        """Sends an action to the environment.

        Args:
            action: The action to send to the environment
        """

    @classmethod
    @abstractmethod
    def from_environment(
        cls,
        env: "Environment[RLState, RLAction]",
        num_workers: int,
    ) -> Sequence["BaseEnvironmentWorker[RLState, RLAction]"]:
        """Creates a worker from an environment.

        Args:
            env: The environment to create the worker from.
            num_workers: The number of workers to create.

        Returns:
            The workers.
        """


class SyncEnvironmentWorker(BaseEnvironmentWorker[RLState, RLAction], Generic[RLState, RLAction]):
    def __init__(self, env: "Environment[RLState, RLAction]", seed: int = 1337) -> None:
        """Defines a synchronous environment worker.

        Args:
            env: The environment to wrap.
            seed: The random seed to use.
        """
        super().__init__()

        self.env = env
        self.seed = seed

        self.state: RLState | SpecialState | None = None

    @classmethod
    def from_environment(
        cls,
        env: "Environment[RLState, RLAction]",
        num_workers: int,
    ) -> Sequence["SyncEnvironmentWorker[RLState, RLAction]"]:
        return [cls(env) for _ in range(num_workers)]

    def cleanup(self) -> None:
        pass

    def get_state(self) -> RLState | SpecialState:
        if self.state is None:
            raise RuntimeError("Environment has not been reset")
        return self.state

    def send_action(self, action: RLAction | SpecialAction) -> None:
        if action == "close":
            raise ValueError("Cannot close a synchronous environment")
        if action == "reset":
            self.state = self.env.reset(self.seed)
        else:
            self.state = self.env.step(action)
        if self.env.terminated(self.state):
            self.state = "terminated"


class AsyncEnvironmentWorker(BaseEnvironmentWorker[RLState, RLAction], Generic[RLState, RLAction]):
    def __init__(
        self,
        env: "Environment[RLState, RLAction]",
        manager: SyncManager,
        rank: int | None = None,
        world_size: int | None = None,
        seed: int = 1337,
        cleanup_time: float = 5.0,
        mode: Mode = "process",
        daemon: bool = True,
    ) -> None:
        """Defines an asynchronous environment worker.

        This worker either runs in a separate thread or process, and is used to
        asynchronously interact with an environment. This is useful for
        environments that are slow to interact with, such as a simulator.

        Args:
            env: The environment to wrap.
            manager: The manager to use for shared memory.
            rank: The rank of the worker.
            world_size: The number of workers.
            seed: The random seed to use.
            cleanup_time: The time to wait for the worker to finish before killing it.
            mode: The mode to use for the worker.
            daemon: Whether to run the worker as a daemon.

        Raises:
            ValueError: If the mode is invalid.
        """
        super().__init__()

        self.cleanup_time = cleanup_time
        self.rank = 0 if rank is None else rank
        self.world_size = 1 if world_size is None else world_size

        self.action_queue: "Queue[RLAction | SpecialAction]" = manager.Queue(maxsize=1)
        self.state_queue: "Queue[RLState | SpecialState]" = manager.Queue(maxsize=1)
        args = env, seed, self.action_queue, self.state_queue, rank, world_size

        self._proc: threading.Thread | mp.Process
        if mode == "thread":
            self._proc = threading.Thread(target=self._thread, args=args, daemon=daemon)
            self._proc.start()
        elif mode == "process":
            self._proc = mp.Process(target=self._thread, args=args, daemon=daemon)
            self._proc.start()
        else:
            raise ValueError(f"Invalid mode: {mode}")

    @classmethod
    def from_environment(
        cls,
        env: "Environment[RLState, RLAction]",
        num_workers: int,
    ) -> Sequence["AsyncEnvironmentWorker[RLState, RLAction]"]:
        manager = mp.Manager()
        return [cls(env, manager, rank=rank, world_size=num_workers) for rank in range(num_workers)]

    def cleanup(self) -> None:
        logger.debug("Cleaning up task...")
        try:
            self.send_action("close")
            self._proc.join(timeout=self.cleanup_time)
            if self._proc.is_alive():
                logger.warning("Process failed to finish after %.2f seconds; killing", self.cleanup_time)
                if isinstance(self._proc, threading.Thread):
                    self._proc._stop()
                else:
                    self._proc.kill()
        except Exception:
            pass

    @classmethod
    def _thread(
        cls,
        env: "Environment[RLState, RLAction]",
        seed: int,
        action_queue: "Queue[RLAction | SpecialAction]",
        state_queue: "Queue[RLState | SpecialState]",
        rank: int | None,
        world_size: int | None,
    ) -> None:
        configure_logging(rank=rank, world_size=world_size)

        while True:
            action = action_queue.get()
            if action == "close":
                logger.debug("Got close action; exiting")
                break
            if action == "reset":
                logger.debug("Got reset action; resetting environment")
                state = env.reset(seed)
            else:
                state = env.step(action)
            if env.terminated(state):
                state_queue.put("terminated")
            else:
                state_queue.put(state)

    def get_state(self) -> RLState | SpecialState:
        return self.state_queue.get()

    def send_action(self, action: RLAction | SpecialAction) -> None:
        if action in ("reset", "close"):
            clear_queue(self.state_queue)
            clear_queue(self.action_queue)
        self.action_queue.put(action)


class WorkerPool(Generic[RLState, RLAction]):
    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @overload
    def get_state(self, timeout: float) -> tuple[RLState | SpecialState, int] | None:
        ...

    @overload
    def get_state(self, timeout: None = None) -> tuple[RLState | SpecialState, int]:
        ...

    def get_state(self, timeout: float | None = None) -> tuple[RLState | SpecialState, int] | None:
        return self._get_state_impl(timeout)

    @abstractmethod
    def _get_state_impl(self, timeout: float | None = None) -> tuple[RLState | SpecialState, int] | None:
        """Returns the current state for one of the workers.

        Args:
            timeout: The timeout for getting the worker state.

        Returns:
            The worker state, or None if we timed out.
        """

    @abstractmethod
    def send_action(self, action: RLAction | SpecialAction, worker_id: int) -> None:
        """Sends an action to the given worker.

        Args:
            action: The action to send.
            worker_id: The ID of the worker to send the action to.
        """

    @classmethod
    @abstractmethod
    def from_workers(
        cls,
        workers: Sequence[BaseEnvironmentWorker[RLState, RLAction]],
    ) -> "WorkerPool[RLState, RLAction]":
        """Creates a worker pool from a list of workers.

        Args:
            workers: The list of workers.

        Returns:
            The worker pool.
        """


class SyncWorkerPool(WorkerPool[RLState, RLAction], Generic[RLState, RLAction]):
    def __init__(self, workers: Sequence[BaseEnvironmentWorker[RLState, RLAction]]) -> None:
        super().__init__()

        self.workers = workers
        self.worker_queue: Deque[int] = collections.deque(range(len(self.workers)))

    def reset(self) -> None:
        for worker in self.workers:
            worker.send_action("reset")

    def __len__(self) -> int:
        return len(self.workers)

    def _get_state_impl(self, timeout: float | None = None) -> tuple[RLState | SpecialState, int] | None:
        i = self.worker_queue.popleft()
        self.worker_queue.append(i)
        return self.workers[i].get_state(), i

    def send_action(self, action: RLAction | SpecialAction, worker_id: int) -> None:
        self.workers[worker_id].send_action(action)

    @classmethod
    def from_workers(
        cls,
        workers: Sequence[BaseEnvironmentWorker[RLState, RLAction]],
    ) -> "SyncWorkerPool[RLState, RLAction]":
        return cls(workers)


class AsyncWorkerPool(WorkerPool[RLState, RLAction], Generic[RLState, RLAction]):
    def __init__(self, workers: Sequence[BaseEnvironmentWorker[RLState, RLAction]], daemon: bool = True) -> None:
        super().__init__()

        self.workers = workers
        self.manager = mp.Manager()
        self.state_queue: "Queue[tuple[RLState | SpecialState, int]]" = self.manager.Queue(maxsize=len(workers))
        self.action_queues: list["Queue[RLAction | SpecialAction]"] = [
            self.manager.Queue(maxsize=1) for _ in range(len(workers))
        ]

        # Starts a thread for each worker.
        self._procs = [
            threading.Thread(
                target=self._thread,
                args=(env_id, worker, self.state_queue, action_queue),
                daemon=daemon,
            )
            for env_id, (worker, action_queue) in enumerate(zip(workers, self.action_queues))
        ]
        for proc in self._procs:
            proc.start()

    def cleanup(self) -> None:
        logger.debug("Cleaning up worker pool...")
        try:
            clear_queue(self.state_queue)
            for action_queue in self.action_queues:
                clear_queue(action_queue)
                action_queue.put("close")
            for proc in self._procs:
                proc.join()
        except Exception:
            pass

    def reset(self) -> None:
        clear_queue(self.state_queue)
        for action_queue in self.action_queues:
            clear_queue(action_queue)
            action_queue.put("reset")

    def __del__(self) -> None:
        self.cleanup()

    def __len__(self) -> int:
        return len(self.workers)

    @classmethod
    def _thread(
        cls,
        env_id: int,
        worker: BaseEnvironmentWorker[RLState, RLAction],
        state_queue: "Queue[tuple[RLState | SpecialState, int]]",
        action_queue: "Queue[RLAction | SpecialAction]",
    ) -> None:
        logger.debug("Starting worker pool thread")

        while True:
            action = action_queue.get()
            if action == "close":
                logger.debug("Got None action; exiting thread")
                worker.cleanup()
                break
            worker.send_action(action)
            state = worker.get_state()
            state_queue.put((state, env_id))

    def _get_state_impl(self, timeout: float | None = None) -> tuple[RLState | SpecialState, int] | None:
        if timeout is None:
            return self.state_queue.get()

        try:
            return self.state_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def send_action(self, action: RLAction | SpecialAction, worker_id: int) -> None:
        self.action_queues[worker_id].put(action)

    @classmethod
    def from_workers(
        cls,
        workers: Sequence[BaseEnvironmentWorker[RLState, RLAction]],
    ) -> "AsyncWorkerPool[RLState, RLAction]":
        return cls(workers)


def get_worker_pool(
    workers: Sequence[BaseEnvironmentWorker[RLState, RLAction]],
    force_sync: bool = False,
) -> WorkerPool[RLState, RLAction]:
    if (len(workers) == 1 and isinstance(workers[0], SyncEnvironmentWorker)) or force_sync:
        return SyncWorkerPool(workers)
    return AsyncWorkerPool(workers)
