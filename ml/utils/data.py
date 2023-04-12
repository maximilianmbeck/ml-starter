from dataclasses import dataclass
from typing import Sequence, TypeVar

from torch.utils.data.dataloader import get_worker_info as _get_worker_info_base

from ml.core.state import Phase

T = TypeVar("T")


@dataclass
class WorkerInfo:
    worker_id: int
    num_workers: int
    in_worker: bool


def get_worker_info() -> WorkerInfo:
    """Gets a typed worker info object which always returns a value.

    Returns:
        The typed worker info object
    """

    if (worker_info := _get_worker_info_base()) is None:
        return WorkerInfo(
            worker_id=0,
            num_workers=1,
            in_worker=False,
        )

    return WorkerInfo(
        worker_id=worker_info.id,
        num_workers=worker_info.num_workers,
        in_worker=True,
    )


def get_dataset_splits(
    items: Sequence[T],
    valid: float | int,
    test: float | int,
) -> tuple[Sequence[T], Sequence[T], Sequence[T]]:
    """Splits a list of items into three sub-lists for train, valid, and test.

    Args:
        items: The list of items to split.
        valid: If a value between 0 and 1, the fraction of items to use for
            the validation set, otherwise the number of items to use for the
            validation set.
        test: If a value between 0 and 1, the fraction of items to use for
            the test set, otherwise the number of items to use for the test
            set.

    Returns:
        A tuple of three lists, one for each phase.

    Raises:
        ValueError: If the split sizes would be invalid.
    """

    num_items = len(items)

    # Converts a fraction to an integer number of items.
    if isinstance(valid, float):
        if 0 > valid or valid > 1:
            raise ValueError(f"Valid fraction must be between 0 and 1, got {valid}")
        valid = int(num_items * valid)
    if isinstance(test, float):
        if 0 > test or test > 1:
            raise ValueError(f"Test fraction must be between 0 and 1, got {test}")
        test = int(num_items * test)

    if valid + test > num_items:
        raise ValueError(f"Invalid number of items: {num_items}, valid: {valid}, test: {test}")

    train_items = items[: num_items - valid - test]
    valid_items = items[num_items - valid - test : num_items - test]
    test_items = items[num_items - test :]

    return train_items, valid_items, test_items


def get_dataset_split_for_phase(
    items: Sequence[T],
    phase: Phase,
    valid: float | int,
    test: float | int,
) -> Sequence[T]:
    """Gets the items for a given phase.

    Args:
        items: The list of items to split.
        phase: The phase to get the items for.
        valid: If a value between 0 and 1, the fraction of items to use for
            the validation set, otherwise the number of items to use for the
            validation set.
        test: If a value between 0 and 1, the fraction of items to use for
            the test set, otherwise the number of items to use for the test
            set.

    Returns:
        The items for the given phase.

    Raises:
        ValueError: If the phase is not valid.
    """

    train_items, valid_items, test_items = get_dataset_splits(items, valid, test)

    match phase:
        case "train":
            return train_items
        case "valid":
            return valid_items
        case "test":
            return test_items
        case _:
            raise ValueError(f"Invalid phase: {phase}")
