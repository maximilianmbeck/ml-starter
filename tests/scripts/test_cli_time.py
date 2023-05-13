"""Test to ensure that the CLI startup time doesn't get annoyingly slow.

This basically just imports the main CLI script and ensures that it doesn't
depend on any of a number of heavy imports, like Torch.
"""

import inspect
import sys
from collections import deque
from typing import Deque

import pytest

DISALLOWED_IMPORTS = {
    "numpy",
    "omegaconf",
    "torch",
    "torchaudio",
    "torchvision",
    "tqdm",
}


@pytest.mark.slow
def test_cli_import_time() -> None:
    """Tests that `ml.scripts.cli` never imports heavy dependencies, like Torch.

    Raises:
        ValueError: If `ml.scripts.cli` imports a disallowed module.
    """
    visited: set[str] = set()
    queue: Deque[str] = deque()
    queue.append("ml.scripts.cli")

    while queue:
        module_name = queue.popleft()
        if module_name in visited:
            continue

        __import__(module_name)
        visited.add(module_name)
        module = sys.modules.get(module_name)
        if module is None:
            continue

        for _, obj in inspect.getmembers(module):
            if inspect.ismodule(obj):
                queue.append(obj.__name__)
            elif inspect.isclass(obj):
                queue.append(obj.__module__)

        if module_name in DISALLOWED_IMPORTS:
            raise ValueError(f"Module {module_name} was imported by ml.scripts.cli")
