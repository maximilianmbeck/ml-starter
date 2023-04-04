import functools
import logging
import os
import re
from pathlib import Path

from ml.core.state import Phase

logger = logging.getLogger(__name__)


@functools.lru_cache
def get_dataset_root() -> Path:
    if "KINETICS_DATASET_ROOT" not in os.environ:
        raise EnvironmentError("KINETICS_DATASET_ROOT not set")

    root = Path(os.environ["KINETICS_DATASET_ROOT"])
    if not root.is_dir():
        raise FileNotFoundError(f"KINETICS_DATASET_ROOT={root} not found")

    return root


def get_phase_name(phase: Phase) -> str:
    match phase:
        case "train":
            return "train"
        case "valid" | "test":
            return "val"
        case _:
            raise ValueError(f"Invalid phase: {phase}")


def get_kinetics_concepts(phase: Phase) -> dict[str, list[str]]:
    root = get_dataset_root()
    phase_name = get_phase_name(phase)

    # Loads the list of videos.
    with open(root / "lists" / f"{phase_name}.txt", "r", encoding="utf-8") as f:
        assert f.readline().strip() == f"{phase_name}:"

        # Reads the list of concepts.
        concepts: dict[str, list[str]] = {}
        for line in f:
            if not line.strip():
                break
            concepts[line.strip()] = []

        while True:
            line = f.readline()
            if not line:
                break
            if (line_re := re.match(rf"{phase_name}/(.+?):\n", line)) is None:
                raise ValueError(f"Unexpected line: {line}")
            concept = line_re.group(1)
            if concept not in concepts:
                raise ValueError(f"Unexpected concept: {concept}")
            for line in f:
                if not line.strip():
                    break
                concepts[concept].append(line.strip())

    return concepts


def get_kinetics_clips(phase: Phase) -> list[Path]:
    root = get_dataset_root()
    phase_name = get_phase_name(phase)
    concepts = get_kinetics_concepts(phase)
    return [
        # root / f"{phase_name}_288px" / concept_video
        root / phase_name / concept_video
        for concept_videos in concepts.values()
        for concept_video in concept_videos
    ]
