"""Functions for staging environments.

Staged environments freeze a copy of the current Python codebase that can be
run in a separate process without worrying that changes to the current codebase
will affect the staged environment.

To save space, it basically copies over whatever modules are in the current
environment (which are a subset of the codebase) to a new directory.
"""

import datetime
import hashlib
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Sequence
from uuid import uuid4

from ml.utils.timer import Timer

logger = logging.getLogger(__name__)

# Date format for staging environments.
DATE_FORMAT = "%Y-%m-%d"


def stage_environment(
    project_roots: str | Path | Sequence[str | Path],
    stage_dir: Path,
    date_format: str = DATE_FORMAT,
) -> Path:
    """Stages the current environment to a root directory.

    Args:
        project_roots: The root directory of the project.
        stage_dir: The root directory for staging environments.
        date_format: The date format to use for the staging directory.

    Returns:
        The stage environment path

    Raises:
        ValueError: If no project root is found.
    """
    if isinstance(project_roots, (str, Path)):
        project_roots = [Path(project_roots)]

    if not project_roots:
        raise ValueError("No project root directories")

    with Timer("getting files to stage", spinner=True):
        for project_root in project_roots:
            fpaths: list[tuple[Path, Path]] = []
            for module in sys.modules.values():
                if (fpath_str := getattr(module, "__file__", None)) is None:
                    continue
                fpath = Path(fpath_str).resolve()
                try:
                    rel_fpath = fpath.relative_to(Path(project_root).parent)
                    fpaths.append((fpath, rel_fpath))
                except ValueError:
                    pass

    assert fpaths, "Couldn't find any file paths to stage!"

    with Timer("computing hash of current environment", spinner=True):
        hashobj = hashlib.md5()
        for fpath, _ in fpaths:
            with open(fpath, "rb") as f:
                while data := f.read(65536):
                    hashobj.update(data)
        hashval = hashobj.hexdigest()

    date_str = datetime.datetime.now().strftime(date_format)
    out_dir = stage_dir / f"{date_str}-{hashval[:10]}"
    if not out_dir.exists():
        with Timer("copying files to staging directory", spinner=True):
            tmp_dir = stage_dir / ".tmp" / str(uuid4())
            if tmp_dir.parent.exists():
                shutil.rmtree(tmp_dir.parent)
            tmp_dir.mkdir(exist_ok=False, parents=True)
            for fpath, rel_fpath in fpaths:
                new_fpath = tmp_dir / rel_fpath
                new_fpath.parent.mkdir(exist_ok=True, parents=True)
                shutil.copyfile(fpath, new_fpath)
            tmp_dir.rename(out_dir)
            tmp_dir.parent.rmdir()

    with Timer("removing old directories", spinner=True):
        cur_time = datetime.datetime.now()
        for dpath in stage_dir.iterdir():
            dir_age = cur_time - datetime.datetime.fromtimestamp(os.stat(dpath).st_mtime)
            if dir_age > datetime.timedelta(days=14):
                logger.info("Removing old staging directory %s", dpath)
                shutil.rmtree(dpath)

    return out_dir
