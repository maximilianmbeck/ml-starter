#!/usr/bin/env python

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    """Copies the empty template repository to a new repository with symlinks.

    Usage:
        python copy_empty.py <new-name>
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("new_name", help="The new repository's name")
    args = parser.parse_args()

    old_path, new_path = Path("empty"), Path(args.new_name)
    assert old_path.is_dir()
    new_path.mkdir(exist_ok=False)

    # First, cleans the empty repository by calling "make clean".
    subprocess.run(["make", "clean"], cwd=old_path)

    # Recursively copies the empty repository, making symlinks for all files.
    for path in old_path.rglob("*"):
        if path.is_file() and not path.is_symlink():
            new_file_path = new_path / path.relative_to(old_path)
            new_file_path.parent.mkdir(parents=True, exist_ok=True)

            # Symlink like "../empty/path/to/file" -> "new/path/to/file".
            # Since the new path is relative to the new repository, we need to
            # go up the directory tree to get to the old repository.
            old_file_path = Path(*([".."] * (len(new_file_path.parts) - 1) + [str(path)]))
            new_file_path.symlink_to(old_file_path)


if __name__ == "__main__":
    main()
