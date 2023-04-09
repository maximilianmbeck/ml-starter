from pathlib import Path


def is_relative_to(a: Path, b: Path) -> bool:
    try:
        a.relative_to(b)
        return True
    except ValueError:
        return False


def get_relative_path(a: Path, bs: list[Path], parent: bool = False) -> Path:
    for b in bs:
        try:
            return a.relative_to(b.parent if parent else b)
        except ValueError:
            pass
    raise ValueError(f"Path {a} is not relative to any of {bs}")
