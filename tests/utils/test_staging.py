import sys
import types
from pathlib import Path

from ml.utils.staging import stage_environment


def test_stage_environment(tmpdir: Path) -> None:
    project_dir, stage_dir = Path(tmpdir / "project"), Path(tmpdir / "stage")
    project_dir.mkdir()
    stage_dir.mkdir()

    for fname in ["a.py", "b.py", "c.py"]:
        with open(project_dir / fname, "w") as f:
            f.write("print('hello world')")

    # Manually adds the new files to `sys.modules`.
    for fname in ["a.py", "b.py", "c.py"]:
        module = types.ModuleType(fname)
        module.__file__ = str(project_dir / fname)
        sys.modules[fname] = module

    stage_dir = stage_environment(project_dir, stage_dir)

    for fname in ["a.py", "b.py", "c.py"]:
        assert (stage_dir / "project" / fname).exists()
