import logging
import shlex
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from ml.utils.distributed import get_rank_optional, get_world_size_optional
from ml.utils.logging import configure_logging
from ml.utils.timer import Timer

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger: logging.Logger = logging.getLogger(__name__)


def cli_main(project_root: Path | str | None = None) -> None:
    configure_logging(rank=get_rank_optional(), world_size=get_world_size_optional())
    logger.info("Command: %s", shlex.join(sys.argv))

    # Import here to avoid slow startup time.
    with Timer("importing", spinner=True):
        from ml.core.env import add_global_tag
        from ml.core.registry import Objects, add_project_dir
        from ml.scripts import launch, resolve, stage, train
        from ml.utils.cli import parse_cli
        from ml.utils.colors import colorize

    with Timer("running pre-flight tasks", spinner=True):
        if project_root is not None:
            add_project_dir(Path(project_root).resolve())

        scripts: dict[str, Callable[[DictConfig], None]] = {
            "train": train.train_main,
            "stage": stage.stage_main,
            "resolve": resolve.resolve_main,
            "launch": launch.launch_main,
        }

        def show_help() -> None:
            script_names = (colorize(script_name, "cyan") for script_name in scripts)
            print(f"Usage: ml < {' / '.join(script_names)} > ...\n", file=sys.stderr)
            for key, func in sorted(scripts.items()):
                if func.__doc__ is None:
                    print(f" ↪ {colorize(key, 'green')}", file=sys.stderr)
                else:
                    docstring = func.__doc__.strip().split("\n")[0]
                    print(f" ↪ {colorize(key, 'green')}: {docstring}", file=sys.stderr)
            print()
            sys.exit(1)

        # Parses the raw command line options.
        args = sys.argv[1:]
        if len(args) == 0:
            show_help()
        option, args = args[0], args[1:]

        # Adds a global tag with the currently-selected option.
        add_global_tag(option)

    if option in scripts:
        # Special handling for multi-processing; don't initialize anything since
        # everything will be initialized inside the child processes.
        with Timer("parsing cli arguments"):
            config = parse_cli(args)
        Objects.update_config(config)
        Objects.resolve_config(config)
        scripts[option](config)
    else:
        print(f"Invalid option: {colorize(option, 'red')}\n", file=sys.stderr)
        show_help()


if __name__ == "__main__":
    cli_main()
