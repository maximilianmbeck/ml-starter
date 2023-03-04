import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from ml.core.registry import register_lr_scheduler
from ml.core.state import State
from ml.utils.argparse import add_args, from_args


def main() -> None:
    """Plots a learning rate schedule.

    Usage:
        python -m ml.lr_schedulers.scripts.plot linear /path/to/save.png
    """

    # Gets the plotting-specific arguments.
    parser = argparse.ArgumentParser(description="Plots a learning rate schedule")
    parser.add_argument("lr_scheduler", help="Which scheduler to plot")
    parser.add_argument("save_path", help="Where to save the plot")
    parser.add_argument("-n", "--num-iters", type=int, default=100_000, help="Number of iterations")
    parser.add_argument("-s", "--stride", type=int, default=100, help="Stride between iterations")
    args, cli_args = parser.parse_known_args()
    save_path = Path(args.save_path)
    save_path.parent.mkdir(exist_ok=True, parents=True)
    num_iters: int = args.num_iters
    stride: int = args.stride

    # Parses config-specific arguments and builds the learning rate scheduler.
    scheduler_cls, scheduler_config_cls = register_lr_scheduler.lookup(args.lr_scheduler)
    scheduler_parser = argparse.ArgumentParser(description=f"Parser for {args.lr_scheduler} scheduler")
    add_args(scheduler_parser, scheduler_config_cls)
    scheduler_args = scheduler_parser.parse_args(cli_args)
    scheduler_config = from_args(scheduler_args, scheduler_config_cls)
    scheduler = scheduler_cls(scheduler_config)

    state = State.init_state()
    xs: list[float] = []
    ys: list[float] = []
    for i in range(0, num_iters, stride):
        state.num_steps = i
        lr_scale = scheduler.get_lr_scale(state)
        xs.append(i)
        ys.append(lr_scale)

    plt.figure()
    plt.plot(xs, ys)
    plt.savefig(save_path)


if __name__ == "__main__":
    main()
