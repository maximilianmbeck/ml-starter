# ML Starter

This is the core code for my ML project template over [here](https://github.com/codekansas/ml-project-template).

## Installation

```bash
pip install ml-starter
```

Alternatively, the latest bleeding edge version can be installed from GitHub:

```bash
pip install git+https://github.com/codekansas/ml-starter.git
```

This expects a project structure like this:

```bash
.
├── configs
│   └── my_config.yaml
└── project
    ├── loggers
    ├── lr_schedulers
    ├── models
    │   └── my_model.py
    ├── optimizers
    ├── scripts
    │   └── cli.py
    ├── tasks
    │   └── my_task.py
    └── trainers
```

The `cli.py` file should look something like this:

```python
from pathlib import Path

from ml.scripts.cli import cli_main as ml_cli_main

PROJECT_ROOT = Path(__file__).parent.parent


def cli_main() -> None:
    ml_cli_main(PROJECT_ROOT)


if __name__ == "__main__":
    cli_main()
```

You can then train a model for your config using this command:

```bash
python -m project.scripts.cli train configs/my_config.yaml
```

This can be made more wieldy by adding it as an entry point to your `setup.cfg` file:

```
[options.entry_points]

console_scripts =
    runml = project.scripts.cli:cli_main
```

The template repository above already does this.
