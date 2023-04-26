Getting Started
===============

The easiest way to get started is to use the template repository `here <https://github.com/codekansas/ml-project-template>`_.

You will need to define some environment variables to let the framework know
where files are. At the bare minimum, you need to define:

.. code-block:: bash

    export RUN_DIR=/path/to/run/dir  # This is where all your training logs and checkpoints will be written
    export EVAL_RUN_DIR=/path/to/eval/run/dir  # This is where all your evaluation logs will be written

It's also a good idea to define these other variables as well:

.. code-block:: bash

    export DATA_DIR=/path/to/data/dir  # This is where your datasets are stored
    export MODEL_DIR=/path/to/model/dir  # This is where your pretrained models are stored

I suggest trying out the examples from the template repository to get a feel
for how things work.

From Scratch
------------

Alternatively, you can install the package directly with pip:

.. code-block:: bash

   pip install ml-starter

This expects a project structure like this:

.. code-block:: bash

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

The `cli.py` file should look something like this:

.. code-block:: python

   from pathlib import Path

   from ml.scripts.cli import cli_main as ml_cli_main

   PROJECT_ROOT = Path(__file__).parent.parent


   def cli_main() -> None:
      ml_cli_main(PROJECT_ROOT)


   if __name__ == "__main__":
      cli_main()

You can then train a model for your config using this command:

.. code-block:: bash

   python -m project.scripts.cli train configs/my_config.yaml

This can be made more wieldy by adding it as an entry point to your `setup.cfg` file:

.. code-block:: ini

   [options.entry_points]

   console_scripts =
      runml = project.scripts.cli:cli_main
