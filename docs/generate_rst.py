"""Script which auto-generates RST files for the API reference."""

import os
import re
from dataclasses import is_dataclass

from ml.core.config import BaseConfig, BaseObject
from ml.core.registry import (
    register_base,
    register_launcher,
    register_lr_scheduler,
    register_model,
    register_optimizer,
    register_task,
)

source_dir = "../ml"

subdirs: list[tuple[str, str]] = [
    ("core", "Core"),
    ("loggers", "Logger"),
    ("lr_schedulers", "LR Scheduler"),
    ("models", "Model"),
    ("optimizers", "Optimizer"),
    ("tasks", "Task"),
    ("trainers", "Trainer"),
    ("launchers", "Launcher"),
    ("utils", "Utility Functions"),
]

registries: dict[str, tuple[str, type[register_base]]] = {
    "launchers": ("Launchers", register_launcher),
    "models": ("Models", register_model),
    "optimizers": ("Optimizers", register_optimizer),
    "lr_schedulers": ("LR Schedulers", register_lr_scheduler),
    "tasks": ("Tasks", register_task),
}


def parse_field_type(field_type_str: str) -> str:
    field_type_str = re.sub(r"<class '(.+)'>", r"\1", field_type_str)
    return field_type_str


def create_config_rst(
    registry_name: str,
    key: str,
    klass: type[BaseObject],
    config_klass: type[BaseConfig] | BaseConfig,
) -> str:
    # Gets each of the fields for the current config dataclass. The field may
    # have a "help" metadata field, which is used as the help string.
    fields: list[tuple[str, str, str, str]] = []
    for field_name, field in config_klass.__dataclass_fields__.items():
        if is_dataclass(field.type):
            continue
        field_type = parse_field_type(str(field.type))
        field_default_value = field.default
        field_value = str(getattr(config_klass, field_name, field_default_value))
        fields.append((field_name, field_type, str(field.metadata.get("help", "")), field_value))

    content = f"""
{key}
{'=' * len(key)}

"""

    if klass.__doc__:
        content += f"``{klass.__doc__.strip()}``\n\n"

    if fields:
        content += """
.. list-table:: Fields
    :widths: 20 20 40 20
    :header-rows: 1

    * - Field
      - Default
      - Description
      - Type
"""
        for field_name, field_type, field_help, field_value in fields:
            content += f"""
    * - {field_name}
      - {field_value}
      - {field_help}
      - {field_type}
"""

    registry_key = f"registry.{registry_name}.{key}"
    with open(f"{registry_key}.rst", "w", encoding="utf-8") as f:
        f.write(content.strip())
    return registry_key


registry_keys: list[str] = []
for registry_key, (registry_name, registry) in registries.items():
    registry_subkeys: list[str] = []

    # Write files for each of the registry configs.
    registry.populate_full_regisry()
    for key, (klass, config_klass) in registry.REGISTRY.items():
        registry_subkeys.append(create_config_rst(registry_key, key, klass, config_klass))

    if not registry_subkeys:
        continue

    # Create a file for the current registry.
    index_content = f"""
{registry_name}
{'=' * len(registry_name)}

.. toctree::
    :maxdepth: 2
    :caption: Contents:

"""
    for registry_subkey in sorted(registry_subkeys):
        index_content += f"    {registry_subkey}\n"

    with open(f"registry.{registry_key}.rst", "w") as f:
        f.write(index_content.strip())
    registry_keys.append(registry_key)


def create_module_rst(module_name: str) -> None:
    content = f"""
{module_name}
{'=' * len(module_name)}

.. automodule:: {module_name}
   :members:
   :undoc-members:
   :show-inheritance:
"""
    with open(f"{module_name}.rst", "w") as f:
        f.write(content.strip())


for subdir, subdir_name in subdirs:
    # Walk the source directory and create RST files for all Python modules
    all_files = []
    for root, _, files in os.walk(os.path.join(source_dir, subdir)):
        if "triton" in root:
            continue
        for file in files:
            if file.endswith(".py") and file != "__init__.py":
                module_path = os.path.join(root, file)
                module_name = os.path.splitext(module_path.replace(os.path.sep, "."))[0].lstrip(".")
                create_module_rst(module_name)
                all_files.append(module_name)

    # Create a file for the current subdirectory
    index_content = f"""
{subdir_name}
{'=' * len(subdir_name)}

.. toctree::
    :maxdepth: 2
    :caption: Contents:

"""
    for file in sorted(all_files):
        index_content += f"    {file}\n"

    with open(f"ml.{subdir}.rst", "w") as f:
        f.write(index_content.strip())


index_content = """
ml-starter
==========

Welcome to the documentation for the `ml-starter` package!

This is the documentation for the `ml-starter project <https://github.com/codekansas/ml-starter>`_.

Additionally, there are some pre-trained models which work with this framework which you can find
in the `ml-pretrained repository <https://github.com/codekansas/ml-pretrained>`_.

.. image:: /_static/github.png
   :target: https://github.com/codekansas/ml-project-template/generate

.. toctree::
   :maxdepth: 3
   :caption: Contents
   :hidden:

   about
   getting_started
"""

index_content += """
.. toctree::
   :maxdepth: 3
   :caption: Configs
   :hidden:

"""

for registry_key in sorted(registry_keys):
    index_content += f"   registry.{registry_key}.rst\n"

index_content += """
.. toctree::
   :maxdepth: 3
   :caption: Reference
   :hidden:

"""

for subdir, subdir_name in subdirs:
    index_content += f"   ml.{subdir}.rst\n"

with open("index.rst", "w") as f:
    f.write(index_content.strip())
