import os

source_dir = "../ml"
subdirs = ["core", "loggers", "lr_schedulers", "models", "optimizers", "tasks", "trainers", "launchers", "utils"]


def create_rst(module_name: str) -> None:
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


for subdir in subdirs:
    # Walk the source directory and create RST files for all Python modules
    all_files = []
    for root, _, files in os.walk(os.path.join(source_dir, subdir)):
        for file in files:
            if file.endswith(".py") and file != "__init__.py":
                module_path = os.path.join(root, file)
                module_name = os.path.splitext(module_path.replace(os.path.sep, "."))[0].lstrip(".")
                create_rst(module_name)
                all_files.append(module_name)

    # Create a file for the current subdirectory
    index_content = f"""
{subdir}
{'=' * len(subdir)}

.. toctree::
    :maxdepth: 2
    :caption: Contents:

"""
    for file in sorted(all_files):
        index_content += f"    {file}\n"

    with open(f"ml.{subdir}.rst", "w") as f:
        f.write(index_content.strip())

index_content = """
Full API Reference
==================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

"""

for subdir in subdirs:
    index_content += f"   ml.{subdir}.rst\n"

with open("ml.rst", "w") as f:
    f.write(index_content.strip())
