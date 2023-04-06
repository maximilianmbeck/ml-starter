import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from ml.utils.colors import colorize


def get_template_choices() -> dict[str, Path]:
    root_dir = Path(__file__).parent.parent / "templates"
    template_choices = {}
    for template_dir in root_dir.iterdir():
        if template_dir.is_dir() and (template_dir / "setup.py").exists():
            template_choices[template_dir.name] = template_dir
    return template_choices


def get_yes_no_input(prompt: str) -> bool:
    prompt_str = f"{prompt} {colorize('[Y/n]', 'white', True)} "
    input_value = input(prompt_str).lower()
    while input_value not in ("q", "y", "n", ""):
        input_value = input(f"Invalid value: {colorize(input_value, 'red')} ◆ {prompt_str}").lower()
    if input_value == "q":
        print(colorize("Quitting...", "red"))
        sys.exit(0)
    return input_value in ("y", "")


def create_new_project() -> None:
    template_choices = get_template_choices()

    parser = argparse.ArgumentParser(description="Create a new template repository")
    parser.add_argument(
        "-t",
        "--template",
        type=str,
        help="The template to use for the new repository",
        choices=list(sorted(template_choices.keys())),
        default="empty",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default="project",
        help="The name of the new repository",
    )
    args = parser.parse_args()

    write_dir = str(Path.cwd() / args.name)
    print(f"Creating new project from template {colorize(args.template, 'green', True)}")
    print(f" ↪ Writing to {colorize(write_dir, 'cyan')}")
    template_dir = template_choices[args.template]

    # Copy tree, resolving symlinks.
    for path in template_dir.rglob("*"):
        if path.is_file():
            new_file_path = Path(write_dir) / path.relative_to(template_dir)
            new_file_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(path.resolve(), new_file_path)

    # Prompts to initialize git repository.
    if get_yes_no_input("Initialize git repository?"):
        print(" ↪ Initializing git repository")
        subprocess.run(["git", "init"], cwd=write_dir)

    # Prompts to install project.
    if get_yes_no_input("Install project?"):
        print(" ↪ Installing project")
        subprocess.run(["pip", "install", "-e", ".[dev]"], cwd=write_dir)
