import functools
import inspect
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Iterator, TypeVar, cast

from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.basecontainer import BaseContainer

from ml.core.config import BaseConfig, BaseObject
from ml.core.env import ShowFullImportError
from ml.utils.colors import colorize
from ml.utils.timer import Timer

if TYPE_CHECKING:
    from ml.launchers.base import BaseLauncher, BaseLauncherConfig
    from ml.loggers.base import BaseLogger, BaseLoggerConfig
    from ml.lr_schedulers.base import BaseLRScheduler, BaseLRSchedulerConfig
    from ml.models.base import BaseModel, BaseModelConfig
    from ml.optimizers.base import BaseOptimizer, BaseOptimizerConfig
    from ml.tasks.base import BaseTask, BaseTaskConfig
    from ml.trainers.base import BaseTrainer, BaseTrainerConfig

logger: logging.Logger = logging.getLogger(__name__)

Entry = TypeVar("Entry", bound=BaseObject)
SpecificEntry = TypeVar("SpecificEntry")
Config = TypeVar("Config", bound=BaseConfig)

# Special key in the config, cooresponding to the reserved keyword in the
# BaseConfig, which is used to reference the object to construct.
NAME_KEY = "name"

# This points to the root directory location for the package.
ROOT_DIR: Path = Path(__file__).parent.parent.resolve()

# Maximum number of days to keep a staging directory around. This should
# correspond to the maximum number of days that an experiment could run.
MAX_STAGING_DAYS = 31


class _ProjectDirs:
    def __init__(self) -> None:
        self.__dir_set: set[Path] = {ROOT_DIR}
        self.__dir_list: list[Path] = [ROOT_DIR]

    @property
    def paths(self) -> list[Path]:
        return self.__dir_list

    def add(self, path: Path) -> None:
        if path in self.__dir_set:
            return
        path = path.resolve()
        self.__dir_set.add(path)
        self.__dir_list.append(path)

    def relative_path(self, p: Path, parent: bool = False) -> Path:
        for d in self.__dir_list:
            try:
                return p.relative_to(d.parent if parent else d)
            except ValueError:
                pass
        raise ValueError(f"Path {p} is not relative to any of {self.__dir_list}")


# Project directories singleton.
project_dirs = _ProjectDirs()

# Some aliases for the project directory accessors.
add_project_dir = project_dirs.add
project_dir_paths = project_dirs.paths


def get_name(key: str, config: BaseContainer) -> str:
    if not isinstance(config, DictConfig):
        raise ValueError(f"Expected {key} config to be a dictionary, got {type(config)}")
    if NAME_KEY not in config:
        raise ValueError(f"Malformed {key} config; missing expected key {NAME_KEY}")
    name = config[NAME_KEY]
    if not isinstance(name, str):
        raise ValueError(f"Expected {key} name to be a string, got {name}")
    return name


def get_names(key: str, config: BaseContainer) -> list[str]:
    if not isinstance(config, ListConfig):
        raise ValueError(f"Expected {key} config to be a list, got {type(config)}")
    names = []
    for i, subconfig in enumerate(config):
        if not isinstance(subconfig, DictConfig):
            raise ValueError(f"Expected {key} config item {i} to be a dictionary, got {type(subconfig)}")
        if NAME_KEY not in subconfig:
            raise ValueError(f"Malformed {key} config item {i}; missing expected key {NAME_KEY}")
        name = subconfig[NAME_KEY]
        if not isinstance(name, str):
            raise ValueError(f"Expected {key} name to be a string, got {name}")
        names.append(name)
    return names


class register_base(ABC, Generic[Entry, Config]):  # pylint: disable=invalid-name
    """Defines the base registry type."""

    REGISTRY: dict[str, tuple[type[Entry], type[Config]]] = {}
    REGISTRY_LOCATIONS: dict[str, Path] = {}

    @classmethod
    @abstractmethod
    def search_directory(cls) -> Path:
        """Returns the directory to search for entries."""

    @classmethod
    @abstractmethod
    def config_key(cls) -> str:
        """Returns the key for the current item from the config."""

    @classmethod
    def registry_path(cls) -> Path:
        return project_dirs.paths[-1] / ".cache" / f"{cls.config_key()}.json"

    @classmethod
    @functools.lru_cache(None)
    def load_registry_locations(cls) -> None:
        registry_path = cls.registry_path()
        if not registry_path.exists():
            return
        with open(registry_path, "r", encoding="utf-8") as f:
            try:
                cached_registry_locations = json.load(f)
            except json.decoder.JSONDecodeError:
                return
        new_locations = {
            key: Path(reg_loc)
            for key, reg_loc in cached_registry_locations.items()
            if key not in cls.REGISTRY_LOCATIONS and Path(reg_loc).is_file()
        }
        cls.REGISTRY_LOCATIONS.update(new_locations)

    @classmethod
    def save_registry_locations(cls) -> None:
        registry_path = cls.registry_path()
        registry_path.parent.mkdir(exist_ok=True, parents=True)
        registry_locations = {k: str(v.resolve()) for k, v in cls.REGISTRY_LOCATIONS.items() if v.is_file()}
        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(registry_locations, f, indent=2)

    @classmethod
    @functools.lru_cache(None)
    def manual_import(cls, path: Path) -> None:
        with Timer(f"importing '{path}'"):
            try:
                rel_path = project_dirs.relative_path(path, parent=True)
                module_name = ".".join(list(rel_path.parts[:-1]) + [rel_path.stem])
                __import__(module_name)

            except Exception:
                if ShowFullImportError.get():
                    logger.exception("Caught exception while importing %s", path)
                else:
                    logger.error(
                        "Caught exception while importing %s (set %s to show the full exception)",
                        path,
                        ShowFullImportError.key,
                    )

    @classmethod
    def populate_registry(cls, name: str) -> None:
        """Populates the registry until it has the requested name available.

        Args:
            name: The name of the registry item to get
        """

        lower_name = name.lower()

        # Check in the existing registry locations.
        if name in cls.REGISTRY_LOCATIONS:
            cls.manual_import(cls.REGISTRY_LOCATIONS[name])
        if name in cls.REGISTRY:
            return

        # First do a quick sweep over the cached registry locations to see if
        # one happens to match the name being imported, since this is likely
        # to be the one we want and it will avoid having to import every file
        # by hand.
        for reg_name, path in cls.REGISTRY_LOCATIONS.items():
            if reg_name.lower().startswith(lower_name):
                cls.manual_import(path)
            if name in cls.REGISTRY:
                return

        # This gets populated the first time we walk the directories, so that
        # the second time we can just iterate through it again.
        subfiles: list[Path] = []

        def iter_directory(*curdirs: Path) -> Iterator[Path]:
            for curdir in curdirs:
                for subpath in curdir.iterdir():
                    if subpath.stem.startswith("__"):
                        continue
                    if subpath.is_file() and subpath.suffix == ".py":
                        subfile = subpath.resolve()
                        subfiles.append(subfile)
                        yield subfile
                    elif subpath.is_dir():
                        yield from iter_directory(subpath)

        # Next sweep over the search directory and check for prefix matches.
        search_dir = cls.search_directory()
        search_dirs = [base_dir / search_dir for base_dir in project_dirs.paths]
        search_dirs = [search_dir for search_dir in search_dirs if search_dir.is_dir()]
        for path in iter_directory(*search_dirs):
            if path.stem.lower().startswith(lower_name) or lower_name.startswith(path.stem.lower()):
                cls.manual_import(path)
                if name in cls.REGISTRY:
                    return

        # Finally, try loading files from the requested import path until
        # we've imported the name that we're looking for.
        for path in subfiles:
            cls.manual_import(path)
            if name in cls.REGISTRY:
                return

    @classmethod
    @functools.lru_cache(None)
    def lookup(cls, name: str) -> tuple[type[Entry], type[Config]]:
        # Just loads the entry, if it already exists.
        if name in cls.REGISTRY:
            return cls.REGISTRY[name]

        # If not found, populates the registry. If still not found, then
        # we're out of luck and should throw an error
        with Timer(f"looking up {name}"):
            cls.load_registry_locations()
            cls.populate_registry(name)
            cls.save_registry_locations()
        if name in cls.REGISTRY:
            return cls.REGISTRY[name]

        options = "\n".join(f" - {k}" for k in sorted(cls.REGISTRY.keys()))
        logger.error("Couldn't locate %s '%s' in:\n%s", cls.config_key(), name, options)
        raise KeyError(f"Couldn't locate {cls.config_key()} '{name}' from {len(cls.REGISTRY)} options")

    @classmethod
    def lookup_path(cls, name: str) -> Path:
        if name in cls.REGISTRY_LOCATIONS:
            return cls.REGISTRY_LOCATIONS[name]

        # If the registry locations haven't been loaded, load them, then
        # check again.
        cls.load_registry_locations()
        if name in cls.REGISTRY_LOCATIONS:
            return cls.REGISTRY_LOCATIONS[name]

        # If the file location. has not been cached, search for it, then
        # cache it for future fast lookup.
        cls.populate_registry(name)
        cls.save_registry_locations()
        if name in cls.REGISTRY_LOCATIONS:
            return cls.REGISTRY_LOCATIONS[name]

        options = "\n".join(f" - {k}" for k in sorted(cls.REGISTRY_LOCATIONS.keys()))
        logger.error("Couldn't locate %s '%s' in:\n%s", cls.config_key(), name, options)
        raise KeyError(f"Couldn't locate {cls.config_key()} '{name}' from {len(cls.REGISTRY_LOCATIONS)} options")

    @classmethod
    def _build_entry_from_name(cls, reg_name: str, reg_cfg: DictConfig, raw_config: DictConfig) -> Entry:
        with Timer(f"building {cls.config_key()}", spinner=True):
            reg_cls, _ = cls.lookup(reg_name)

        with Timer(f"building {reg_name}", spinner=True):
            reg_obj = reg_cls(reg_cfg)
            if isinstance(reg_obj, BaseObject):
                reg_obj.set_raw_config(raw_config)

        return reg_obj

    @classmethod
    def build_entry(cls, raw_config: DictConfig) -> Entry | None:
        if cls.config_key() not in raw_config:
            return None
        with Timer(f"getting {cls.config_key()} name", spinner=True):
            reg_name = get_name(cls.config_key(), raw_config[cls.config_key()])
        return cls._build_entry_from_name(reg_name, raw_config[cls.config_key()], raw_config)

    @classmethod
    def build_entry_non_null(cls, raw_config: DictConfig) -> Entry:
        if (entry := cls.build_entry(raw_config)) is None:
            raise ValueError(f"Missing {cls.config_key()} in config")
        return entry

    @classmethod
    def update_config(cls, raw_config: DictConfig) -> None:
        if cls.config_key() not in raw_config:
            return

        with Timer(f"updating {cls.config_key()} config", spinner=True):
            reg_cfg = raw_config[cls.config_key()]
            reg_name = get_name(cls.config_key(), reg_cfg)
            _, reg_cfg_cls = cls.lookup(reg_name)
            reg_cfg = OmegaConf.merge(OmegaConf.structured(reg_cfg_cls), reg_cfg)
            raw_config[cls.config_key()] = reg_cfg

    @classmethod
    def resolve_config(cls, raw_config: DictConfig) -> None:
        if cls.config_key() not in raw_config:
            return

        with Timer(f"resolving {cls.config_key()} config", spinner=True):
            reg_cfg = raw_config[cls.config_key()]
            reg_name = get_name(cls.config_key(), reg_cfg)
            _, reg_cfg_cls = cls.lookup(reg_name)
            reg_cfg_cls.resolve(reg_cfg)
            raw_config[cls.config_key()] = reg_cfg

    def __init__(self, name: str, config: type[Config]) -> None:
        self.name = name
        self.config = config

    def __call__(self, entry: SpecificEntry) -> SpecificEntry:
        if self.name in self.REGISTRY:
            # raise RuntimeError(f"Found duplicate names: {self.name}")
            logger.warning("Found duplicate names: %s", self.name)
            return entry

        registry_location = Path(inspect.getfile(cast(type[Entry], entry)))

        # Adds the registry entry and the entry's location to their respective
        # dictionaries. We overwrite any outdated cache entries.
        self.REGISTRY[self.name] = cast(tuple[type[Entry], type[Config]], (entry, self.config))
        self.REGISTRY_LOCATIONS[self.name] = registry_location

        # Adds all default configurations as well.
        for key, default_cfg in self.config.get_defaults().items():
            self.REGISTRY[key] = cast(tuple[type[Entry], type[Config]], (entry, default_cfg))
            self.REGISTRY_LOCATIONS[key] = registry_location

        return entry


class multi_register_base(register_base[Entry, Config], Generic[Entry, Config]):  # pylint: disable=invalid-name
    """Defines a registry which produces multiple objects."""

    @classmethod
    def build_entry(cls, raw_config: DictConfig) -> Entry | None:
        raise NotImplementedError("`build_entry` not implemented; use `build_entries` instead")

    @classmethod
    def build_entry_non_null(cls, raw_config: DictConfig) -> Entry:
        raise NotImplementedError("`build_entry_non_null` not implemented; use `build_entries_non_null` instead")

    @classmethod
    def update_config(cls, raw_config: DictConfig) -> None:
        raise NotImplementedError("`update_config` not implemented; use `update_configs` instead")

    @classmethod
    def resolve_config(cls, raw_config: DictConfig) -> None:
        raise NotImplementedError("`resolve_config` not implemented; use `resolve_configs` instead")

    @classmethod
    def build_entries(cls, raw_config: DictConfig) -> list[Entry] | None:
        if cls.config_key() not in raw_config:
            return None

        # Attempts to build a single entry first.
        try:
            entry = super().build_entry(raw_config)
            return [entry] if entry is not None else None
        except ValueError:
            pass

        entries: list[Entry] = []
        reg_names = get_names(cls.config_key(), raw_config[cls.config_key()])
        for i, reg_name in enumerate(reg_names):
            entries.append(cls._build_entry_from_name(reg_name, raw_config[cls.config_key()][i], raw_config))

        return entries

    @classmethod
    def build_entries_non_null(cls, raw_config: DictConfig) -> list[Entry]:
        entries = cls.build_entries(raw_config)
        if entries is None:
            raise ValueError(f"Missing {cls.config_key()} in config")
        return entries

    @classmethod
    def update_configs(cls, raw_config: DictConfig) -> None:
        if cls.config_key() not in raw_config:
            return

        # Treat as a single entry first.
        try:
            super().update_config(raw_config)
            return
        except ValueError:
            pass

        with Timer(f"updating {cls.config_key()} config", spinner=True):
            reg_cfgs = raw_config[cls.config_key()]
            reg_names = get_names(cls.config_key(), reg_cfgs)
            for i, (reg_name, reg_cfg) in enumerate(zip(reg_names, reg_cfgs)):
                _, reg_cfg_cls = cls.lookup(reg_name)
                reg_cfg = OmegaConf.merge(OmegaConf.structured(reg_cfg_cls), reg_cfg)
                reg_cfgs[i] = reg_cfg
            raw_config[cls.config_key()] = reg_cfgs

    @classmethod
    def resolve_configs(cls, raw_config: DictConfig) -> None:
        if cls.config_key() not in raw_config:
            return

        # Treat as a single entry first.
        try:
            super().resolve_config(raw_config)
            return
        except ValueError:
            pass

        with Timer(f"resolving {cls.config_key()} config", spinner=True):
            reg_cfgs = raw_config[cls.config_key()]
            reg_names = get_names(cls.config_key(), reg_cfgs)
            for i, (reg_name, reg_cfg) in enumerate(zip(reg_names, reg_cfgs)):
                _, reg_cfg_cls = cls.lookup(reg_name)
                reg_cfg_cls.resolve(reg_cfg)
                reg_cfgs[i] = reg_cfg
            raw_config[cls.config_key()] = reg_cfgs


class register_model(register_base["BaseModel", "BaseModelConfig"]):  # pylint: disable=invalid-name
    """Defines a registry for holding modules."""

    REGISTRY: dict[str, tuple[type["BaseModel"], type["BaseModelConfig"]]] = {}
    REGISTRY_LOCATIONS: dict[str, Path] = {}

    @classmethod
    def search_directory(cls) -> Path:
        return Path("models")

    @classmethod
    def config_key(cls) -> str:
        return "model"


class register_task(register_base["BaseTask", "BaseTaskConfig"]):  # pylint: disable=invalid-name
    """Defines a registry for holding tasks."""

    REGISTRY: dict[str, tuple[type["BaseTask"], type["BaseTaskConfig"]]] = {}
    REGISTRY_LOCATIONS: dict[str, Path] = {}

    @classmethod
    def search_directory(cls) -> Path:
        return Path("tasks")

    @classmethod
    def config_key(cls) -> str:
        return "task"


class register_trainer(register_base["BaseTrainer", "BaseTrainerConfig"]):  # pylint: disable=invalid-name
    """Defines a registry for holding trainers."""

    REGISTRY: dict[str, tuple[type["BaseTrainer"], type["BaseTrainerConfig"]]] = {}
    REGISTRY_LOCATIONS: dict[str, Path] = {}

    @classmethod
    def search_directory(cls) -> Path:
        return Path("trainers")

    @classmethod
    def config_key(cls) -> str:
        return "trainer"


class register_optimizer(register_base["BaseOptimizer", "BaseOptimizerConfig"]):  # pylint: disable=invalid-name
    """Defines a registry for holding optimizers."""

    REGISTRY: dict[str, tuple[type["BaseOptimizer"], type["BaseOptimizerConfig"]]] = {}
    REGISTRY_LOCATIONS: dict[str, Path] = {}

    @classmethod
    def search_directory(cls) -> Path:
        return Path("optimizers")

    @classmethod
    def config_key(cls) -> str:
        return "optimizer"


class register_lr_scheduler(register_base["BaseLRScheduler", "BaseLRSchedulerConfig"]):  # pylint: disable=invalid-name
    """Defines a registry for holding learning rate schedulers."""

    REGISTRY: dict[str, tuple[type["BaseLRScheduler"], type["BaseLRSchedulerConfig"]]] = {}
    REGISTRY_LOCATIONS: dict[str, Path] = {}

    @classmethod
    def search_directory(cls) -> Path:
        return Path("lr_schedulers")

    @classmethod
    def config_key(cls) -> str:
        return "lr_scheduler"


class register_logger(multi_register_base["BaseLogger", "BaseLoggerConfig"]):  # pylint: disable=invalid-name
    """Defines a registry for holding loggers."""

    REGISTRY: dict[str, tuple[type["BaseLogger"], type["BaseLoggerConfig"]]] = {}
    REGISTRY_LOCATIONS: dict[str, Path] = {}

    @classmethod
    def search_directory(cls) -> Path:
        return Path("loggers")

    @classmethod
    def config_key(cls) -> str:
        return "logger"


class register_launcher(register_base["BaseLauncher", "BaseLauncherConfig"]):  # pylint: disable=invalid-name
    REGISTRY: dict[str, tuple[type["BaseLauncher"], type["BaseLauncherConfig"]]] = {}
    REGISTRY_LOCATIONS: dict[str, Path] = {}

    @classmethod
    def search_directory(cls) -> Path:
        return Path("launchers")

    @classmethod
    def config_key(cls) -> str:
        return "launcher"


@dataclass(frozen=True)
class Objects:
    raw_config: DictConfig
    model: "BaseModel | None" = None
    task: "BaseTask | None" = None
    trainer: "BaseTrainer | None" = None
    optimizer: "BaseOptimizer | None" = None
    lr_scheduler: "BaseLRScheduler | None" = None
    logger: "list[BaseLogger] | None" = None
    launcher: "BaseLauncher | None" = None

    def __post_init__(self) -> None:
        if self.trainer is not None:
            if self.logger is not None:
                self.trainer.add_loggers(self.logger)

    def summarize(self) -> str:
        parts: dict[str, tuple[str, str]] = {}
        if self.model is not None:
            parts["Model"] = (
                inspect.getfile(self.model.__class__),
                f"{self.model.__class__.__module__}.{self.model.__class__.__name__}",
            )
        if self.task is not None:
            parts["Task"] = (
                inspect.getfile(self.task.__class__),
                f"{self.task.__class__.__module__}.{self.task.__class__.__name__}",
            )
        if self.trainer is not None:
            parts["Trainer"] = (
                inspect.getfile(self.trainer.__class__),
                f"{self.trainer.__class__.__module__}.{self.trainer.__class__.__name__}",
            )
        if self.optimizer is not None:
            parts["Optimizer"] = (
                inspect.getfile(self.optimizer.__class__),
                f"{self.optimizer.__class__.__module__}.{self.optimizer.__class__.__name__}",
            )
        if self.lr_scheduler is not None:
            parts["LR Scheduler"] = (
                inspect.getfile(self.lr_scheduler.__class__),
                f"{self.lr_scheduler.__class__.__module__}.{self.lr_scheduler.__class__.__name__}",
            )
        if self.launcher is not None:
            parts["Launcher"] = (
                inspect.getfile(self.launcher.__class__),
                f"{self.launcher.__class__.__module__}.{self.launcher.__class__.__name__}",
            )
        return "Components:" + "".join(
            f"\n ↪ {colorize(k, 'green')}: {colorize(v[1], 'cyan')} ({colorize(v[0], 'blue')})"
            for k, v in parts.items()
        )

    @classmethod
    def update_config(cls, config: DictConfig) -> None:
        """Updates the config in-place.

        Args:
            config: The config to update
        """

        # Pre-builds the config using the structured configs.
        register_model.update_config(config)
        register_task.update_config(config)
        register_trainer.update_config(config)
        register_optimizer.update_config(config)
        register_lr_scheduler.update_config(config)
        register_logger.update_configs(config)
        register_launcher.update_config(config)

    @classmethod
    def resolve_config(cls, config: DictConfig) -> None:
        """Resolves the config in-place.

        Args:
            config: The config to resolve
        """

        # Resolves the final config once all structured configs have been merged.
        OmegaConf.resolve(config)

        # Runs object-specific resolutions.
        register_model.resolve_config(config)
        register_task.resolve_config(config)
        register_trainer.resolve_config(config)
        register_optimizer.resolve_config(config)
        register_lr_scheduler.resolve_config(config)
        register_logger.resolve_configs(config)
        register_launcher.resolve_config(config)

    @classmethod
    def parse_raw_config(cls, config: DictConfig) -> "Objects":
        """Parses a raw config to the objects it contains.

        Args:
            config: The raw DictConfig to parse

        Returns:
            The parsed Objects dataclass
        """

        model = register_model.build_entry(config)
        task = register_task.build_entry(config)
        trainer = register_trainer.build_entry(config)
        optimizer = register_optimizer.build_entry(config)
        lr_scheduler = register_lr_scheduler.build_entry(config)
        loggers = register_logger.build_entries(config)
        launcher = register_launcher.build_entry(config)

        objs = Objects(
            raw_config=config,
            model=model,
            task=task,
            trainer=trainer,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            logger=loggers,
            launcher=launcher,
        )

        logger.info("%s", objs.summarize())

        return objs

    @classmethod
    def from_config_file(cls, config_path: str | Path, **overrides: Any) -> "Objects":
        config = OmegaConf.load(config_path)
        config = OmegaConf.merge(config, DictConfig(overrides))
        return cls.parse_raw_config(cast(DictConfig, config))
