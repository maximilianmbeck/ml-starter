import functools
import json
import logging
import pickle
from typing import Any, Callable, List, TypeVar

from ml.core.env import get_cache_dir

logger = logging.getLogger(__name__)

Object = TypeVar("Object", bound=Any)


class cached_object:  # pylint: disable=invalid-name
    """Defines a wrapper for caching function calls to a file location."""

    def __init__(self, cache_key: str, ext: str = "pkl", ignore: bool = False) -> None:
        self.cache_key = cache_key
        self.ext = ext
        self.obj = None
        self.ignore = ignore

        assert ext in ("json", "pkl"), f"Unexpected extension: {ext}"

    def __call__(self, func: Callable[..., Object]) -> Callable[..., Object]:
        """Returns a wrapped function that caches the return value.

        Args:
            func: The function to cache, which returns the object to load

        Returns:
            A cached version of the same function
        """

        @functools.wraps(func)
        def call_function_cached(*args: Any, **kwargs: Any) -> Object:
            if self.obj is not None:
                return self.obj

            keys: List[str] = []
            for arg in args:
                keys += [str(arg)]
            for key, val in sorted(kwargs.items()):
                keys += [f"{key}_{val}"]
            key = ".".join(keys)

            fpath = get_cache_dir() / self.cache_key / f"{key}.{self.ext}"

            if fpath.is_file() and not self.ignore:
                logger.debug("Loading cached object from %s", fpath)
                if self.ext == "json":
                    with open(fpath, "r", encoding="utf-8") as f:
                        return json.load(f)
                if self.ext == "pkl":
                    with open(fpath, "rb") as fb:
                        return pickle.load(fb)
                raise NotImplementedError(f"Can't load extension {self.ext}")

            self.obj = func(*args, **kwargs)

            logger.debug("Saving cached object to %s", fpath)
            fpath.parent.mkdir(exist_ok=True, parents=True)
            if self.ext == "json":
                with open(fpath, "w", encoding="utf-8") as f:
                    json.dump(self.obj, f)
                    return self.obj
            if self.ext == "pkl":
                with open(fpath, "wb") as fb:
                    pickle.dump(self.obj, fb)
                    return self.obj
            raise NotImplementedError(f"Can't save extension {self.ext}")

        return call_function_cached
