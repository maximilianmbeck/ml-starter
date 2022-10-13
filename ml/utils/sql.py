from pathlib import Path
from typing import Any, Iterable, List

import aiosqlite as asql

DEFAULT_PRAGMAS = (
    "PRAGMA journal_mode = OFF",
    "PRAGMA synchronous = 0",
    "PRAGMA cache_size = 1000000",
    "PRAGMA locking_mode = EXCLUSIVE",
    "PRAGMA temp_store = MEMORY",
)


def as_in_set(items: Iterable[Any]) -> str:
    """Formats a set of items for a `WHERE x IN {items}` query.

    Example:
        query = f"SELECT * FROM table WHERE x IN {as_in_set(('a', 'b', 'c'))}"

    Args:
        items: The set of items

    Returns:
        The formatted string
    """

    return "(" + ", ".join(f"'{i}'" for i in items) + ")"


async def sqlite_connect_async(
    db_path: str | Path,
    *,
    readonly: bool = False,
    immutable: bool = False,
    pragmas: Iterable[str] = DEFAULT_PRAGMAS,
    check_exists: bool = False,
) -> asql.Connection:
    """Returns an async connection to a SQLite database.

    Args:
        db_path: The path to the database file
        readonly: If the file should be opened in read-only mode
        immutable: If the database is immutable
        pragmas: Pragmas to apply on opening the connection
        check_exists: If set, make sure the database exists

    Returns:
        The database connection
    """

    if immutable:
        readonly = True

    options: List[str] = []
    if readonly:
        options += ["mode=ro"]
    if immutable:
        options += ["immutable=1"]

    if check_exists:
        assert Path(db_path).exists(), f"Missing {db_path}"

    db_path_uri = f"file:///{db_path}"
    if options:
        db_path_uri = f"{db_path_uri}?{'&'.join(options)}"
    conn = await asql.connect(db_path_uri, uri=True)

    # Pragmas can't be applied in read-only mode.
    if not readonly:
        async with conn.cursor() as cursor:
            for pragma in pragmas:
                await cursor.execute(pragma)

    return conn
