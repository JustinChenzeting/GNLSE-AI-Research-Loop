"""File a saved campaign under its controls and optimization method."""

import os
from pathlib import Path


def file_saved_run(directory, parameters, method):
    """Make an existing campaign available from its parameter and method folder."""
    directory = Path(directory).resolve(strict=True)
    runs = Path(__file__).resolve().parents[1] / "output" / "organized" / "runs"
    shelf = runs / parameters / method
    shelf.mkdir(parents=True, exist_ok=True)
    entry = shelf / directory.name
    try:
        entry.symlink_to(os.path.relpath(directory, shelf), target_is_directory=True)
    except FileExistsError:
        if not entry.is_dir() or not entry.samefile(directory):
            raise FileExistsError(f"Another campaign already uses {entry}") from None
    return entry
