# utils/root_locator.py
from pathlib import Path
from typing import Iterable, Optional


class RootLocator:
    """
    Utility class to reliably locate the project root directory.
    Searches upwards from the current file for common marker files/folders.
    Caches the result for performance.
    """

    _root: Optional[Path] = None

    @classmethod
    def get_root(
        cls,
        markers: Iterable[str] = None,
        start_from: Path = None
    ) -> Path:
        """
        Returns the project root directory.

        Args:
            markers: Iterable of marker names to search for (e.g., '.git', 'README.md', 'pyproject.toml').
                     Defaults to common project markers.
            start_from: Path to start searching from. Defaults to the directory of this file.

        Returns:
            Path: The project root directory.

        Raises:
            FileNotFoundError: If no marker is found up the directory tree.
        """
        if cls._root is not None:
            return cls._root

        if markers is None:
            markers = (
                ".git",               # Git repository
                "README.md",          # Common in most projects
                "pyproject.toml",     # Poetry/PyPA projects
                "requirements.txt",   # Classic Python projects
                "setup.py",           # Setuptools projects
                "setup.cfg",
                ".project_root",      # Custom empty marker file (optional)
            )

        start_path = start_from or Path(__file__).resolve().parent
        current = start_path

        while True:
            for marker in markers:
                if (current / marker).exists():
                    cls._root = current
                    return current

            if current.parent == current:  # Reached filesystem root
                break
            current = current.parent

        raise FileNotFoundError(
            f"Project root not found. Searched up from {start_path}. "
            f"Consider adding one of the markers: {', '.join(markers)}"
        )

    @classmethod
    def reset_cache(cls) -> None:
        """Clear the cached root (useful for testing or dynamic environments)."""
        cls._root = None


# Convenience global instance
ROOT_DIR = RootLocator.get_root()