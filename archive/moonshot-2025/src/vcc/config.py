"""Configuration helpers for locating local data and resource directories."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

DATA_ENV_VAR = "VCC_DATA_DIR"
RESOURCES_ENV_VAR = "VCC_RESOURCES_DIR"
DATA_PROMPT = "Enter the path to the VCC data directory: "
TRAIN_FILE = "adata_Training.h5ad"


class ConfigurationError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


def _resolve_dir(raw_path: str, *, var_name: str) -> Path:
    """Expand and validate the directory supplied in environment variables."""
    path = Path(raw_path).expanduser()
    if not path.exists():
        raise ConfigurationError(f"{var_name}={path} does not exist")
    if not path.is_dir():
        raise ConfigurationError(f"{var_name}={path} is not a directory")
    return path


def _prompt_for_dir(message: str) -> Path:
    """Prompt the user interactively for a directory path."""
    if not sys.stdin.isatty():
        raise ConfigurationError(
            "Cannot prompt for the VCC data directory because stdin is not interactive. "
            "Set the VCC_DATA_DIR environment variable instead."
        )
    try:
        response = input(message).strip()
    except EOFError as exc:
        raise ConfigurationError(
            "No data directory provided. Set VCC_DATA_DIR or run again in an interactive shell."
        ) from exc
    if not response:
        raise ConfigurationError(
            "No data directory entered. Set VCC_DATA_DIR or rerun and provide a path."
        )
    return _resolve_dir(response, var_name="supplied path")


def _discover_data_dir() -> Optional[Path]:
    """Heuristically locate the data directory relative to the repo layout."""
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root / "data",
        repo_root / "datasets",
        repo_root / "vcc_data",
        repo_root.parent / "vcc_data",
        Path.cwd() / "data",
        Path.cwd() / "datasets",
        Path.cwd().parent / "vcc_data",
    ]

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / TRAIN_FILE).exists():
            return candidate
    return None


def get_data_dir(
    env_var: str = DATA_ENV_VAR,
    *,
    default: Optional[str] = None,
    prompt_if_missing: bool = True,
) -> Path:
    """Return the configured data directory.

    Order of resolution:
    1. Environment variable (if set)
    2. Optional default string passed by the caller
    3. Interactive prompt (if `prompt_if_missing=True`)
    """
    value = os.getenv(env_var)
    candidates: list[str | Path] = []
    if value:
        candidates.append(value)
    if default:
        candidates.append(default)
    autodetected = _discover_data_dir()
    if autodetected is not None:
        candidates.append(autodetected)

    for candidate in candidates:
        try:
            return _resolve_dir(str(candidate), var_name=env_var if isinstance(candidate, str) else "auto-detected path")
        except ConfigurationError:
            continue

    if prompt_if_missing:
        return _prompt_for_dir(DATA_PROMPT)
    raise ConfigurationError(f"Environment variable {env_var} is not set")


def get_resources_dir(
    env_var: str = RESOURCES_ENV_VAR,
    *,
    default: Optional[str] = None,
    prompt_if_missing: bool = False,
) -> Path:
    """Return the configured resources directory."""
    value = os.getenv(env_var, default)
    if value is not None:
        return _resolve_dir(value, var_name=env_var)
    if prompt_if_missing:
        return _prompt_for_dir("Enter the path to the VCC resources directory: ")
    raise ConfigurationError(f"Environment variable {env_var} is not set")
