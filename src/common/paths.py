from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Caminhos canônicos do projeto."""
    root: Path
    input_dir: Path
    output_dir: Path
    logs_dir: Path


def get_project_root() -> Path:
    """
    Resolve a raiz do projeto assumindo que este arquivo está em:
    <root>/src/common/paths.py
    """
    return Path(__file__).resolve().parents[2]


PATHS = ProjectPaths(
    root=get_project_root(),
    input_dir=get_project_root() / "input",
    output_dir=get_project_root() / "output",
    logs_dir=get_project_root() / "output" / "logs",
)

INPUT_DIR: Path = PATHS.input_dir
OUTPUT_DIR: Path = PATHS.output_dir
LOGS_DIR: Path = PATHS.logs_dir


def ensure_directories() -> None:
    """Garante que as pastas essenciais existam."""
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)