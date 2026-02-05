from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from common.paths import LOGS_DIR, ensure_directories


def setup_logging(log_name: str = "app", level: int = logging.INFO) -> None:
    """Configura logging com console + arquivo rotativo em output/logs."""
    ensure_directories()

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Evita duplicar handlers se o Streamlit recarregar
    if root_logger.handlers:
        return

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(fmt)
    root_logger.addHandler(sh)

    log_file: Path = LOGS_DIR / f"{log_name}.log"
    fh = RotatingFileHandler(
        filename=str(log_file),
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    fh.setLevel(level)
    fh.setFormatter(fmt)
    root_logger.addHandler(fh)

