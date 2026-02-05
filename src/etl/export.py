from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def export_tables(
    output_dir: Path,
    posts: pd.DataFrame,
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    prefix: str,
) -> dict[str, Path]:
    """
    Exporta tabelas em CSV para output_dir/tables.
    Retorna paths gerados.
    """
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    p_posts = tables_dir / f"{prefix}_posts.csv"
    p_daily = tables_dir / f"{prefix}_daily.csv"
    p_weekly = tables_dir / f"{prefix}_weekly.csv"

    posts.to_csv(p_posts, index=False, encoding="utf-8")
    daily.to_csv(p_daily, index=False, encoding="utf-8")
    weekly.to_csv(p_weekly, index=False, encoding="utf-8")

    paths["posts"] = p_posts
    paths["daily"] = p_daily
    paths["weekly"] = p_weekly

    logger.info("Export OK: %s", {k: str(v) for k, v in paths.items()})
    return paths
