from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict

import pandas as pd

logger = logging.getLogger(__name__)

INVALID_FILENAME_CHARS = r'[<>:"/\\|?*\x00-\x1F]'


def sanitize_filename(name: str) -> str:
    """Sanitiza nome para ser válido no Windows."""
    s = re.sub(INVALID_FILENAME_CHARS, "_", str(name))
    s = s.strip().strip(".")
    s = re.sub(r"_+", "_", s)
    return s or "file"


def _mpl():
    """
    Importa matplotlib de forma segura para servidor (sem backend GUI).
    Importar aqui evita problemas de backend no Streamlit.
    """
    import matplotlib

    matplotlib.use("Agg")  # backend não interativo
    import matplotlib.pyplot as plt  # noqa: WPS433

    return plt


def export_test_png(output_dir: Path, prefix: str) -> Path:
    """Gera um PNG simples para validar export (sem Kaleido)."""
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    prefix_safe = sanitize_filename(prefix)
    out_path = charts_dir / f"{prefix_safe}__matplotlib_test.png"

    plt = _mpl()
    fig = plt.figure(figsize=(10, 5), dpi=160)
    ax = fig.add_subplot(111)
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
    ax.set_title("matplotlib test")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    logger.info("Matplotlib test PNG OK | %s | bytes=%d", str(out_path), out_path.stat().st_size)
    return out_path


def _save_placeholder(out_path: Path, title: str, message: str) -> None:
    plt = _mpl()
    fig = plt.figure(figsize=(10, 5), dpi=160)
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.5, 0.6, title, ha="center", va="center", fontsize=16, fontweight="bold")
    ax.text(0.5, 0.4, message, ha="center", va="center", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def export_charts_png(
    output_dir: Path,
    prefix: str,
    top_reach: pd.DataFrame,
    top_eng: pd.DataFrame,
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
) -> Dict[str, Path]:
    """
    Exporta os gráficos principais como PNG via Matplotlib (robusto no Windows).
    Retorna dict: nome_do_grafico -> path PNG.
    """
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    prefix_safe = sanitize_filename(prefix)
    paths: Dict[str, Path] = {}

    plt = _mpl()

    def _save(fig, key: str) -> Path:
        out_path = charts_dir / f"{prefix_safe}__{sanitize_filename(key)}.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        paths[key] = out_path
        return out_path

    logger.info("Matplotlib PNG export START | prefix=%s | charts_dir=%s", prefix_safe, str(charts_dir))

    # ---------- Top Reach (barh)
    key = "top_reach"
    out_path = charts_dir / f"{prefix_safe}__{key}.png"
    if len(top_reach) == 0 or "post_id" not in top_reach.columns or "reach" not in top_reach.columns:
        _save_placeholder(out_path, "Top 5 — Reach", "Sem dados suficientes para export.")
        paths[key] = out_path
    else:
        d = top_reach.sort_values("reach", ascending=True)
        fig = plt.figure(figsize=(14, 7), dpi=160)
        ax = fig.add_subplot(111)
        ax.barh(d["post_id"].astype(str), d["reach"])
        ax.set_title("Top 5 — Reach")
        ax.set_xlabel("reach")
        ax.set_ylabel("post_id")
        fig.tight_layout()
        _save(fig, key)

    # ---------- Top Engagement (barh)
    key = "top_engagement"
    out_path = charts_dir / f"{prefix_safe}__{key}.png"
    if len(top_eng) == 0 or "post_id" not in top_eng.columns or "engagement" not in top_eng.columns:
        _save_placeholder(out_path, "Top 5 — Engagement", "Sem dados suficientes para export.")
        paths[key] = out_path
    else:
        d = top_eng.sort_values("engagement", ascending=True)
        fig = plt.figure(figsize=(14, 7), dpi=160)
        ax = fig.add_subplot(111)
        ax.barh(d["post_id"].astype(str), d["engagement"])
        ax.set_title("Top 5 — Engagement")
        ax.set_xlabel("engagement")
        ax.set_ylabel("post_id")
        fig.tight_layout()
        _save(fig, key)

    # ---------- Daily series (linhas)
    def _line(df: pd.DataFrame, x: str, y: str, title: str, key: str) -> None:
        out_path2 = charts_dir / f"{prefix_safe}__{key}.png"
        if len(df) == 0 or x not in df.columns or y not in df.columns:
            _save_placeholder(out_path2, title, "Sem dados suficientes para export.")
            paths[key] = out_path2
            return
        fig = plt.figure(figsize=(14, 7), dpi=160)
        ax = fig.add_subplot(111)
        ax.plot(df[x], df[y])
        ax.set_title(title)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        _save(fig, key)

    _line(daily, "date", "reach_total", "Reach total por dia", "daily_reach_total")
    _line(daily, "date", "engagement_total", "Engajamento total por dia", "daily_engagement_total")
    _line(daily, "date", "engagement_rate", "Engajamento/Reach por dia", "daily_engagement_rate")

    # ---------- Weekly series (barras + linha)
    def _bar(df: pd.DataFrame, x: str, y: str, title: str, key: str) -> None:
        out_path2 = charts_dir / f"{prefix_safe}__{key}.png"
        if len(df) == 0 or x not in df.columns or y not in df.columns:
            _save_placeholder(out_path2, title, "Sem dados suficientes para export.")
            paths[key] = out_path2
            return
        fig = plt.figure(figsize=(14, 7), dpi=160)
        ax = fig.add_subplot(111)
        ax.bar(df[x].astype(str), df[y])
        ax.set_title(title)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        _save(fig, key)

    _bar(weekly, "week_start", "reach_total", "Reach total por semana", "weekly_reach_total")
    _bar(weekly, "week_start", "engagement_total", "Engajamento total por semana", "weekly_engagement_total")
    _line(weekly, "week_start", "engagement_rate", "Engajamento/Reach por semana", "weekly_engagement_rate")

    # validação rápida
    for k, p in paths.items():
        if not p.exists():
            raise RuntimeError(f"PNG não foi criado: {p}")

    logger.info("Matplotlib PNG export OK | files=%d", len(paths))
    return paths
