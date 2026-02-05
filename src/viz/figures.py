from __future__ import annotations

from typing import Dict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _safe_caption(df: pd.DataFrame) -> list[str]:
    """Garante hover_data estável mesmo com caption vazio."""
    if "caption" not in df.columns:
        return []
    return df["caption"].astype(str).fillna("").tolist()


def fig_top_posts_bar(
    top_df: pd.DataFrame,
    metric: str,
    title: str,
) -> go.Figure:
    """
    Bar horizontal para Top posts.
    Espera top_df com colunas: post_id, metric, caption, created_at (se existir).
    """
    if len(top_df) == 0:
        return px.bar(title=title)

    # Ordena para ficar bonito no bar horizontal (menor -> maior)
    d = top_df.sort_values(metric, ascending=True)

    hover_cols = []
    if "caption" in d.columns:
        hover_cols.append("caption")
    if "created_at" in d.columns:
        hover_cols.append("created_at")

    fig = px.bar(
        d,
        x=metric,
        y="post_id",
        orientation="h",
        hover_data=hover_cols,
        title=title,
    )
    fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    return fig


def fig_daily_line(daily: pd.DataFrame, y: str, title: str) -> go.Figure:
    """Linha diária (date vs y)."""
    if len(daily) == 0:
        return px.line(title=title)

    fig = px.line(daily, x="date", y=y, title=title)
    fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    return fig


def fig_weekly_bar(weekly: pd.DataFrame, y: str, title: str) -> go.Figure:
    """Bar semanal (week_start vs y)."""
    if len(weekly) == 0:
        return px.bar(title=title)

    fig = px.bar(weekly, x="week_start", y=y, title=title)
    fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    return fig


def fig_weekly_line(weekly: pd.DataFrame, y: str, title: str) -> go.Figure:
    """Linha semanal (week_start vs y)."""
    if len(weekly) == 0:
        return px.line(title=title)

    fig = px.line(weekly, x="week_start", y=y, title=title)
    fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    return fig


def build_all_figures(
    top_reach: pd.DataFrame,
    top_eng: pd.DataFrame,
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
) -> Dict[str, go.Figure]:
    """
    Cria um dicionário padronizado de figuras para export e exibição.
    Keys viram nomes de arquivo (com prefixo) no export.
    """
    figs: Dict[str, go.Figure] = {}

    figs["top_reach"] = fig_top_posts_bar(top_reach, metric="reach", title="Top 5 — Reach")
    figs["top_engagement"] = fig_top_posts_bar(top_eng, metric="engagement", title="Top 5 — Engagement")

    figs["daily_reach_total"] = fig_daily_line(daily, y="reach_total", title="Reach total por dia")
    figs["daily_engagement_total"] = fig_daily_line(daily, y="engagement_total", title="Engajamento total por dia")
    figs["daily_engagement_rate"] = fig_daily_line(daily, y="engagement_rate", title="Engajamento/Reach por dia")

    figs["weekly_reach_total"] = fig_weekly_bar(weekly, y="reach_total", title="Reach total por semana")
    figs["weekly_engagement_total"] = fig_weekly_bar(weekly, y="engagement_total", title="Engajamento total por semana")
    figs["weekly_engagement_rate"] = fig_weekly_line(weekly, y="engagement_rate", title="Engajamento/Reach por semana")

    return figs
