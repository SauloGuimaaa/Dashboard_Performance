from __future__ import annotations

import pandas as pd


def daily_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega por dia:
    - reach_total
    - engagement_total
    - posts
    - engagement_rate (média ponderada por reach)
    """
    if len(df) == 0:
        return pd.DataFrame(columns=["date", "reach_total", "engagement_total", "posts", "engagement_rate"])

    d = df.copy()
    d = d.dropna(subset=["created_at"])
    if len(d) == 0:
        return pd.DataFrame(columns=["date", "reach_total", "engagement_total", "posts", "engagement_rate"])

    d["date"] = d["created_at"].dt.date

    g = d.groupby("date", as_index=False).agg(
        reach_total=("reach", "sum"),
        engagement_total=("engagement", "sum"),
        posts=("post_id", "count"),
    )

    # engagement rate ponderado: sum(engagement)/sum(reach)
    g["engagement_rate"] = g.apply(
        lambda r: (float(r["engagement_total"]) / float(r["reach_total"])) if r["reach_total"] > 0 else 0.0,
        axis=1,
    )

    return g.sort_values("date")


def weekly_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega por semana (ISO):
    - week_start (segunda-feira)
    - reach_total
    - engagement_total
    - posts
    - engagement_rate ponderado
    """
    if len(df) == 0:
        return pd.DataFrame(columns=["week_start", "reach_total", "engagement_total", "posts", "engagement_rate"])

    d = df.copy()
    d = d.dropna(subset=["created_at"])
    if len(d) == 0:
        return pd.DataFrame(columns=["week_start", "reach_total", "engagement_total", "posts", "engagement_rate"])

    # week start (segunda)
    d["week_start"] = d["created_at"].dt.to_period("W-MON").apply(lambda p: p.start_time.date())

    g = d.groupby("week_start", as_index=False).agg(
        reach_total=("reach", "sum"),
        engagement_total=("engagement", "sum"),
        posts=("post_id", "count"),
    )

    g["engagement_rate"] = g.apply(
        lambda r: (float(r["engagement_total"]) / float(r["reach_total"])) if r["reach_total"] > 0 else 0.0,
        axis=1,
    )

    return g.sort_values("week_start")
