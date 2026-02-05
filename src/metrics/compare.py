from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class PeriodKpis:
    posts: int
    reach_total: int
    engagement_total: int
    engagement_rate_avg: float  # média simples do dataset filtrado (% em 0..1)
    reach_avg: float
    engagement_avg: float


def compute_kpis(df: pd.DataFrame) -> PeriodKpis:
    """Calcula KPIs principais para um período (df já filtrado)."""
    if len(df) == 0:
        return PeriodKpis(
            posts=0,
            reach_total=0,
            engagement_total=0,
            engagement_rate_avg=0.0,
            reach_avg=0.0,
            engagement_avg=0.0,
        )

    posts = int(len(df))
    reach_total = int(df["reach"].sum()) if "reach" in df.columns else 0
    engagement_total = int(df["engagement"].sum()) if "engagement" in df.columns else 0
    engagement_rate_avg = float(df["engagement_rate"].mean()) if "engagement_rate" in df.columns else 0.0
    reach_avg = float(df["reach"].mean()) if "reach" in df.columns else 0.0
    engagement_avg = float(df["engagement"].mean()) if "engagement" in df.columns else 0.0

    return PeriodKpis(
        posts=posts,
        reach_total=reach_total,
        engagement_total=engagement_total,
        engagement_rate_avg=engagement_rate_avg,
        reach_avg=reach_avg,
        engagement_avg=engagement_avg,
    )


def pct_change(new: float, old: float) -> Optional[float]:
    """Retorna variação percentual (ex.: 0.12 = +12%)."""
    if old == 0:
        return None
    return (new - old) / old
