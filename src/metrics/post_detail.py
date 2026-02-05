from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class PostSummary:
    post_id: str
    created_at: Optional[pd.Timestamp]
    caption: str
    reach: int
    engagement: int
    engagement_rate: float
    likes: int
    comments: int
    shares: int
    saves: int


def get_post_summary(df: pd.DataFrame, post_id: str) -> Optional[PostSummary]:
    """Retorna resumo do post_id selecionado (pega a primeira linha)."""
    if len(df) == 0 or "post_id" not in df.columns:
        return None

    d = df[df["post_id"].astype(str) == str(post_id)]
    if len(d) == 0:
        return None

    row = d.iloc[0]

    def _i(col: str) -> int:
        return int(row[col]) if col in d.columns and pd.notna(row[col]) else 0

    def _f(col: str) -> float:
        return float(row[col]) if col in d.columns and pd.notna(row[col]) else 0.0

    created = row["created_at"] if "created_at" in d.columns and pd.notna(row["created_at"]) else None
    caption = str(row["caption"]) if "caption" in d.columns and pd.notna(row["caption"]) else ""

    return PostSummary(
        post_id=str(row["post_id"]),
        created_at=created,
        caption=caption,
        reach=_i("reach"),
        engagement=_i("engagement"),
        engagement_rate=_f("engagement_rate"),
        likes=_i("likes"),
        comments=_i("comments"),
        shares=_i("shares"),
        saves=_i("saves"),
    )
