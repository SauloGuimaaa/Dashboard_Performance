from __future__ import annotations

from typing import Iterable

import pandas as pd


def _unique_preserve_order(items: Iterable[str]) -> list[str]:
    """Remove duplicatas preservando a ordem."""
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def top_posts(df: pd.DataFrame, metric: str, n: int = 5) -> pd.DataFrame:
    """
    Retorna Top N posts ordenados por `metric` (desc), com um conjunto consistente de colunas.

    Importante:
    - Garante que NÃO haja colunas duplicadas no resultado (Streamlit/PyArrow exige colunas únicas).
    """
    if metric not in df.columns:
        raise ValueError(f"Métrica '{metric}' não existe no DataFrame. Colunas: {list(df.columns)}")

    preferred = [
        "post_id",
        "created_at",
        "caption",
        metric,          # pode ser 'reach' ou 'engagement'
        "reach",
        "engagement",
        "likes",
        "comments",
        "shares",
        "saves",
        "engagement_rate",
    ]

    # 1) remove duplicatas (ex.: metric='reach' duplicando 'reach')
    preferred = _unique_preserve_order(preferred)

    # 2) mantém só colunas existentes
    cols = [c for c in preferred if c in df.columns]

    out = df.sort_values(metric, ascending=False, na_position="last").head(n)[cols].copy()
    return out
