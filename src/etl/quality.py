from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd


@dataclass(frozen=True)
class QualityReport:
    """Relatório de qualidade do dataset."""
    row_count: int
    invalid_created_at: int
    invalid_created_at_pct: float
    missing_post_id: int
    missing_post_id_pct: float
    negative_counts: Dict[str, int]
    duplicate_post_id: int
    duplicate_post_id_pct: float


COUNT_COLS: Tuple[str, ...] = ("reach", "likes", "comments", "shares", "saves", "engagement")


def compute_quality(df: pd.DataFrame) -> QualityReport:
    """Calcula indicadores de qualidade para o schema alvo."""
    n = int(len(df))
    if n == 0:
        return QualityReport(
            row_count=0,
            invalid_created_at=0,
            invalid_created_at_pct=0.0,
            missing_post_id=0,
            missing_post_id_pct=0.0,
            negative_counts={c: 0 for c in COUNT_COLS},
            duplicate_post_id=0,
            duplicate_post_id_pct=0.0,
        )

    invalid_created_at = int(df["created_at"].isna().sum()) if "created_at" in df.columns else n
    missing_post_id = int((df["post_id"].astype(str).str.strip() == "").sum()) if "post_id" in df.columns else n

    negative_counts: Dict[str, int] = {}
    for c in COUNT_COLS:
        if c in df.columns:
            negative_counts[c] = int((df[c] < 0).sum())
        else:
            negative_counts[c] = 0

    duplicate_post_id = 0
    if "post_id" in df.columns:
        s = df["post_id"].astype(str).str.strip()
        duplicate_post_id = int(s.duplicated(keep=False).sum())

    return QualityReport(
        row_count=n,
        invalid_created_at=invalid_created_at,
        invalid_created_at_pct=(invalid_created_at / n) * 100.0,
        missing_post_id=missing_post_id,
        missing_post_id_pct=(missing_post_id / n) * 100.0,
        negative_counts=negative_counts,
        duplicate_post_id=duplicate_post_id,
        duplicate_post_id_pct=(duplicate_post_id / n) * 100.0,
    )


def quality_warnings(q: QualityReport) -> List[str]:
    """Gera avisos baseados no relatório."""
    warnings: List[str] = []

    if q.row_count == 0:
        warnings.append("Dataset vazio após filtros.")
        return warnings

    if q.invalid_created_at_pct > 10:
        warnings.append(f"Muitas datas inválidas: {q.invalid_created_at_pct:.1f}% (ver coluna created_at).")

    if q.missing_post_id_pct > 0:
        warnings.append(f"Há post_id vazio em {q.missing_post_id_pct:.1f}% das linhas.")

    if q.duplicate_post_id_pct > 0:
        warnings.append(f"Há post_id duplicado em {q.duplicate_post_id_pct:.1f}% das linhas (pode ser export com granularidade diferente).")

    neg_total = sum(q.negative_counts.values())
    if neg_total > 0:
        warnings.append("Há contagens negativas em alguma métrica (isso geralmente indica parsing errado).")

    return warnings
