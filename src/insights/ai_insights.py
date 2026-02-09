from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from typing import Any

import httpx
import pandas as pd
from openai import OpenAI


@dataclass(frozen=True)
class InsightRequest:
    """Payload with structured stats for LLM summarization."""

    summary: dict[str, Any]
    top_reach: list[dict[str, Any]]
    top_engagement: list[dict[str, Any]]
    post_types: list[dict[str, Any]]
    captions_sample: list[dict[str, Any]]


def _safe_sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if df.empty:
        return df
    n = min(n, len(df))
    return df.sample(n=n, random_state=42)


def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.columns.is_unique:
        return df
    return df.loc[:, ~df.columns.duplicated()].copy()


def _select_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    available = [col for col in columns if col in df.columns]
    return df.loc[:, available]


def build_insight_request(df: pd.DataFrame, max_posts: int = 10) -> InsightRequest:
    """Builds a compact summary payload for the AI prompt."""
    df = _dedupe_columns(df)
    created_at = df["created_at"] if "created_at" in df else pd.Series([], dtype="datetime64[ns]")
    summary = {
        "rows": int(len(df)),
        "date_min": None if created_at.isna().all() else str(created_at.min()),
        "date_max": None if created_at.isna().all() else str(created_at.max()),
        "reach_total": int(df["reach"].sum()) if "reach" in df else 0,
        "engagement_total": int(df["engagement"].sum()) if "engagement" in df else 0,
        "views_total": int(df["views"].sum()) if "views" in df else 0,
    }

    top_reach = (
        _select_columns(
            df.sort_values("reach", ascending=False).head(max_posts),
            ["post_id", "created_at", "post_type", "reach", "engagement", "views", "permalink", "caption"],
        )
        .assign(created_at=lambda d: d["created_at"].astype("string") if "created_at" in d else d)
        .fillna("")
        .to_dict("records")
        if "reach" in df
        else []
    )

    top_engagement = (
        _select_columns(
            df.sort_values("engagement", ascending=False).head(max_posts),
            ["post_id", "created_at", "post_type", "reach", "engagement", "views", "permalink", "caption"],
        )
        .assign(created_at=lambda d: d["created_at"].astype("string") if "created_at" in d else d)
        .fillna("")
        .to_dict("records")
        if "engagement" in df
        else []
    )

    post_types = (
        df["post_type"].fillna("(vazio)").value_counts().reset_index().rename(columns={"index": "post_type", "post_type": "count"}).to_dict("records")
        if "post_type" in df
        else []
    )

    captions_sample = (
        _select_columns(_safe_sample(df, n=max_posts), ["post_id", "caption", "post_type", "permalink", "created_at"])
        .assign(created_at=lambda d: d["created_at"].astype("string") if "created_at" in d else d)
        .fillna("")
        .to_dict("records")
        if "caption" in df
        else []
    )

    return InsightRequest(
        summary=summary,
        top_reach=top_reach,
        top_engagement=top_engagement,
        post_types=post_types,
        captions_sample=captions_sample,
    )


def generate_instagram_insights(
    df: pd.DataFrame,
    api_key: str,
    model: str = "gpt-4o-mini",
    max_posts: int = 10,
) -> str:
    """Generate insights text for Instagram performance using OpenAI."""
    payload = build_insight_request(df, max_posts=max_posts)
    payload_json = json.dumps(asdict(payload), ensure_ascii=False)

    prompt = (
        "Você é um(a) especialista em performance digital no Instagram. "
        "Analise o dataset e forneça insights acionáveis sobre alcance, views e engajamento. "
        "Use bullet points, destaque oportunidades e recomendações práticas (ex.: formatos, horários, temas). "
        "Considere também as legendas e categorias de post. "
        "Responda em português, de forma objetiva e estruturada.\n\n"
        "Dados estruturados (JSON):\n"
        f"{payload_json}"
    )

    http_client = httpx.Client(timeout=httpx.Timeout(30.0), trust_env=False)
    client = OpenAI(api_key=api_key, http_client=http_client)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Você é um(a) especialista em performance digital no Instagram."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content or ""
