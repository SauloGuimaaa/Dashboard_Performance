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
    categories: list[dict[str, Any]]
    category_metrics: list[dict[str, Any]]
    top_by_category: dict[str, list[dict[str, Any]]]
    category_rankings: dict[str, list[dict[str, Any]]]
    top20_distribution: list[dict[str, Any]]


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


def _truncate_text(value: Any, limit: int) -> Any:
    if not isinstance(value, str):
        return value
    if len(value) <= limit:
        return value
    return value[:limit] + "…"


def _truncate_records(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return records
    out: list[dict[str, Any]] = []
    for rec in records:
        out.append({k: _truncate_text(v, limit) for k, v in rec.items()})
    return out


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).lower()


def _build_category_map(raw: str) -> dict[str, list[str]]:
    categories: dict[str, list[str]] = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        name, keywords = line.split(":", 1)
        tokens = [k.strip().lower() for k in keywords.split(",") if k.strip()]
        if name.strip() and tokens:
            categories[name.strip()] = tokens
    return categories


def _categorize_caption(caption: str, categories: dict[str, list[str]]) -> str:
    if not categories:
        return "Sem categoria"
    text = _normalize_text(caption)
    for name, keywords in categories.items():
        if any(k in text for k in keywords):
            return name
    return "Outros"


def build_insight_request(
    df: pd.DataFrame,
    max_posts: int = 10,
    caption_char_limit: int = 280,
    category_rules: dict[str, list[str]] | None = None,
) -> InsightRequest:
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
    top_reach = _truncate_records(top_reach, caption_char_limit)

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
    top_engagement = _truncate_records(top_engagement, caption_char_limit)

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
    captions_sample = _truncate_records(captions_sample, caption_char_limit)

    categories = category_rules or {}
    if "caption" in df:
        df["_category"] = df["caption"].apply(lambda c: _categorize_caption(c, categories))
    else:
        df["_category"] = "Sem categoria"

    categories_summary = (
        df["_category"].value_counts().reset_index().rename(columns={"index": "category", "_category": "count"}).to_dict("records")
    )

    category_metrics = []
    for category, group in df.groupby("_category"):
        posts_count = int(len(group))
        reach_total = int(group["reach"].sum()) if "reach" in group else 0
        engagement_total = int(group["engagement"].sum()) if "engagement" in group else 0
        views_total = int(group["views"].sum()) if "views" in group else 0
        reach_avg = float(reach_total / posts_count) if posts_count else 0.0
        engagement_avg = float(engagement_total / posts_count) if posts_count else 0.0
        views_avg = float(views_total / posts_count) if posts_count else 0.0
        category_metrics.append(
            {
                "category": category,
                "posts": posts_count,
                "reach_total": reach_total,
                "engagement_total": engagement_total,
                "views_total": views_total,
                "reach_avg": reach_avg,
                "engagement_avg": engagement_avg,
                "views_avg": views_avg,
                "engagement_rate": float(
                    (group["engagement"].sum() / group["reach"].sum()) if "reach" in group and group["reach"].sum() else 0.0
                ),
            }
        )

    category_rankings: dict[str, list[dict[str, Any]]] = {
        "views_avg": sorted(category_metrics, key=lambda x: x.get("views_avg", 0.0), reverse=True),
        "reach_avg": sorted(category_metrics, key=lambda x: x.get("reach_avg", 0.0), reverse=True),
        "engagement_avg": sorted(category_metrics, key=lambda x: x.get("engagement_avg", 0.0), reverse=True),
    }

    top_by_category: dict[str, list[dict[str, Any]]] = {}
    for category, group in df.groupby("_category"):
        records = (
            _select_columns(
                group.sort_values("reach", ascending=False).head(max_posts),
                ["post_id", "created_at", "post_type", "reach", "engagement", "views", "permalink", "caption"],
            )
            .assign(created_at=lambda d: d["created_at"].astype("string") if "created_at" in d else d)
            .fillna("")
            .to_dict("records")
        )
        top_by_category[category] = _truncate_records(records, caption_char_limit)

    top20_distribution = []
    if "reach" in df:
        top20 = df.sort_values("reach", ascending=False).head(20)
        top20_counts = top20["_category"].value_counts().reset_index()
        top20_distribution = top20_counts.rename(columns={"index": "category", "_category": "count"}).to_dict("records")

    return InsightRequest(
        summary=summary,
        top_reach=top_reach,
        top_engagement=top_engagement,
        post_types=post_types,
        captions_sample=captions_sample,
        categories=categories_summary,
        category_metrics=category_metrics,
        top_by_category=top_by_category,
        category_rankings=category_rankings,
        top20_distribution=top20_distribution,
    )


def generate_instagram_insights(
    df: pd.DataFrame,
    api_key: str,
    model: str = "gpt-4o-mini",
    max_posts: int = 10,
    category_rules_text: str | None = None,
) -> str:
    """Generate insights text for Instagram performance using OpenAI."""
    category_rules = _build_category_map(category_rules_text or "")
    payload_json = ""
    for candidate in [max_posts, 100, 50, 20, 10, 5]:
        if candidate <= 0:
            continue
        payload = build_insight_request(
            df,
            max_posts=min(candidate, max_posts),
            category_rules=category_rules,
        )
        payload_json = json.dumps(asdict(payload), ensure_ascii=False)
        if len(payload_json) <= 12000:
            break

    prompt = (
        "Você é um(a) estrategista de performance digital para Instagram. "
        "Gere um relatório executivo, direto e profissional, baseado em dados. "
        "Siga a estrutura obrigatória abaixo e use números concretos do JSON.\n\n"
        "Estrutura obrigatória:\n"
        "1) Pergunta estratégica (1 frase)\n"
        "2) Resumo executivo (3-5 bullets)\n"
        "3) Ranking de performance por categoria (views_avg, reach_avg, engagement_avg)\n"
        "4) Categorias vencedoras + justificativa (2-4 categorias)\n"
        "5) Categorias críticas (queda/baixa eficiência) + hipótese de causa\n"
        "6) Dominância no Top 20 (participação por categoria)\n"
        "7) Mix recomendado (proporção sugerida por categoria, com ação: AUMENTAR/MANTER/REDUZIR)\n"
        "8) Próximos passos (3-5 ações práticas)\n\n"
        "Considere legendas, tipo de post e top posts por categoria. "
        "Evite generalidades. Se faltar dado, diga explicitamente.\n\n"
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
