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
    overall_metrics: dict[str, Any]
    format_metrics: list[dict[str, Any]]
    temporal_day: list[dict[str, Any]]
    temporal_hour: list[dict[str, Any]]


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


def _ensure_metrics(df: pd.DataFrame) -> pd.DataFrame:
    dff = df.copy()
    for col in ["likes", "comments", "shares", "saves", "reach", "views"]:
        if col not in dff.columns:
            dff[col] = 0
    if "engagement" not in dff.columns:
        dff["engagement"] = dff["likes"] + dff["comments"] + dff["shares"] + dff["saves"]
    dff["engagement_rate"] = (
        (dff["engagement"] / dff["reach"].replace({0: pd.NA})).fillna(0.0) if "reach" in dff else 0.0
    )
    dff["virality"] = (
        (dff["shares"] / dff["reach"].replace({0: pd.NA}) * 1000).fillna(0.0) if "reach" in dff else 0.0
    )
    dff["save_value"] = (
        (dff["saves"] / dff["engagement"].replace({0: pd.NA}) * 100).fillna(0.0) if "engagement" in dff else 0.0
    )
    dff["community_index"] = (
        (dff["comments"] / dff["likes"].replace({0: pd.NA}) * 100).fillna(0.0) if "likes" in dff else 0.0
    )
    return dff


def build_insight_request(
    df: pd.DataFrame,
    max_posts: int = 10,
    caption_char_limit: int = 280,
    category_rules: dict[str, list[str]] | None = None,
) -> InsightRequest:
    """Builds a compact summary payload for the AI prompt."""
    df = _dedupe_columns(df)
    df = _ensure_metrics(df)
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
        eng_std = float(group["engagement"].std()) if posts_count > 1 else 0.0
        eng_median = float(group["engagement"].median()) if posts_count else 0.0
        cv = float((eng_std / engagement_avg) * 100) if engagement_avg else 0.0
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
                "engagement_median": eng_median,
                "engagement_std": eng_std,
                "cv": cv,
                "engagement_rate": float(
                    (group["engagement"].sum() / group["reach"].sum()) if "reach" in group and group["reach"].sum() else 0.0
                ),
                "virality_avg": float(group["virality"].mean()) if "virality" in group else 0.0,
                "save_value_avg": float(group["save_value"].mean()) if "save_value" in group else 0.0,
                "community_index_avg": float(group["community_index"].mean()) if "community_index" in group else 0.0,
                "likes_total": int(group["likes"].sum()) if "likes" in group else 0,
                "comments_total": int(group["comments"].sum()) if "comments" in group else 0,
                "shares_total": int(group["shares"].sum()) if "shares" in group else 0,
                "saves_total": int(group["saves"].sum()) if "saves" in group else 0,
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

    overall_metrics = {
        "engagement_avg": float(df["engagement"].mean()) if "engagement" in df else 0.0,
        "engagement_median": float(df["engagement"].median()) if "engagement" in df else 0.0,
        "engagement_rate_avg": float(df["engagement_rate"].mean()) if "engagement_rate" in df else 0.0,
        "virality_avg": float(df["virality"].mean()) if "virality" in df else 0.0,
        "save_value_avg": float(df["save_value"].mean()) if "save_value" in df else 0.0,
        "community_index_avg": float(df["community_index"].mean()) if "community_index" in df else 0.0,
    }

    format_metrics = []
    if "post_type" in df:
        for ptype, group in df.groupby("post_type"):
            format_metrics.append(
                {
                    "post_type": ptype,
                    "posts": int(len(group)),
                    "engagement_avg": float(group["engagement"].mean()) if "engagement" in group else 0.0,
                    "engagement_rate_avg": float(group["engagement_rate"].mean()) if "engagement_rate" in group else 0.0,
                    "reach_avg": float(group["reach"].mean()) if "reach" in group else 0.0,
                    "views_avg": float(group["views"].mean()) if "views" in group else 0.0,
                }
            )

    temporal_day = []
    if "created_at" in df:
        dff = df.dropna(subset=["created_at"]).copy()
        if not dff.empty:
            dff["weekday"] = dff["created_at"].dt.day_name()
            for day, group in dff.groupby("weekday"):
                temporal_day.append(
                    {
                        "weekday": day,
                        "posts": int(len(group)),
                        "engagement_avg": float(group["engagement"].mean()),
                        "engagement_rate_avg": float(group["engagement_rate"].mean()),
                        "reach_avg": float(group["reach"].mean()),
                    }
                )

    temporal_hour = []
    if "created_at" in df:
        dff = df.dropna(subset=["created_at"]).copy()
        if not dff.empty:
            dff["hour"] = dff["created_at"].dt.hour
            for hour, group in dff.groupby("hour"):
                temporal_hour.append(
                    {
                        "hour": int(hour),
                        "posts": int(len(group)),
                        "engagement_avg": float(group["engagement"].mean()),
                        "engagement_rate_avg": float(group["engagement_rate"].mean()),
                    }
                )

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
        overall_metrics=overall_metrics,
        format_metrics=format_metrics,
        temporal_day=temporal_day,
        temporal_hour=temporal_hour,
    )


def generate_instagram_insights(
    df: pd.DataFrame,
    api_key: str,
    model: str = "gpt-4o-mini",
    max_posts: int = 10,
    category_rules_text: str | None = None,
    vehicle_name: str | None = None,
    objective: str | None = None,
    benchmark_engagement_rate: float | None = None,
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

    vehicle = vehicle_name or "Veículo"
    objective_text = objective or "Maximizar engajamento e alcance orgânico"
    benchmark_text = (
        f"Benchmark de taxa de engajamento: {benchmark_engagement_rate:.2f}%."
        if benchmark_engagement_rate is not None
        else "Benchmark não informado."
    )

    prompt = (
        "Você é Dr. Ricardo Mendes, consultor sênior em estratégia digital para veículos de notícias. "
        f"Cliente: {vehicle}. Objetivo: {objective_text}. {benchmark_text}\n\n"
        "Gere um relatório executivo de alto valor, baseado em dados, seguindo o framework O-P-E-A "
        "(Observação, Por quê, E então, Ação) em todos os insights. "
        "Sempre quantifique com comparações vs média geral e vs benchmark quando disponível.\n\n"
        "Estrutura obrigatória:\n"
        "1) Pergunta estratégica (1 frase)\n"
        "2) Sumário executivo (máx 150 palavras) + Top 3 insights\n"
        "3) Ranking por categoria (views_avg, reach_avg, engagement_avg, performance vs média)\n"
        "4) Categorias vencedoras (Top 3) com oportunidades quantificadas e exemplos\n"
        "5) Categorias críticas (Bottom 3) com diagnóstico e plano de recuperação SMART\n"
        "6) Formatos (Reels/Carrossel/Imagem): comparativo e mix recomendado\n"
        "7) Tempo (dia da semana + horários top): recomendações práticas\n"
        "8) Dominância no Top 20 e padrão identificado\n"
        "9) Mix recomendado por categoria (AUMENTAR/MANTER/REDUZIR) com impacto estimado\n"
        "10) Quick wins (5 ações executáveis)\n"
        "11) Plano de ação 90 dias (3 sprints)\n"
        "12) KPIs de acompanhamento e conclusão\n\n"
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


def generate_instagram_comparison_insights(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    api_key: str,
    model: str = "gpt-4o-mini",
    max_posts: int = 10,
    category_rules_text: str | None = None,
    vehicle_name: str | None = None,
    objective: str | None = None,
    benchmark_engagement_rate: float | None = None,
    label_a: str | None = None,
    label_b: str | None = None,
) -> str:
    """Generate comparative insights for two datasets (A vs B)."""
    category_rules = _build_category_map(category_rules_text or "")
    payload_a = build_insight_request(
        _ensure_metrics(_dedupe_columns(df_a)),
        max_posts=max_posts,
        category_rules=category_rules,
    )
    payload_b = build_insight_request(
        _ensure_metrics(_dedupe_columns(df_b)),
        max_posts=max_posts,
        category_rules=category_rules,
    )
    payload_json = json.dumps(
        {
            "period_a": asdict(payload_a),
            "period_b": asdict(payload_b),
        },
        ensure_ascii=False,
    )

    vehicle = vehicle_name or "Veículo"
    objective_text = objective or "Aumentar performance comparando períodos"
    benchmark_text = (
        f"Benchmark de taxa de engajamento: {benchmark_engagement_rate:.2f}%."
        if benchmark_engagement_rate is not None
        else "Benchmark não informado."
    )
    period_a = label_a or "Período A"
    period_b = label_b or "Período B"

    prompt = (
        "Você é Dr. Ricardo Mendes, consultor sênior em estratégia digital para veículos de notícias. "
        f"Cliente: {vehicle}. Objetivo: {objective_text}. {benchmark_text}\n\n"
        "Compare dois períodos (A vs B) e gere um relatório executivo comparativo. "
        "Use o framework O-P-E-A em cada insight e quantifique diferenças (% e deltas). "
        "Identifique gargalos, boas práticas e o que mudou no mix de conteúdo.\n\n"
        "Estrutura obrigatória:\n"
        "1) Pergunta estratégica (comparativa)\n"
        "2) Sumário executivo com diferenças-chave (3-5 bullets)\n"
        "3) Vencedores e perdedores por categoria (delta de views_avg, reach_avg, engagement_avg)\n"
        "4) Mudanças no Top 20 (dominância por categoria)\n"
        "5) Mudanças no mix recomendado (A vs B) com ações\n"
        "6) Gargalos identificados (formato, horário, categoria, copy)\n"
        "7) Boas práticas observadas (replicar)\n"
        "8) Plano de ação 30 dias + KPIs\n\n"
        f"Período A: {period_a}\n"
        f"Período B: {period_b}\n\n"
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
