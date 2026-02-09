# app/dashboard.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
import os
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px


# =========================
# Paths / Logging
# =========================
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.logging_config import setup_logging  # noqa: E402
from common.paths import INPUT_DIR, OUTPUT_DIR  # noqa: E402
from insights.ai_insights import generate_instagram_insights  # noqa: E402

CHARTS_DIR = OUTPUT_DIR / "charts"
TABLES_DIR = OUTPUT_DIR / "tables"

setup_logging(log_name="dashboard")
log = logging.getLogger("dashboard")


# =========================
# Sidebar / Files
# =========================
def _safe_filename(name: str) -> str:
    name = name.strip().replace("\\", "_").replace("/", "_")
    name = re.sub(r"[^\w\-. ]+", "_", name, flags=re.UNICODE)
    if not name.lower().endswith(".csv"):
        name = f"{name}.csv"
    return name


def _list_csv_files(input_dir: Path, include_test: bool) -> list[Path]:
    input_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(input_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if include_test:
        return files
    return [p for p in files if not p.name.lower().startswith("sample_")]


def sidebar_file_manager(input_dir: Path, logger: logging.Logger) -> tuple[list[Path], bool]:
    st.sidebar.header("Entrada (/input)")
    st.sidebar.caption("Suba um CSV e salve em /input, ou selecione arquivos já existentes.")

    include_test = st.sidebar.checkbox("Mostrar arquivos de teste (sample_*.csv)", value=False, key="show_tests")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Upload (opcional)")
    up = st.sidebar.file_uploader("Enviar CSV", type=["csv"], key="uploader")
    if up is not None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        suggested = _safe_filename(f"{Path(up.name).stem}_{ts}.csv")
        out_name = st.sidebar.text_input("Nome do arquivo ao salvar em /input", value=suggested, key="save_name")
        if st.sidebar.button("Salvar em /input", type="primary", key="save_btn"):
            input_dir.mkdir(parents=True, exist_ok=True)
            out_path = input_dir / _safe_filename(out_name)
            out_path.write_bytes(up.getbuffer())
            st.sidebar.success(f"Salvo: {out_path.name}")
            logger.info(f"UPLOAD_SAVED | {out_path}")

    files = _list_csv_files(input_dir, include_test=include_test)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Arquivos disponíveis")
    if not files:
        st.sidebar.warning("Nenhum CSV em /input.")
    else:
        st.sidebar.write(f"{len(files)} arquivo(s) detectado(s).")

    return files, include_test


# =========================
# CSV read + Standardization
# =========================
def read_csv_robust(path: Path) -> pd.DataFrame:
    """
    Lê CSV de forma robusta:
    - tenta encodings comuns
    - tenta separadores ',' e ';'
    - lê tudo como string (evita post_id virar float/scientific)
    """
    encodings = ["utf-8-sig", "utf-8", "cp1252"]
    seps = [",", ";"]
    last_err: Exception | None = None

    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(
                    path,
                    encoding=enc,
                    sep=sep,
                    dtype=str,
                    keep_default_na=False,  # vazio vira "" (a gente trata depois)
                )
                if df.shape[1] <= 1:
                    continue
                return df
            except Exception as e:
                last_err = e

    raise RuntimeError(f"Falha ao ler CSV '{path.name}'. Erro final: {last_err!r}")


def _blank_to_na(s: pd.Series | None) -> pd.Series | None:
    if s is None:
        return None
    out = s.astype(str).str.strip()
    out = out.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "null": pd.NA})
    return out


def _coalesce(a: pd.Series | None, b: pd.Series | None, index: pd.Index) -> pd.Series:
    a = _blank_to_na(a)
    b = _blank_to_na(b)
    if a is None and b is None:
        return pd.Series(pd.NA, index=index)
    if a is None:
        return b.reindex(index)
    if b is None:
        return a.reindex(index)
    return a.where(a.notna(), b).reindex(index)


def _to_int(s: pd.Series | None, index: pd.Index) -> pd.Series:
    s = _blank_to_na(s)
    if s is None:
        return pd.Series(0, index=index, dtype="Int64")

    # remove milhares e normaliza decimal pt-br (se existir)
    x = (
        s.astype(str)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(x, errors="coerce").fillna(0).round(0).astype("Int64").reindex(index)


def parse_published_at(series: pd.Series) -> pd.Series:
    """
    Parse robusto para 'Horário de publicação' (Meta).
    Aceita MM/DD, DD/MM, com/sem segundos, e fallback dayfirst.
    Nunca dá raise: retorna NaT onde falhar.
    """
    raw = _blank_to_na(series)  # pode virar NA
    if raw is None:
        return pd.Series(pd.NaT, dtype="datetime64[ns]")

    raw = raw.astype("string")
    raw = raw.str.replace(r"\s*(UTC|GMT.*)$", "", regex=True)

    fmts = [
        "%m/%d/%Y %H:%M",
        "%d/%m/%Y %H:%M",
        "%m/%d/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
    ]

    parsed = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")
    for fmt in fmts:
        p = pd.to_datetime(raw, format=fmt, errors="coerce")
        parsed = parsed.fillna(p)

    # fallbacks
    parsed = parsed.fillna(pd.to_datetime(raw, errors="coerce", dayfirst=False))
    parsed = parsed.fillna(pd.to_datetime(raw, errors="coerce", dayfirst=True))

    return parsed


def standardize_meta_business_instagram(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Standardiza export do Meta Business Suite (Instagram) para schema alvo do projeto.
    """
    cols = set(df_raw.columns)
    required = {"Horário de publicação", "Alcance"}
    if not required.issubset(cols):
        raise ValueError(f"Não parece export do Meta (Instagram). Faltando: {sorted(required - cols)}")

    post_id = _coalesce(df_raw.get("Identificação do post"), df_raw.get("Post ID"), df_raw.index)
    account_name = _coalesce(df_raw.get("Nome da conta"), df_raw.get("Account name"), df_raw.index)
    caption = _coalesce(df_raw.get("Descrição"), df_raw.get("Caption"), df_raw.index)
    permalink = _coalesce(df_raw.get("Link permanente"), df_raw.get("Permalink"), df_raw.index)
    post_type_raw = _coalesce(df_raw.get("Tipo de post"), df_raw.get("Post type"), df_raw.index)

    created_at = parse_published_at(df_raw["Horário de publicação"])

    post_type = (
        _blank_to_na(post_type_raw).astype("string").fillna("")
        .str.lower()
        .str.replace("carrossel do instagram", "carousel", regex=False)
        .str.replace("reel do instagram", "reel", regex=False)
        .str.replace("imagem do instagram", "image", regex=False)
    )

    out = pd.DataFrame(
        {
            "post_id": _blank_to_na(post_id),
            "account_name": _blank_to_na(account_name),
            "caption": _blank_to_na(caption),
            "created_at": created_at,
            "permalink": _blank_to_na(permalink),
            "post_type": post_type,

            "reach": _to_int(df_raw.get("Alcance"), df_raw.index),
            "views": _to_int(_coalesce(df_raw.get("Visualizações"), df_raw.get("Views"), df_raw.index), df_raw.index),
            "likes": _to_int(_coalesce(df_raw.get("Curtidas"), df_raw.get("Likes"), df_raw.index), df_raw.index),
            "comments": _to_int(df_raw.get("Comentários"), df_raw.index),
            "shares": _to_int(df_raw.get("Compartilhamentos"), df_raw.index),
            "saves": _to_int(df_raw.get("Salvamentos"), df_raw.index),
            "follows": _to_int(_coalesce(df_raw.get("Seguimentos"), df_raw.get("Follows"), df_raw.index), df_raw.index),

            "_published_at_raw": _blank_to_na(df_raw.get("Horário de publicação")),
            "_source": "meta_business_instagram",
        }
    )

    # post_id final como string
    out["post_id"] = out["post_id"].astype("string").str.strip()
    out.loc[out["post_id"].isin(["", "nan", "None"]), "post_id"] = pd.NA

    out["engagement"] = (out["likes"] + out["comments"] + out["shares"] + out["saves"]).astype("Int64")
    denom = out["reach"].replace({0: pd.NA}).astype("Float64")
    out["engagement_rate"] = (out["engagement"].astype("Float64") / denom).fillna(0.0)

    out["_published_at_parse_ok"] = out["created_at"].notna()

    bad = int((~out["_published_at_parse_ok"]).sum())
    if bad > 0:
        log.warning(f"DATE_PARSE_WARN | bad_rows={bad}")

    return out


def standardize_already_schema(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Caso o CSV já esteja no schema alvo (post_id, created_at, reach, likes, etc).
    """
    df = df_raw.copy()

    # normaliza nomes comuns
    rename = {
        "created_at": "created_at",
        "post_id": "post_id",
        "reach": "reach",
        "likes": "likes",
        "comments": "comments",
        "shares": "shares",
        "saves": "saves",
        "views": "views",
        "engagement": "engagement",
        "engagement_rate": "engagement_rate",
        "caption": "caption",
        "permalink": "permalink",
        "post_type": "post_type",
    }
    # mantém como está, mas garante colunas mínimas
    for col in list(rename.keys()):
        if col not in df.columns:
            df[col] = pd.NA

    # parse datetime
    df["created_at"] = pd.to_datetime(_blank_to_na(df["created_at"]), errors="coerce", dayfirst=True)

    # ints
    for c in ["reach", "views", "likes", "comments", "shares", "saves"]:
        df[c] = _to_int(df[c], df.index)

    if "engagement" in df.columns:
        df["engagement"] = _to_int(df["engagement"], df.index)
    else:
        df["engagement"] = (df["likes"] + df["comments"] + df["shares"] + df["saves"]).astype("Int64")

    denom = df["reach"].replace({0: pd.NA}).astype("Float64")
    if "engagement_rate" in df.columns:
        er = pd.to_numeric(_blank_to_na(df["engagement_rate"]), errors="coerce")
        df["engagement_rate"] = er.fillna((df["engagement"].astype("Float64") / denom)).fillna(0.0)
    else:
        df["engagement_rate"] = (df["engagement"].astype("Float64") / denom).fillna(0.0)

    df["_published_at_parse_ok"] = df["created_at"].notna()
    df["_source"] = "already_schema"
    return df


def standardize_posts_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    if "Horário de publicação" in df_raw.columns and "Alcance" in df_raw.columns:
        return standardize_meta_business_instagram(df_raw)

    # tentativa de schema alvo
    if "created_at" in df_raw.columns and "reach" in df_raw.columns:
        return standardize_already_schema(df_raw)

    raise ValueError("CSV desconhecido. Esperado export do Meta (Instagram) ou um CSV já no schema alvo.")


def file_signature(p: Path) -> tuple[str, int]:
    st_mtime = int(p.stat().st_mtime_ns) if p.exists() else 0
    return (str(p.resolve()), st_mtime)


@st.cache_data(show_spinner=False)
def load_and_standardize_csv(path_str: str, mtime_ns: int) -> pd.DataFrame:
    p = Path(path_str)
    df_raw = read_csv_robust(p)
    df = standardize_posts_df(df_raw)
    df["_file"] = p.name
    return df


def load_one_csv(p: Path) -> pd.DataFrame:
    path_str, mt = file_signature(p)
    df = load_and_standardize_csv(path_str, mt)
    return df


def dedupe_posts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dedupe seguro para dataset mesclado:
    - chave preferencial: post_id
    - fallback: permalink
    - fallback: created_at + caption
    Mantém a linha com maior reach e, em empate, maior engagement.
    """
    d = df.copy()

    key = d["post_id"].astype("string")
    key = key.where(key.notna(), d["permalink"].astype("string"))
    fallback = (d["created_at"].astype("string").fillna("") + "||" + d["caption"].astype("string").fillna(""))
    key = key.where(key.notna(), fallback)

    d["_dedupe_key"] = key.fillna(fallback)

    # rank por reach, engagement
    d["_rank_reach"] = d["reach"].astype("Int64").fillna(0)
    d["_rank_eng"] = d["engagement"].astype("Int64").fillna(0)

    d = d.sort_values(["_dedupe_key", "_rank_reach", "_rank_eng"], ascending=[True, False, False])
    d = d.drop_duplicates(subset=["_dedupe_key"], keep="first")

    d = d.drop(columns=["_dedupe_key", "_rank_reach", "_rank_eng"], errors="ignore")
    return d


@st.cache_data(show_spinner=False)
def load_merge_standardize(paths_and_mtimes: tuple[tuple[str, int], ...]) -> pd.DataFrame:
    dfs = []
    for path_str, mt in paths_and_mtimes:
        dfs.append(load_and_standardize_csv(path_str, mt))
    merged = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    if len(merged) > 0:
        merged = dedupe_posts(merged)
    return merged


# =========================
# Metrics / Aggregations
# =========================
@dataclass(frozen=True)
class KPIs:
    posts: int
    reach_total: int
    engagement_total: int
    engagement_rate: float


def compute_kpis(dff: pd.DataFrame) -> KPIs:
    posts = int(len(dff))
    reach = int(dff["reach"].sum()) if posts else 0
    eng = int(dff["engagement"].sum()) if posts else 0
    er = (eng / reach) if reach > 0 else 0.0
    return KPIs(posts=posts, reach_total=reach, engagement_total=eng, engagement_rate=er)


def fmt_int(x: int) -> str:
    return f"{x:,}".replace(",", ".")


def fmt_pct(x: float) -> str:
    return f"{x:.2%}"


def daily_agg(dff: pd.DataFrame) -> pd.DataFrame:
    daily = (
        dff.assign(pub_date=dff["created_at"].dt.date)
        .groupby("pub_date", as_index=False)
        .agg(posts=("post_id", "count"), reach=("reach", "sum"), engagement=("engagement", "sum"))
        .sort_values("pub_date")
    )
    daily["engagement_rate"] = np.where(daily["reach"] > 0, daily["engagement"] / daily["reach"], 0.0)
    return daily


def weekly_agg(dff: pd.DataFrame) -> pd.DataFrame:
    iso = dff["created_at"].dt.isocalendar()
    w = (
        dff.assign(iso_year=iso["year"], iso_week=iso["week"])
        .groupby(["iso_year", "iso_week"], as_index=False)
        .agg(posts=("post_id", "count"), reach=("reach", "sum"), engagement=("engagement", "sum"))
        .sort_values(["iso_year", "iso_week"])
    )
    w["engagement_rate"] = np.where(w["reach"] > 0, w["engagement"] / w["reach"], 0.0)
    w["year_week"] = w["iso_year"].astype(str) + "-W" + w["iso_week"].astype(str).str.zfill(2)
    return w


def top_posts(dff: pd.DataFrame, metric: str, n: int = 5) -> pd.DataFrame:
    cols = ["post_id", "created_at", "post_type", "reach", "engagement", "views", "likes", "comments", "shares", "saves", "permalink", "_file"]
    cols = [c for c in cols if c in dff.columns]
    return dff.sort_values(metric, ascending=False).head(n)[cols]


def filter_period(dff: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    mask = (dff["created_at"].dt.date >= start) & (dff["created_at"].dt.date <= end)
    return dff.loc[mask].copy()


def month_start_end(y: int, m: int) -> tuple[date, date]:
    start = date(y, m, 1)
    if m == 12:
        end = date(y, 12, 31)
    else:
        end = (pd.Timestamp(y, m + 1, 1) - pd.Timedelta(days=1)).date()
    return start, end


def available_years(dff: pd.DataFrame) -> list[int]:
    yrs = sorted(dff["created_at"].dt.year.dropna().astype(int).unique().tolist())
    return yrs


# =========================
# Exports
# =========================
def export_tables(prefix: str, df_posts: pd.DataFrame, daily: pd.DataFrame, weekly: pd.DataFrame) -> list[Path]:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out1 = TABLES_DIR / f"{prefix}__posts.csv"
    out2 = TABLES_DIR / f"{prefix}__daily.csv"
    out3 = TABLES_DIR / f"{prefix}__weekly.csv"
    df_posts.to_csv(out1, index=False, encoding="utf-8-sig")
    daily.to_csv(out2, index=False, encoding="utf-8-sig")
    weekly.to_csv(out3, index=False, encoding="utf-8-sig")
    return [out1, out2, out3]


def export_pngs_matplotlib(prefix: str, daily: pd.DataFrame) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []

    fig1 = plt.figure()
    plt.plot(daily["pub_date"], daily["reach"])
    plt.title("Reach por dia (posts publicados)")
    plt.xticks(rotation=35, ha="right")
    out1 = CHARTS_DIR / f"{prefix}__daily_reach.png"
    fig1.savefig(out1, dpi=160, bbox_inches="tight")
    plt.close(fig1)
    files.append(out1)

    fig2 = plt.figure()
    plt.plot(daily["pub_date"], daily["engagement_rate"])
    plt.title("Engagement rate por dia (Eng/Reach)")
    plt.xticks(rotation=35, ha="right")
    out2 = CHARTS_DIR / f"{prefix}__daily_engagement_rate.png"
    fig2.savefig(out2, dpi=160, bbox_inches="tight")
    plt.close(fig2)
    files.append(out2)

    return files


def list_recent(dir_path: Path, pattern: str, n: int = 20) -> list[str]:
    dir_path.mkdir(parents=True, exist_ok=True)
    files = sorted(dir_path.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)[:n]
    return [f.name for f in files]


# =========================
# UI helpers (time filters)
# =========================
def time_filter_ui(prefix: str, dff: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Retorna (df_filtrado, label_periodo).
    """
    dff = dff.loc[dff["created_at"].notna()].copy()
    if dff.empty:
        return dff, "Sem dados com data válida."

    min_dt = dff["created_at"].min().date()
    max_dt = dff["created_at"].max().date()

    mode = st.selectbox(
        "Filtro de tempo",
        options=["Tudo", "Ano", "Mês", "Semana (ISO)", "Dia", "Período livre"],
        index=0,
        key=f"{prefix}_mode",
    )

    if mode == "Tudo":
        return dff, f"{min_dt} → {max_dt}"

    if mode == "Ano":
        yrs = available_years(dff)
        year = st.selectbox("Ano", options=yrs, index=len(yrs) - 1, key=f"{prefix}_year")
        start, end = date(int(year), 1, 1), date(int(year), 12, 31)
        return filter_period(dff, start, end), f"{start} → {end}"

    if mode == "Mês":
        yrs = available_years(dff)
        year = st.selectbox("Ano", options=yrs, index=len(yrs) - 1, key=f"{prefix}_myear")
        month = st.selectbox("Mês", options=list(range(1, 13)), index=0, key=f"{prefix}_month")
        start, end = month_start_end(int(year), int(month))
        return filter_period(dff, start, end), f"{start} → {end}"

    if mode == "Semana (ISO)":
        iso = dff["created_at"].dt.isocalendar()
        dff["_iso_year"] = iso["year"].astype(int)
        dff["_iso_week"] = iso["week"].astype(int)
        dff["_year_week"] = dff["_iso_year"].astype(str) + "-W" + dff["_iso_week"].astype(str).str.zfill(2)
        options = sorted(dff["_year_week"].dropna().unique().tolist())
        selected = st.selectbox("Semana (ISO)", options=options, index=len(options) - 1, key=f"{prefix}_week")
        out = dff.loc[dff["_year_week"] == selected].copy()
        out = out.drop(columns=["_iso_year", "_iso_week", "_year_week"], errors="ignore")
        return out, f"Semana {selected}"

    if mode == "Dia":
        day = st.date_input("Dia", value=max_dt, min_value=min_dt, max_value=max_dt, key=f"{prefix}_day")
        out = dff.loc[dff["created_at"].dt.date == day].copy()
        return out, f"Dia {day}"

    # período livre
    start, end = st.date_input(
        "Intervalo",
        value=(min_dt, max_dt),
        min_value=min_dt,
        max_value=max_dt,
        key=f"{prefix}_range",
    )
    return filter_period(dff, start, end), f"{start} → {end}"


def compare_period_ui(prefix: str, dff: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """
    UI de comparação A vs B dentro do MESMO dataset (geralmente mesclado).
    """
    dff = dff.loc[dff["created_at"].notna()].copy()
    if dff.empty:
        return dff, dff, "Sem dados", "Sem dados"

    min_dt = dff["created_at"].min().date()
    max_dt = dff["created_at"].max().date()

    mode = st.selectbox(
        "Modo de comparação",
        options=["Ano vs Ano", "Mês vs Mês", "Período livre (A e B)"],
        index=0,
        key=f"{prefix}_cmp_mode",
    )

    yrs = available_years(dff)
    if not yrs:
        return dff, dff, "Sem datas", "Sem datas"

    if mode == "Ano vs Ano":
        year_a = st.selectbox("Ano A", options=yrs, index=len(yrs) - 1, key=f"{prefix}_ya")
        year_b = st.selectbox("Ano B", options=yrs, index=max(len(yrs) - 2, 0), key=f"{prefix}_yb")
        a = filter_period(dff, date(int(year_a), 1, 1), date(int(year_a), 12, 31))
        b = filter_period(dff, date(int(year_b), 1, 1), date(int(year_b), 12, 31))
        return a, b, f"{year_a}", f"{year_b}"

    if mode == "Mês vs Mês":
        month = st.selectbox("Mês", options=list(range(1, 13)), index=0, key=f"{prefix}_m")
        year_a = st.selectbox("Ano A", options=yrs, index=len(yrs) - 1, key=f"{prefix}_mya")
        year_b = st.selectbox("Ano B", options=yrs, index=max(len(yrs) - 2, 0), key=f"{prefix}_myb")
        sa, ea = month_start_end(int(year_a), int(month))
        sb, eb = month_start_end(int(year_b), int(month))
        a = filter_period(dff, sa, ea)
        b = filter_period(dff, sb, eb)
        return a, b, f"{year_a}-{int(month):02d}", f"{year_b}-{int(month):02d}"

    # período livre
    (sa, ea) = st.date_input("Período A", value=(min_dt, max_dt), min_value=min_dt, max_value=max_dt, key=f"{prefix}_ra")
    (sb, eb) = st.date_input("Período B", value=(min_dt, max_dt), min_value=min_dt, max_value=max_dt, key=f"{prefix}_rb")
    a = filter_period(dff, sa, ea)
    b = filter_period(dff, sb, eb)
    return a, b, f"{sa}→{ea}", f"{sb}→{eb}"


def ai_insights_ui(prefix: str, dff: pd.DataFrame) -> None:
    st.subheader("Insights com IA")
    st.caption("Gere insights automáticos com base nos dados filtrados.")

    api_key_default = os.getenv("OPENAI_API_KEY", "")
    api_key = st.text_input(
        "OpenAI API Key",
        value=api_key_default,
        type="password",
        key=f"{prefix}_api_key",
        help="Defina sua chave para gerar insights. Ela não é salva no projeto.",
    )
    model = st.text_input(
        "Modelo",
        value="gpt-4o-mini",
        key=f"{prefix}_model",
        help="Ex.: gpt-4o-mini, gpt-4o, ou outro modelo disponível na sua conta.",
    )
    max_posts = st.slider(
        "Quantidade de posts para análise (amostra)",
        min_value=5,
        max_value=200,
        value=10,
        step=1,
        key=f"{prefix}_max_posts",
    )

    if st.button("Gerar insights", key=f"{prefix}_run"):
        if dff.empty:
            st.warning("Sem dados após filtros. Ajuste o período ou selecione outros arquivos.")
            return
        if not api_key.strip():
            st.warning("Informe sua OpenAI API Key para continuar.")
            return
        with st.spinner("Gerando insights..."):
            insights = generate_instagram_insights(dff, api_key=api_key.strip(), model=model.strip(), max_posts=max_posts)
        st.markdown(insights)


# =========================
# App
# =========================
def main() -> None:
    st.set_page_config(page_title="Dashboard Performance", layout="wide")

    files, include_test = sidebar_file_manager(INPUT_DIR, logger=log)

    if st.sidebar.button("Limpar cache do Streamlit"):
        st.cache_data.clear()
        st.sidebar.success("Cache limpo. Recarregue a página se necessário.")

    if not files:
        st.title("Dashboard Performance")
        st.warning("Nenhum CSV em /input. Faça upload e salve para começar.")
        st.stop()

    # =========================
    # TABS
    # =========================
    tab1, tab2 = st.tabs(["Analisar 1 CSV", "Comparar (multi CSV)"])

    # ---------------------------------
    # TAB 1: Analisa 1 arquivo
    # ---------------------------------
    with tab1:
        st.title("Dashboard Performance")
        st.caption("Análise de 1 CSV por vez (ano / mês / semana / dia / intervalo).")

        file_names = [p.name for p in files]
        sel_name = st.selectbox("Escolha o CSV para analisar", options=file_names, index=0, key="single_file")
        p = INPUT_DIR / sel_name

        try:
            df = load_one_csv(p)
        except Exception as e:
            log.exception(f"SINGLE_LOAD_FAIL | file={p.name}")
            st.error("Falha ao carregar/standardizar o CSV.")
            st.exception(e)
            st.stop()

        # diagnóstico de datas inválidas
        bad = int((~df["_published_at_parse_ok"]).sum()) if "_published_at_parse_ok" in df.columns else 0
        if bad > 0:
            st.warning(f"{bad} linha(s) com data inválida. Elas serão ignoradas nos filtros/séries.")
            with st.expander("Ver linhas problemáticas", expanded=False):
                st.dataframe(df.loc[~df["_published_at_parse_ok"], ["post_id", "_published_at_raw", "reach", "permalink", "_file"]], use_container_width=True)

        # filtros gerais
        st.subheader("Filtros gerais")
        c1, c2, c3 = st.columns(3)
        with c1:
            types = sorted(df["post_type"].dropna().astype(str).unique().tolist())
            selected_types = st.multiselect("Tipo de post", options=types, default=types, key="single_types")
        with c2:
            reach_floor = st.number_input("Reach mínimo", min_value=0, value=0, step=100, key="single_reach_min")
        with c3:
            q = st.text_input("Buscar na legenda (caption)", value="", key="single_q").strip().lower()

        df_base = df.loc[df["created_at"].notna()].copy()
        if selected_types:
            df_base = df_base.loc[df_base["post_type"].astype(str).isin(selected_types)]
        if reach_floor > 0:
            df_base = df_base.loc[df_base["reach"].astype("Int64") >= int(reach_floor)]
        if q:
            df_base = df_base.loc[df_base["caption"].astype(str).str.lower().str.contains(re.escape(q), na=False)]

        st.subheader("Filtro de tempo (neste CSV)")
        df_f, label = time_filter_ui("single_time", df_base)
        st.caption(f"Período selecionado: {label} | Linhas: {len(df_f)}")

        # KPIs
        k = compute_kpis(df_f)
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Posts", fmt_int(k.posts))
        a2.metric("Reach", fmt_int(k.reach_total))
        a3.metric("Engagement", fmt_int(k.engagement_total))
        a4.metric("Engagement rate", fmt_pct(k.engagement_rate))

        # tendências
        st.subheader("Evolução")
        daily = daily_agg(df_f)
        weekly = weekly_agg(df_f)

        colA, colB = st.columns(2)
        fig_d = px.line(daily, x="pub_date", y="reach", markers=True, title="Reach por dia")
        colA.plotly_chart(fig_d, use_container_width=True)

        fig_w = px.bar(weekly, x="year_week", y="reach", title="Reach por semana (ISO)")
        colB.plotly_chart(fig_w, use_container_width=True)

        # top posts
        st.subheader("Top posts")
        t1, t2 = st.columns(2)
        t1.markdown("**Top 10 por Reach**")
        t1.dataframe(top_posts(df_f, "reach", n=10), use_container_width=True, hide_index=True)
        t2.markdown("**Top 10 por Engagement**")
        t2.dataframe(top_posts(df_f, "engagement", n=10), use_container_width=True, hide_index=True)

        ai_insights_ui("single_ai", df_f)

        # export
        st.subheader("Export para /output (este CSV filtrado)")
        prefix_default = f"{Path(p.name).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        prefix = st.text_input("Prefixo", value=prefix_default, key="single_prefix")

        b1, b2, b3 = st.columns(3)
        do_tables = b1.button("Exportar tabelas (CSV)", key="single_tables")
        do_pngs = b2.button("Exportar PNGs (Matplotlib)", key="single_pngs")
        do_all = b3.button("Exportar tudo", key="single_all")

        exported: list[Path] = []
        if do_tables or do_all:
            exported += export_tables(prefix, df_f, daily, weekly)
            st.success("Tabelas exportadas.")
        if do_pngs or do_all:
            exported += export_pngs_matplotlib(prefix, daily)
            st.success("PNGs exportados.")

        if exported:
            st.code("\n".join([str(x) for x in exported]))

        st.caption("Últimos PNGs em /output/charts:")
        st.write(list_recent(CHARTS_DIR, "*.png", n=15))
        st.caption("Últimos CSVs em /output/tables:")
        st.write(list_recent(TABLES_DIR, "*.csv", n=15))

    # ---------------------------------
    # TAB 2: Compara multi CSV (merge)
    # ---------------------------------
    with tab2:
        st.title("Comparar (multi CSV)")
        st.caption("Selecione vários CSVs (ex.: 2024 + 2025), mescle em um dataset e compare A vs B (Ano vs Ano / Mês vs Mês).")

        file_names = [p.name for p in files]
        selected_files = st.multiselect(
            "Selecione os CSVs que entram no dataset mesclado",
            options=file_names,
            default=file_names[:2] if len(file_names) >= 2 else file_names,
            key="multi_files",
        )

        if not selected_files:
            st.warning("Selecione pelo menos 1 CSV.")
            st.stop()

        paths = [INPUT_DIR / n for n in selected_files]
        sig = tuple(file_signature(p) for p in paths)

        try:
            dfm = load_merge_standardize(sig)
        except Exception as e:
            log.exception("MULTI_LOAD_FAIL")
            st.error("Falha ao carregar/mesclar os CSVs.")
            st.exception(e)
            st.stop()

        st.success(f"Dataset mesclado carregado: {len(dfm)} linha(s) | Arquivos: {len(selected_files)}")
        st.caption("Dica: se você escolheu 2024 e 2025 aqui, agora o 'Ano A' e 'Ano B' vão aparecer corretamente (sem travar).")

        # filtros gerais (antes da comparação)
        st.subheader("Filtros gerais (aplicados antes de A vs B)")
        c1, c2, c3 = st.columns(3)
        with c1:
            types = sorted(dfm["post_type"].dropna().astype(str).unique().tolist())
            selected_types = st.multiselect("Tipo de post", options=types, default=types, key="multi_types")
        with c2:
            reach_floor = st.number_input("Reach mínimo", min_value=0, value=0, step=100, key="multi_reach_min")
        with c3:
            q = st.text_input("Buscar na legenda (caption)", value="", key="multi_q").strip().lower()

        base = dfm.loc[dfm["created_at"].notna()].copy()
        if selected_types:
            base = base.loc[base["post_type"].astype(str).isin(selected_types)]
        if reach_floor > 0:
            base = base.loc[base["reach"].astype("Int64") >= int(reach_floor)]
        if q:
            base = base.loc[base["caption"].astype(str).str.lower().str.contains(re.escape(q), na=False)]

        st.subheader("Comparação A vs B (dentro do dataset mesclado)")
        dfa, dfb, label_a, label_b = compare_period_ui("multi_cmp", base)

        # KPIs A vs B
        k_a = compute_kpis(dfa)
        k_b = compute_kpis(dfb)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Posts (A)", fmt_int(k_a.posts))
        m2.metric("Reach (A)", fmt_int(k_a.reach_total))
        m3.metric("Engagement (A)", fmt_int(k_a.engagement_total))
        m4.metric("Eng. rate (A)", fmt_pct(k_a.engagement_rate))
        st.caption(f"A: {label_a} | B: {label_b} | A linhas={len(dfa)} | B linhas={len(dfb)}")

        st.subheader("Evolução (Reach por dia)")
        da = daily_agg(dfa)
        db = daily_agg(dfb)
        ca, cb = st.columns(2)
        ca.plotly_chart(px.line(da, x="pub_date", y="reach", markers=True, title=f"Reach por dia — A ({label_a})"), use_container_width=True)
        cb.plotly_chart(px.line(db, x="pub_date", y="reach", markers=True, title=f"Reach por dia — B ({label_b})"), use_container_width=True)

        st.subheader("Top posts — A vs B")
        t1, t2 = st.columns(2)
        t1.markdown("**Top 10 por Reach — A**")
        t1.dataframe(top_posts(dfa, "reach", n=10), use_container_width=True, hide_index=True)
        t2.markdown("**Top 10 por Reach — B**")
        t2.dataframe(top_posts(dfb, "reach", n=10), use_container_width=True, hide_index=True)

        u1, u2 = st.columns(2)
        u1.markdown("**Top 10 por Engagement — A**")
        u1.dataframe(top_posts(dfa, "engagement", n=10), use_container_width=True, hide_index=True)
        u2.markdown("**Top 10 por Engagement — B**")
        u2.dataframe(top_posts(dfb, "engagement", n=10), use_container_width=True, hide_index=True)

        ai_insights_ui("multi_ai_a", dfa)
        ai_insights_ui("multi_ai_b", dfb)

        st.subheader("Export para /output (A e B)")
        prefix_default = f"merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        prefix = st.text_input("Prefixo", value=prefix_default, key="multi_prefix")

        b1, b2, b3 = st.columns(3)
        do_tables = b1.button("Exportar tabelas (CSV)", key="multi_tables")
        do_pngs = b2.button("Exportar PNGs (Matplotlib)", key="multi_pngs")
        do_all = b3.button("Exportar tudo", key="multi_all")

        exported: list[Path] = []
        if do_tables or do_all:
            exported += export_tables(prefix + "__A", dfa, daily_agg(dfa), weekly_agg(dfa))
            exported += export_tables(prefix + "__B", dfb, daily_agg(dfb), weekly_agg(dfb))
            st.success("Tabelas exportadas (A e B).")
        if do_pngs or do_all:
            exported += export_pngs_matplotlib(prefix + "__A", daily_agg(dfa))
            exported += export_pngs_matplotlib(prefix + "__B", daily_agg(dfb))
            st.success("PNGs exportados (A e B).")

        if exported:
            st.code("\n".join([str(x) for x in exported]))

        st.caption("Últimos PNGs em /output/charts:")
        st.write(list_recent(CHARTS_DIR, "*.png", n=15))
        st.caption("Últimos CSVs em /output/tables:")
        st.write(list_recent(TABLES_DIR, "*.csv", n=15))


if __name__ == "__main__":
    main()
