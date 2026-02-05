from __future__ import annotations

import csv
import logging
import re
import unicodedata
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import pandas as pd
from pandas.errors import ParserError

logger = logging.getLogger(__name__)


# -----------------------------
# Schema alvo (canônico)
# -----------------------------
REQUIRED_COLUMNS: Tuple[str, ...] = (
    "post_id",
    "created_at",
    "caption",
    "reach",
    "likes",
    "comments",
    "shares",
    "saves",
)

# Aliases comuns (PT-BR + EN) de exports
COLUMN_ALIASES: Mapping[str, Sequence[str]] = {
    "post_id": [
        "post_id",
        "post id",
        "id",
        "media id",
        "media_id",
        "id da publicacao",
        "id da publicação",
        "id da postagem",
        "id do post",
        "identificador",
        "codigo",
        "código",
    ],
    "created_at": [
        "created_at",
        "created at",
        "date",
        "data",
        "data de publicacao",
        "data de publicação",
        "data/hora",
        "data hora",
        "timestamp",
        "published at",
        "hora",
    ],
    "caption": [
        "caption",
        "legenda",
        "texto",
        "descricao",
        "descrição",
        "message",
        "conteudo",
        "conteúdo",
        "post text",
        "titulo",
        "título",
    ],
    "reach": [
        "reach",
        "alcance",
        "total reach",
        "accounts reached",
        "contas alcancadas",
        "contas alcançadas",
    ],
    "likes": [
        "likes",
        "curtidas",
        "curtida",
        "me gusta",
    ],
    "comments": [
        "comments",
        "comentarios",
        "comentários",
    ],
    "shares": [
        "shares",
        "compartilhamentos",
        "compartilhamento",
        "shared",
    ],
    "saves": [
        "saves",
        "salvamentos",
        "salvamento",
        "saved",
    ],
}


def normalize_colname(name: str) -> str:
    """
    Normaliza nomes de colunas para facilitar mapeamento:
    - lowercase
    - remove acentos
    - troca não-alfanum por underscore
    - colapsa underscores repetidos
    """
    s = str(name).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _build_alias_map(aliases: Mapping[str, Sequence[str]]) -> Dict[str, str]:
    """Cria mapa: alias_normalizado -> canonical."""
    out: Dict[str, str] = {}
    for canonical, names in aliases.items():
        for n in names:
            out[normalize_colname(n)] = canonical
    return out


ALIAS_MAP = _build_alias_map(COLUMN_ALIASES)


def read_csv_robust(path: Path) -> pd.DataFrame:
    """
    Lê CSV com tolerância a:
    - separador ; ou ,
    - utf-8 / utf-8-sig / latin-1
    - mantém tudo como string inicialmente (dtype=str)

    Além disso, tem fallback para CSV com aspas quebradas:
    - 1ª tentativa: parsing normal (respeita quotes)
    - 2ª tentativa: modo leniente (ignora quotes como delimitador)
    """
    encodings = ["utf-8", "utf-8-sig", "latin-1"]
    last_err: Exception | None = None

    # Tentativa A: parsing normal (padrão)
    def _try_normal(enc: str) -> pd.DataFrame:
        return pd.read_csv(
            path,
            sep=None,            # autodetect ; ou ,
            engine="python",
            encoding=enc,
            dtype=str,
            na_filter=False,
            on_bad_lines="warn",
        )

    # Tentativa B: modo leniente (ignora aspas como quotes)
    # Isso costuma salvar CSV “sujo” que tem " sobrando dentro de texto.
    def _try_lenient(enc: str) -> pd.DataFrame:
        return pd.read_csv(
            path,
            sep=None,
            engine="python",
            encoding=enc,
            dtype=str,
            na_filter=False,
            on_bad_lines="warn",
            quoting=csv.QUOTE_NONE,   # <- ignora quotes
        )

    for enc in encodings:
        # 1) normal
        try:
            df = _try_normal(enc)
            logger.info("CSV lido (normal): %s (encoding=%s) | linhas=%s cols=%s", path.name, enc, len(df), len(df.columns))
            return df
        except ParserError as e:
            last_err = e
        except Exception as e:
            last_err = e

        # 2) leniente
        try:
            df = _try_lenient(enc)
            logger.warning("CSV lido (lenient/QUOTE_NONE): %s (encoding=%s) | linhas=%s cols=%s", path.name, enc, len(df), len(df.columns))
            return df
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Falha ao ler CSV '{path.name}'. Erro final: {last_err!r}")


def parse_count(value: object) -> int:
    """
    Converte contagens em int. Tolera:
    - "", None -> 0
    - "1.234" / "1,234" / "1 234" -> 1234
    - "1,2 mil" / "1.2k" -> 1200
    - "3M" / "2,5 mi" -> 2500000
    """
    if value is None:
        return 0

    if isinstance(value, (int, float)):
        if pd.isna(value):
            return 0
        return int(round(float(value)))

    s = str(value).strip().lower()
    if s == "" or s in {"nan", "none", "null", "-"}:
        return 0

    s = re.sub(r"\s+", "", s)

    # k/m (pt/en)
    m = re.match(r"^([0-9]+(?:[.,][0-9]+)?)((k|mil)|(m|mi|milhao|milhoes|milhões))$", s)
    if m:
        num = m.group(1).replace(".", "").replace(",", ".")
        mult_raw = m.group(2)
        mult = 1
        if mult_raw in {"k", "mil"}:
            mult = 1_000
        elif mult_raw in {"m", "mi", "milhao", "milhões", "milhoes"}:
            mult = 1_000_000
        return int(round(float(num) * mult))

    # remove tudo que não for dígito (tolerante a separadores)
    digits = re.sub(r"[^0-9]", "", s)
    if digits == "":
        return 0
    return int(digits)


def map_and_normalize_columns(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Renomeia colunas para schema alvo usando aliases.
    Lança ValueError com mensagem útil se faltar coluna obrigatória.
    """
    original_cols = list(df_raw.columns)
    normalized_cols = {c: normalize_colname(c) for c in original_cols}

    rename_map: Dict[str, str] = {}
    used_canon: set[str] = set()

    for orig, norm in normalized_cols.items():
        canonical = ALIAS_MAP.get(norm)
        if not canonical:
            continue
        if canonical in used_canon:
            logger.warning("Coluna duplicada mapeando para '%s'. Mantendo a primeira; ignorando: %s", canonical, orig)
            continue
        rename_map[orig] = canonical
        used_canon.add(canonical)

    df = df_raw.rename(columns=rename_map).copy()

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        msg = (
            "CSV não tem colunas suficientes para o schema alvo.\n"
            f"Faltando (obrigatórias): {missing}\n\n"
            f"Colunas originais: {original_cols}\n"
            f"Colunas normalizadas: {[normalized_cols[c] for c in original_cols]}\n\n"
            "Como corrigir:\n"
            "- Se o export usa nomes diferentes, adicione o nome em COLUMN_ALIASES no arquivo src/etl/posts_ingest.py\n"
            "- Ou ajuste o CSV para conter as colunas mínimas do schema alvo.\n"
        )
        raise ValueError(msg)

    return df


def coerce_types_and_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Converte tipos e cria métricas derivadas (engagement, engagement_rate)."""
    out = df.copy()

    # limpeza básica de strings (remove CR/LF e aspas sobrando nas pontas)
    out["post_id"] = out["post_id"].astype(str).fillna("").str.strip().str.strip('"')
    out["caption"] = (
        out["caption"]
        .astype(str)
        .fillna("")
        .str.replace("\r", " ", regex=False)
        .str.replace("\n", " ", regex=False)
        .str.strip()
        .str.strip('"')
    )

    # datetime (tolerante a formatos variados)
    out["created_at"] = pd.to_datetime(out["created_at"], errors="coerce", dayfirst=True)

    # contagens
    for col in ["reach", "likes", "comments", "shares", "saves"]:
        out[col] = out[col].map(parse_count).astype("int64")

    out["engagement"] = (out["likes"] + out["comments"] + out["shares"] + out["saves"]).astype("int64")
    out["engagement_rate"] = out.apply(
        lambda r: float(r["engagement"]) / float(r["reach"]) if r["reach"] > 0 else 0.0,
        axis=1,
    )

    return out


def load_posts_csv(path: Path) -> pd.DataFrame:
    """
    Pipeline mínimo:
    1) read_csv_robust
    2) map_and_normalize_columns (schema alvo)
    3) coerce_types_and_metrics (tipos + métricas)
    """
    df_raw = read_csv_robust(path)
    df_mapped = map_and_normalize_columns(df_raw)
    df_final = coerce_types_and_metrics(df_mapped)

    # ordenação padrão
    df_final = df_final.sort_values("created_at", ascending=False, na_position="last")

    logger.info("Pipeline OK: %s | linhas=%s", path.name, len(df_final))
    return df_final
