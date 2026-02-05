from __future__ import annotations

import json
import sys
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio


def _set_kaleido_timeout(seconds: int) -> None:
    """
    Tenta configurar timeout do Kaleido se disponível.
    (Nem toda versão expõe isso, então é best-effort.)
    """
    try:
        kaleido = getattr(pio, "kaleido", None)
        scope = getattr(kaleido, "scope", None) if kaleido else None
        if scope is not None and hasattr(scope, "default_timeout"):
            scope.default_timeout = seconds
    except Exception:
        pass


def main() -> int:
    """
    CLI:
      python png_worker.py <figure_json_path> <out_png_path> <width> <height> <scale>
    """
    if len(sys.argv) != 6:
        print("Usage: python png_worker.py <figure_json_path> <out_png_path> <width> <height> <scale>", file=sys.stderr)
        return 2

    fig_json_path = Path(sys.argv[1])
    out_png_path = Path(sys.argv[2])
    width = int(sys.argv[3])
    height = int(sys.argv[4])
    scale = int(sys.argv[5])

    _set_kaleido_timeout(30)

    fig_dict = json.loads(fig_json_path.read_text(encoding="utf-8"))
    fig = go.Figure(fig_dict)
    fig.update_layout(width=width, height=height)

    out_png_path.parent.mkdir(parents=True, exist_ok=True)

    # Export PNG (ponto que estava “pendurando” dentro do Streamlit)
    pio.write_image(fig, str(out_png_path), format="png", width=width, height=height, scale=scale)

    if not out_png_path.exists():
        print(f"ERROR: output not created: {out_png_path}", file=sys.stderr)
        return 3

    print(f"OK: {out_png_path} ({out_png_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
