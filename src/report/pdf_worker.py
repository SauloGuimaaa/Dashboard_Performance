from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    """
    CLI:
      python pdf_worker.py <input_docx> <output_pdf>
    Requer: docx2pdf + Microsoft Word instalado (Windows).
    """
    if len(sys.argv) != 3:
        print("Usage: python pdf_worker.py <input_docx> <output_pdf>", file=sys.stderr)
        return 2

    docx_path = Path(sys.argv[1])
    pdf_path = Path(sys.argv[2])

    if not docx_path.exists():
        print(f"ERROR: docx not found: {docx_path}", file=sys.stderr)
        return 3

    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    from docx2pdf import convert  # import local

    # docx2pdf: se output é arquivo, passa path do pdf.
    convert(str(docx_path), str(pdf_path))

    if not pdf_path.exists():
        print(f"ERROR: pdf not created: {pdf_path}", file=sys.stderr)
        return 4

    print(f"OK: {pdf_path} ({pdf_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
