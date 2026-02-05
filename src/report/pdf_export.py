from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def convert_docx_to_pdf(docx_path: Path, timeout_seconds: int = 60) -> Path:
    """
    Converte DOCX para PDF usando docx2pdf em subprocesso com timeout.
    Requer Microsoft Word instalado (Windows).
    """
    if not docx_path.exists():
        raise FileNotFoundError(f"DOCX não encontrado: {docx_path}")

    pdf_path = docx_path.with_suffix(".pdf")
    worker = Path(__file__).with_name("pdf_worker.py")
    if not worker.exists():
        raise RuntimeError(f"pdf_worker.py não encontrado: {worker}")

    cmd = [sys.executable, str(worker), str(docx_path), str(pdf_path)]
    logger.info("PDF convert START | %s -> %s | timeout=%ss", str(docx_path), str(pdf_path), timeout_seconds)

    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds)
    except subprocess.TimeoutExpired as e:
        msg = f"Timeout convertendo PDF após {timeout_seconds}s. Verifique se o Word está instalado e não travou."
        logger.exception(msg)
        raise RuntimeError(msg) from e

    if p.returncode != 0:
        msg = f"Falha ao converter PDF.\nSTDOUT:\n{p.stdout}\n\nSTDERR:\n{p.stderr}"
        logger.error(msg)
        raise RuntimeError(msg)

    if not pdf_path.exists():
        raise RuntimeError(f"PDF não foi criado: {pdf_path}")

    logger.info("PDF convert OK | %s | bytes=%d", str(pdf_path), pdf_path.stat().st_size)
    return pdf_path
