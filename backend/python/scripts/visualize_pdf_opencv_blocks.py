#!/usr/bin/env python3
"""CLI: visualize ``PyMuPDFOpenCVProcessor`` blocks (highlights + type labels).

Run from ``backend/python`` with the package on ``PYTHONPATH``::

    cd backend/python
    PYTHONPATH=. python scripts/visualize_pdf_opencv_blocks.py /path/to/file.pdf -o /tmp/out

Table extraction calls LLMs in production; this script mocks those calls so it runs offline.

Optional: set env ``PIPESHUB_DEBUG_PDF_BLOCKS_OUT=/tmp/overlays`` while parsing PDFs in the app to
write the same PNGs automatically (see ``parse_file.handle_pdf``).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


def _ensure_pythonpath() -> None:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


async def _run(input_pdf: Path, output_dir: Path, dpi: int) -> None:
    _ensure_pythonpath()

    log = logging.getLogger("visualize_pdf")

    from app.modules.parsers.pdf.pymupdf_opencv_processor import PyMuPDFOpenCVProcessor
    from app.modules.parsers.pdf.visualize_opencv_blocks import write_pdf_blocks_debug

    raw = input_pdf.read_bytes()
    mock_config = AsyncMock()

    processor = PyMuPDFOpenCVProcessor(logger=log, config=mock_config)

    mock_response = MagicMock()
    mock_response.summary = ""
    mock_response.headers = []

    with patch(
        "app.modules.parsers.pdf.pymupdf_opencv_processor.get_table_summary_n_headers",
        new_callable=AsyncMock,
        return_value=mock_response,
    ), patch(
        "app.modules.parsers.pdf.pymupdf_opencv_processor.get_rows_text",
        new_callable=AsyncMock,
        return_value=([], []),
    ):
        container = await processor.load_document(input_pdf.name, raw)

    stem = input_pdf.stem
    written = write_pdf_blocks_debug(
        raw,
        container,
        output_dir,
        dpi=dpi,
        stem=stem,
    )
    log.info("Wrote %s page image(s) under %s", len(written), output_dir)
    for p in written:
        log.info("  %s", p)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="Input PDF path")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for PNG output (created if missing)",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Rasterization DPI (default 150)")
    args = parser.parse_args()
    if not args.pdf.is_file():
        raise SystemExit(f"Not a file: {args.pdf}")
    asyncio.run(_run(args.pdf, args.output_dir, args.dpi))


if __name__ == "__main__":
    main()
