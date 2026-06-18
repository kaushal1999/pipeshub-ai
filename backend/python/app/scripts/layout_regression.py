"""
Layout region regression tests for extract_layout_regions.

Compares extract_layout_regions output against JSON snapshots so you can
catch unintended changes after editing opencv_layout_analyzer.py.

Typical workflow
----------------
1. Put your reference PDFs in a folder (only *.pdf at the top level).

2. Generate or refresh snapshots after you are happy with current output:
       cd backend/python
       python -m app.scripts.layout_regression --pdf-dir path/to/pdfs --update

3. After code changes, verify nothing regressed:
       python -m app.scripts.layout_regression --pdf-dir path/to/pdfs

   Or via pytest (uses tests/fixtures/layout_regression/pdfs by default):
       pytest tests/unit/modules/parsers/test_layout_regression.py -v

Snapshots are written next to the PDFs by default:
    <pdf-dir>/snapshots/<stem>.json

Each snapshot stores per-page region metadata (types, bboxes, text, tables,
list items). Image crops are fingerprinted with SHA-256 rather than stored
inline so the JSON stays readable and diff-friendly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pdfplumber

from app.modules.parsers.pdf.opencv_layout_analyzer import (
    LayoutRegion,
    extract_layout_regions,
)

DEFAULT_FIXTURE_PDF_DIR = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "fixtures"
    / "layout_regression"
    / "pdfs"
)


def _round_bbox(bbox: Tuple[float, float, float, float]) -> List[float]:
    return [round(float(v), 2) for v in bbox]


def region_to_snapshot_dict(region: LayoutRegion) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "type": region.type.value,
        "bbox": _round_bbox(region.bbox),
        "text": region.text,
        "font_size": round(float(region.font_size), 2),
        "is_bold": bool(region.is_bold),
        "list_items": list(region.list_items),
    }
    if region.table_grid is not None:
        payload["table_grid"] = region.table_grid
    if region.image_data is not None:
        payload["image_sha256"] = hashlib.sha256(region.image_data).hexdigest()
        payload["image_size_bytes"] = len(region.image_data)
        payload["image_ext"] = region.image_ext
    return payload


def extract_page_snapshot(
    pdf_path: Path,
    page_index: int,
    page: Any,
) -> Dict[str, Any]:
    regions = extract_layout_regions(page, pdf_path=str(pdf_path))
    return {
        "page_index": page_index,
        "page_width": round(float(page.width), 2),
        "page_height": round(float(page.height), 2),
        "region_count": len(regions),
        "regions": [region_to_snapshot_dict(r) for r in regions],
    }


def extract_pdf_snapshot(pdf_path: Path) -> Dict[str, Any]:
    pages: List[Dict[str, Any]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages):
            pages.append(extract_page_snapshot(pdf_path, page_index, page))
    return {
        "source": pdf_path.name,
        "page_count": len(pages),
        "pages": pages,
    }


def snapshot_path_for_pdf(pdf_path: Path, snapshot_dir: Path) -> Path:
    return snapshot_dir / f"{pdf_path.stem}.json"


def write_snapshot(pdf_path: Path, snapshot_dir: Path) -> Path:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    out_path = snapshot_path_for_pdf(pdf_path, snapshot_dir)
    payload = extract_pdf_snapshot(pdf_path)
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return out_path


def _format_diff(expected: Any, actual: Any, path: str = "") -> List[str]:
    if type(expected) is not type(actual):
        return [f"{path or '$'}: type {type(expected).__name__} != {type(actual).__name__}"]

    if isinstance(expected, dict):
        lines: List[str] = []
        keys = sorted(set(expected) | set(actual))
        for key in keys:
            subpath = f"{path}.{key}" if path else key
            if key not in expected:
                lines.append(f"{subpath}: unexpected key in actual")
            elif key not in actual:
                lines.append(f"{subpath}: missing key in actual")
            else:
                lines.extend(_format_diff(expected[key], actual[key], subpath))
        return lines

    if isinstance(expected, list):
        lines = []
        if len(expected) != len(actual):
            lines.append(f"{path}: length {len(expected)} != {len(actual)}")
        for idx, (exp_item, act_item) in enumerate(zip(expected, actual)):
            lines.extend(_format_diff(exp_item, act_item, f"{path}[{idx}]"))
        return lines

    if expected != actual:
        return [f"{path}: {expected!r} != {actual!r}"]
    return []


def compare_snapshots(expected: Dict[str, Any], actual: Dict[str, Any]) -> List[str]:
    return _format_diff(expected, actual)


def check_pdf(pdf_path: Path, snapshot_dir: Path) -> Tuple[bool, List[str]]:
    snap_path = snapshot_path_for_pdf(pdf_path, snapshot_dir)
    if not snap_path.is_file():
        return False, [f"missing snapshot: {snap_path}"]

    expected = json.loads(snap_path.read_text(encoding="utf-8"))
    actual = extract_pdf_snapshot(pdf_path)
    diffs = compare_snapshots(expected, actual)
    if not diffs:
        return True, []
    return False, [f"{pdf_path.name}:"] + [f"  {line}" for line in diffs]


def discover_pdfs(pdf_dir: Path) -> List[Path]:
    if not pdf_dir.is_dir():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    return pdfs


def resolve_snapshot_dir(pdf_dir: Path, snapshot_dir: Optional[Path]) -> Path:
    return snapshot_dir if snapshot_dir is not None else pdf_dir / "snapshots"


def run_regression(
    pdf_dir: Path,
    *,
    snapshot_dir: Optional[Path] = None,
    update: bool = False,
) -> int:
    pdfs = discover_pdfs(pdf_dir)
    if not pdfs:
        print(f"No PDF files found in {pdf_dir}", file=sys.stderr)
        return 1

    snapshots = resolve_snapshot_dir(pdf_dir, snapshot_dir)
    failures: List[str] = []

    for pdf_path in pdfs:
        if update:
            out_path = write_snapshot(pdf_path, snapshots)
            print(f"updated {out_path.relative_to(pdf_dir)}")
            continue

        ok, messages = check_pdf(pdf_path, snapshots)
        if ok:
            print(f"ok   {pdf_path.name}")
        else:
            failures.extend(messages)
            print(f"FAIL {pdf_path.name}")

    if update:
        print(f"\nWrote {len(pdfs)} snapshot(s) to {snapshots}")
        return 0

    if failures:
        print("\nRegression failures:", file=sys.stderr)
        for line in failures:
            print(line, file=sys.stderr)
        print(
            f"\nIf the changes are intentional, refresh snapshots with:\n"
            f"  python -m app.scripts.layout_regression "
            f'--pdf-dir "{pdf_dir}" --update',
            file=sys.stderr,
        )
        return 1

    print(f"\nAll {len(pdfs)} PDF(s) matched their snapshots.")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Regression-test extract_layout_regions against JSON snapshots.",
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=DEFAULT_FIXTURE_PDF_DIR,
        help=f"Directory containing *.pdf files (default: {DEFAULT_FIXTURE_PDF_DIR})",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=None,
        help="Directory for JSON snapshots (default: <pdf-dir>/snapshots)",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Write or overwrite snapshots from current extractor output",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return run_regression(
        args.pdf_dir.resolve(),
        snapshot_dir=args.snapshot_dir.resolve() if args.snapshot_dir else None,
        update=args.update,
    )


if __name__ == "__main__":
    raise SystemExit(main())
