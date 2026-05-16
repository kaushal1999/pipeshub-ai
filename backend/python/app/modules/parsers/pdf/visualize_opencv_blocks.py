"""Debug visualization: render OpenCV/pdfplumber PDF blocks on page images.

Used by ``scripts/visualize_pdf_opencv_blocks.py`` and optionally when
``PIPESHUB_DEBUG_PDF_BLOCKS_OUT`` is set during ``parse_file`` PDF handling.
"""

from __future__ import annotations

import colorsys
import hashlib
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import pdfplumber
from PIL import Image, ImageDraw, ImageFont

from app.models.blocks import (
    Block,
    BlockGroup,
    BlocksContainer,
    BlockSubType,
    BlockType,
    CitationMetadata,
    GroupType,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Ann:
    page: int
    rect_pdf: tuple[float, float, float, float]  # x0, top, x1, bottom
    label: str


def _stable_color(label: str) -> tuple[int, int, int]:
    h = int(hashlib.md5(label.encode("utf-8"), usedforsecurity=False).hexdigest()[:8], 16)
    hf = (h % 360) / 360.0
    r_f, g_f, b_f = colorsys.hsv_to_rgb(hf, 0.55, 0.92)
    return (int(r_f * 255), int(g_f * 255), int(b_f * 255))


def _citation_to_rect_pdf(
    cite: CitationMetadata | None,
    page_dims: dict[int, tuple[float, float]],
) -> tuple[int, tuple[float, float, float, float]] | None:
    if cite is None or cite.page_number is None or not cite.bounding_boxes:
        return None
    pno = cite.page_number
    if pno not in page_dims:
        return None
    w, h = page_dims[pno]
    xs = [pt.x * w for pt in cite.bounding_boxes]
    ys = [pt.y * h for pt in cite.bounding_boxes]
    rect = (min(xs), min(ys), max(xs), max(ys))
    return (pno, rect)


def _parent_groups_by_index(container: BlocksContainer) -> dict[int, BlockGroup]:
    """Map ``BlockGroup.index`` -> group (only entries with non-None index)."""
    out: dict[int, BlockGroup] = {}
    for g in container.block_groups:
        if g.index is not None:
            out[g.index] = g
    return out


def _list_item_ordinal_by_block_index(container: BlocksContainer) -> dict[int, int]:
    """1-based item position within each list group, keyed by ``Block.index``."""
    by_parent: defaultdict[int, list[Block]] = defaultdict(list)
    for b in container.blocks:
        if (
            b.parent_index is not None
            and b.sub_type == BlockSubType.LIST_ITEM
        ):
            by_parent[b.parent_index].append(b)
    ordinals: dict[int, int] = {}
    for children in by_parent.values():
        children.sort(key=lambda c: c.index)
        for i, c in enumerate(children, start=1):
            ordinals[c.index] = i
    return ordinals


def _block_label(
    b: Block,
    *,
    parents: dict[int, BlockGroup],
    list_ordinals: dict[int, int],
) -> str:
    parts = [b.type.value]
    if b.sub_type is not None:
        parts.append(b.sub_type.value)
    base = " / ".join(parts)

    if b.type == BlockType.TABLE_ROW and isinstance(b.data, dict):
        row_n = b.data.get("row_number")
        total: int | None = None
        if b.parent_index is not None:
            parent = parents.get(b.parent_index)
            if parent and parent.table_metadata is not None:
                total = parent.table_metadata.num_of_rows
        if row_n is not None:
            if total is not None:
                return f"{base} (row {row_n}/{total})"
            return f"{base} (row {row_n})"

    if (
        b.sub_type == BlockSubType.LIST_ITEM
        and b.parent_index is not None
    ):
        n = list_ordinals.get(b.index)
        total_items: int | None = None
        parent = parents.get(b.parent_index)
        if parent and parent.list_metadata is not None:
            total_items = parent.list_metadata.item_count
        if n is not None:
            if total_items is not None:
                return f"{base} (item {n}/{total_items})"
            return f"{base} (item {n})"

    return base


def _group_label(g: BlockGroup) -> str:
    base = f"group:{g.type.value}"
    if g.type == GroupType.TABLE and g.table_metadata is not None:
        tm = g.table_metadata
        rows, cols = tm.num_of_rows, tm.num_of_cols
        fragments: list[str] = []
        if rows is not None:
            fragments.append(f"{rows} rows")
        if cols is not None:
            fragments.append(f"{cols} cols")
        if fragments:
            return f"{base} ({' × '.join(fragments)})"
    if g.type in (
        GroupType.LIST,
        GroupType.ORDERED_LIST,
    ) and g.list_metadata is not None:
        ic = g.list_metadata.item_count
        if ic is not None:
            return f"{base} ({ic} items)"
    return base


def collect_annotations_for_pdf(
    container: BlocksContainer, page_dims: dict[int, tuple[float, float]]
) -> list[_Ann]:
    anns: list[_Ann] = []
    parents = _parent_groups_by_index(container)
    list_ordinals = _list_item_ordinal_by_block_index(container)

    def add_cite(cite: CitationMetadata | None, label: str) -> None:
        got = _citation_to_rect_pdf(cite, page_dims)
        if got is None:
            return
        pno, rect = got
        anns.append(_Ann(page=pno, rect_pdf=rect, label=label))

    for bg in container.block_groups:
        add_cite(bg.citation_metadata, _group_label(bg))

    for b in container.blocks:
        add_cite(
            b.citation_metadata,
            _block_label(
                b, parents=parents, list_ordinals=list_ordinals
            ),
        )

    return anns


def _pdf_rect_to_pixels(
    rect_pdf: tuple[float, float, float, float],
    page_w_pt: float,
    page_h_pt: float,
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = rect_pdf
    sx = img_w / page_w_pt
    sy = img_h / page_h_pt
    px0 = int(round(x0 * sx))
    py0 = int(round(y0 * sy))
    px1 = int(round(x1 * sx))
    py1 = int(round(y1 * sy))
    return (min(px0, px1), min(py0, py1), max(px0, px1), max(py0, py1))


def _load_font(size: int = 11) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_pdf_blocks_overlay(
    pdf_bytes: bytes,
    container: BlocksContainer,
    *,
    dpi: int = 150,
) -> dict[int, Image.Image]:
    """Return mapping page_number -> RGB image with outlines and corner labels."""
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        page_dims: dict[int, tuple[float, float]] = {
            i + 1: (float(p.width), float(p.height)) for i, p in enumerate(pdf.pages)
        }

    anns = collect_annotations_for_pdf(container, page_dims)
    by_page: dict[int, list[_Ann]] = {}
    for a in anns:
        by_page.setdefault(a.page, []).append(a)

    out: dict[int, Image.Image] = {}
    font = _load_font()

    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            pno = i + 1
            pil = page.to_image(resolution=dpi).original
            if pil.mode != "RGB":
                pil = pil.convert("RGB")
            draw = ImageDraw.Draw(pil)

            page_list = list(by_page.get(pno, []))

            def area_pdf(ann: _Ann) -> float:
                x0, y0, x1, y1 = ann.rect_pdf
                return max(0.0, x1 - x0) * max(0.0, y1 - y0)

            page_list.sort(key=area_pdf)

            pw, ph = page_dims[pno]
            iw, ih = pil.size

            for ann in page_list:
                color = _stable_color(ann.label)
                px = _pdf_rect_to_pixels(ann.rect_pdf, pw, ph, iw, ih)
                draw.rectangle(px, outline=color, width=3)

                short = (
                    ann.label if len(ann.label) <= 88 else ann.label[:85] + "…"
                )
                tx, ty = px[0] + 2, px[1] + 2
                bbox = draw.textbbox((tx, ty), short, font=font)
                pad = 2
                draw.rectangle(
                    (
                        bbox[0] - pad,
                        bbox[1] - pad,
                        bbox[2] + pad,
                        bbox[3] + pad,
                    ),
                    fill=(25, 25, 25),
                )
                draw.text((tx, ty), short, fill=(255, 255, 255), font=font)

            out[pno] = pil

    return out


def write_pdf_blocks_debug(
    pdf_bytes: bytes,
    container: BlocksContainer,
    output_dir: Path,
    *,
    dpi: int = 150,
    stem: str = "debug",
) -> list[Path]:
    """Write ``{stem}_page_NNN.png`` for every page."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images = render_pdf_blocks_overlay(pdf_bytes, container, dpi=dpi)
    written: list[Path] = []
    for pno in sorted(images.keys()):
        path = output_dir / f"{stem}_page_{pno:03d}.png"
        images[pno].save(path, format="PNG")
        written.append(path)
        logger.info("Wrote block overlay: %s", path)
    return written


def maybe_write_pdf_debug_overlay(
    pdf_bytes: bytes,
    container: BlocksContainer,
    *,
    stem: str,
    log: logging.Logger | None = None,
    dpi: int = 150,
) -> None:
    """If ``PIPESHUB_DEBUG_PDF_BLOCKS_OUT`` is set, write PNG overlays under that directory.

    Output path: ``{env_dir}/{stem}_blocks/{stem}_page_NNN.png`` (same layout as ``parse_file``).
    """
    dbg_dir = os.environ.get("PIPESHUB_DEBUG_PDF_BLOCKS_OUT", "").strip()
    if not dbg_dir:
        return
    try:
        out_base = Path(dbg_dir).expanduser()
        out_base.mkdir(parents=True, exist_ok=True)
        subdir = out_base / f"{stem}_blocks"
        write_pdf_blocks_debug(
            pdf_bytes, container, subdir, dpi=dpi, stem=stem
        )
    except Exception as exc:
        if log is not None:
            log.warning(
                "PIPESHUB_DEBUG_PDF_BLOCKS_OUT overlay skipped: %s", exc
            )
