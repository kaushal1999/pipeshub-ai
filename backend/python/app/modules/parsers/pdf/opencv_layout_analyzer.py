from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from io import BytesIO
from operator import itemgetter
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from pdfplumber.page import Page as PdfplumberPage
from pdfplumber.utils import cluster_objects

DEFAULT_RENDER_DPI = 150
DPI_SCALE = 72.0
MIN_TABLE_AREA_RATIO = 0.002
MAX_TABLE_AREA_RATIO = 0.7
MIN_IMAGE_AREA_RATIO = 0.003
MIN_TEXT_AREA_RATIO = 0.0005
HEADING_FONT_SIZE_RATIO = 1.3
BOLD_FLAG = 0b10000
OVERLAP_THRESHOLD = 0.5
TABLE_CELL_COUNT_THRESHOLD = 4
MIN_TABLE_GRID_LINES = 3
HORIZONTAL_KERNEL_SCALE = 40
VERTICAL_KERNEL_SCALE = 40
TEXT_DILATE_KERNEL_WIDTH_FRAC = 1 / 40
TEXT_DILATE_KERNEL_HEIGHT_FRAC = 1 / 150
BLOCK_MATCH_OVERLAP_THRESHOLD = 0.3
MIN_LIST_LINES = 2
LIST_BULLET_PATTERNS = ("•", "●", "○", "■", "□", "▪", "▫", "–", "—", "-")
ORDERED_LIST_PATTERN_CHARS = frozenset("0123456789.)")
MIN_GRAPHIC_TEXT_COVERAGE = 0.02


def _fontname_is_bold(fontname: str) -> bool:
    fn = fontname.lower()
    return "bold" in fn or "black" in fn


def _build_text_dict_from_pdfplumber_page(page: PdfplumberPage) -> Dict[str, Any]:
    """Build a PyMuPDF-compatible text dict from a pdfplumber page (blocks/lines/spans)."""
    words = page.extract_words(extra_attrs=["fontname", "size"], keep_blank_chars=False)
    if not words:
        return {"blocks": []}
    word_lines = cluster_objects(words, itemgetter("doctop"), 5)
    blocks: List[Dict[str, Any]] = []
    for word_line in word_lines:
        ordered = sorted(word_line, key=itemgetter("x0"))
        x0 = min(w["x0"] for w in ordered)
        top = min(w["top"] for w in ordered)
        x1 = max(w["x1"] for w in ordered)
        bottom = max(w["bottom"] for w in ordered)
        block_bbox = (x0, top, x1, bottom)
        spans: List[Dict[str, Any]] = []
        for w in ordered:
            fn = str(w.get("fontname") or "")
            flags = BOLD_FLAG if _fontname_is_bold(fn) else 0
            sz = float(w.get("size") or 0)
            if sz <= 0:
                sz = float(w.get("bottom", bottom) - w.get("top", top))
            spans.append({"text": w["text"], "size": sz, "flags": flags})
        blocks.append(
            {"type": 0, "bbox": block_bbox, "lines": [{"spans": spans}]}
        )
    return {"blocks": blocks}


class LayoutRegionType(str, Enum):
    TEXT = "text"
    HEADING = "heading"
    TABLE = "table"
    IMAGE = "image"
    LIST = "list"
    ORDERED_LIST = "ordered_list"


@dataclass
class LayoutRegion:
    type: LayoutRegionType
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1) in PDF points
    text: str = ""
    font_size: float = 0.0
    is_bold: bool = False
    image_data: Optional[bytes] = None
    image_ext: str = "png"
    table_grid: Optional[List[List[str]]] = None
    list_items: List[str] = field(default_factory=list)


def _rect_area(bbox: Tuple[float, float, float, float]) -> float:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def _overlap_ratio(
    a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]
) -> float:
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_a = _rect_area(a)
    if area_a == 0:
        return 0.0
    return inter / area_a


def _pixel_to_pdf(val: float, dpi: int) -> float:
    return val * DPI_SCALE / dpi


def _pil_from_pdf_stream(stream: Any) -> Optional[Any]:
    """Extract a PIL Image from a pdfminer PDFStream.

    Handles DCTDecode (JPEG) and JPXDecode (JPEG2000) natively via PIL.
    Returns None for other encodings, triggering page-crop fallback.
    """
    from PIL import Image as _PILImage

    try:
        data = stream.get_data()
    except Exception:
        return None
    if not data or len(data) < 8:
        return None
    try:
        img = _PILImage.open(BytesIO(data))
        img.load()
        return img
    except Exception:
        return None


def _count_distinct_lines(projection: np.ndarray) -> int:
    """Count distinct contiguous runs of True values in a 1-D boolean array."""
    if projection.size == 0:
        return 0
    transitions = np.diff(projection.astype(np.int8))
    rising_edges = int(np.sum(transitions == 1))
    return rising_edges + (1 if projection[0] else 0)


def _reading_order_key(region: LayoutRegion) -> Tuple[float, float]:
    return (region.bbox[1], region.bbox[0])


class OpenCVLayoutAnalyzer:
    """Lightweight ML-free layout analysis using OpenCV morphological operations.

    Text metrics come from pdfplumber word extraction; page images use pdfplumber's
    pypdfium2 rasterizer at *render_dpi*.
    """

    def __init__(self, logger: logging.Logger, render_dpi: int = DEFAULT_RENDER_DPI) -> None:
        self.logger = logger
        self.render_dpi = render_dpi

    def _render_page_to_image(self, page: PdfplumberPage) -> np.ndarray:
        """Rasterize using pdfplumber (pypdfium2); matches prior ~150 DPI OpenCV input."""
        page_image = page.to_image(resolution=self.render_dpi)
        pil_im = page_image.original
        if pil_im.mode != "RGB":
            pil_im = pil_im.convert("RGB")
        return np.asarray(pil_im)

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10
        )
        kernel = np.ones((2, 2), np.uint8)
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    def _detect_table_regions(
        self, binary: np.ndarray, page_width_pt: float, page_height_pt: float
    ) -> List[Tuple[float, float, float, float]]:
        h, w = binary.shape
        page_area = page_width_pt * page_height_pt

        horiz_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (max(w // HORIZONTAL_KERNEL_SCALE, 1), 1)
        )
        horiz_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel)

        vert_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, max(h // VERTICAL_KERNEL_SCALE, 1))
        )
        vert_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vert_kernel)

        grid_mask = cv2.add(horiz_lines, vert_lines)
        grid_mask = cv2.dilate(grid_mask, np.ones((3, 3), np.uint8), iterations=2)

        contours, _ = cv2.findContours(
            grid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        table_rects: List[Tuple[float, float, float, float]] = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            pdf_bbox = (
                _pixel_to_pdf(x, self.render_dpi),
                _pixel_to_pdf(y, self.render_dpi),
                _pixel_to_pdf(x + cw, self.render_dpi),
                _pixel_to_pdf(y + ch, self.render_dpi),
            )
            region_area = _rect_area(pdf_bbox)
            if region_area < page_area * MIN_TABLE_AREA_RATIO:
                continue

            # Reject regions covering most of the page (borders / frames)
            if region_area > page_area * MAX_TABLE_AREA_RATIO:
                self.logger.debug(
                    f"Skipping oversized table candidate "
                    f"({region_area / page_area:.0%} of page) — likely a border"
                )
                continue

            # Verify real grid structure: need multiple horizontal AND vertical
            # internal lines, not just a surrounding border.
            roi_h = horiz_lines[y : y + ch, x : x + cw]
            roi_v = vert_lines[y : y + ch, x : x + cw]

            h_projection = np.any(roi_h > 0, axis=1)
            v_projection = np.any(roi_v > 0, axis=0)
            num_h_lines = _count_distinct_lines(h_projection)
            num_v_lines = _count_distinct_lines(v_projection)

            if num_h_lines < MIN_TABLE_GRID_LINES or num_v_lines < MIN_TABLE_GRID_LINES:
                self.logger.debug(
                    f"Skipping table candidate with insufficient grid "
                    f"(h_lines={num_h_lines}, v_lines={num_v_lines})"
                )
                continue

            roi = grid_mask[y : y + ch, x : x + cw]
            inner_contours, _ = cv2.findContours(
                cv2.bitwise_not(roi), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            if len(inner_contours) >= TABLE_CELL_COUNT_THRESHOLD:
                table_rects.append(pdf_bbox)

        return table_rects

    def _detect_text_regions(
        self,
        binary: np.ndarray,
        table_rects: List[Tuple[float, float, float, float]],
        page_width_pt: float,
        page_height_pt: float,
    ) -> List[Tuple[float, float, float, float]]:
        h, w = binary.shape
        page_area = page_width_pt * page_height_pt
        dilate_w = max(int(w * TEXT_DILATE_KERNEL_WIDTH_FRAC), 5)
        dilate_h = max(int(h * TEXT_DILATE_KERNEL_HEIGHT_FRAC), 3)
        dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (dilate_w, dilate_h)
        )
        dilated = cv2.dilate(binary, dilate_kernel, iterations=2)

        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        text_rects: List[Tuple[float, float, float, float]] = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            pdf_bbox = (
                _pixel_to_pdf(x, self.render_dpi),
                _pixel_to_pdf(y, self.render_dpi),
                _pixel_to_pdf(x + cw, self.render_dpi),
                _pixel_to_pdf(y + ch, self.render_dpi),
            )
            if _rect_area(pdf_bbox) < page_area * MIN_TEXT_AREA_RATIO:
                continue
            in_table = any(
                _overlap_ratio(pdf_bbox, tr) > OVERLAP_THRESHOLD
                for tr in table_rects
            )
            if not in_table:
                text_rects.append(pdf_bbox)

        return text_rects

    def _extract_image_regions(
        self,
        page: PdfplumberPage,
        table_rects: List[Tuple[float, float, float, float]],
        page_width_pt: float,
        page_height_pt: float,
    ) -> List[Dict[str, Any]]:
        page_area = page_width_pt * page_height_pt
        images: List[Dict[str, Any]] = []
        for img_obj in page.images:
            pdf_bbox = (img_obj["x0"], img_obj["top"], img_obj["x1"], img_obj["bottom"])
            if _rect_area(pdf_bbox) < page_area * MIN_IMAGE_AREA_RATIO:
                continue
            in_table = any(
                _overlap_ratio(pdf_bbox, tr) > OVERLAP_THRESHOLD
                for tr in table_rects
            )
            if in_table:
                continue

            pil_img = None

            stream = img_obj.get("stream")
            if stream is not None:
                pil_img = _pil_from_pdf_stream(stream)

            if pil_img is None:
                try:
                    cropped = page.crop(pdf_bbox, strict=False)
                    pil_img = cropped.to_image(resolution=self.render_dpi).original
                except Exception as e:
                    self.logger.warning(f"Could not rasterize image region: {e}")
                    continue

            try:
                if pil_img.mode not in ("RGB", "L"):
                    pil_img = pil_img.convert("RGB")
                buf = BytesIO()
                pil_img.save(buf, format="PNG")
                images.append(
                    {
                        "bbox": pdf_bbox,
                        "data": buf.getvalue(),
                        "ext": "png",
                    }
                )
            except Exception as e:
                self.logger.warning(f"Could not encode image: {e}")
        return images

    def _get_text_blocks_for_region(
        self,
        region_bbox: Tuple[float, float, float, float],
        text_dict: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        matched: List[Dict[str, Any]] = []
        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            bb = block.get("bbox")
            if not bb:
                continue
            block_bbox = (bb[0], bb[1], bb[2], bb[3])
            if _overlap_ratio(block_bbox, region_bbox) > BLOCK_MATCH_OVERLAP_THRESHOLD:
                matched.append(block)
        return matched

    def _extract_text_and_metadata(
        self, blocks: List[Dict[str, Any]]
    ) -> Tuple[str, float, bool]:
        texts: List[str] = []
        total_size = 0.0
        count = 0
        any_bold = False
        for block in blocks:
            for line in block.get("lines", []):
                line_text_parts: List[str] = []
                for span in line.get("spans", []):
                    span_text = span.get("text", "")
                    line_text_parts.append(span_text)
                    size = span.get("size", 0)
                    total_size += size
                    count += 1
                    if span.get("flags", 0) & BOLD_FLAG:
                        any_bold = True
                line_text = " ".join(line_text_parts).strip()
                if line_text:
                    texts.append(line_text)
        avg_size = total_size / count if count > 0 else 0
        return "\n".join(texts), avg_size, any_bold

    def _compute_median_font_size(self, text_dict: Dict[str, Any]) -> float:
        sizes: List[float] = []
        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    s = span.get("size", 0)
                    if s > 0:
                        sizes.append(s)
        if not sizes:
            return 12.0
        sizes.sort()
        mid = len(sizes) // 2
        if len(sizes) % 2 == 0:
            return (sizes[mid - 1] + sizes[mid]) / 2
        return sizes[mid]

    def _classify_list_type(self, text: str) -> Optional[LayoutRegionType]:
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if len(lines) < MIN_LIST_LINES:
            return None
        bullet_count = sum(
            1 for line in lines if any(line.startswith(p) for p in LIST_BULLET_PATTERNS)
        )
        if bullet_count >= len(lines) * 0.6:
            return LayoutRegionType.LIST

        ordered_count = 0
        for line in lines:
            stripped = line.lstrip()
            if stripped and stripped[0].isdigit():
                j = 1
                while j < len(stripped) and stripped[j] in ORDERED_LIST_PATTERN_CHARS:
                    j += 1
                if j < len(stripped) and stripped[j] == " ":
                    ordered_count += 1
        if ordered_count >= len(lines) * 0.6:
            return LayoutRegionType.ORDERED_LIST
        return None

    def analyze_page(
        self, page: PdfplumberPage
    ) -> List[LayoutRegion]:
        page_w = float(page.width)
        page_h = float(page.height)

        img = self._render_page_to_image(page)
        binary = self._preprocess(img)

        table_rects = self._detect_table_regions(binary, page_w, page_h)
        text_rects = self._detect_text_regions(binary, table_rects, page_w, page_h)
        image_infos = self._extract_image_regions(page, table_rects, page_w, page_h)

        text_dict = _build_text_dict_from_pdfplumber_page(page)
        median_font_size = self._compute_median_font_size(text_dict)
        page_area = page_w * page_h

        regions: List[LayoutRegion] = []

        regions.extend(
            LayoutRegion(type=LayoutRegionType.TABLE, bbox=tb) for tb in table_rects
        )

        for ib in image_infos:
            regions.append(
                LayoutRegion(
                    type=LayoutRegionType.IMAGE,
                    bbox=ib["bbox"],
                    image_data=ib["data"],
                    image_ext=ib["ext"],
                )
            )

        image_bboxes = [ib["bbox"] for ib in image_infos]
        for tr in text_rects:
            in_image = any(
                _overlap_ratio(tr, ib) > OVERLAP_THRESHOLD for ib in image_bboxes
            )
            if in_image:
                continue

            matched_blocks = self._get_text_blocks_for_region(tr, text_dict)
            region_area = _rect_area(tr)

            if not matched_blocks and region_area >= page_area * MIN_IMAGE_AREA_RATIO:
                text_coverage_in_region = 0.0
                if region_area > 0:
                    for block in text_dict.get("blocks", []):
                        if block.get("type") != 0:
                            continue
                        bb = block.get("bbox")
                        if not bb:
                            continue
                        ix0 = max(bb[0], tr[0])
                        iy0 = max(bb[1], tr[1])
                        ix1 = min(bb[2], tr[2])
                        iy1 = min(bb[3], tr[3])
                        text_coverage_in_region += max(0, ix1 - ix0) * max(0, iy1 - iy0)

                if text_coverage_in_region / max(region_area, 1) >= MIN_GRAPHIC_TEXT_COVERAGE:
                    continue

                try:
                    cropped = page.crop(tr, strict=False)
                    pil = cropped.to_image(resolution=self.render_dpi).original
                    if pil.mode != "RGB":
                        pil = pil.convert("RGB")
                    buf = BytesIO()
                    pil.save(buf, format="PNG")
                    regions.append(LayoutRegion(
                        type=LayoutRegionType.IMAGE,
                        bbox=tr,
                        image_data=buf.getvalue(),
                        image_ext="png",
                    ))
                    image_bboxes.append(tr)
                except Exception as e:
                    self.logger.debug(f"Could not rasterize graphic region: {e}")
                continue

            if not matched_blocks:
                continue
            text, avg_size, is_bold = self._extract_text_and_metadata(matched_blocks)
            if not text.strip():
                continue

            list_type = self._classify_list_type(text)
            if list_type is not None:
                items = [ln.strip() for ln in text.split("\n") if ln.strip()]
                regions.append(
                    LayoutRegion(
                        type=list_type,
                        bbox=tr,
                        text=text,
                        font_size=avg_size,
                        is_bold=is_bold,
                        list_items=items,
                    )
                )
            elif avg_size >= median_font_size * HEADING_FONT_SIZE_RATIO or (
                is_bold and avg_size >= median_font_size * 1.1 and "\n" not in text.strip()
            ):
                regions.append(
                    LayoutRegion(
                        type=LayoutRegionType.HEADING,
                        bbox=tr,
                        text=text,
                        font_size=avg_size,
                        is_bold=is_bold,
                    )
                )
            else:
                regions.append(
                    LayoutRegion(
                        type=LayoutRegionType.TEXT,
                        bbox=tr,
                        text=text,
                        font_size=avg_size,
                        is_bold=is_bold,
                    )
                )

        self._collect_unclaimed_text_blocks(
            text_dict, regions, table_rects, image_bboxes, page_w, page_h
        )

        regions.sort(key=_reading_order_key)
        return regions

    def _collect_unclaimed_text_blocks(
        self,
        text_dict: Dict[str, Any],
        regions: List[LayoutRegion],
        table_rects: List[Tuple[float, float, float, float]],
        image_bboxes: List[Tuple[float, float, float, float]],
        page_w: float,
        page_h: float,
    ) -> None:
        """Pick up any parsed text blocks not covered by existing regions."""
        existing_bboxes = [r.bbox for r in regions]
        median_size = self._compute_median_font_size(text_dict)

        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            bb = block.get("bbox")
            if not bb:
                continue
            block_bbox = (bb[0], bb[1], bb[2], bb[3])

            claimed = any(
                _overlap_ratio(block_bbox, eb) > BLOCK_MATCH_OVERLAP_THRESHOLD for eb in existing_bboxes
            )
            if claimed:
                continue
            in_table = any(
                _overlap_ratio(block_bbox, tr) > OVERLAP_THRESHOLD for tr in table_rects
            )
            if in_table:
                continue
            in_image = any(
                _overlap_ratio(block_bbox, ib) > OVERLAP_THRESHOLD for ib in image_bboxes
            )
            if in_image:
                continue

            text, avg_size, is_bold = self._extract_text_and_metadata([block])
            if not text.strip():
                continue

            if avg_size >= median_size * HEADING_FONT_SIZE_RATIO or (
                is_bold and avg_size >= median_size * 1.1 and "\n" not in text.strip()
            ):
                rtype = LayoutRegionType.HEADING
            else:
                rtype = LayoutRegionType.TEXT

            regions.append(
                LayoutRegion(
                    type=rtype,
                    bbox=block_bbox,
                    text=text,
                    font_size=avg_size,
                    is_bold=is_bold,
                )
            )
