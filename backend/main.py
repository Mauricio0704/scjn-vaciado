"""OCR pipeline: Imagen -> EasyOCR -> Parser -> Pandas -> Excel."""

from __future__ import annotations

from io import BytesIO
import re
import unicodedata
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
from zipfile import ZipFile

import cv2
import numpy as np
import pandas as pd


OUTPUT_COLUMNS = [
    "Comisionado",
    "Area u Organo Jurisdiccional",
    "Dias que comprende la comision",
    "Localidad",
    "Hospedaje",
    "Alimentacion",
    "Combustible / Pasajes",
    "Recorrido Int. / Taxis",
    "Peaje",
    "Total",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}
DATAFRAME_COLUMNS = ["Archivo"] + OUTPUT_COLUMNS

_ocr_reader = None


def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        _ocr_reader = easyocr.Reader(["es", "en"], gpu=False)
    return _ocr_reader


def load_image_bytes(image_bytes: bytes, image_name: str):
    """Load an image from raw bytes."""
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Image file not found or cannot be read: {image_name}")
    return image


def preprocess_image(image):
    """Preprocess the image for better OCR results."""
    return image


def run_ocr(image) -> str:
    """Extract text using EasyOCR."""
    reader = _get_ocr_reader()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*pin_memory.*MPS.*")
        results = reader.readtext(image, detail=0, paragraph=True)
    text = "\n".join(results)
    print("=== OCR RAW TEXT ===")
    print(text)
    print("=== END OCR TEXT ===")
    return text


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value)
    without_accents = "".join(
        ch for ch in normalized if unicodedata.category(ch) != "Mn"
    )
    compact = re.sub(r"\s+", " ", without_accents).strip().lower()
    return compact


_AMOUNT_RE = re.compile(r"\d{1,3}(?:,\d{3})*(?:\.\d{2})|\d+(?:\.\d{2})")

# Ordered list of (output_key, regex_pattern) for the five amount columns.
# The order here matches the expected left-to-right column order in the document.
_AMOUNT_COLUMNS: List[Tuple[str, str]] = [
    ("Hospedaje", r"hospedaje"),
    ("Alimentacion", r"al[il1]mentac"),
    ("Combustible / Pasajes", r"combust[il1]ble|pasajes"),
    ("Recorrido Int. / Taxis", r"recorr[il1]do|tax[il1]s"),
    ("Peaje", r"peaje"),
]


def _extract_line_value(lines: List[str], patterns: List[str]) -> str:
    """Return the value that follows a label, with next-line fallback.

    ``patterns`` are regex patterns matched against the normalised line.
    Common OCR substitutions (l/i/1) should be expressed as ``[il1]``.
    """
    for i, line in enumerate(lines):
        normalized_line = _normalize_text(line)
        for pattern in patterns:
            m = re.search(pattern, normalized_line)
            if m:
                # Use the match end-position against the original line; lengths
                # stay equal because _normalize_text only removes combining marks
                # (same visible-character count) and compresses interior spaces.
                orig_after = line[m.end():].strip(" :-\t")
                if orig_after:
                    return orig_after
                # Label only on this line — value is on the next
                if i + 1 < len(lines):
                    return lines[i + 1].strip()
    return ""


def _extract_table_amounts(lines: List[str]) -> Dict[str, str]:
    """Extract the five per-diem amounts from the table header + value rows.

    The document lays out column headers on one line and the corresponding
    numeric values on the next.  We detect the header row by requiring at
    least two column-keyword matches, then map each header's left-to-right
    position to the corresponding amount on the following value row.
    """
    for i, line in enumerate(lines):
        normalized = _normalize_text(line)
        col_positions = []
        for field_name, pattern in _AMOUNT_COLUMNS:
            m = re.search(pattern, normalized)
            if m:
                col_positions.append((m.start(), field_name))

        if len(col_positions) < 2:
            continue

        # Find the first subsequent line that has at least as many amounts
        # as columns detected in the header row.
        for j in range(i + 1, min(i + 4, len(lines))):
            amounts = _AMOUNT_RE.findall(lines[j])
            if len(amounts) >= len(col_positions):
                col_positions.sort()
                return {name: amounts[k] for k, (_, name) in enumerate(col_positions)}

    return {}


def _extract_last_amount(lines: List[str], patterns: List[str]) -> str:
    """Return the last amount on the matching line, or the first on the next."""
    for i, line in enumerate(lines):
        normalized_line = _normalize_text(line)
        if any(re.search(p, normalized_line) for p in patterns):
            amounts = _AMOUNT_RE.findall(line)
            if amounts:
                return amounts[-1]
            # Amount is on the next line (e.g. "Total\n2,545.20")
            if i + 1 < len(lines):
                amounts = _AMOUNT_RE.findall(lines[i + 1])
                if amounts:
                    return amounts[-1]
    return ""


def parse_ocr_text(text: str) -> Dict[str, str]:
    """Parse target fields from OCR output."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    table = _extract_table_amounts(lines)

    return {
        "Comisionado": _extract_line_value(lines, [r"com[il1]s[il1]onado"]),
        "Area u Organo Jurisdiccional": _extract_line_value(
            lines, [r"area u.{0,5}rgano.{0,15}ur[il1]sd[il1]cc[il1]onal"]
        ),
        "Dias que comprende la comision": _extract_line_value(
            lines, [r"d[il1]as que comprende la com[il1]s[il1]on"]
        ),
        "Localidad": _extract_line_value(lines, [r"local[il1]dad"]),
        "Hospedaje": table.get("Hospedaje", ""),
        "Alimentacion": table.get("Alimentacion", ""),
        "Combustible / Pasajes": table.get("Combustible / Pasajes", ""),
        "Recorrido Int. / Taxis": table.get("Recorrido Int. / Taxis", ""),
        "Peaje": table.get("Peaje", ""),
        "Total": _extract_last_amount(lines, [r"total"]),
    }


def process_image_bytes(
    image_bytes: bytes, image_name: str, lang: str = "es"
) -> Dict[str, str]:
    """Process one uploaded image from raw bytes."""
    image = load_image_bytes(image_bytes, image_name)
    preprocessed = preprocess_image(image)
    text = run_ocr(preprocessed)
    parsed_record = parse_ocr_text(text)
    parsed_record["Archivo"] = image_name
    return parsed_record


def extract_images_from_zip(zip_bytes: bytes) -> List[Tuple[str, bytes]]:
    """Extract supported images from a zip archive."""
    extracted_images: List[Tuple[str, bytes]] = []
    with ZipFile(BytesIO(zip_bytes)) as archive:
        for member in sorted(archive.namelist()):
            member_path = Path(member)
            if (
                member.endswith("/")
                or member_path.suffix.lower() not in IMAGE_EXTENSIONS
            ):
                continue
            extracted_images.append((member_path.name, archive.read(member)))
    return extracted_images


def build_dataframe(records: Sequence[Dict[str, str]]) -> pd.DataFrame:
    """Convert parsed records to a DataFrame."""
    return pd.DataFrame(
        [
            [record.get(column, "") for column in DATAFRAME_COLUMNS]
            for record in records
        ],
        columns=DATAFRAME_COLUMNS,
    )


def dataframe_to_excel_bytes(dataframe: pd.DataFrame) -> bytes:
    """Export a DataFrame to an in-memory Excel file."""
    buffer = BytesIO()
    dataframe.to_excel(buffer, index=False)
    buffer.seek(0)
    return buffer.getvalue()
