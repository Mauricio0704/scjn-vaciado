"""OCR pipeline: Imagen -> OpenCV -> Tesseract -> Parser -> Pandas -> Excel."""

from __future__ import annotations

from io import BytesIO
import re
import unicodedata
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple
from zipfile import ZipFile

import cv2
import numpy as np
import pandas as pd
import pytesseract


OUTPUT_COLUMNS = [
    "Comisionado",
    "Area u Organo Jurisdiccional",
    "Dias que comprende la comision",
    "Localidad",
    "Hospedaje $",
    "Alimentacion $",
    "Combustible / Pasajes $",
    "Recorrido Int. / Taxis $",
    "Peaje $",
    "Total $",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}
DATAFRAME_COLUMNS = ["Archivo"] + OUTPUT_COLUMNS


def load_image_bytes(image_bytes: bytes, image_name: str):
    """Load an image from raw bytes."""
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Image file not found or cannot be read: {image_name}")
    return image


def preprocess_image(image):
    """Convert image to high-contrast black/white for OCR."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray_image, 9, 75, 75)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    sharpened = cv2.filter2D(denoised, -1, kernel)
    thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[
        1
    ]
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = thresh.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(
        thresh, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )

    return deskewed


def run_tesseract(image, lang: str = "spa") -> str:
    """Extract OCR text in Spanish."""
    config = "--oem 3 --psm 6"
    return pytesseract.image_to_string(image, lang=lang, config=config)


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value)
    without_accents = "".join(
        ch for ch in normalized if unicodedata.category(ch) != "Mn"
    )
    compact = re.sub(r"\s+", " ", without_accents).strip().lower()
    return compact


def _extract_line_value(lines: List[str], patterns: List[str]) -> str:
    for line in lines:
        normalized_line = _normalize_text(line)
        for pattern in patterns:
            if pattern in normalized_line:
                split_match = re.split(pattern, normalized_line, maxsplit=1)
                if len(split_match) < 2:
                    continue
                prefix = line[: len(line) - len(split_match[1])]
                raw_suffix = line[len(prefix) :].strip(" :-\t")
                return raw_suffix
    return ""


def _extract_last_amount(lines: List[str], patterns: List[str]) -> str:
    amount_regex = re.compile(r"\d{1,3}(?:,\d{3})*(?:\.\d{2})|\d+(?:\.\d{2})")
    for line in lines:
        normalized_line = _normalize_text(line)
        if any(pattern in normalized_line for pattern in patterns):
            amounts = amount_regex.findall(line)
            if amounts:
                return amounts[-1]
    return ""


def parse_ocr_text(text: str) -> Dict[str, str]:
    """Parse target fields from OCR output."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    return {
        "Comisionado": _extract_line_value(lines, ["comisionado"]),
        "Area u Organo Jurisdiccional": _extract_line_value(
            lines, ["area u organo jurisdiccional"]
        ),
        "Dias que comprende la comision": _extract_line_value(
            lines, ["dias que comprende la comision"]
        ),
        "Localidad": _extract_line_value(lines, ["localidad"]),
        "Hospedaje $": _extract_last_amount(lines, ["hospedaje"]),
        "Alimentacion $": _extract_last_amount(lines, ["alimentacion"]),
        "Combustible / Pasajes $": _extract_last_amount(
            lines, ["combustible", "pasajes"]
        ),
        "Recorrido Int. / Taxis $": _extract_last_amount(lines, ["recorrido", "taxis"]),
        "Peaje $": _extract_last_amount(lines, ["peaje"]),
        "Total $": _extract_last_amount(lines, ["total"]),
    }


def process_image_bytes(
    image_bytes: bytes, image_name: str, lang: str = "spa"
) -> Dict[str, str]:
    """Process one uploaded image from raw bytes."""
    image = load_image_bytes(image_bytes, image_name)
    bw_image = preprocess_image(image)
    text = run_tesseract(bw_image, lang=lang)
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
