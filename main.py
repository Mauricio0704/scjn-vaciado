"""OCR pipeline: Imagen -> OpenCV -> Tesseract -> Parser -> Pandas -> Excel."""

from __future__ import annotations

from io import BytesIO
import re
import unicodedata
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple
from zipfile import ZipFile

import cv2
import numpy as np
import pandas as pd
import pytesseract

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"


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


def load_image(image_path: str):
    """Step 1: Load source image from disk."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image file not found or cannot be read: {image_path}")
    return image


def load_image_bytes(image_bytes: bytes, image_name: str):
    """Load an image from raw bytes."""
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Image file not found or cannot be read: {image_name}")
    return image


def preprocess_image(image):
    """Step 2: Convert image to high-contrast black/white for OCR."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray_image, 9, 75, 75)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    
    sharpened = cv2.filter2D(denoised, -1, kernel)
    thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))  # Find non-zero pixel coordinates.
    angle = cv2.minAreaRect(coords)[-1]  # Calculate the skew angle
    
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        
    (h, w) = image.shape[:2]
        
    (h, w) = thresh.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return deskewed


def run_tesseract(image, lang: str = "spa") -> str:
    """Step 3: Extract OCR text in Spanish."""
    config = "--oem 3 --psm 6"
    return pytesseract.image_to_string(image, lang=lang, config=config)


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value)
    without_accents = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
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
    """Step 4: Parse target fields from OCR output."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    parsed = {
        "Comisionado": _extract_line_value(lines, ["comisionado"]),
        "Area u Organo Jurisdiccional": _extract_line_value(lines, ["area u organo jurisdiccional"]),
        "Dias que comprende la comision": _extract_line_value(lines, ["dias que comprende la comision"]),
        "Localidad": _extract_line_value(lines, ["localidad"]),
        "Hospedaje $": _extract_last_amount(lines, ["hospedaje"]),
        "Alimentacion $": _extract_last_amount(lines, ["alimentacion"]),
        "Combustible / Pasajes $": _extract_last_amount(lines, ["combustible", "pasajes"]),
        "Recorrido Int. / Taxis $": _extract_last_amount(lines, ["recorrido", "taxis"]),
        "Peaje $": _extract_last_amount(lines, ["peaje"]),
        "Total $": _extract_last_amount(lines, ["total"]),
    }

    return parsed


def process_image(image, image_name: str, lang: str = "spa") -> Dict[str, str]:
    """Run the OCR pipeline for a single decoded image."""
    bw_image = preprocess_image(image)
    text = run_tesseract(bw_image, lang=lang)
    parsed_record = parse_ocr_text(text)
    parsed_record["Archivo"] = image_name
    return parsed_record


def process_image_path(image_path: Path, lang: str = "spa") -> Dict[str, str]:
    """Process one image from disk."""
    image = load_image(str(image_path))
    return process_image(image, image_path.name, lang=lang)


def process_image_bytes(image_bytes: bytes, image_name: str, lang: str = "spa") -> Dict[str, str]:
    """Process one uploaded image from raw bytes."""
    image = load_image_bytes(image_bytes, image_name)
    return process_image(image, image_name, lang=lang)


def build_dataframe(records: Sequence[Dict[str, str]]) -> pd.DataFrame:
    """Step 5: Convert parsed records to a DataFrame."""
    return pd.DataFrame(
        [[record.get(column, "") for column in DATAFRAME_COLUMNS] for record in records],
        columns=DATAFRAME_COLUMNS,
    )


def export_to_excel(dataframe: pd.DataFrame, output_path: str) -> None:
    """Step 6: Export DataFrame to Excel file."""
    dataframe.to_excel(output_path, index=False)


def dataframe_to_excel_bytes(dataframe: pd.DataFrame) -> bytes:
    """Export a DataFrame to an in-memory Excel file."""
    buffer = BytesIO()
    dataframe.to_excel(buffer, index=False)
    buffer.seek(0)
    return buffer.getvalue()


def list_image_files(directory: Path) -> List[Path]:
    """Return supported image files in a directory."""
    if not directory.exists() or not directory.is_dir():
        raise ValueError(f"Directory does not exist or is not valid: {directory}")
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def extract_images_from_zip(zip_bytes: bytes) -> List[Tuple[str, bytes]]:
    """Extract supported images from a zip archive."""
    extracted_images: List[Tuple[str, bytes]] = []
    with ZipFile(BytesIO(zip_bytes)) as archive:
        for member in sorted(archive.namelist()):
            member_path = Path(member)
            if member.endswith("/") or member_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            extracted_images.append((member_path.as_posix(), archive.read(member)))
    return extracted_images


def process_uploaded_images(
    uploads: Sequence[Tuple[str, bytes]],
    lang: str = "spa",
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> pd.DataFrame:
    """Process image uploads provided as (file_name, bytes)."""
    records: List[Dict[str, str]] = []
    total = len(uploads)
    for index, (image_name, image_bytes) in enumerate(uploads, start=1):
        records.append(process_image_bytes(image_bytes, image_name, lang=lang))
        if progress_callback is not None:
            progress_callback(index, total, image_name)
    return build_dataframe(records)


def process_directory(
    directory: Path,
    lang: str = "spa",
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> pd.DataFrame:
    """Process all supported images in a directory."""
    image_files = list_image_files(directory)
    records: List[Dict[str, str]] = []
    total = len(image_files)
    for index, image_path in enumerate(image_files, start=1):
        records.append(process_image_path(image_path, lang=lang))
        if progress_callback is not None:
            progress_callback(index, total, image_path.name)
    return build_dataframe(records)


def main() -> None:
    tests_dir = Path("tests")
    excel_path = "output.xlsx"

    image_files = list_image_files(tests_dir)
    if not image_files:
        print(f"No images found in {tests_dir}/")
        return

    def report_progress(current: int, total: int, image_name: str) -> None:
        print(f"Processing {image_name} ({current}/{total}) ...")

    dataframe = process_directory(tests_dir, progress_callback=report_progress)
    export_to_excel(dataframe, excel_path)

    print("\nParsed records:")
    print(dataframe.to_string(index=False))
    print(f"\nExcel generated: {excel_path}")


if __name__ == "__main__":
    main()