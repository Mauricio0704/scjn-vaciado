"""FastAPI application for the SCJN OCR pipeline."""

from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from main import (
    IMAGE_EXTENSIONS,
    build_dataframe,
    dataframe_to_excel_bytes,
    extract_images_from_zip,
    process_image_bytes,
)

app = FastAPI(title="SCJN Vaciado OCR API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4321",
        "http://localhost:3000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_ORIGINS_ENV = None


@app.on_event("startup")
def _configure_cors_from_env() -> None:
    """Allow overriding CORS origins via FRONTEND_URL env var."""
    import os

    frontend_url = os.getenv("FRONTEND_URL")
    if frontend_url:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[frontend_url],
            allow_methods=["*"],
            allow_headers=["*"],
        )


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/process")
async def process_images(files: list[UploadFile] = File(...)):
    """Process uploaded images and/or zip files, return parsed JSON rows."""
    uploads: list[tuple[str, bytes]] = []

    for file in files:
        content = await file.read()
        filename = file.filename or "unknown"

        if filename.lower().endswith(".zip"):
            uploads.extend(extract_images_from_zip(content))
        else:
            ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
            if ext not in IMAGE_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Tipo de archivo no soportado: {filename}",
                )
            uploads.append((filename, content))

    if not uploads:
        raise HTTPException(status_code=400, detail="No se enviaron imagenes validas.")

    records = []
    for image_name, image_bytes in uploads:
        try:
            records.append(process_image_bytes(image_bytes, image_name))
        except Exception as exc:
            records.append({"Archivo": image_name, "error": str(exc)})

    return {"records": records, "total": len(records)}


@app.post("/process/excel")
async def process_images_excel(files: list[UploadFile] = File(...)):
    """Process uploaded images and return an Excel file."""
    uploads: list[tuple[str, bytes]] = []

    for file in files:
        content = await file.read()
        filename = file.filename or "unknown"

        if filename.lower().endswith(".zip"):
            uploads.extend(extract_images_from_zip(content))
        else:
            ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
            if ext not in IMAGE_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Tipo de archivo no soportado: {filename}",
                )
            uploads.append((filename, content))

    if not uploads:
        raise HTTPException(status_code=400, detail="No se enviaron imagenes validas.")

    records = []
    for image_name, image_bytes in uploads:
        records.append(process_image_bytes(image_bytes, image_name))

    dataframe = build_dataframe(records)
    excel_bytes = dataframe_to_excel_bytes(dataframe)

    return Response(
        content=excel_bytes,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=output.xlsx"},
    )
