from __future__ import annotations

from typing import List, Tuple

import streamlit as st

from main import dataframe_to_excel_bytes, extract_images_from_zip, process_uploaded_images


def collect_uploads() -> List[Tuple[str, bytes]]:
    uploads: List[Tuple[str, bytes]] = []

    image_files = st.file_uploader(
        "Imagenes",
        type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"],
        accept_multiple_files=True,
        help="Sube una o varias imagenes.",
    )
    if image_files:
        for image_file in image_files:
            uploads.append((image_file.name, image_file.getvalue()))

    zip_file = st.file_uploader(
        "Carpeta comprimida (.zip)",
        type=["zip"],
        help="Sube la carpeta como archivo zip.",
    )
    if zip_file is not None:
        uploads.extend(extract_images_from_zip(zip_file.getvalue()))

    return uploads


def main() -> None:
    st.set_page_config(page_title="SCJN OCR", page_icon="📄", layout="centered")
    st.title("Vaciado viáticos")
    st.write(
        "Sube una o varias imágenes, o una carpeta comprimida con imágenes. "
        "La aplicacion procesara cada archivo y generara una fila en Excel por imagen."
    )

    uploads = collect_uploads()
    process_clicked = st.button("Procesar imagenes", type="primary", disabled=not uploads)

    if not uploads:
        st.info("Sube al menos una imagen o un archivo zip con imagenes para continuar.")
        return

    st.caption(f"{len(uploads)} imagen(es) listas para procesar")

    if not process_clicked:
        return

    progress_bar = st.progress(0)
    status = st.empty()

    def report_progress(current: int, total: int, image_name: str) -> None:
        progress_bar.progress(current / total, text=f"Procesando {image_name} ({current}/{total})")
        status.write(f"Procesada: {image_name}")

    try:
        dataframe = process_uploaded_images(uploads, progress_callback=report_progress)
    except Exception as exc:
        st.error(f"El procesamiento fallo: {exc}")
        return

    progress_bar.progress(1.0, text="Procesamiento completado")
    excel_bytes = dataframe_to_excel_bytes(dataframe)

    st.success("El archivo Excel se genero correctamente.")
    st.dataframe(dataframe, use_container_width=True)
    st.download_button(
        label="Descargar Excel",
        data=excel_bytes,
        file_name="output.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


if __name__ == "__main__":
    main()