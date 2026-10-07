import os
import uuid
from pathlib import Path

from docling.datamodel.base_models import (
    ConversionStatus,
    InputFormat,
)
from docling.datamodel.pipeline_options import (
    NativePdfPipelineOptions,
)
from docling.document_converter import (
    DocumentConverter,
    NativePdfFormatOption,
)
from haystack import Document


PDF_MIME = "application/pdf"
DOCX_MIME = (
    "application/vnd.openxmlformats-officedocument."
    "wordprocessingml.document"
)

CHUNK_CHAR_LIMIT = 1500

_converter: DocumentConverter | None = None


def source_type_for(
    filename: str | None,
    mime_type: str | None = None,
) -> str | None:
    """Возвращает pdf или docx, если формат поддерживается."""

    extension = Path(filename or "").suffix.lower()

    if extension == ".pdf":
        return "pdf"

    if extension == ".docx":
        return "docx"

    mime = (mime_type or "").lower()

    if mime == PDF_MIME:
        return "pdf"

    if mime == DOCX_MIME:
        return "docx"

    return None


def build_converter() -> DocumentConverter:
    """PDF читается без OCR и layout-модели. DOCX идёт штатным парсером."""

    pdf_options = NativePdfPipelineOptions(
        generate_picture_images=False,
        generate_page_images=False,
    )

    return DocumentConverter(
        allowed_formats=[
            InputFormat.PDF,
            InputFormat.DOCX,
        ],
        format_options={
            InputFormat.PDF: NativePdfFormatOption(
                pipeline_options=pdf_options,
            ),
        },
    )


def get_converter() -> DocumentConverter:
    global _converter

    if _converter is None:
        _converter = build_converter()

    return _converter


def convert_file(path: str):
    result = get_converter().convert(path)

    if result.status not in (
        ConversionStatus.SUCCESS,
        ConversionStatus.PARTIAL_SUCCESS,
    ):
        raise RuntimeError(
            "Docling не смог разобрать документ"
        )

    if result.document is None:
        raise RuntimeError(
            "Docling не вернул содержимое документа"
        )

    return result.document


def page_texts(document) -> list[tuple[int, str]]:
    """Текст по страницам. page_no — параметр export_to_markdown."""

    texts: list[tuple[int, str]] = []
    pages = getattr(document, "pages", None) or {}

    for page_no in sorted(pages):
        text = document.export_to_markdown(
            page_no=int(page_no),
        ).strip()

        if text:
            texts.append((int(page_no), text))

    if texts:
        return texts

    whole_document = document.export_to_markdown().strip()

    if whole_document:
        return [(1, whole_document)]

    return []


def split_page_text(text: str) -> list[str]:
    cleaned = text.strip()

    if not cleaned:
        return []

    if len(cleaned) <= CHUNK_CHAR_LIMIT:
        return [cleaned]

    parts = []
    start = 0

    while start < len(cleaned):
        part = cleaned[start:start + CHUNK_CHAR_LIMIT].strip()

        if part:
            parts.append(part)

        start += CHUNK_CHAR_LIMIT

    return parts


def documents_from_pages(
    pages: list[tuple[int, str]],
    filename: str,
    user_id: str | int,
    source_type: str,
) -> list[Document]:
    documents = []
    chunk_number = 1

    for page_number, page_text in pages:
        for part in split_page_text(page_text):
            documents.append(
                Document(
                    id=str(uuid.uuid4()),
                    content=part,
                    meta={
                        "user_id": str(user_id),
                        "filename": filename,
                        "page_number": page_number,
                        "chunk_number": chunk_number,
                        "source_type": source_type,
                    },
                )
            )
            chunk_number += 1

    return documents


def documents_from_file(
    path: str,
    filename: str,
    user_id: str | int,
    source_type: str,
) -> tuple[list[Document], str]:
    document = convert_file(path)
    pages = page_texts(document)
    haystack_documents = documents_from_pages(
        pages,
        filename=filename,
        user_id=user_id,
        source_type=source_type,
    )
    summary_source = document.export_to_markdown().strip()

    if not summary_source:
        summary_source = "\n\n".join(
            page_text for _, page_text in pages
        )

    return haystack_documents, summary_source


def first_sentence(text: str) -> str:
    cleaned = " ".join(
        (text or "").replace("\n", " ").split()
    )

    if not cleaned:
        return (
            "Краткое содержание документа "
            "получить не удалось."
        )

    for index, char in enumerate(cleaned):
        if char in ".!?":
            return cleaned[: index + 1].strip()

    if cleaned[-1] not in ".!?":
        return f"{cleaned}."

    return cleaned


def temp_directory() -> str:
    directory = os.getenv("DOCUMENT_TMP_DIR", "/tmp")
    os.makedirs(directory, exist_ok=True)
    return directory
