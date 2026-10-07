import os
import tempfile
import unittest
import zipfile
from pathlib import Path


REQUIRED_META = (
    "user_id",
    "filename",
    "page_number",
    "chunk_number",
    "source_type",
)


def connection_pairs(pipeline) -> set[tuple[str, str]]:
    pairs = set()

    for sender, receiver, key in pipeline.graph.edges(keys=True):
        output_name, input_name = key.split("/", 1)
        pairs.add(
            (f"{sender}.{output_name}", f"{receiver}.{input_name}")
        )

    return pairs


def write_minimal_pdf(path: Path, sentence: str) -> None:
    safe = (
        sentence.replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
    )
    stream = f"BT /F1 18 Tf 36 90 Td ({safe}) Tj ET\n".encode(
        "latin-1",
        errors="replace",
    )
    objects = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Count 1 /Kids [3 0 R] >>\nendobj\n",
        (
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R "
            b"/MediaBox [0 0 612 792] /Contents 4 0 R "
            b"/Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        ),
        b"4 0 obj\n<< /Length %d >>\nstream\n%s\nendstream\nendobj\n"
        % (len(stream), stream),
        (
            b"5 0 obj\n<< /Type /Font /Subtype /Type1 "
            b"/BaseFont /Helvetica >>\nendobj\n"
        ),
    ]
    header = b"%PDF-1.4\n"
    body = b""
    offsets = [0]

    for obj in objects:
        offsets.append(len(header) + len(body))
        body += obj

    xref_pos = len(header) + len(body)
    xref = [
        b"xref\n",
        f"0 {len(offsets)}\n".encode(),
        b"0000000000 65535 f \n",
    ]

    for offset in offsets[1:]:
        xref.append(f"{offset:010d} 00000 n \n".encode())

    trailer = (
        f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n"
    ).encode()
    path.write_bytes(header + body + b"".join(xref) + trailer)


def write_minimal_docx(path: Path, sentence: str) -> None:
    document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>{sentence}</w:t></w:r></w:p>
  </w:body>
</w:document>
"""
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>
"""
    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
"""

    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", rels)
        archive.writestr("word/document.xml", document_xml)


class ImportTests(unittest.TestCase):
    def test_new_modules_import(self):
        import bot.telegram_bot
        import components.document_processor
        import components.document_store
        import main
        import pipelines.ingestion
        import pipelines.rag

        self.assertTrue(hasattr(main, "main"))
        self.assertTrue(
            hasattr(bot.telegram_bot, "create_bot")
        )
        self.assertTrue(
            hasattr(
                components.document_processor,
                "documents_from_pages",
            )
        )
        self.assertTrue(
            hasattr(pipelines.ingestion, "build_ingestion_pipeline")
        )
        self.assertTrue(
            hasattr(pipelines.rag, "build_rag_pipeline")
        )


class PipelineTests(unittest.TestCase):
    def test_ingestion_pipeline_uses_openai_embedder(self):
        from haystack.components.embedders import (
            OpenAITextEmbedder,
        )

        from pipelines.ingestion import build_ingestion_pipeline

        pipeline = build_ingestion_pipeline()
        embedder = pipeline.get_component("embedder")
        writer = pipeline.get_component("writer")

        self.assertIsInstance(
            embedder.text_embedder,
            OpenAITextEmbedder,
        )
        self.assertEqual(
            embedder.text_embedder.model,
            os.getenv(
                "EMBEDDING_MODEL",
                "text-embedding-3-small",
            ),
        )
        self.assertNotIn(
            "sentence_transformers",
            embedder.text_embedder.__class__.__module__,
        )
        self.assertEqual(writer.document_store.dimension, 1536)
        self.assertEqual(writer.document_store.metric, "cosine")
        self.assertEqual(
            writer.document_store.namespace,
            os.getenv(
                "DOCUMENT_NAMESPACE",
                "haystack-documents",
            ),
        )

        self.assertIn(
            ("embedder.documents", "writer.documents"),
            connection_pairs(pipeline),
        )

    def test_rag_pipeline_connects(self):
        from haystack.components.builders import PromptBuilder
        from haystack.components.embedders import (
            OpenAITextEmbedder,
        )
        from haystack.components.generators.chat import (
            OpenAIChatGenerator,
        )

        from pipelines.rag import build_rag_pipeline

        pipeline = build_rag_pipeline()

        self.assertIsInstance(
            pipeline.get_component("text_embedder"),
            OpenAITextEmbedder,
        )
        self.assertIsInstance(
            pipeline.get_component("prompt_builder"),
            PromptBuilder,
        )
        self.assertIsInstance(
            pipeline.get_component("llm"),
            OpenAIChatGenerator,
        )
        self.assertEqual(
            pipeline.get_component("retriever").document_store.namespace,
            os.getenv(
                "DOCUMENT_NAMESPACE",
                "haystack-documents",
            ),
        )

        connections = connection_pairs(pipeline)
        self.assertIn(
            (
                "text_embedder.embedding",
                "retriever.query_embedding",
            ),
            connections,
        )
        self.assertIn(
            (
                "retriever.documents",
                "prompt_builder.documents",
            ),
            connections,
        )
        self.assertIn(
            ("prompt_builder.prompt", "llm.messages"),
            connections,
        )


class DocumentMetadataTests(unittest.TestCase):
    def test_chunk_metadata(self):
        from components.document_processor import (
            documents_from_pages,
            first_sentence,
            source_type_for,
        )

        documents = documents_from_pages(
            pages=[
                (2, "Первая страница про яблоки."),
                (3, "Вторая страница про груши."),
            ],
            filename="notes.pdf",
            user_id=42,
            source_type="pdf",
        )

        self.assertEqual(len(documents), 2)

        for document in documents:
            for key in REQUIRED_META:
                self.assertIn(key, document.meta)

        self.assertEqual(documents[0].meta["user_id"], "42")
        self.assertEqual(documents[0].meta["filename"], "notes.pdf")
        self.assertEqual(documents[0].meta["page_number"], 2)
        self.assertEqual(documents[0].meta["chunk_number"], 1)
        self.assertEqual(documents[0].meta["source_type"], "pdf")
        self.assertEqual(documents[1].meta["page_number"], 3)
        self.assertEqual(documents[1].meta["chunk_number"], 2)
        self.assertEqual(source_type_for("file.pdf", None), "pdf")
        self.assertEqual(
            source_type_for(
                "file.docx",
                None,
            ),
            "docx",
        )
        self.assertEqual(
            source_type_for(
                "file.bin",
                "application/pdf",
            ),
            "pdf",
        )
        self.assertIsNone(source_type_for("notes.txt", "text/plain"))
        self.assertEqual(
            first_sentence("Первое предложение. Второе тоже."),
            "Первое предложение.",
        )

    def test_docling_reads_pdf_and_docx(self):
        from components.document_processor import (
            documents_from_file,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf_path = root / "sample.pdf"
            docx_path = root / "sample.docx"
            write_minimal_pdf(
                pdf_path,
                "Hello from the Docling PDF fixture.",
            )
            write_minimal_docx(
                docx_path,
                "Тестовый DOCX про поиск по своим данным.",
            )

            pdf_documents, pdf_summary = documents_from_file(
                str(pdf_path),
                filename="sample.pdf",
                user_id="user-pdf",
                source_type="pdf",
            )
            docx_documents, docx_summary = documents_from_file(
                str(docx_path),
                filename="sample.docx",
                user_id="user-docx",
                source_type="docx",
            )

        self.assertTrue(pdf_documents)
        self.assertTrue(pdf_summary)
        self.assertTrue(docx_documents)
        self.assertTrue(docx_summary)

        for document in pdf_documents + docx_documents:
            for key in REQUIRED_META:
                self.assertIn(key, document.meta)
            self.assertIsInstance(
                document.meta["page_number"],
                int,
            )
            self.assertGreaterEqual(
                document.meta["page_number"],
                1,
            )
            self.assertIsInstance(
                document.meta["chunk_number"],
                int,
            )

        self.assertEqual(
            pdf_documents[0].meta["source_type"],
            "pdf",
        )
        self.assertEqual(
            docx_documents[0].meta["user_id"],
            "user-docx",
        )
        self.assertIn(
            "DOCX",
            docx_documents[0].content,
        )


if __name__ == "__main__":
    unittest.main()
