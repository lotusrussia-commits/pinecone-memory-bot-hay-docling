from haystack import Document, Pipeline, component
from haystack.components.embedders import OpenAITextEmbedder
from haystack.components.generators.chat import (
    OpenAIChatGenerator,
)
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret

from components.document_processor import (
    documents_from_file,
    first_sentence,
)
from components.document_store import (
    build_document_store,
    embedding_model_name,
    openai_base_url,
    user_filename_filter,
)


SUMMARY_PROMPT = (
    "Сделай краткое содержание документа. "
    "Ответь ровно одним предложением на русском языке. "
    "Не используй список и не пиши второе предложение.\n\n"
    "{text}"
)


@component
class OpenAIChunkEmbedder:
    """Считает embedding каждого чанка через OpenAITextEmbedder."""

    def __init__(self, text_embedder: OpenAITextEmbedder):
        self.text_embedder = text_embedder

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]):
        embedded = []

        for document in documents:
            if not document.content or not document.content.strip():
                continue

            result = self.text_embedder.run(
                text=document.content,
            )
            document.embedding = result["embedding"]
            embedded.append(document)

        return {"documents": embedded}


def build_text_embedder() -> OpenAITextEmbedder:
    return OpenAITextEmbedder(
        api_key=Secret.from_env_var("OPENAI_API_KEY"),
        api_base_url=openai_base_url(),
        model=embedding_model_name(),
    )


def build_chat_generator() -> OpenAIChatGenerator:
    return OpenAIChatGenerator(
        api_key=Secret.from_env_var("OPENAI_API_KEY"),
        api_base_url=openai_base_url(),
        model="gpt-5.4-nano",
    )


def build_ingestion_pipeline(
    document_store=None,
) -> Pipeline:
    store = document_store or build_document_store()
    pipeline = Pipeline()

    pipeline.add_component(
        "embedder",
        OpenAIChunkEmbedder(build_text_embedder()),
    )
    pipeline.add_component(
        "writer",
        DocumentWriter(
            document_store=store,
            policy=DuplicatePolicy.OVERWRITE,
        ),
    )
    pipeline.connect(
        "embedder.documents",
        "writer.documents",
    )

    return pipeline


def summarize_document(text: str) -> str:
    source = (text or "").strip()

    if not source:
        return first_sentence("")

    generator = build_chat_generator()
    result = generator.run(
        messages=SUMMARY_PROMPT.format(text=source[:6000]),
    )
    reply = result["replies"][0].text

    return first_sentence(reply)


def ingest_document(
    path: str,
    user_id: str | int,
    filename: str,
    source_type: str,
) -> str:
    """Разбирает файл, пишет чанки в Pinecone и возвращает одно предложение."""

    documents, summary_source = documents_from_file(
        path,
        filename=filename,
        user_id=user_id,
        source_type=source_type,
    )

    if not documents:
        raise RuntimeError(
            "В документе не найден текст для сохранения"
        )

    document_store = build_document_store()
    document_store.delete_by_filter(
        filters=user_filename_filter(user_id, filename),
    )

    pipeline = build_ingestion_pipeline(document_store)
    pipeline.run({
        "embedder": {
            "documents": documents,
        },
    })

    try:
        return summarize_document(summary_source)
    except Exception:
        return (
            "Документ сохранён, но краткое содержание "
            "составить не удалось."
        )
