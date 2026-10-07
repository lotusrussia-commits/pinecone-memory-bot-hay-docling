import os

from dotenv import load_dotenv
from haystack.utils import Secret
from haystack_integrations.document_stores.pinecone import (
    PineconeDocumentStore,
)


load_dotenv()


DOCUMENTS_NAMESPACE = "haystack-documents"
EMBEDDING_DIMENSION = 1536
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def documents_namespace() -> str:
    return os.getenv("DOCUMENT_NAMESPACE", DOCUMENTS_NAMESPACE)


def embedding_model_name() -> str:
    return os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def openai_base_url() -> str | None:
    return os.getenv("OPENAI_BASE_URL")


def similarity_threshold() -> float:
    raw_value = os.getenv("SIMILARITY_THRESHOLD", "0.5")

    try:
        return float(raw_value)
    except ValueError:
        return 0.5


def pinecone_secret() -> Secret:
    # Ключ читается только при реальном запросе в Pinecone.
    return Secret.from_env_var("PINECONE_API_KEY")


def build_document_store() -> PineconeDocumentStore:
    """Хранилище чанков документов. Индекс не создаётся до первой записи."""

    return PineconeDocumentStore(
        api_key=pinecone_secret(),
        index=os.getenv(
            "PINECONE_INDEX_NAME",
            "pinecone-memory-bot",
        ),
        namespace=documents_namespace(),
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
    )


def user_filter(user_id: str | int) -> dict:
    return {
        "field": "user_id",
        "operator": "==",
        "value": str(user_id),
    }


def user_filename_filter(
    user_id: str | int,
    filename: str,
) -> dict:
    return {
        "operator": "AND",
        "conditions": [
            user_filter(user_id),
            {
                "field": "filename",
                "operator": "==",
                "value": filename,
            },
        ],
    }


def clear_user_documents(user_id: str | int) -> int:
    """Удаляет чанки документов только этого пользователя."""

    document_store = build_document_store()

    return document_store.delete_by_filter(
        filters=user_filter(user_id)
    )
