import os
import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv
from haystack import Document
from haystack.components.embedders import OpenAITextEmbedder
from haystack_integrations.document_stores.pinecone import (
    PineconeDocumentStore,
)
from haystack_integrations.components.retrievers.pinecone import (
    PineconeEmbeddingRetriever,
)
from haystack.utils import Secret


load_dotenv()


class PineconeMemory:
    """Память персонального ассистента на Pinecone."""

    def __init__(self):
        self.index_name = os.getenv("PINECONE_INDEX_NAME")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_base_url = os.getenv("OPENAI_BASE_URL")

        if not self.index_name:
            raise ValueError(
                "PINECONE_INDEX_NAME не найден в .env"
            )

        if not self.pinecone_api_key:
            raise ValueError(
                "PINECONE_API_KEY не найден в .env"
            )

        if not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY не найден в .env"
            )

        if not self.openai_base_url:
            raise ValueError(
                "OPENAI_BASE_URL не найден в .env"
            )

        self.document_store = PineconeDocumentStore(
            index=self.index_name,
            namespace="haystack-memory",
            api_key=Secret.from_token(
                self.pinecone_api_key
            ),
            dimension=1536,
        )

        self.embedder = OpenAITextEmbedder(
            api_key=Secret.from_env_var("OPENAI_API_KEY"),
            api_base_url=self.openai_base_url,
            model=os.getenv(
                "EMBEDDING_MODEL",
                "text-embedding-3-small",
            ),
        )

        self.retriever = PineconeEmbeddingRetriever(
            document_store=self.document_store,
            top_k=5,
        )

        self.similarity_threshold = self._read_threshold()

    @staticmethod
    def _read_threshold() -> float:
        raw_value = os.getenv("SIMILARITY_THRESHOLD", "0.5")

        try:
            return float(raw_value)
        except ValueError:
            return 0.5

    def save_message(
        self,
        text: str,
        user_id: str | int,
        role: str,
    ) -> None:
        """Сохраняет реплику пользователя или ассистента."""

        if not text or not text.strip():
            return

        text = text.strip()

        result = self.embedder.run(text)

        embedding = result["embedding"]

        document = Document(
            id=str(uuid.uuid4()),
            content=text,
            embedding=embedding,
            meta={
                "type": role,
                "user_id": str(user_id),
                "created_at": datetime.now(
                    timezone.utc
                ).isoformat(),
            },
        )

        self.document_store.write_documents(
            [document]
        )

    def search(
        self,
        query: str,
        user_id: str | int,
        top_k: int = 5,
    ) -> list[Document]:
        """Ищет похожие реплики только этого пользователя."""

        if not query or not query.strip():
            return []

        result = self.embedder.run(query)

        query_embedding = result["embedding"]

        retrieved = self.retriever.run(
            query_embedding=query_embedding,
            filters={
                "field": "user_id",
                "operator": "==",
                "value": str(user_id),
            },
            top_k=top_k,
        )

        documents = retrieved["documents"]

        return [
            document
            for document in documents
            if document.score is not None
            and document.score >= self.similarity_threshold
        ]

    def clear(self, user_id: str | int) -> int:
        """Удаляет документы только этого пользователя."""

        return self.document_store.delete_by_filter(
            filters={
                "field": "user_id",
                "operator": "==",
                "value": str(user_id),
            }
        )