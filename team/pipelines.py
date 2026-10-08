import os

from dotenv import load_dotenv
from haystack import Document, Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.embedders import (
    OpenAIDocumentEmbedder,
    OpenAITextEmbedder,
)
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import ChatMessage
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret
from haystack_integrations.components.retrievers.pinecone import (
    PineconeEmbeddingRetriever,
)
from haystack_integrations.document_stores.pinecone import (
    PineconeDocumentStore,
)

load_dotenv()


TEAM_NAMESPACE = os.getenv(
    "TEAM_NAMESPACE",
    "haystack-team",
)

EMBEDDING_DIMENSION = 1536

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_OPENAI_MODEL = "gpt-5.4-nano"

TEAM_TOP_K = 8


def embedding_model_name() -> str:
    return os.getenv(
        "EMBEDDING_MODEL",
        DEFAULT_EMBEDDING_MODEL,
    )


def openai_model_name() -> str:
    return os.getenv(
        "OPENAI_MODEL",
        DEFAULT_OPENAI_MODEL,
    )


def openai_base_url() -> str:
    value = os.getenv("OPENAI_BASE_URL")

    if not value:
        raise ValueError(
            "OPENAI_BASE_URL not found in .env"
        )

    return value


def build_team_document_store() -> PineconeDocumentStore:
    index_name = os.getenv("PINECONE_INDEX_NAME")

    if not index_name:
        raise ValueError(
            "PINECONE_INDEX_NAME not found in .env"
        )

    return PineconeDocumentStore(
        api_key=Secret.from_env_var(
            "PINECONE_API_KEY"
        ),
        index=index_name,
        namespace=TEAM_NAMESPACE,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
    )


def build_document_embedder() -> OpenAIDocumentEmbedder:
    return OpenAIDocumentEmbedder(
        api_key=Secret.from_env_var("OPENAI_API_KEY"),
        api_base_url=openai_base_url(),
        model=embedding_model_name(),
        batch_size=32,
        progress_bar=False,
    )


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
        model=openai_model_name(),
    )


def build_indexing_pipeline(
    document_store: PineconeDocumentStore | None = None,
) -> Pipeline:
    store = document_store or build_team_document_store()

    pipeline = Pipeline()

    pipeline.add_component(
        "embedder",
        build_document_embedder(),
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


QUERY_TEMPLATE = [
    ChatMessage.from_system(
        """
You are an AI assistant for a company team chat.

Answer only using the retrieved team-chat context.

Always consider:
- who wrote each message;
- the message content;
- the order of discussion;
- timestamps when available.

Do not invent facts.

If the answer is not present in the retrieved context,
say that the required information was not found
in the chat history.

Answer in Russian.
        """.strip()
    ),
    ChatMessage.from_user(
        """
Retrieved team-chat messages:

{% for doc in documents %}
[{{ doc.meta.get("display_name", "Participant") }}
{% if doc.meta.get("username") %}
@{{ doc.meta.get("username") }}
{% endif %}
| {{ doc.meta.get("created_at", "unknown time") }}]

{{ doc.content }}

{% endfor %}

Question:
{{ query }}
        """.strip()
    ),
]


def build_query_pipeline(
    document_store: PineconeDocumentStore | None = None,
) -> Pipeline:
    store = document_store or build_team_document_store()

    pipeline = Pipeline()

    pipeline.add_component(
        "text_embedder",
        build_text_embedder(),
    )

    pipeline.add_component(
        "retriever",
        PineconeEmbeddingRetriever(
            document_store=store,
            top_k=TEAM_TOP_K,
        ),
    )

    pipeline.add_component(
        "prompt_builder",
        ChatPromptBuilder(
            template=QUERY_TEMPLATE,
        ),
    )

    pipeline.add_component(
        "llm",
        build_chat_generator(),
    )

    pipeline.connect(
        "text_embedder.embedding",
        "retriever.query_embedding",
    )

    pipeline.connect(
        "retriever.documents",
        "prompt_builder.documents",
    )

    pipeline.connect(
        "prompt_builder.prompt",
        "llm.messages",
    )

    return pipeline


SUMMARIZATION_TEMPLATE = [
    ChatMessage.from_system(
        """
You are an AI assistant for a company team chat.

Analyze the complete discussion.

Prepare a concise final summary.

Include:
1. the discussion topic;
2. the main positions of participants;
3. what the team agreed on;
4. actions or decisions that should happen next.

If there was a disagreement,
briefly describe the opposing positions
and then give a balanced opinion.

Do not invent information.

Answer in Russian.
        """.strip()
    ),
    ChatMessage.from_user(
        """
Complete team-chat discussion:

{{ conversation }}

Prepare the final discussion summary.
        """.strip()
    ),
]


def build_summarization_pipeline() -> Pipeline:
    pipeline = Pipeline()

    pipeline.add_component(
        "prompt_builder",
        ChatPromptBuilder(
            template=SUMMARIZATION_TEMPLATE,
        ),
    )

    pipeline.add_component(
        "llm",
        build_chat_generator(),
    )

    pipeline.connect(
        "prompt_builder.prompt",
        "llm.messages",
    )

    return pipeline


def make_team_document(
    text: str,
    user_id: str | int,
    display_name: str,
    username: str | None,
    chat_id: str | int,
    created_at: str,
    message_id: str | int,
) -> Document:
    return Document(
        content=text.strip(),
        meta={
            "user_id": str(user_id),
            "display_name": display_name,
            "username": username or "",
            "chat_id": str(chat_id),
            "created_at": created_at,
            "message_id": str(message_id),
            "source": "telegram_team_chat",
        },
    )


def team_chat_filter(chat_id: str | int) -> dict:
    return {
        "field": "chat_id",
        "operator": "==",
        "value": str(chat_id),
    }


def extract_text_from_replies(result: dict) -> str:
    replies = result.get("replies", [])

    if not replies:
        return ""

    return (replies[0].text or "").strip()
