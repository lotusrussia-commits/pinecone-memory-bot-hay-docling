from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.generators.chat import (
    OpenAIChatGenerator,
)
from haystack_integrations.components.retrievers.pinecone import (
    PineconeEmbeddingRetriever,
)

from components.document_store import (
    build_document_store,
    similarity_threshold,
    user_filter,
)
from pipelines.ingestion import (
    build_chat_generator,
    build_text_embedder,
)


RAG_TOP_K = 5

RAG_TEMPLATE = """
Ты отвечаешь на вопрос только по фрагментам загруженных документов.
Если во фрагментах нет ответа, напиши ровно одно предложение:
"В загруженных документах ответа не найдено."
Не придумывай факты вне фрагментов.
Отвечай на русском языке.

Фрагменты:
{% for doc in documents %}
[{{ doc.meta.filename }}, страница {{ doc.meta.page_number }}, фрагмент {{ doc.meta.chunk_number }}]
{{ doc.content }}
{% endfor %}

Вопрос: {{ query }}
""".strip()


def build_rag_pipeline(document_store=None) -> Pipeline:
    store = document_store or build_document_store()
    pipeline = Pipeline()

    pipeline.add_component(
        "text_embedder",
        build_text_embedder(),
    )
    pipeline.add_component(
        "retriever",
        PineconeEmbeddingRetriever(
            document_store=store,
            top_k=RAG_TOP_K,
        ),
    )
    pipeline.add_component(
        "prompt_builder",
        PromptBuilder(template=RAG_TEMPLATE),
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


def _relevant(documents: list) -> list:
    threshold = similarity_threshold()

    return [
        document
        for document in documents
        if document.score is not None
        and document.score >= threshold
    ]


def answer_from_documents(
    query: str,
    user_id: str | int,
    pipeline: Pipeline | None = None,
    chat_generator: OpenAIChatGenerator | None = None,
) -> str | None:
    """
    Ищет чанки этого user_id.
    Если релевантных нет, возвращает None: вопрос уйдёт обычному агенту.
    """

    question = (query or "").strip()

    if not question:
        return None

    rag_pipeline = pipeline or build_rag_pipeline()
    embedder = rag_pipeline.get_component("text_embedder")
    retriever = rag_pipeline.get_component("retriever")
    prompt_builder = rag_pipeline.get_component(
        "prompt_builder",
    )
    generator = chat_generator or rag_pipeline.get_component(
        "llm",
    )

    embedding = embedder.run(text=question)["embedding"]
    retrieved = retriever.run(
        query_embedding=embedding,
        filters=user_filter(user_id),
        top_k=RAG_TOP_K,
    )["documents"]
    documents = _relevant(retrieved)

    if not documents:
        return None

    prompt = prompt_builder.run(
        documents=documents,
        query=question,
    )["prompt"]
    replies = generator.run(messages=prompt)["replies"]

    return replies[0].text
