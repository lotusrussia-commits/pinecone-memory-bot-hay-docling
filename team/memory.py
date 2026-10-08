from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union

from haystack import Document

from team.pipelines import (
    build_indexing_pipeline,
    build_query_pipeline,
    build_summarization_pipeline,
    extract_text_from_replies,
    make_team_document,
    team_chat_filter,
)


ID = Union[str, int]


@dataclass
class ListeningSession:
    chat_id: str
    messages: List[Document]


class TeamMemory:
    """
    Memory for a Telegram team chat.

    Indexing Pipeline:
        Document -> OpenAI embedding -> Pinecone

    Query Pipeline:
        Question -> embedding -> Pinecone -> LLM

    Summarization Pipeline:
        Conversation -> prompt -> LLM
    """

    def __init__(self):
        self.indexing_pipeline = build_indexing_pipeline()
        self.query_pipeline = build_query_pipeline()
        self.summarization_pipeline = build_summarization_pipeline()

        self._sessions: Dict[str, ListeningSession] = {}
        self._lock = threading.Lock()

    def start_listening(self, chat_id: ID) -> None:
        key = str(chat_id)

        with self._lock:
            self._sessions[key] = ListeningSession(
                chat_id=key,
                messages=[],
            )

    def is_listening(self, chat_id: ID) -> bool:
        key = str(chat_id)

        with self._lock:
            return key in self._sessions

    def add_message(
        self,
        text: str,
        user_id: ID,
        display_name: str,
        username: Optional[str],
        chat_id: ID,
        message_id: ID,
        created_at: Optional[str] = None,
    ) -> bool:
        clean_text = (text or "").strip()

        if not clean_text:
            return False

        chat_key = str(chat_id)

        with self._lock:
            if chat_key not in self._sessions:
                return False

        if created_at is None:
            created_at = datetime.now(
                timezone.utc
            ).isoformat()

        document = make_team_document(
            text=clean_text,
            user_id=user_id,
            display_name=display_name,
            username=username,
            chat_id=chat_id,
            created_at=created_at,
            message_id=message_id,
        )

        self.indexing_pipeline.run(
            {
                "embedder": {
                    "documents": [document],
                }
            }
        )

        with self._lock:
            session = self._sessions.get(chat_key)

            if session is not None:
                session.messages.append(document)

        return True

    @staticmethod
    def _conversation_text(
        documents: List[Document],
    ) -> str:
        lines = []

        for document in documents:
            display_name = document.meta.get(
                "display_name",
                "\u0423\u0447\u0430\u0441\u0442\u043d\u0438\u043a",
            )

            username = document.meta.get(
                "username",
                "",
            )

            created_at = document.meta.get(
                "created_at",
                "",
            )

            if username:
                speaker = (
                    f"{display_name} "
                    f"(@{username})"
                )
            else:
                speaker = display_name

            lines.append(
                f"[{created_at}] "
                f"{speaker}: "
                f"{document.content}"
            )

        return "\n".join(lines)

    def stop_listening(self, chat_id: ID) -> str:
        chat_key = str(chat_id)

        with self._lock:
            session = self._sessions.pop(
                chat_key,
                None,
            )

        if session is None:
            return (
                "\u0420\u0435\u0436\u0438\u043c \u0441\u043b\u0443\u0448\u0430\u043d\u0438\u044f "
                "\u0434\u043b\u044f \u044d\u0442\u043e\u0433\u043e \u0447\u0430\u0442\u0430 "
                "\u0441\u0435\u0439\u0447\u0430\u0441 \u043d\u0435 \u0432\u043a\u043b\u044e\u0447\u0451\u043d."
            )

        if not session.messages:
            return (
                "\u0420\u0435\u0436\u0438\u043c \u0441\u043b\u0443\u0448\u0430\u043d\u0438\u044f "
                "\u043e\u0441\u0442\u0430\u043d\u043e\u0432\u043b\u0435\u043d, "
                "\u043d\u043e \u0441\u043e\u043e\u0431\u0449\u0435\u043d\u0438\u0439 "
                "\u0434\u043b\u044f \u0430\u043d\u0430\u043b\u0438\u0437\u0430 \u043d\u0435\u0442."
            )

        conversation = self._conversation_text(
            session.messages
        )

        result = self.summarization_pipeline.run(
            {
                "prompt_builder": {
                    "conversation": conversation,
                }
            }
        )

        summary = extract_text_from_replies(
            result["llm"]
        )

        if not summary:
            return (
                "u041fu0435u0440u0435u043fu0438u0441u043au0430 u0441u043eu0445u0440u0430u043du0435u043du0430, "
                "u043du043e u0438u0442u043eu0433 u0441u0444u043eu0440u043cu0438u0440u043eu0432u0430u0442u044c "
                "u043du0435 u0443u0434u0430u043bu043eu0441u044c."
            )

        return summary

    def ask(
        self,
        query: str,
        chat_id: ID,
    ) -> str:
        question = (query or "").strip()

        if not question:
            return "\u041d\u0430\u043f\u0438\u0448\u0438 \u0432\u043e\u043f\u0440\u043e\u0441."

        result = self.query_pipeline.run(
            {
                "text_embedder": {
                    "text": question,
                },
                "retriever": {
                    "filters": team_chat_filter(chat_id),
                },
                "prompt_builder": {
                    "query": question,
                },
            }
        )

        answer = extract_text_from_replies(
            result["llm"]
        )

        if answer:
            return answer

        return (
            "\u041d\u0435 \u0443\u0434\u0430\u043b\u043e\u0441\u044c \u0441\u0444\u043e\u0440\u043c\u0438\u0440\u043e\u0432\u0430\u0442\u044c "
            "\u043e\u0442\u0432\u0435\u0442 \u043f\u043e \u043a\u043e\u043d\u0442\u0435\u043a\u0441\u0442\u0443 \u0447\u0430\u0442\u0430."
        )


team_memory = TeamMemory()
