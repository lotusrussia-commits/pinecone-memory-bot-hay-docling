import os
import threading

from dotenv import load_dotenv
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.tools import ComponentTool
from haystack.utils import Secret

from app.memory import PineconeMemory
from app.tools import CatFactTool, DogImageAnalyzerTool, WeatherTool


load_dotenv()


OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

if not OPENAI_BASE_URL:
    raise ValueError(
        "OPENAI_BASE_URL не найден в .env"
    )


HISTORY_LIMIT = 6

_history: dict[str, list[dict[str, str]]] = {}
_history_lock = threading.Lock()


weather_tool = ComponentTool(
    component=WeatherTool(),
    name="weather_tool",
    description=(
        "Получает текущую погоду в указанном городе. "
        "Используй этот инструмент, когда пользователь "
        "спрашивает о текущей погоде."
    ),
)


cat_fact_tool = ComponentTool(
    component=CatFactTool(),
    name="cat_fact_tool",
    description=(
        "Получает случайный интересный факт о кошках. "
        "Используй этот инструмент, когда пользователь "
        "просит рассказать факт о кошках или просит "
        "интересный факт про кошку."
    ),
)


dog_image_tool = ComponentTool(
    component=DogImageAnalyzerTool(),
    name="dog_image_analyzer",
    description=(
        "Анализирует локальный файл изображения. "
        "Вызывай этот инструмент, когда в сообщении "
        "есть путь к изображению. Аргумент image_path — "
        "путь к файлу на диске."
    ),
)


chat_generator = OpenAIChatGenerator(
    api_key=Secret.from_env_var("OPENAI_API_KEY"),
    api_base_url=OPENAI_BASE_URL,
    model="gpt-5.4-nano",
)


agent = Agent(
    chat_generator=chat_generator,
    system_prompt=(
        "Ты персональный ассистент. "
        "Отвечай на русском языке.\n\n"

        "У тебя есть следующие инструменты:\n"
        "1. weather_tool — получает текущую погоду "
        "в указанном городе. Обязательно используй его, "
        "если пользователь спрашивает о текущей погоде.\n"
        "2. cat_fact_tool — получает случайный интересный "
        "факт о кошках. Используй его, когда пользователь "
        "просит факт о кошках.\n"
        "3. dog_image_analyzer — анализирует изображение "
        "по локальному пути. Если в сообщении есть путь "
        "к файлу, обязательно вызови этот инструмент "
        "и ответь по его результату.\n\n"

        "Последние реплики чата нужны для коротких "
        "уточнений, например «а завтра?». "
        "Если в новой реплике нет города или темы, "
        "бери их из этих реплик и не переспрашивай "
        "то, что уже сказано. "
        "Контекст из памяти используй, только если он "
        "относится к текущему вопросу.\n\n"

        "Не придумывай данные о погоде самостоятельно. "
        "Если пользователь спрашивает о погоде, используй "
        "weather_tool."
    ),
    tools=[
        weather_tool,
        cat_fact_tool,
        dog_image_tool,
    ],
)


memory = PineconeMemory()


def _history_key(
    user_id: str | int,
    chat_id: str | int | None,
) -> str:
    if chat_id is None:
        return str(user_id)

    return str(chat_id)


def _history_text(chat_key: str) -> str:
    with _history_lock:
        turns = list(_history.get(chat_key, []))

    if not turns:
        return ""

    lines = []

    for turn in turns:
        if turn["role"] == "user":
            speaker = "Пользователь"
        else:
            speaker = "Ассистент"

        lines.append(f"{speaker}: {turn['text']}")

    return (
        "\n\nПоследние реплики этого чата:\n"
        + "\n".join(lines)
    )


def _remember_turn(
    chat_key: str,
    role: str,
    text: str,
) -> None:
    with _history_lock:
        turns = _history.setdefault(chat_key, [])

        turns.append({
            "role": role,
            "text": text,
        })

        del turns[:-HISTORY_LIMIT]


def clear_memory(
    user_id: str | int,
    chat_id: str | int | None = None,
) -> int:
    """Очищает долгую память пользователя и реплики этого чата."""

    deleted = memory.clear(user_id)

    chat_key = _history_key(user_id, chat_id)

    with _history_lock:
        _history.pop(chat_key, None)

    return deleted


def ask_agent(
    text: str,
    user_id: str | int,
    chat_id: str | int | None = None,
    image_path: str | None = None,
) -> str:
    """
    Ищет память до сохранения текущего сообщения,
    добавляет короткие реплики чата и отвечает через агента.

    В Pinecone сохраняются только сообщения пользователя.
    Ответ ассистента хранится только в короткой памяти
    текущего процесса.
    """

    user_text = (text or "").strip()

    if not user_text:
        return "Напиши сообщение или отправь фотографию."

    chat_key = _history_key(user_id, chat_id)

    memories = []

    if user_text:
        memories = memory.search(
        user_text,
        user_id=user_id,
        top_k=5,
    )

    context = ""

    if memories:
        context_lines = []

        for document in memories:
            context_lines.append(
                f"- Пользователь: {document.content}"
            )

        context = (
            "\n\nКонтекст из памяти пользователя:\n"
            + "\n".join(context_lines)
        )

    history = _history_text(chat_key)

    # В Pinecone сохраняем только текст сообщения пользователя.
    if user_text:
        memory.save_message(
        user_text,
        user_id=user_id,
        role="user_message",
    )

    prompt_parts = []

    if history:
        prompt_parts.append(history.strip())

    if context:
        prompt_parts.append(context.strip())

    prompt_parts.append(user_text)

    if image_path:
        prompt_parts.append(
            "К сообщению приложено изображение. "
            f"Путь к файлу: {image_path}. "
            "Обязательно вызови dog_image_analyzer "
            "с этим путём и ответь по его результату."
        )

    prompt = "\n\n".join(prompt_parts)

    result = agent.run(
        messages=[
            ChatMessage.from_user(prompt)
        ]
    )

    answer = result["last_message"].text

    # Ответ ассистента НЕ сохраняем в Pinecone,
    # потому что по заданию долговременная память
    # должна содержать только сообщения пользователя.

    _remember_turn(
        chat_key,
        "user",
        user_text,
    )

    _remember_turn(
        chat_key,
        "assistant",
        answer,
    )

    return answer