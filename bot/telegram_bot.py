import os
from pathlib import Path

import telebot
from dotenv import load_dotenv

from components.document_processor import (
    source_type_for,
    temp_directory,
)
from components.document_store import clear_user_documents


load_dotenv()


def _public_document_error(error: Exception) -> str:
    message = str(error).strip().splitlines()
    detail = message[0] if message else "неизвестная ошибка"

    if len(detail) > 200:
        detail = detail[:200] + "..."

    return f"❌ Не удалось обработать документ: {detail}"


def create_bot() -> telebot.TeleBot:
    token = os.getenv("TELEGRAM_BOT_TOKEN")

    if not token:
        raise ValueError(
            "TELEGRAM_BOT_TOKEN не найден в .env"
        )

    bot = telebot.TeleBot(token)

    @bot.message_handler(commands=["start"])
    def start_command(message):
        bot.reply_to(
            message,
            "Привет! 👋\n\n"
            "Я персональный AI-ассистент с памятью на Pinecone.\n\n"
            "Я умею:\n"
            "🐱 рассказывать факты о кошках;\n"
            "🌤 показывать текущую погоду;\n"
            "🧠 помнить твои сообщения;\n"
            "🐶 анализировать изображения собак;\n"
            "📄 искать ответы в твоих PDF и DOCX.\n\n"
            "Попробуй спросить меня о погоде, "
            "попросить рассказать факт о кошках, "
            "отправить фотографию собаки "
            "или загрузить документ.\n\n"
            "Команды:\n"
            "/start — запустить бота\n"
            "/help — показать помощь\n"
            "/clear — очистить твою память",
        )

    @bot.message_handler(commands=["help"])
    def help_command(message):
        bot.reply_to(
            message,
            "🤖 Что я умею:\n\n"
            "🌤 Погода\n"
            "Например: «Какая сейчас погода в Москве?»\n\n"
            "🐱 Факты о кошках\n"
            "Например: «Расскажи интересный факт о кошках»\n\n"
            "🧠 Память\n"
            "Я помню твои сообщения и свои ответы. "
            "Память разделена по пользователям.\n\n"
            "🐶 Изображения собак\n"
            "Отправь мне фотографию собаки, "
            "и я попробую определить её породу, "
            "окрас и внешний вид.\n\n"
            "📄 Документы\n"
            "Отправь PDF или DOCX. "
            "Я разберу файл и дальше буду отвечать "
            "по его содержимому.\n\n"
            "/clear — очистить твою память "
            "и загруженные документы",
        )

    @bot.message_handler(commands=["clear"])
    def clear_command(message):
        try:
            from app.agent import clear_memory

            clear_memory(
                user_id=message.from_user.id,
                chat_id=message.chat.id,
            )
            clear_user_documents(message.from_user.id)

            bot.reply_to(
                message,
                "🧹 Твоя память и загруженные документы очищены.",
            )

        except Exception as error:
            print(f"Ошибка очистки памяти: {error}")

            bot.reply_to(
                message,
                "Не удалось очистить память.",
            )

    @bot.message_handler(content_types=["photo"])
    def handle_photo_message(message):
        image_path = None

        try:
            from app.agent import ask_agent

            bot.send_chat_action(
                message.chat.id,
                "typing",
            )

            photo = message.photo[-1]
            file_info = bot.get_file(photo.file_id)
            downloaded_file = bot.download_file(
                file_info.file_path
            )

            image_path = os.path.join(
                temp_directory(),
                f"telegram_dog_{message.message_id}.jpg",
            )

            with open(image_path, "wb") as file:
                file.write(downloaded_file)

            caption = message.caption or ""

            response = ask_agent(
                caption,
                user_id=message.from_user.id,
                chat_id=message.chat.id,
                image_path=image_path,
            )

            bot.reply_to(message, response)

        except Exception as error:
            print(f"Ошибка анализа изображения: {error}")

            bot.reply_to(
                message,
                "Не удалось проанализировать изображение. "
                "Попробуй отправить другую фотографию.",
            )

        finally:
            if image_path and os.path.exists(image_path):
                os.remove(image_path)

    @bot.message_handler(content_types=["document"])
    def handle_document_message(message):
        document_path = None

        try:
            from pipelines.ingestion import ingest_document

            incoming = message.document
            filename = Path(
                incoming.file_name or "document"
            ).name
            source_type = source_type_for(
                filename,
                incoming.mime_type,
            )

            if source_type is None:
                bot.reply_to(
                    message,
                    "❌ Не удалось обработать документ: "
                    "поддерживаются только PDF и DOCX.",
                )
                return

            bot.reply_to(
                message,
                "📄 Получил документ. Начинаю обработку...",
            )
            bot.send_chat_action(
                message.chat.id,
                "typing",
            )

            file_info = bot.get_file(incoming.file_id)
            downloaded_file = bot.download_file(
                file_info.file_path
            )
            document_path = os.path.join(
                temp_directory(),
                f"doc_{message.from_user.id}_"
                f"{message.message_id}.{source_type}",
            )

            with open(document_path, "wb") as file:
                file.write(downloaded_file)

            summary = ingest_document(
                document_path,
                user_id=message.from_user.id,
                filename=filename,
                source_type=source_type,
            )

            bot.reply_to(message, "✅ Документ готов.")
            bot.reply_to(message, summary)

        except Exception as error:
            print(f"Ошибка обработки документа: {error}")

            bot.reply_to(
                message,
                _public_document_error(error),
            )

        finally:
            if document_path and os.path.exists(document_path):
                os.remove(document_path)

    @bot.message_handler(
        content_types=["text"],
        func=lambda message: True,
    )
    def handle_text_message(message):
        user_text = message.text

        if not user_text or not user_text.strip():
            return

        try:
            from app.agent import ask_agent
            from pipelines.rag import answer_from_documents

            bot.send_chat_action(
                message.chat.id,
                "typing",
            )

            document_answer = answer_from_documents(
                user_text,
                user_id=message.from_user.id,
            )

            if document_answer is None:
                response = ask_agent(
                    user_text,
                    user_id=message.from_user.id,
                    chat_id=message.chat.id,
                )
            else:
                response = document_answer

            bot.reply_to(message, response)

        except Exception as error:
            print(f"Ошибка обработки сообщения: {error}")

            bot.reply_to(
                message,
                "Произошла ошибка при обработке сообщения. "
                "Попробуй ещё раз.",
            )

    return bot


def main() -> None:
    load_dotenv()

    # Собираем агента до polling, чтобы ошибка .env была видна сразу.
    import app.agent  # noqa: F401

    bot = create_bot()

    print("Telegram-бот запущен...")
    bot.infinity_polling()
