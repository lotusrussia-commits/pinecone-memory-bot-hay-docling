import os
import re
from datetime import datetime, timezone

import telebot
from dotenv import load_dotenv

from team.memory import team_memory


load_dotenv()


def create_bot():
    token = os.getenv("TELEGRAM_BOT_TOKEN")

    if not token:
        raise ValueError(
            "TELEGRAM_BOT_TOKEN not found in .env"
        )

    bot = telebot.TeleBot(token)

    bot_info = bot.get_me()
    bot_id = bot_info.id
    bot_username = bot_info.username or ""

    print(
        "Telegram bot:",
        "@" + bot_username,
        "id=" + str(bot_id),
    )

    def is_team_chat(message):
        return message.chat.type in {
            "group",
            "supergroup",
        }

    def display_name(user):
        parts = []

        if user.first_name:
            parts.append(user.first_name)

        if user.last_name:
            parts.append(user.last_name)

        if parts:
            return " ".join(parts)

        return str(user.id)

    def created_at(message):
        return datetime.fromtimestamp(
            message.date,
            tz=timezone.utc,
        ).isoformat()

    def mentions_bot(message):
        text = (message.text or "").strip().lower()

        if bot_username:
            mention = "@" + bot_username.lower()

            if mention in text:
                return True

        reply = message.reply_to_message

        if reply is None:
            return False

        reply_user = reply.from_user

        if reply_user is None:
            return False

        return reply_user.id == bot_id

    def clean_question(text):
        question = (text or "").strip()

        if bot_username:
            pattern = (
                r"@" + re.escape(bot_username) + r"\b"
            )

            question = re.sub(
                pattern,
                "",
                question,
                flags=re.IGNORECASE,
            ).strip()

        return question

    @bot.message_handler(
        commands=["start"],
    )
    def start_command(message):
        bot.reply_to(
            message,
            "Team AI assistant is running.",
        )

    @bot.message_handler(
        commands=["help"],
    )
    def help_command(message):
        bot.reply_to(
            message,
            "Commands:\n"
            "/start_listening - start collecting chat context\n"
            "/stop_listening - stop and summarize discussion\n\n"
            "Mention the bot to ask a question about the chat context.",
        )

    @bot.message_handler(
        commands=["start_listening"],
    )
    def start_listening_command(message):
        if not is_team_chat(message):
            bot.reply_to(
                message,
                "This command is available only in a group chat.",
            )
            return

        team_memory.start_listening(
            message.chat.id,
        )

        bot.reply_to(
            message,
            "Listening mode is ON. "
            "Messages from this chat will be saved to Pinecone.",
        )

    @bot.message_handler(
        commands=["stop_listening"],
    )
    def stop_listening_command(message):
        if not is_team_chat(message):
            bot.reply_to(
                message,
                "This command is available only in a group chat.",
            )
            return

        bot.send_chat_action(
            message.chat.id,
            "typing",
        )

        bot.reply_to(
            message,
            "Listening mode is OFF. "
            "Analyzing the discussion...",
        )

        try:
            summary = team_memory.stop_listening(
                message.chat.id,
            )

            bot.send_message(
                message.chat.id,
                "Discussion summary:\n\n" + summary,
            )

        except Exception as error:
            print(
                "Summarization error:",
                repr(error),
            )

            bot.send_message(
                message.chat.id,
                "Could not create the discussion summary.",
            )

    @bot.message_handler(
        content_types=["text"],
    )
    def text_message(message):
        text = (message.text or "").strip()

        if not text:
            return

        if not is_team_chat(message):
            return

        if mentions_bot(message):
            question = clean_question(text)

            if not question:
                question = (
                    "Analyze the context of this team chat "
                    "and tell me what you think."
                )

            bot.send_chat_action(
                message.chat.id,
                "typing",
            )

            try:
                answer = team_memory.ask(
                    question,
                    message.chat.id,
                )

                bot.reply_to(
                    message,
                    answer,
                )

            except Exception as error:
                print(
                    "Query error:",
                    repr(error),
                )

                bot.reply_to(
                    message,
                    "Could not answer using the chat context.",
                )

            return

        if not team_memory.is_listening(
            message.chat.id,
        ):
            return

        try:
            saved = team_memory.add_message(
                text=text,
                user_id=message.from_user.id,
                display_name=display_name(
                    message.from_user,
                ),
                username=message.from_user.username,
                chat_id=message.chat.id,
                message_id=message.message_id,
                created_at=created_at(message),
            )

            if saved:
                print(
                    "Saved team message:",
                    message.chat.id,
                    message.from_user.id,
                )

        except Exception as error:
            print(
                "Indexing error:",
                repr(error),
            )

    return bot


def main():
    bot = create_bot()

    print("Team Telegram bot started.")
    print(
        "Use /start_listening in a group "
        "to start collecting messages."
    )

    bot.infinity_polling(
        skip_pending=True,
    )


if __name__ == "__main__":
    main()
