from app.agent import ask_agent


def main():
    question = "Какая сейчас погода в Москве?"

    print(f"Вопрос: {question}")
    print()

    answer = ask_agent(
        question,
        user_id="test-agent",
        chat_id="test-agent",
    )

    print("Ответ агента:")
    print(answer)
    print()

    follow_up = "А завтра?"

    print(f"Уточнение: {follow_up}")
    print()

    follow_up_answer = ask_agent(
        follow_up,
        user_id="test-agent",
        chat_id="test-agent",
    )

    print("Ответ на уточнение:")
    print(follow_up_answer)


if __name__ == "__main__":
    main()