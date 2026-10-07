import time
import uuid

from app.memory import PineconeMemory


PHRASE = (
    "Мою собаку зовут Барсик и она живёт в Казани"
)
QUERY = "Как зовут мою собаку, которая живёт в Казани"


def contains_phrase(documents) -> bool:
    return any(
        document.content == PHRASE
        for document in documents
    )


def search_until(memory, query, user_id, attempts=5):
    found = []

    for attempt in range(attempts):
        found = memory.search(query, user_id=user_id)

        if contains_phrase(found):
            return found

        if attempt < attempts - 1:
            time.sleep(1)

    return found


def main():
    memory = PineconeMemory()
    user_a = f"memory-check-a-{uuid.uuid4()}"
    user_b = f"memory-check-b-{uuid.uuid4()}"

    print(f"Порог сходства: {memory.similarity_threshold}")
    print(f"Пользователь A: {user_a}")
    print(f"Пользователь B: {user_b}")
    print()

    try:
        memory.save_message(
            PHRASE,
            user_id=user_a,
            role="user_message",
        )

        found_a = search_until(memory, QUERY, user_a)
        found_b = memory.search(QUERY, user_id=user_b)

        print("Поиск пользователя A:")
        for document in found_a:
            print(
                f"- score={document.score:.3f} "
                f"user_id={document.meta.get('user_id')} "
                f"{document.content}"
            )

        print("Поиск пользователя B:")
        if found_b:
            for document in found_b:
                print(
                    f"- score={document.score:.3f} "
                    f"user_id={document.meta.get('user_id')} "
                    f"{document.content}"
                )
        else:
            print("- пусто")

        if not contains_phrase(found_a):
            raise SystemExit(
                "Память пользователя A не нашла сохранённую фразу."
            )

        if contains_phrase(found_b):
            raise SystemExit(
                "Фраза пользователя A попала в память пользователя B."
            )

        deleted = memory.clear(user_a)
        print()
        print(f"Удалено документов пользователя A: {deleted}")

        after_clear = []

        for attempt in range(5):
            after_clear = memory.search(
                PHRASE,
                user_id=user_a,
            )

            if not contains_phrase(after_clear):
                break

            if attempt < 4:
                time.sleep(1)

        if contains_phrase(after_clear):
            raise SystemExit(
                "После очистки фраза пользователя A всё ещё находится."
            )

        print()
        print("Проверка памяти прошла.")

    finally:
        memory.clear(user_a)
        memory.clear(user_b)


if __name__ == "__main__":
    main()
