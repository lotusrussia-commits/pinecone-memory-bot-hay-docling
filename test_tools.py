import sys

from app.tools import CatFactTool, DogImageAnalyzerTool


def test_cat_fact():
    print("=== Проверка CatFactTool ===")

    result = CatFactTool().run()

    print(result["result"])

    if not result["result"]:
        raise SystemExit("CatFactTool вернул пустой результат.")

    print("CatFactTool: OK\n")


def test_dog_image(image_path: str):
    print("=== Проверка DogImageAnalyzerTool ===")

    result = DogImageAnalyzerTool().run(image_path)

    print(result["result"])

    if not result["result"]:
        raise SystemExit(
            "DogImageAnalyzerTool вернул пустой результат."
        )

    print("DogImageAnalyzerTool: OK\n")


if __name__ == "__main__":
    test_cat_fact()

    if len(sys.argv) > 1:
        test_dog_image(sys.argv[1])
    else:
        print(
            "DogImageAnalyzerTool: пропущен — "
            "не указан путь к фотографии."
        )