import os

import requests
from dotenv import load_dotenv
from haystack import component
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage, ImageContent
from haystack.utils import Secret


load_dotenv()


@component
class WeatherTool:
    """Получает текущую погоду по названию города."""

    @component.output_types(result=str)
    def run(self, city: str):
        """Возвращает текущую погоду для указанного города."""

        if not city or not city.strip():
            return {
                "result": "Не удалось получить погоду: город не указан."
            }

        city = city.strip()

        try:
            geocoding_response = requests.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={
                    "name": city,
                    "count": 1,
                    "language": "ru",
                    "format": "json",
                },
                timeout=10,
            )

            geocoding_response.raise_for_status()
            geocoding_data = geocoding_response.json()

            results = geocoding_data.get("results")

            if not results:
                return {
                    "result": f"Не удалось найти город: {city}."
                }

            location = results[0]

            latitude = location["latitude"]
            longitude = location["longitude"]
            found_city = location["name"]
            country = location.get("country", "")

            weather_response = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                    "current": (
                        "temperature_2m,"
                        "relative_humidity_2m,"
                        "apparent_temperature,"
                        "weather_code,"
                        "wind_speed_10m"
                    ),
                    "timezone": "auto",
                },
                timeout=10,
            )

            weather_response.raise_for_status()
            weather_data = weather_response.json()

            current = weather_data["current"]

            temperature = current["temperature_2m"]
            feels_like = current["apparent_temperature"]
            humidity = current["relative_humidity_2m"]
            wind_speed = current["wind_speed_10m"]
            weather_code = current["weather_code"]

            description = self._weather_description(weather_code)

            weather_text = (
                f"Погода в городе {found_city}"
                f"{', ' + country if country else ''}:\n"
                f"🌡 Температура: {temperature} °C\n"
                f"🌡 Ощущается как: {feels_like} °C\n"
                f"☁️ Состояние: {description}\n"
                f"💧 Влажность: {humidity}%\n"
                f"💨 Ветер: {wind_speed} км/ч"
            )

            return {
                "result": weather_text
            }

        except requests.RequestException as error:
            return {
                "result": (
                    "Не удалось получить данные о погоде: "
                    f"{error}"
                )
            }

        except (KeyError, TypeError, ValueError) as error:
            return {
                "result": (
                    "Ошибка обработки данных о погоде: "
                    f"{error}"
                )
            }

    @staticmethod
    def _weather_description(code: int) -> str:
        """Преобразует код Open-Meteo в понятное описание."""

        descriptions = {
            0: "ясно",
            1: "преимущественно ясно",
            2: "переменная облачность",
            3: "пасмурно",
            45: "туман",
            48: "изморозь и туман",
            51: "лёгкая морось",
            53: "морось",
            55: "сильная морось",
            61: "небольшой дождь",
            63: "дождь",
            65: "сильный дождь",
            71: "небольшой снег",
            73: "снег",
            75: "сильный снег",
            80: "небольшие ливни",
            81: "ливни",
            82: "сильные ливни",
            95: "гроза",
            96: "гроза с небольшим градом",
            99: "гроза с сильным градом",
        }

        return descriptions.get(
            code,
            "неизвестные погодные условия",
        )


@component
class CatFactTool:
    """Получает случайный интересный факт о кошках."""

    @component.output_types(result=str)
    def run(self):
        """Возвращает случайный факт о кошках."""

        try:
            response = requests.get(
                "https://catfact.ninja/fact",
                timeout=10,
            )

            response.raise_for_status()

            data = response.json()
            fact = data.get("fact")

            if not fact:
                return {
                    "result": "Не удалось получить факт о кошке."
                }

            return {
                "result": f"🐱 Факт о кошках: {fact}"
            }

        except requests.RequestException as error:
            return {
                "result": (
                    "Не удалось получить факт о кошке: "
                    f"{error}"
                )
            }

        except (KeyError, TypeError, ValueError) as error:
            return {
                "result": (
                    "Ошибка обработки факта о кошке: "
                    f"{error}"
                )
            }


@component
class DogImageAnalyzerTool:
    """Анализирует изображение с помощью мультимодальной модели."""

    def __init__(self):
        self._generator = None

    def _get_generator(self):
        """Создаёт чат-модель один раз и переиспользует её."""

        if self._generator is not None:
            return self._generator

        openai_base_url = os.getenv("OPENAI_BASE_URL")

        if not openai_base_url:
            return None

        self._generator = OpenAIChatGenerator(
            api_key=Secret.from_env_var(
                "OPENAI_API_KEY"
            ),
            api_base_url=openai_base_url,
            model="gpt-5.4-nano",
        )

        return self._generator

    @component.output_types(result=str)
    def run(self, image_path: str):
        """Анализирует локальный файл изображения."""

        if not image_path:
            return {
                "result": "Путь к изображению не указан."
            }

        if not os.path.isfile(image_path):
            return {
                "result": (
                    "Не удалось найти изображение для анализа."
                )
            }

        generator = self._get_generator()

        if generator is None:
            return {
                "result": (
                    "OPENAI_BASE_URL не найден в .env."
                )
            }

        try:
            image = ImageContent.from_file_path(
                image_path,
                detail="low",
            )

            message = ChatMessage.from_user(
                content_parts=[
                    (
                        "Проанализируй изображение. "
                        "Определи, есть ли на нём собака. "
                        "Если на изображении есть собака, "
                        "кратко опиши её внешний вид, "
                        "окрас и предполагаемую породу. "
                        "Если породу определить невозможно, "
                        "честно скажи об этом. "
                        "Отвечай на русском языке."
                    ),
                    image,
                ],
            )

            response = generator.run(
                messages=[message]
            )

            return {
                "result": response["replies"][0].text
            }

        except Exception as error:
            return {
                "result": (
                    "Не удалось проанализировать изображение: "
                    f"{error}"
                )
            }
