# Pinecone Memory Bot

Telegram-бот на [Haystack](https://haystack.deepset.ai/) и [Pinecone](https://www.pinecone.io/). В одном индексе живут три независимых хранилища: память личного диалога, чанки загруженных PDF и DOCX, сообщения командного чата. Embeddings считает `text-embedding-3-small` через OpenAI-compatible API (`OPENAI_BASE_URL`). Ответы пишет `gpt-5.4-nano`.

В репозитории две точки входа:

| Команда | Где работает | Что делает |
| --- | --- | --- |
| `python main.py` | личный чат | ассистент, документы, фото |
| `python bot.py` | группа | слушает обсуждение, отвечает по контексту, пишет итог |

Docker запускает личный бот: `CMD` в образе — `python main.py`.

## Личный ассистент

Текстовый вопрос сначала ищет ответ в PDF и DOCX этого пользователя. Если близких фрагментов нет, запрос уходит агенту.

Агент умеет:

- текущую погоду по названию города (Open-Meteo);
- случайный факт о кошках (Cat Facts API);
- разбор фотографии: есть ли собака, окрас, предполагаемая порода;
- помнить реплики пользователя и свои ответы;
- держать короткий контекст чата, чтобы понимать уточнения вроде «а завтра?».

Команды: `/start`, `/help`, `/clear`.

`/clear` удаляет память диалога и чанки документов только этого пользователя.

## Командный чат

Бот работает в группе и супергруппе. В личке команды прослушивания не принимаются, обычный текст игнорируется.

1. `/start_listening` включает запись. Каждое текстовое сообщение группы получает embedding и пишется в Pinecone.
2. Упоминание `@бот` или ответ на его сообщение — вопрос по истории этого чата. Ответ строится только по найденным сообщениям.
3. `/stop_listening` выключает запись и присылает итог обсуждения: тема, позиции, договорённости и следующие шаги.

Пока запись выключена, сообщения в индекс не попадают. Вопрос по уже сохранённой истории можно задать и после остановки.

Итог и ответы на вопросы пишутся по-русски. Если в найденном контексте ответа нет, модель так и говорит, не додумывая факты.

## Как устроены данные

Один индекс Pinecone, размерность 1536, метрика cosine. Namespace не смешиваются.

| Namespace | Переменная | Что лежит |
| --- | --- | --- |
| `haystack-memory` | фиксировано в коде | реплики личного диалога |
| `haystack-documents` | `DOCUMENT_NAMESPACE` | чанки PDF и DOCX |
| `haystack-team` | `TEAM_NAMESPACE` | сообщения группы |

Поиск по документам фильтруется по `user_id`. Поиск по командному чату — по `chat_id`.

### Личный вопрос

```text
текст
  |
  +-- embedding
  +-- чанки этого user_id, top 5
  |
  +-- score >= SIMILARITY_THRESHOLD
  |     PromptBuilder -> gpt-5.4-nano
  |
  +-- релевантных чанков нет
        Haystack Agent
          weather_tool
          cat_fact_tool
          dog_image_analyzer
```

Порог по умолчанию — `0.45`. В ответ попадают до 5 фрагментов. Если их нет, агент отвечает сам.

Загрузка файла:

```text
PDF или DOCX
  -> Docling
  -> страницы и чанки (~1500 символов)
  -> embedding
  -> Pinecone, namespace документов
  -> одно предложение с содержанием
```

PDF читается через `NativePdfPipeline` как текст, без OCR. DOCX разбирает штатный парсер Docling. У чанка в metadata есть `user_id`, `filename`, `page_number`, `chunk_number`, `source_type`.

Фото скачивается во временный файл, путь передаётся агенту, файл удаляется после ответа.

### Командный вопрос

```text
сообщение группы
  -> OpenAIDocumentEmbedder
  -> Pinecone, namespace haystack-team

вопрос (@бот или reply)
  -> embedding
  -> до 8 сообщений этого chat_id
  -> промпт с автором, @username и временем
  -> gpt-5.4-nano

/stop_listening
  -> полная лента текущей сессии
  -> отдельный промпт итога
  -> gpt-5.4-nano
```

Итог строится по сообщениям, накопленным с `/start_listening`, а не по всему индексу. Сессия живёт в памяти процесса: перезапуск бота её сбрасывает. Уже записанные в Pinecone сообщения при этом остаются, и по ним можно спрашивать.

Повторный `/start_listening` начинает новую сессию для итога. Старые векторы этого чата в индексе не стираются.

## Стек

| Компонент | Версия | Роль |
| --- | --- | --- |
| Python | 3.12 | язык, образ `python:3.12-slim` |
| haystack-ai | 3.3.0 | агент, пайплайны, embeddings, чат |
| pinecone-haystack | 6.4.1 | хранилище и поиск |
| pinecone | 10.0.0 | клиент векторной базы |
| pyTelegramBotAPI | 4.37.0 | Telegram |
| requests | 2.34.2 | погода и факты о кошках |
| python-dotenv | 1.2.4 | переменные окружения |
| docling | отдельно | разбор PDF и DOCX |

Версии из таблицы, кроме Docling и Python, закреплены в `requirements.txt`.

Разбор документов импортирует `docling`. Этого пакета в `requirements.txt` нет: личный бот, память и инструменты ставятся без него. PDF и DOCX заработают после установки Docling. На macOS 12 Intel для используемого Docling нет совместимого колеса `docling-parse`, поэтому разбор файлов рассчитан на Linux-контейнер. В образе для этого стоят `libgl1`, `libglib2.0-0` и `libgomp1`.

Запросы к модели идут на `OPENAI_BASE_URL`. В примере это ProxyAPI.

## Структура

```text
.
├── main.py                 # личный бот: python main.py
├── bot.py                  # командный бот: python bot.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
│
├── app/                    # личный ассистент
│   ├── agent.py            # Haystack Agent, история, /clear
│   ├── memory.py           # namespace haystack-memory
│   ├── tools.py            # погода, кошки, фото
│   └── telegram_bot.py     # совместимый запуск того же main()
│
├── bot/
│   └── telegram_bot.py     # личный Telegram: текст, фото, документы
│
├── components/
│   ├── document_processor.py
│   └── document_store.py
│
├── pipelines/
│   ├── ingestion.py        # чанки -> embeddings -> Pinecone
│   └── rag.py              # вопрос -> чанки -> ответ
│
├── team/
│   ├── memory.py           # сессия прослушивания
│   └── pipelines.py        # индекс, поиск, итог
│
├── test_agent.py
├── test_memory.py
├── test_tools.py
└── test_documents.py
```

`python -m app.telegram_bot` и `python -m bot.telegram_bot` вызывают тот же личный бот, что и `main.py`.

## Переменные окружения

```bash
cp .env.example .env
```

| Переменная | Нужна | Назначение |
| --- | --- | --- |
| `PINECONE_API_KEY` | да | ключ Pinecone |
| `PINECONE_INDEX_NAME` | да | индекс 1536 / cosine. Если индекса ещё нет, клиент создаст его при первом обращении |
| `OPENAI_API_KEY` | да | чат и embeddings |
| `OPENAI_BASE_URL` | да | базовый URL, например `https://api.proxyapi.ru/openai/v1` |
| `TELEGRAM_BOT_TOKEN` | да для бота | токен BotFather |
| `EMBEDDING_MODEL` | нет | по умолчанию `text-embedding-3-small` |
| `OPENAI_MODEL` | нет | модель командного бота, по умолчанию `gpt-5.4-nano` |
| `SIMILARITY_THRESHOLD` | нет | порог чанков документов, по умолчанию `0.45` |
| `DOCUMENT_NAMESPACE` | нет | по умолчанию `haystack-documents` |
| `TEAM_NAMESPACE` | нет | по умолчанию `haystack-team` |

Модель личного агента в `app/agent.py` задана как `gpt-5.4-nano` и переменную `OPENAI_MODEL` не читает.

Секреты лежат в `.env`. В образ они не копируются: compose передаёт их через `env_file`.

## Запуск

Нужны Python 3.12, заполненный `.env` и индекс Pinecone.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Личный бот:

```bash
python main.py
```

Командный бот — отдельный процесс и тот же токен. Два polling-клиента с одним токеном одновременно работать не будут: для группы запускайте `bot.py`, для личного чата — `main.py`.

```bash
python bot.py
```

Группа: добавьте бота, отключите privacy mode в BotFather (`/setprivacy` → Disable), иначе Telegram не отдаёт обычные сообщения группы и запись не увидит чужой текст. Упоминания и ответы на сообщения бота privacy mode не скрывает.

### Docker

Образ собирается под `linux/amd64` — так Docling ставится на Intel Mac.

```bash
docker compose up --build
```

Без compose:

```bash
docker build --platform linux/amd64 -t pinecone-memory-bot .
docker run --rm --platform linux/amd64 \
  --env-file .env \
  -v bot_tmp:/tmp \
  pinecone-memory-bot
```

Контейнер поднимает личный бот. Командный запускается локально: `python bot.py`.

Чтобы внутри образа заработали PDF и DOCX, в образ нужно добавить пакет `docling` поверх `requirements.txt`.

## Проверка без Telegram

Память, агент и инструменты ходят в Pinecone и API из `.env`:

```bash
python test_memory.py
python test_agent.py
python test_tools.py
```

`test_agent.py` спрашивает погоду в Москве и следом «А завтра?». `test_tools.py` без аргумента берёт факт о кошках; путь к картинке проверяет разбор фото:

```bash
python test_tools.py /path/to/dog.jpg
```

Разбор PDF и DOCX — в контейнере, где есть Docling:

```bash
docker run --rm --platform linux/amd64 \
  pinecone-memory-bot \
  python -m unittest test_documents.py
```

Тест проверяет импорты, связи пайплайнов и metadata чанков. Запись в Pinecone из него не обязательна.

## Что бот отвечает в Telegram

Личный чат:

1. PDF или DOCX — «📄 Получил документ. Начинаю обработку...», затем «✅ Документ готов.» и одно предложение.
2. Другой тип файла — сообщение об ошибке, без traceback.
3. Вопрос по файлу опирается на чанки. Если их нет, отвечают погода, факт о кошках или обычный диалог.
4. Фото уходит в разбор изображения.
5. `/clear` чистит память диалога и документы этого пользователя. Командный namespace не трогает.

Группа:

1. `/start_listening` — запись включена.
2. Сообщения копятся до `/stop_listening`.
3. `/stop_listening` — запись выключена и приходит итог. Пустая сессия об этом сообщает отдельно.
4. `@бот вопрос` или ответ на сообщение бота — ответ по истории этого `chat_id`.
