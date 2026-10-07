# Pinecone Memory Bot + Docling

Telegram-бот с персональным ассистентом на Haystack и поиском по своим PDF и DOCX. Текстовые сообщения сначала ищут ответ в загруженных документах этого пользователя. Если релевантных фрагментов нет, запрос обрабатывает прежний агент: погода, факты о кошках, память диалога и разбор фотографий. Embeddings считаются моделью `text-embedding-3-small` через OpenAI-compatible API (`OPENAI_BASE_URL`), без локальных `sentence-transformers`.

На macOS 12 Intel для используемой версии Docling нет совместимого колеса `docling-parse`, поэтому Docling запускается внутри Linux-контейнера.

## Возможности

* ответы в Telegram через Haystack Agent и модель `gpt-5.4-nano`;
* текущая погода по названию города через Open-Meteo;
* случайный факт о кошках через Cat Facts API;
* разбор фотографии: есть ли на ней собака, окрас и предполагаемая порода;
* долговременная память сообщений пользователя в Pinecone, namespace `haystack-memory`;
* короткий контекст чата для уточнений вроде «а завтра?»;
* загрузка PDF и DOCX, разбор Docling, чанки и embeddings в отдельном namespace;
* ответ по найденным фрагментам документов этого `user_id`;
* команды `/start`, `/help` и `/clear`.

## Как это работает

```text
Telegram
   |
   +-- document --> Docling --> chunks --> embeddings --> Pinecone
   |
   +-- question --> embedder --> retriever --> context --> LLM
                                      |
                                      +--> answer
```

Загрузка документа:

```text
PDF или DOCX
    |
    +-- Docling (NativePdfPipeline, текст из PDF)
    +-- страницы и чанки
    +-- OpenAITextEmbedder, text-embedding-3-small
    +-- Pinecone, namespace haystack-documents
    +-- одно предложение с кратким содержанием
```

Вопрос пользователя:

```text
текст
  |
  +-- embedding вопроса
  +-- поиск чанков только этого user_id
  |
  +-- есть релевантные фрагменты --> PromptBuilder --> OpenAIChatGenerator
  |
  +-- фрагментов нет --> прежний Haystack Agent
                            ├── weather_tool
                            ├── cat_fact_tool
                            └── dog_image_analyzer
```

Фотография по-прежнему скачивается во временный файл, путь передаётся агенту, файл удаляется после ответа. Память диалога и чанки документов лежат в одном индексе Pinecone, но в разных namespace и не смешиваются.

## Стек

| Компонент | Версия | Роль |
| --- | --- | --- |
| Python | 3.12 | язык, образ `python:3.12-slim` |
| haystack-ai | 3.3.0 | агент, пайплайны, embeddings, чат |
| pinecone-haystack | 6.4.1 | хранилище и поиск |
| pinecone | 10.0.0 | клиент векторной базы |
| docling | 2.134.0 | разбор PDF и DOCX |
| pyTelegramBotAPI | 4.37.0 | Telegram |
| requests | 2.34.2 | погода и факты о кошках |
| python-dotenv | 1.2.4 | переменные окружения |

Модель ответов и разбора изображений — `gpt-5.4-nano`. Embeddings — `text-embedding-3-small`, размерность 1536, метрика cosine. Запросы к OpenAI идут через `OPENAI_BASE_URL` (ProxyAPI).

PDF обрабатывается через `NativePdfPipeline`; для текущей версии используется извлечение текста из PDF без OCR. DOCX читает штатный парсер Docling.

## Структура проекта

```text
pinecone-memory-bot-hay-docling/
│
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── main.py
├── .env.example
├── test_documents.py
├── test_agent.py
├── test_memory.py
├── test_tools.py
│
├── app/
│   ├── agent.py
│   ├── memory.py
│   ├── tools.py
│   └── telegram_bot.py
│
├── bot/
│   └── telegram_bot.py
│
├── components/
│   ├── document_processor.py
│   └── document_store.py
│
└── pipelines/
    ├── ingestion.py
    └── rag.py
```

### `main.py`

Точка входа контейнера: `python main.py`. Запускает Telegram polling. Сам по себе импорт polling не стартует.

### `bot/telegram_bot.py`

Обработчики Telegram.

* `/start` и `/help` — прежние возможности плюс загрузка документов;
* `/clear` — память диалога этого пользователя и его чанки документов;
* текст — сначала поиск по документам, иначе прежний агент;
* фото — прежний разбор изображения;
* документ — PDF или DOCX.

### `components/document_processor.py`

Docling `DocumentConverter`. Для PDF используется `NativePdfFormatOption`. Текст страницы берётся через `export_to_markdown(page_no=...)`. Длинная страница режется на чанки. В metadata каждого чанка есть `user_id`, `filename`, `page_number`, `chunk_number` и `source_type`.

### `components/document_store.py`

`PineconeDocumentStore` для существующего индекса `PINECONE_INDEX_NAME`. Namespace по умолчанию `haystack-documents`. Размерность 1536, метрика cosine. Индекс не создаётся заново при каждой загрузке: подключение ленивое, а если индекса нет, его создаст клиент Pinecone один раз со указанными dimension и metric.

### `pipelines/ingestion.py`

Пайплайн `OpenAIChunkEmbedder` → `DocumentWriter`. Внутри эмбеддер — `OpenAITextEmbedder`. После записи бот просит модель одно предложение на русском.

### `pipelines/rag.py`

Пайплайн:

* `OpenAITextEmbedder`
* `PineconeEmbeddingRetriever`
* `PromptBuilder`
* `OpenAIChatGenerator`

Поиск фильтруется по `user_id`. Если score ниже `SIMILARITY_THRESHOLD`, фрагмент не считается релевантным и вопрос уходит агенту. Если фрагменты есть, ответ строится только по ним. Когда ответа в тексте нет, модель пишет, что в загруженных документах его не найдено.

### `app/`

Прежний слой: `PineconeMemory`, агент, погода, факты о кошках и разбор фотографий. `app/telegram_bot.py` остаётся совместимой командой `python -m app.telegram_bot` и вызывает тот же `main()`.

## Переменные окружения

```bash
cp .env.example .env
```

```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=pinecone-memory-bot

OPENAI_BASE_URL=https://api.proxyapi.ru/openai/v1
OPENAI_API_KEY=your_proxyapi_key
EMBEDDING_MODEL=text-embedding-3-small

TELEGRAM_BOT_TOKEN=your_telegram_bot_token

SIMILARITY_THRESHOLD=0.5
DOCUMENT_NAMESPACE=haystack-documents
```

| Переменная | Обязательна | Назначение |
| --- | --- | --- |
| `PINECONE_API_KEY` | да | ключ Pinecone |
| `PINECONE_INDEX_NAME` | да | существующий индекс, 1536, cosine |
| `OPENAI_API_KEY` | да | ключ чата и embeddings |
| `OPENAI_BASE_URL` | да | базовый URL ProxyAPI |
| `EMBEDDING_MODEL` | нет | по умолчанию `text-embedding-3-small` |
| `TELEGRAM_BOT_TOKEN` | да для бота | токен BotFather |
| `SIMILARITY_THRESHOLD` | нет | минимальный score, по умолчанию `0.5` |
| `DOCUMENT_NAMESPACE` | нет | namespace документов, по умолчанию `haystack-documents` |

Секреты остаются в `.env` и в контейнер попадают через `env_file`, а не через образ.

## Запуск в Docker

Сборка образа для Intel Mac:

```bash
docker build --platform linux/amd64 -t pinecone-memory-bot-hay-docling .
```

Проверка импортов и пайплайнов без Telegram polling:

```bash
docker run --rm --platform linux/amd64 \
  pinecone-memory-bot-hay-docling \
  python -m unittest test_documents.py
```

Запуск бота:

```bash
docker compose up --build
```

Эквивалент без compose:

```bash
docker run --rm --platform linux/amd64 \
  --env-file .env \
  -v bot_tmp:/tmp \
  pinecone-memory-bot-hay-docling
```

В Telegram:

1. Отправьте PDF или DOCX.
2. Бот пишет «📄 Получил документ. Начинаю обработку...».
3. После разбора: «✅ Документ готов.» и одно предложение с содержанием.
4. Задайте вопрос по файлу. Ответ опирается на найденные чанки.
5. «Какая сейчас погода в Москве?» и «Расскажи факт о кошках» по-прежнему вызывают инструменты, если в документах нет близкого фрагмента.
6. Фотография анализируется как раньше.
7. `/clear` удаляет память диалога и документы этого пользователя.

Неподдерживаемый файл отклоняется сообщением об ошибке, без traceback.

## Локальные проверки без Docling

Агент, память и инструменты по-прежнему можно запускать из `.venv` на Mac, если Docling туда не устанавливать:

```bash
python test_agent.py
python test_memory.py
python test_tools.py
```

Разбор PDF и DOCX проверяется только внутри контейнера, командой `unittest` выше.

## Безопасность

`.gitignore` исключает окружение, секреты, кэш и временные файлы. `.dockerignore` не копирует `.env` и `.venv` в образ. `/clear` удаляет документы текущего пользователя, а не весь индекс.
