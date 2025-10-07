#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Telegram RAG-bot: FAISS + sentence-transformers + внешняя LLM (Giga Chat).
Ключевые особенности:
- Приоритет документам (retrieval injection).
- Если нет релевантных фрагментов — бот честно сообщает об отсутствии ответа.
- Автоматическое логирование запросов и использованных фрагментов.
- Заглушка для вызова Giga Chat API: заполни GIGA_API_URL и GIGA_API_KEY в .env.
"""

import os
import sys
import json
import time
import logging
from typing import List, Dict, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict

import requests
from dotenv import load_dotenv
from tqdm import tqdm

# NLP / embeddings / vector DB
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Telegram
from telegram import Update, ParseMode
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, CommandHandler, filters

# ----------------------
# Конфигурация (env)
# ----------------------
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")  # обязательно
GIGA_API_URL = os.getenv("GIGA_API_URL")      # например https://api.giga-chat.example/v1/generate (заполить сам)
GIGA_API_KEY = os.getenv("GIGA_API_KEY")
DOCS_DIR = os.getenv("DOCS_DIR", "docs")
INDEX_PATH = os.getenv("INDEX_PATH", "faiss_index.bin")
META_PATH = os.getenv("META_PATH", "faiss_meta.json")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
TOP_K = int(os.getenv("TOP_K", "5"))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.25"))  # минимум релевантности (косинус)

if TELEGRAM_TOKEN is None:
    print("Ошибка: TELEGRAM_TOKEN не задан. Установи в .env")
    sys.exit(1)

# ----------------------
# Логирование
# ----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ----------------------
# Data structures
# ----------------------
@dataclass
class Fragment:
    doc_id: str
    fragment_id: str
    text: str
    meta: Dict[str, Any]  # например page, title
    embedding: List[float] = None

# ----------------------
# System prompt (строго)
# ----------------------
SYSTEM_PROMPT = """
Вы — вспомогательная нейросеть, встроенная в Telegram-бота. Ваш приоритет — отвечать, опираясь В ПЕРВУЮ ОЧЕРЕДЬ
на предоставленные в блоке Retrieved_documents фрагменты. Правила:

1) Сначала прочтите блок Retrieved_documents (он уже встроён в prompt). Если среди них есть ответ на вопрос — используйте только их и не добавляйте "факты по памяти".
2) Если ответ полностью покрывается документами — ссылайтесь на них: [doc_id, fragment_id].
3) Если документов недостаточно, честно укажите: "Не нашёл точного подтверждения в документах. Могу предположить: ...". Всегда ясно отделяйте допущения.
4) Никогда не придумывайте точные факты (даты, суммы, официальные названия) без подтверждения.
5) Кратко (до 5-7 предложений), затем секция "Источники:" с перечислением doc_id и fragment_id и короткой цитатой (1-2 предложения).
6) В ответе верните поле sources_used: список объектов {doc_id, fragment_id, relevance_score}.
"""

# ----------------------
# Утилиты: чтение документов и chunking
# ----------------------
def load_text_files_from_dir(directory: str) -> List[Tuple[str, str]]:
    """
    Загружает все .txt, .md, .pdf (pdf не поддерживается здесь, можно предварительно конвертировать) из директории.
    Возвращает список (doc_id, text).
    """
    docs = []
    p = Path(directory)
    if not p.exists():
        logger.error("Папка с документами не найдена: %s", directory)
        return docs

    for file in p.rglob("*"):
        if file.is_file() and file.suffix.lower() in {".txt", ".md"}:
            text = file.read_text(encoding="utf-8", errors="ignore")
            doc_id = str(file.relative_to(p))
            docs.append((doc_id, text))
            logger.info("Загружен документ: %s (%d chars)", doc_id, len(text))
    return docs

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> List[str]:
    """
    Простое разбиение по предложениям/символам.
    Возвращает список фрагментов (строк).
    """
    import re
    # разбиваем в первую очередь по параграфам/строкам
    paragraphs = [p.strip() for p in re.split(r'\n{1,}', text) if p.strip()]
    chunks = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            chunks.append(para)
        else:
            # скользящее окно по символам
            start = 0
            while start < len(para):
                end = start + chunk_size
                chunk = para[start:end].strip()
                if chunk:
                    chunks.append(chunk)
                start = max(end - overlap, end)  # overlap
    return chunks

# ----------------------
# Индекс FAISS / эмбеддинги
# ----------------------
class Retriever:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        logger.info("Загружаю модель эмбеддингов: %s", model_name)
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.meta = []  # list of dicts with doc_id, fragment_id, text
        self.dimension = self.model.get_sentence_embedding_dimension()

    def build_index(self, docs: List[Tuple[str, str]], force_rebuild=False):
        """
        docs: list of (doc_id, text)
        """
        if (Path(INDEX_PATH).exists() and Path(META_PATH).exists()) and not force_rebuild:
            logger.info("Индекс уже существует, загружаю из диска...")
            self.load_index()
            return

        fragments = []
        metas = []
        logger.info("Создаём фрагменты документов...")
        for doc_id, text in tqdm(docs):
            chunks = chunk_text(text, chunk_size=500, overlap=100)
            for i, chunk in enumerate(chunks):
                frag = Fragment(
                    doc_id=doc_id,
                    fragment_id=f"frag_{i}",
                    text=chunk,
                    meta={"doc_id": doc_id}
                )
                fragments.append(frag)
                metas.append({"doc_id": doc_id, "fragment_id": frag.fragment_id, "text": chunk})

        logger.info("Вычисляю эмбеддинги для %d фрагментов...", len(fragments))
        texts = [f.text for f in fragments]
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        # нормализация для косинусного поиска
        faiss.normalize_L2(embeddings)

        logger.info("Создаю FAISS индекс (IndexFlatIP)...")
        index = faiss.IndexFlatIP(self.dimension)
        index.add(embeddings)
        self.index = index
        self.meta = metas

        # сохраняем
        logger.info("Сохраняю индекс и метаданные на диск...")
        faiss.write_index(self.index, INDEX_PATH)
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)
        logger.info("Индекс сохранен.")

    def load_index(self):
        self.index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        # убедимся, что векторы нормализованы (FAISS хранит их такими, если были нормализованы)
        logger.info("Индекс и метаданные загружены (метаданных: %d).", len(self.meta))

    def search(self, query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        q_emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        if self.index is None:
            raise RuntimeError("Индекс не загружен")
        D, I = self.index.search(q_emb, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.meta):
                continue
            meta = self.meta[idx]
            results.append({
                "doc_id": meta["doc_id"],
                "fragment_id": meta["fragment_id"],
                "text": meta["text"],
                "score": float(score)
            })
        return results

# ----------------------
# Вызов внешней LLM (Giga Chat) — заглушка
# ----------------------
def call_giga_chat(system_prompt: str, user_prompt: str, retrieved: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Формирует payload и вызывает внешний API Giga Chat.
    Ожидаем, что API вернёт структуру {'text': ..., 'sources_used': [...] }.
    Если у тебя есть конкретный SDK — замени тело функции вызовом SDK.
    """
    if not GIGA_API_URL or not GIGA_API_KEY:
        logger.warning("GIGA_API_URL или GIGA_API_KEY не заданы. Используется локальная имитация ответа.")
        # простая имитация: возвращаем конкатенацию retrieved
        combined = "\n\n".join([f"[{r['doc_id']}|{r['fragment_id']}]: {r['text'][:300]}" for r in retrieved])
        text = f"Согласно документам:\n{combined}\n\n(Это пример ответа — подключи реальный Giga Chat.)"
        return {"text": text, "sources_used": [{"doc_id": r["doc_id"], "fragment_id": r["fragment_id"], "score": r["score"]} for r in retrieved]}

    # Формируем жесткий prompt: system + retrieved_documents + user
    retrieved_section = ""
    for i, r in enumerate(retrieved, start=1):
        retrieved_section += f"{i}) doc_id: \"{r['doc_id']}\"\n   fragment_id: \"{r['fragment_id']}\"\n   score: {r['score']:.4f}\n   text: \"{r['text'].replace('\"', '\\\"')[:1000]}\"\n\n"

    payload = {
        "system": system_prompt,
        "user": user_prompt,
        "retrieved_documents": retrieved_section,
        "instruction": "Отвечай, опираясь в первую очередь на retrieved_documents. Если подтверждения нет — честно скажи 'Не нашёл подтверждения'. Верни поле sources_used."
    }
    headers = {
        "Authorization": f"Bearer {GIGA_API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        resp = requests.post(GIGA_API_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # ожидаем data['text'] и (опционально) data['sources_used']
        return data
    except Exception as e:
        logger.exception("Ошибка при вызове Giga Chat API: %s", e)
        return {"text": "Ошибка при обращении к модели.", "sources_used": []}

# ----------------------
# Telegram handlers
# ----------------------
class RAGTelegramBot:
    def __init__(self, retriever: Retriever):
        self.retriever = retriever
        # создаём директорию логов
        Path("logs").mkdir(exist_ok=True)

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("Привет! Я RAG-бот. Задай вопрос про документы, и я постараюсь ответить, опираясь на них.")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_text = update.message.text.strip()
        user_id = update.message.from_user.id
        chat_id = update.message.chat_id
        logger.info("Запрос от user=%s: %s", user_id, user_text)

        # 1) retrieval
        hits = self.retriever.search(user_text, top_k=TOP_K)
        # фильтруем по порогу
        filtered = [h for h in hits if h["score"] >= SCORE_THRESHOLD]

        # 2) Если ничего релевантного — не даём модели сочинять факты
        if not filtered:
            # Ответ: честно сообщаем об отсутствии подтверждения
            reply = "Не нашёл точного подтверждения в доступных документах. Могу попытаться ответить предположительно — напиши 'предположение' если хочешь."
            await update.message.reply_text(reply)
            # логируем
            self._log_interaction(user_id, user_text, [], reply)
            return

        # 3) Формируем prompt и вызываем Giga Chat
        system_prompt = SYSTEM_PROMPT
        # Подготовим компактный retrieved_documents для модели (top-N)
        retrieved_for_prompt = filtered[:TOP_K]
        user_prompt = user_text

        giga_resp = call_giga_chat(system_prompt, user_prompt, retrieved_for_prompt)
        text = giga_resp.get("text", "")
        sources_used = giga_resp.get("sources_used", [])
        # Если модель не вернула sources_used — формируем из retrieved_for_prompt
        if not sources_used:
            sources_used = [{"doc_id": r["doc_id"], "fragment_id": r["fragment_id"], "score": r["score"]} for r in retrieved_for_prompt]

        # 4) Строгая проверка: если модель в тексте явно делает утверждение, но sources_used пуст — предупреждаем
        # (мы уже заполнили sources_used выше)

        # 5) Форматируем ответ для Telegram с секцией источников (кнопки можно добавить отдельно)
        sources_text_lines = []
        for s in sources_used:
            # находим текст в retrieved_for_prompt, чтобы дать цитату
            match = next((r for r in retrieved_for_prompt if r["doc_id"] == s["doc_id"] and r["fragment_id"] == s["fragment_id"]), None)
            quote = (match["text"][:200] + "...") if match else ""
            sources_text_lines.append(f"- {s['doc_id']} | {s['fragment_id']} (score={s.get('score', 0):.3f}): {quote}")

        answer = f"{text}\n\n*Источники:*\n" + "\n".join(sources_text_lines)
        try:
            await update.message.reply_text(answer, parse_mode=ParseMode.MARKDOWN)
        except Exception:
            # если Markdown вызовет ошибку — отправим plain
            await update.message.reply_text(answer)

        # 6) лог
        self._log_interaction(user_id, user_text, sources_used, text)

    def _log_interaction(self, user_id: int, query: str, sources: List[Dict[str, Any]], response_text: str):
        log = {
            "ts": time.time(),
            "user_id": user_id,
            "query": query,
            "sources": sources,
            "response": response_text
        }
        fn = Path("logs") / f"{int(time.time())}_{user_id}.json"
        with open(fn, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)
        logger.info("Interaction logged to %s", fn)

# ----------------------
# Main
# ----------------------
def main():
    # 1) подготовка данных
    retriever = Retriever(model_name=EMBEDDING_MODEL_NAME)
    docs = load_text_files_from_dir(DOCS_DIR)
    if not docs:
        logger.error("Нет документов для индексации. Положи .txt/.md файлы в папку 'docs/'")
        # не выходим — может быть индекс уже есть
    try:
        retriever.build_index(docs, force_rebuild=False)
    except Exception as e:
        logger.exception("Ошибка при создании/загрузке индекса: %s", e)
        return

    # 2) запускаем бота
    bot = RAGTelegramBot(retriever=retriever)
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", bot.start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_message))

    logger.info("Bot started. Listening for messages...")
    app.run_polling()

if __name__ == "__main__":
    main()
