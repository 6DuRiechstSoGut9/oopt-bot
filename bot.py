# bot.py
# Telegram-бот (python-telegram-bot v20+) для поиска по документам через эмбеддинги
# Под Hugging Face Spaces: использует polling (application.run_polling())
# Поддерживает .txt и .docx, экономит память через numpy.memmap (embeddings.dat)

import os
import logging
import glob
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# docx reader
from docx import Document

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    filters,
)

# ---------- CONFIG ----------
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    # лучше коротко логировать — при запуске в Spaces переменную задай в UI
    raise RuntimeError("TELEGRAM_BOT_TOKEN не установлен в окружении")

MODEL_NAME = os.environ.get("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
DOCS_DIR = Path(os.environ.get("DOCS_DIR", "documents"))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 200))  # символов на кусок (меньше -> больше чанков)
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 8))    # батч для encode
EMBED_DEVICE = os.environ.get("EMBED_DEVICE", "cpu")  # "cpu" (Spaces free обычно CPU)
TOP_K = int(os.environ.get("TOP_K", 3))

# Файлы индекса на диске
EMB_PATH = Path("embeddings.dat")
TEXTS_PATH = Path("texts.json")
META_PATH = Path("index_meta.json")

# ----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------- Utils for reading files ----------
def read_txt(path: Path) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        with open(path, "r", encoding="latin-1") as f:
            return f.read()


def read_docx(path: Path) -> str:
    doc = Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n".join(paragraphs)


def load_documents_texts(directory: Path, chunk_size: int = 2000) -> List[str]:
    """
    Проходит по папке и собирает список текстовых чанков.
    Поддерживает .txt и .docx.
    """
    directory.mkdir(parents=True, exist_ok=True)
    texts: List[str] = []

    # Список файлов с расширением в порядке сортировки
    files = sorted(directory.glob("*"))
    for path in files:
        if path.suffix.lower() == ".txt":
            content = read_txt(path).strip()
        elif path.suffix.lower() == ".docx":
            content = read_docx(path).strip()
        else:
            # пропускаем неподдерживаемые расширения
            continue

        if not content:
            continue

        # Простая нарезка по символам (можно улучшить на sentences)
        start = 0
        n = len(content)
        while start < n:
            end = min(n, start + chunk_size)
            chunk = content[start:end].strip()
            if chunk and len(chunk) >= 30:  # пропускаем очень короткие куски
                texts.append(chunk)
            start = end

    if not texts:
        texts.append("Пример содержимого. Добавьте файлы .txt или .docx в папку documents для индексации.")
    logger.info(f"Найдено {len(texts)} чанков в папке '{directory}'")
    return texts


# ---------- Embedding index (memmap-backed) ----------
class SimpleEmbeddingIndex:
    def __init__(self, model_name: str, device: str = "cpu"):
        logger.info("Загружаем tokenizer и model (use_fast=False)...")
        # use_fast=False для совместимости с окружениями без tokenizers
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.device = device

        self.texts: List[str] = []
        self.emb_shape: Optional[tuple] = None  # (n, dim)
        self.emb_path = EMB_PATH

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode_batch(self, texts_batch: List[str]) -> np.ndarray:
        """Возвращает L2-normalized numpy array shape (batch, dim)"""
        self.model.eval()
        with torch.no_grad():
            enc = self.tokenizer(
                texts_batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            pooled = self.mean_pooling(out, attention_mask)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            arr = pooled.cpu().numpy().astype(np.float32)
            # явное удаление тензоров
            del enc, input_ids, attention_mask, out, pooled
            return arr

    def build_from_texts_memmap(self, texts: List[str], batch_size: int = 8):
        """
        Построение индекса с записью в numpy.memmap (embeddings.dat).
        texts -> список строк (чанков)
        """
        n = len(texts)
        dim = self.model.config.hidden_size
        logger.info(f"Запускаем построение индекса: {n} текстов, dim={dim}")

        # Создаём memmap
        logger.info(f"Создаём memmap файл {self.emb_path} размера ({n}, {dim})")
        mm = np.memmap(self.emb_path, dtype="float32", mode="w+", shape=(n, dim))

        # Итерация по батчам и запись в memmap
        for i in range(0, n, batch_size):
            batch = texts[i : i + batch_size]
            arr = self.encode_batch(batch)  # (b, dim)
            mm[i : i + arr.shape[0], :] = arr
            mm.flush()
            logger.info(f"Записано батч {i}..{i+arr.shape[0]}")

        # освобождаем memmap
        del mm
        # Сохраняем тексты в JSON и мета
        with open(TEXTS_PATH, "w", encoding="utf-8") as f:
            json.dump(texts, f, ensure_ascii=False)
        meta = {"n": n, "dim": dim, "model_name": MODEL_NAME, "chunk_size": CHUNK_SIZE}
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        logger.info("Построение индекса завершено и сохранено на диск.")

    def load_index(self):
        """
        Загружает индекс (texts.json + embeddings.dat) в объект:
        texts -> list, embeddings -> numpy.memmap (r)
        """
        if not TEXTS_PATH.exists() or not self.emb_path.exists() or not META_PATH.exists():
            raise FileNotFoundError("Индекс не найден на диске.")
        with open(TEXTS_PATH, "r", encoding="utf-8") as f:
            self.texts = json.load(f)
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        n = meta["n"]
        dim = meta["dim"]
        # открываем memmap в режиме чтения
        emb = np.memmap(self.emb_path, dtype="float32", mode="r", shape=(n, dim))
        self.emb_shape = (n, dim)
        logger.info(f"Индекс загружен: {n} embeddings, dim={dim}")
        return emb

    def search(self, query: str, top_k: int = 3, emb_memmap: Optional[np.memmap] = None):
        """
        Поиск: q -> encode -> dot(embs, q)
        emb_memmap можно передать, чтобы не загружать внутри
        """
        if emb_memmap is None:
            if not self.emb_path.exists():
                return []
            # Загружаем memmap
            with open(META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
            n, dim = meta["n"], meta["dim"]
            emb_memmap = np.memmap(self.emb_path, dtype="float32", mode="r", shape=(n, dim))
        # вычисляем эмбед для запроса
        q_emb = self.encode_batch([query])[0]  # shape (dim,)
        # cosine similarity since stored vectors are normalized
        sims = emb_memmap @ q_emb  # shape (n,)
        if sims.size == 0:
            return []
        # топ-k через argpartition (быстрее, чем argsort для больших n)
        if top_k >= sims.size:
            idx = np.argsort(-sims)
        else:
            part = np.argpartition(-sims, top_k - 1)[:top_k]
            idx = part[np.argsort(-sims[part])]
        results = [(int(i), float(sims[i])) for i in idx[:top_k]]
        return results


# ---------- Bot logic ----------
class BotApp:
    def __init__(self):
        self.index = SimpleEmbeddingIndex(MODEL_NAME, device=EMBED_DEVICE)
        self.emb_memmap: Optional[np.memmap] = None
        # Попробуем загрузить индекс с диска; если нет — построим
        try:
            self.emb_memmap = self.index.load_index()
            # texts загружены внутри load_index
        except Exception as e:
            logger.info(f"Индекс не найден или не загружен: {e}. Будем строить заново.")
            # Загружаем тексты из документов
            texts = load_documents_texts(DOCS_DIR, chunk_size=CHUNK_SIZE)
            # Построение индекса (будет записан в файлы)
            self.index.build_from_texts_memmap(texts, batch_size=BATCH_SIZE)
            # После построения — загрузим memmap и texts
            self.emb_memmap = self.index.load_index()

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "Бот запущен. Отправь вопрос — я найду похожие фрагменты.\nКоманда /reload — переиндексировать документы."
        )

    async def reload_docs(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("Начинаю переиндексацию... Это может занять время.")
        # перестроим индекс (в главном потоке — может занять несколько минут)
        texts = load_documents_texts(DOCS_DIR, chunk_size=CHUNK_SIZE)
        self.index.build_from_texts_memmap(texts, batch_size=BATCH_SIZE)
        self.emb_memmap = self.index.load_index()
        await update.message.reply_text(f"Документы переиндексированы. Фрагментов: {len(self.index.texts)}")

    async def handle_msg(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        text = (update.message.text or "").strip()
        if not text:
            await update.message.reply_text("Пустое сообщение — пришли текст вопроса.")
            return
        if self.emb_memmap is None:
            await update.message.reply_text("Индекс недоступен. Запросите /reload.")
            return
        results = self.index.search(text, top_k=TOP_K, emb_memmap=self.emb_memmap)
        if not results:
            await update.message.reply_text("Ничего не найдено.")
            return
        reply_lines = []
        for idx, score in results:
            snippet = self.index.texts[idx]
            score_s = f"{score:.4f}"
            snippet_short = snippet if len(snippet) <= 800 else snippet[:800] + "..."
            reply_lines.append(f"#{idx} (sim={score_s}):\n{snippet_short}\n")
        # Если сообщение длинное, отправляем частями
        reply = "\n\n".join(reply_lines)
        if len(reply) <= 4000:
            await update.message.reply_text(reply)
        else:
            # разбиваем на части до 4000 символов
            for i in range(0, len(reply), 3800):
                await update.message.reply_text(reply[i : i + 3800])


# ---------- Entrypoint ----------
def main():
    app_logic = BotApp()
    application = ApplicationBuilder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", app_logic.start))
    application.add_handler(CommandHandler("reload", app_logic.reload_docs))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, app_logic.handle_msg))

    logger.info("Запуск бота в polling режиме (для Hugging Face Spaces).")
    # polling — проще на Spaces (не нужен webhook/порт)
    application.run_polling()


if __name__ == "__main__":
    main()
