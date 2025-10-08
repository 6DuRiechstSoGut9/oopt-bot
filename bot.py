# bot.py
import os
import logging
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

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
    raise RuntimeError("TELEGRAM_BOT_TOKEN не установлен в окружении")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DOCS_DIR = Path("documents")
BATCH_SIZE = 8
TOP_K = 5

# Файлы индекса
EMB_PATH = Path("embeddings.dat")
TEXTS_PATH = Path("texts.json")
META_PATH = Path("meta.json")

# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def read_txt_file(path: Path) -> str:
    """Чтение txt файлов с разными кодировками"""
    for encoding in ['utf-8', 'cp1251', 'windows-1251', 'latin-1']:
        try:
            with open(path, "r", encoding=encoding) as f:
                content = f.read().strip()
                if content and len(content) > 10:
                    logger.info(f"✅ Успешно прочитан {path.name} в кодировке {encoding}")
                    return content
        except Exception as e:
            continue
    
    logger.error(f"❌ Не удалось прочитать {path.name} ни в одной кодировке")
    return ""


def load_documents() -> List[str]:
    """Загрузка только TXT документов"""
    DOCS_DIR.mkdir(exist_ok=True)
    texts = []
    
    # Только txt файлы
    txt_files = sorted(DOCS_DIR.glob("*.txt"))
    
    logger.info(f"🔍 Найдено TXT файлов: {len(txt_files)}")
    
    for path in txt_files:
        logger.info(f"📖 Чтение файла: {path.name}")
        content = read_txt_file(path)
        
        if content:
            # Разбиваем на абзацы и фильтруем короткие
            paragraphs = [p.strip() for p in content.split('\n') if len(p.strip()) > 30]
            texts.extend(paragraphs)
            logger.info(f"✅ Добавлено {len(paragraphs)} абзацев из {path.name}")
        else:
            logger.warning(f"⚠️ Файл {path.name} пустой или не читается")
    
    if not texts:
        texts = [
            "Добавьте .txt файлы в папку documents для поиска.",
            "Файлы должны быть в кодировке UTF-8, CP1251 или Windows-1251.",
            "Бот готов к работе, но нужны документы для поиска."
        ]
        logger.warning("⚠️ Используются демо-тексты, так как документы не найдены")
    
    logger.info(f"📊 Всего текстовых фрагментов: {len(texts)}")
    return texts


class EmbeddingIndex:
    def __init__(self):
        logger.info("🤖 Загрузка модели для семантического поиска...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME)
        self.texts = []

    def encode_batch(self, texts_batch: List[str]) -> np.ndarray:
        """Кодирование текстов в эмбеддинги"""
        with torch.no_grad():
            enc = self.tokenizer(
                texts_batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            outputs = self.model(**enc)
            embeddings = self.mean_pooling(outputs, enc['attention_mask'])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            return embeddings.numpy().astype(np.float32)

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def build_index(self, texts: List[str]):
        """Построение поискового индекса"""
        logger.info(f"🏗️ Начало построения индекса для {len(texts)} текстов...")
        
        self.texts = texts
        n = len(texts)
        dim = self.model.config.hidden_size
        
        # Создаем memmap файл для эмбеддингов
        embeddings = np.memmap(EMB_PATH, dtype=np.float32, mode='w+', shape=(n, dim))
        
        # Обрабатываем батчами с прогрессом
        for i in range(0, n, BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            batch_emb = self.encode_batch(batch)
            embeddings[i:i + len(batch)] = batch_emb
            
            if i % (BATCH_SIZE * 5) == 0:  # Логируем каждые 5 батчей
                logger.info(f"📦 Обработано {min(i + BATCH_SIZE, n)}/{n} текстов")
        
        embeddings.flush()
        del embeddings
        
        # Сохраняем тексты и метаданные
        with open(TEXTS_PATH, 'w', encoding='utf-8') as f:
            json.dump(texts, f, ensure_ascii=False, indent=2)
        
        meta = {
            "n_texts": n,
            "dim": dim,
            "model": MODEL_NAME
        }
        with open(META_PATH, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)
        
        logger.info("✅ Индекс успешно построен!")

    def load_index(self):
        """Загрузка индекса с диска"""
        if not TEXTS_PATH.exists():
            raise FileNotFoundError("Файл texts.json не найден")
        
        with open(TEXTS_PATH, 'r', encoding='utf-8') as f:
            self.texts = json.load(f)
        
        with open(META_PATH, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        
        n = meta["n_texts"]
        dim = meta["dim"]
        embeddings = np.memmap(EMB_PATH, dtype=np.float32, mode='r', shape=(n, dim))
        
        logger.info(f"📂 Индекс загружен: {n} текстовых фрагментов")
        return embeddings

    def search(self, query: str, top_k: int = 5):
        """Семантический поиск по индексу"""
        if not self.texts:
            return []
        
        embeddings = self.load_index()
        query_emb = self.encode_batch([query])[0]
        
        # Вычисляем косинусное сходство
        scores = embeddings @ query_emb
        
        # Находим топ-K результатов
        if top_k >= len(scores):
            top_indices = np.argsort(-scores)
        else:
            top_indices = np.argpartition(-scores, top_k)[:top_k]
            top_indices = top_indices[np.argsort(-scores[top_indices])]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0.1:  # Минимальный порог сходства
                results.append((int(idx), float(scores[idx])))
        
        return results


class BotApp:
    def __init__(self):
        self.index = None
        self.is_ready = False
        self.init_bot()

    def init_bot(self):
        """Инициализация бота с принудительной переиндексацией"""
        try:
            logger.info("🚀 Инициализация Telegram бота...")
            
            # Всегда перестраиваем индекс
            texts = load_documents()
            self.index = EmbeddingIndex()
            self.index.build_index(texts)
            
            self.is_ready = True
            logger.info("✅ Бот успешно инициализирован и готов к работе!")
            
        except Exception as e:
            logger.error(f"❌ Критическая ошибка инициализации: {e}")
            self.is_ready = False

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_ready:
            await update.message.reply_text("🔄 Бот还在初始化，请稍候...")
            return
            
        await update.message.reply_text(
            "🤖 *Бот готов к работе!*\n\n"
            "Отправьте мне вопрос или фразу для поиска по документам.\n"
            "Я найду наиболее релевантные фрагменты текста.\n\n"
            "_Используется семантический поиск на основе нейросетей_",
            parse_mode='Markdown'
        )

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_ready:
            await update.message.reply_text("⏳ Бот还在 загрузки，请稍等...")
            return
            
        query = update.message.text.strip()
        if not query:
            await update.message.reply_text("📝 Введите текст для поиска")
            return
            
        try:
            logger.info(f"🔍 Поиск запроса: {query}")
            results = self.index.search(query, TOP_K)
            
            if not results:
                await update.message.reply_text(
                    "❌ По вашему запросу ничего не найдено.\n"
                    "Попробуйте переформулировать вопрос или проверьте наличие документов."
                )
                return
                
            response = "📄 *Результаты поиска:*\n\n"
            for i, (idx, score) in enumerate(results, 1):
                snippet = self.index.texts[idx]
                response += f"*{i}.* Сходство: `{score:.3f}`\n"
                response += f"{snippet}\n\n"
                
            # Обрезаем если слишком длинное сообщение
            if len(response) > 4000:
                response = response[:4000] + "\n\n... (сообщение обрезано)"
                
            await update.message.reply_text(response, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"❌ Ошибка при поиске: {e}")
            await update.message.reply_text(
                "⚠️ Произошла ошибка при поиске. Попробуйте еще раз или перезапустите бота командой /start"
            )


def main():
    """Основная функция запуска бота"""
    logger.info("🎯 Запуск Telegram бота...")
    
    try:
        bot_app = BotApp()
        application = ApplicationBuilder().token(TOKEN).build()
        
        # Регистрируем обработчики
        application.add_handler(CommandHandler("start", bot_app.start))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot_app.handle_message))
        
        logger.info("✅ Бот запущен в режиме polling")
        application.run_polling()
        
    except Exception as e:
        logger.critical(f"💥 Критическая ошибка при запуске бота: {e}")
        raise


if __name__ == "__main__":
    main()
