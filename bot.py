# bot.py
import os
import logging
import json
from pathlib import Path
from typing import List

from RAG import rag_answer, build_index

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    filters,
)

# ---------- CONFIG ----------
TOKEN = input("TOKEN:") #os.environ.get("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN не установлен в окружении")

INDEX_PATH = "faiss.index"
META_PATH = "chunks_meta.json"
DOCS_DIR = "documents/"
#MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5

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


class BotApp:
    def __init__(self):
        self.index = None
        self.is_ready = False
        self.init_bot()

    def init_bot(self):
        """Инициализация бота с принудительной переиндексацией"""
        try:
            logger.info("🚀 Инициализация Telegram бота...")

            if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
                print("Строим индекс из папки:", DOCS_DIR)
                build_index(DOCS_DIR, index_path=INDEX_PATH, meta_path=META_PATH)
            else:
                print("Индекс найден, пропускаем сборку.")

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
            results = rag_answer(query, index_path = INDEX_PATH, meta_path= META_PATH, top_k = TOP_K)
            
            if not results:
                await update.message.reply_text(
                    "❌ По вашему запросу ничего не найдено.\n"
                    "Попробуйте переформулировать вопрос или проверьте наличие документов."
                )
                return
                
            response = "📄 *Результаты поиска:*\n\n"
            for i, r in enumerate(results, 1):
                m = r["meta"]
                score = r.get("score", 0.0)
                rerank = r.get("rerank_score", 0.0)
                if ( rerank//100>score ):
                    score=rerank if rerank!=999 else 1
                response += f"*{i}.* Сходство: `{score:.3f}`\n"
                response += f"{r['text']}\n\n"
                
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
