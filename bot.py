import os
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
from flask import Flask
from waitress import serve

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logger = logging.getLogger(__name__)

# Flask app для health checks
app = Flask(__name__)

@app.route('/')
def health_check():
    return "🤖 Бот ООПТ Вологодской области работает!"

@app.route('/health')
def health():
    return "OK"

# Глобальная переменная для бота
application = None

async def start(update: Update, context: CallbackContext):
    """Обработчик команды /start"""
    welcome_text = """
🤖 **Бот ООПТ Вологодской области**

Я помогу вам:
• Найти информацию об ООПТ
• Ответить на вопросы о природных территориях  
• Принять жалобу о нарушениях

📋 **Команды:**
/search - поиск по ООПТ
/report - сообщить о нарушении
/help - помощь

Задайте вопрос об ООПТ Вологодской области!
    """
    await update.message.reply_text(welcome_text)

async def help_command(update: Update, context: CallbackContext):
    """Обработчик команды /help"""
    help_text = """
📖 **Помощь по боту:**

🔍 **Поиск информации:**
• Задайте вопрос об ООПТ
• Укажите название, район, площадь
• Спросите о режиме охраны

📝 **Жалобы о нарушениях:**
• Используйте /report для подачи жалобы
• Опишите проблему подробно
• Приложите фото если есть
    """
    await update.message.reply_text(help_text)

async def handle_message(update: Update, context: CallbackContext):
    """Обработчик текстовых сообщений"""
    user_question = update.message.text
    answer = f"""
🔍 **Ваш вопрос:** {user_question}

📚 **Ответ:** Бот работает! Режим интеллектуального поиска по ООПТ Вологодской области активирован.

*В настоящее время идет настройка интеграции с базой знаний*
    """
    await update.message.reply_text(answer)

def main():
    """Основная функция"""
    TOKEN = os.getenv('TELEGRAM_TOKEN')
    
    if not TOKEN:
        logger.error("❌ TELEGRAM_TOKEN не найден!")
        return
    
    # Создаем приложение
    application = Application.builder().token(TOKEN).build()
    
    # Добавляем обработчики
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Запускаем бота в фоне
    import threading
    
    def run_bot():
        logger.info("🤖 Запуск Telegram бота...")
        application.run_polling()
    
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()
    
    # Запускаем веб-сервер
    logger.info("🌐 Запуск веб-сервера на порту 8000...")
    serve(app, host='0.0.0.0', port=8000)

if __name__ == '__main__':
    main()