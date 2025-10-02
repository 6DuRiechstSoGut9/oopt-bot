import os
import logging
import threading
import telebot
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

# Инициализация бота
TOKEN = os.getenv('TELEGRAM_TOKEN')
if not TOKEN:
    logger.error("❌ TELEGRAM_TOKEN не найден!")
else:
    bot = telebot.TeleBot(TOKEN)

@bot.message_handler(commands=['start'])
def send_welcome(message):
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
    bot.reply_to(message, welcome_text)

@bot.message_handler(commands=['help'])
def send_help(message):
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
    bot.reply_to(message, help_text)

@bot.message_handler(func=lambda message: True)
def echo_all(message):
    user_question = message.text
    answer = f"""
🔍 **Ваш вопрос:** {user_question}

📚 **Ответ:** Бот работает! Режим интеллектуального поиска по ООПТ Вологодской области активирован.

*В настоящее время идет настройка интеграции с базой знаний*
    """
    bot.reply_to(message, answer)

def run_bot():
    """Запуск бота в отдельном потоке"""
    if TOKEN:
        logger.info("🤖 Запуск Telegram бота...")
        bot.infinity_polling()
    else:
        logger.error("❌ Не удалось запустить бота: токен не найден")

def main():
    """Основная функция"""
    # Запускаем бота в фоне
    if TOKEN:
        bot_thread = threading.Thread(target=run_bot, daemon=True)
        bot_thread.start()
        logger.info("✅ Бот запущен в фоновом режиме")
    else:
        logger.error("❌ TELEGRAM_TOKEN не установлен!")
    
    # Запускаем веб-сервер
    logger.info("🌐 Запуск веб-сервера на порту 8000...")
    serve(app, host='0.0.0.0', port=8000)

if __name__ == '__main__':
    main()