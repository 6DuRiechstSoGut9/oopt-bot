import os
import logging
import threading
import telebot
from flask import Flask
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
PORT = int(os.environ.get('PORT', 8000))

@app.route('/')
def home():
    return "🤖 Бот ООПТ работает!"

TOKEN = os.getenv('TELEGRAM_TOKEN')
bot = telebot.TeleBot(TOKEN)

# Простое хранилище документов в памяти
documents = {
    "Верховинский лес": "Особо охраняемая природная территория. Птицы: зяблик, пеночка-теньковка, глухарь, рябчик, ястреб-перепелятник, большой пестрый дятел.",
    "Шимозерский заказник": "Расположен в Вытегорском районе. Площадь: 45 000 га. Обитают медведи, лоси, рыси, бобры.",
    "Русский Север": "Национальный парк. Создан в 1992 году. Площадь: 166 400 га."
}

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    welcome_text = """🤖 Бот ООПТ Вологодской области

Доступные команды:
/start - начать работу
/status - статус бота
/list - список ООПТ

Просто напиши название ООПТ и получи информацию!"""
    
    bot.reply_to(message, welcome_text)

@bot.message_handler(commands=['status'])
def send_status(message):
    bot.reply_to(message, "✅ Бот работает исправно")

@bot.message_handler(commands=['list'])
def send_list(message):
    oopt_list = "\n".join([f"• {name}" for name in documents.keys()])
    bot.reply_to(message, f"📋 Список ООПТ:\n\n{oopt_list}")

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    user_query = message.text.strip().lower()
    
    # Простой поиск по ключевым словам
    found_results = []
    for name, info in documents.items():
        if user_query in name.lower() or any(word in info.lower() for word in user_query.split()):
            found_results.append((name, info))
    
    if found_results:
        response = "📄 Найдена информация:\n\n"
        for name, info in found_results:
            response += f"**{name}**\n{info}\n\n"
        bot.reply_to(message, response)
    else:
        bot.reply_to(message, "❌ Информация не найдена. Попробуйте /list для просмотра доступных ООПТ")

def run_bot():
    """Запуск бота в отдельном потоке"""
    logger.info("🤖 Запуск Telegram бота...")
    try:
        bot.infinity_polling()
    except Exception as e:
        logger.error(f"Ошибка бота: {e}")

def main():
    # Запускаем бота в отдельном потоке
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()
    
    # Запускаем Flask приложение
    from waitress import serve
    serve(app, host='0.0.0.0', port=PORT)

if __name__ == '__main__':
    main()
