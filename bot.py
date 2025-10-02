import os
import logging
import threading
import telebot
import requests
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
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

if not TOKEN:
    logger.error("❌ TELEGRAM_TOKEN не найден!")
else:
    bot = telebot.TeleBot(TOKEN)

def get_deepseek_answer(question: str) -> str:
    """Получение ответа от DeepSeek API"""
    if not DEEPSEEK_API_KEY:
        return "❌ API ключ DeepSeek не настроен"
    
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
        }
        
        # Системный промпт с контекстом ООПТ
        system_prompt = """
Ты - эксперт по особо охраняемым природным территориям (ООПТ) Вологодской области. 
Отвечай ТОЛЬКО на основе предоставленной базы знаний об ООПТ.

Основные категории ООПТ в Вологодской области:
- Государственные природные заказники (региональные)
- Памятники природы
- Охраняемые природные комплексы
- Туристско-рекреационные местности

Если вопрос не связан с ООПТ Вологодской области, вежливо откажись отвечать.
"""
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            "temperature": 0.1,
            "max_tokens": 1000
        }
        
        response = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        return result['choices'][0]['message']['content']
        
    except Exception as e:
        logger.error(f"Ошибка DeepSeek API: {e}")
        return f"⚠️ Произошла ошибка при обработке запроса. Попробуйте позже."

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

💡 **Примеры вопросов:**
• "Какие ООПТ в Вытегорском районе?"
• "Расскажи о заказнике Модно"
• "Площадь каких ООПТ больше 10000 га?"

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

🌍 **Примеры запросов:**
• "ООПТ Белозерского района"
• "Заказники площадью более 5000 га" 
• "Памятники природы в Устюженском районе"
    """
    bot.reply_to(message, help_text)

@bot.message_handler(commands=['search'])
def search_command(message):
    bot.reply_to(message, "🔍 **Режим поиска активирован**\n\nЗадайте ваш вопрос об ООПТ Вологодской области:")

@bot.message_handler(func=lambda message: True)
def handle_all_messages(message):
    user_question = message.text
    
    # Показываем что бот думает
    bot.send_chat_action(message.chat.id, 'typing')
    
    # Получаем ответ от DeepSeek
    answer = get_deepseek_answer(user_question)
    
    # Отправляем ответ
    bot.reply_to(message, answer)

def run_bot():
    """Запуск бота в отдельном потоке"""
    if TOKEN:
        logger.info("🤖 Запуск Telegram бота с DeepSeek...")
        bot.infinity_polling()
    else:
        logger.error("❌ Не удалось запустить бота: токен не найден")

def main():
    """Основная функция"""
    # Проверяем настройки
    if not TOKEN:
        logger.error("❌ TELEGRAM_TOKEN не установлен!")
    if not DEEPSEEK_API_KEY:
        logger.warning("⚠️ DEEPSEEK_API_KEY не установлен - бот будет работать в ограниченном режиме")
    else:
        logger.info("✅ DeepSeek API настроен")
    
    # Запускаем бота в фоне
    if TOKEN:
        bot_thread = threading.Thread(target=run_bot, daemon=True)
        bot_thread.start()
        logger.info("✅ Бот запущен в фоновом режиме")
    
    # Запускаем веб-сервер
    logger.info("🌐 Запуск веб-сервера на порту 8000...")
    serve(app, host='0.0.0.0', port=8000)

if __name__ == '__main__':
    main()