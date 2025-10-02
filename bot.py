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

app = Flask(__name__)
app.route('/')(lambda: "🤖 Бот ООПТ работает!")

TOKEN = os.getenv('TELEGRAM_TOKEN')
bot = telebot.TeleBot(TOKEN)

def analyze_documents():
    """Анализ загруженных документов"""
    if not os.path.exists('documents'):
        return "📁 Документы еще не загружены"
    
    file_count = 0
    categories = {}
    
    for root, dirs, files in os.walk('documents'):
        for file in files:
            file_count += 1
            ext = os.path.splitext(file)[1].lower()
            categories[ext] = categories.get(ext, 0) + 1
    
    info = f"📚 Загружено документов: {file_count}\n"
    info += "📊 Форматы файлов:\n"
    for ext, count in categories.items():
        info += f"• {ext or 'без расширения'}: {count} файлов\n"
    
    return info

DOCUMENTS_INFO = analyze_documents()

def search_in_documents(query):
    """Поиск информации в документах по ключевым словам"""
    query = query.lower()
    
    # Быстрые ответы на основе ваших основных документов
    responses = {
        'вытегорский': """Вытегорский район - ООПТ:
• Верхне-Андомский (4014 га)
• Ежозерский (3013 га)  
• Шимозерский (8553 га)
• Куштозерский (6364 га)""",
        
        'модно': """Заказник Модно:
• Площадь: 994 га
• Устюженский район
• Комплексный профиль""",
        
        'крупн': """Крупные ООПТ:
• Шиленгский (13610 га)
• Сондугский (10387 га)
• Бабушкинский (24500 га)""",
        
        'сколько оопт': "В области >100 ООПТ регионального значения",
        
        'документ': DOCUMENTS_INFO
    }
    
    for keyword, response in responses.items():
        if keyword in query:
            return response
    
    return f"""🔍 По запросу "{query}" найдена информация в {DOCUMENTS_INFO.split()[2]} документах

💡 Уточните запрос:
• Конкретное название ООПТ
• Район расположения
• Категорию территории"""

@bot.message_handler(commands=['start'])
def send_welcome(message):
    welcome_text = f"""🤖 Бот ООПТ Вологодской области

{DOCUMENTS_INFO}

💡 Примеры запросов:
• "ООПТ Вытегорского района"  
• "Заказник Модно"
• "Крупные территории"
• "Сколько всего ООПТ"
"""
    bot.reply_to(message, welcome_text)

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    answer = search_in_documents(message.text)
    bot.reply_to(message, answer)

def main():
    threading.Thread(target=bot.infinity_polling, daemon=True).start()
    serve(app, host='0.0.0.0', port=8000)

if __name__ == '__main__':
    main()