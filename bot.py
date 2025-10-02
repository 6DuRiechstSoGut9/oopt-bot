import os
import logging
import threading
import telebot
from flask import Flask
from waitress import serve
from dotenv import load_dotenv
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import docx

# Загружаем переменные окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.route('/')(lambda: "🤖 Бот ООПТ работает!")

TOKEN = os.getenv('TELEGRAM_TOKEN')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')

bot = telebot.TeleBot(TOKEN)

class SimpleRAGSystem:
    def __init__(self):
        self.documents = []
        self.chunks = []
        
    def extract_text_from_file(self, file_path):
        """Извлечение текста из разных форматов файлов"""
        text = ""
        try:
            if file_path.endswith('.pdf'):
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
            elif file_path.endswith('.docx'):
                doc = docx.Document(file_path)
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
            elif file_path.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
        except Exception as e:
            logger.error(f"Ошибка чтения файла {file_path}: {e}")
        return text
    
    def load_documents(self):
        """Загружаем и обрабатываем все документы"""
        if not os.path.exists('documents'):
            logger.warning("Папка documents не найдена")
            return
        
        self.documents = []
        self.chunks = []
        
        for root, dirs, files in os.walk('documents'):
            for file in files:
                if file.endswith(('.pdf', '.docx', '.txt')):
                    file_path = os.path.join(root, file)
                    logger.info(f"Обрабатываем файл: {file}")
                    
                    text = self.extract_text_from_file(file_path)
                    if text and len(text.strip()) > 50:
                        # Просто сохраняем текст для поиска по ключевым словам
                        self.chunks.append({
                            'text': text,
                            'file': file,
                            'preview': text[:200] + '...' if len(text) > 200 else text
                        })
        
        logger.info(f"Загружено документов: {len(self.chunks)}")
    
    def search_in_documents(self, query, top_k=3):
        """Простой поиск по ключевым словам"""
        if not self.chunks:
            return []
        
        query_lower = query.lower()
        results = []
        
        for chunk in self.chunks:
            text_lower = chunk['text'].lower()
            # Простой подсчет совпадений ключевых слов
            score = sum(1 for word in query_lower.split() if word in text_lower and len(word) > 3)
            
            if score > 0:
                results.append({
                    'text': chunk['text'],
                    'file': chunk['file'],
                    'score': score,
                    'preview': chunk['preview']
                })
        
        # Сортируем по релевантности и берем топ-K
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def ask_deepseek(self, query, context):
        """Запрос к DeepSeek API"""
        if not DEEPSEEK_API_KEY:
            return "❌ API ключ DeepSeek не настроен."
        
        prompt = f"""На основе предоставленной информации об ООПТ Вологодской области ответь на вопрос.

КОНТЕКСТ:
{context}

ВОПРОС: {query}

Ответь максимально информативно на основе контекста. Если информации нет, скажи об этом."""

        try:
            headers = {
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "Ты помощник по ООПТ Вологодской области."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1000,
                "temperature": 0.3
            }
            
            response = requests.post(
                "https://api.deepseek.com/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                return f"❌ Ошибка API: {response.status_code}"
                
        except Exception as e:
            return f"❌ Ошибка соединения: {str(e)}"

# Инициализация системы
rag_system = SimpleRAGSystem()

def initialize_system():
    """Инициализация системы в отдельном потоке"""
    logger.info("Начинаем загрузку документов...")
    rag_system.load_documents()
    logger.info(f"✅ Система готова! Загружено документов: {len(rag_system.chunks)}")

# Запускаем инициализацию
threading.Thread(target=initialize_system, daemon=True).start()

@bot.message_handler(commands=['start'])
def send_welcome(message):
    welcome_text = """🤖 Бот ООПТ Вологодской области

📚 Интеллектуальный поиск по документам об ООПТ
💡 Задавайте вопросы на естественном языке!

Примеры:
• "Какие ООПТ в Вытегорском районе?"
• "Расскажи о заказнике Модно"
• "Сколько всего ООПТ в области?"

Использует DeepSeek для генерации ответов."""
    bot.reply_to(message, welcome_text)

@bot.message_handler(commands=['status'])
def send_status(message):
    status_text = f"""📊 Статус системы:

• Загружено документов: {len(rag_system.chunks)}
• DeepSeek API: {'✅ Настроен' if DEEPSEEK_API_KEY else '❌ Не настроен'}

Готов к работе!"""
    bot.reply_to(message, status_text)

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    bot.send_chat_action(message.chat.id, 'typing')
    
    if not rag_system.chunks:
        bot.reply_to(message, "🔄 Документы загружаются... Попробуйте через минуту.")
        return
    
    # Поиск в документах
    results = rag_system.search_in_documents(message.text)
    
    if results:
        # Собираем контекст из найденных документов
        context = "\n\n".join([f"Из {result['file']}:\n{result['text']}" for result in results])
        
        # Генерируем ответ через DeepSeek
        answer = rag_system.ask_deepseek(message.text, context)
    else:
        answer = "❌ В документах не найдено информации по вашему запросу."
    
    bot.reply_to(message, answer)

def main():
    logger.info("Запуск бота...")
    threading.Thread(target=bot.infinity_polling, daemon=True).start()
    serve(app, host='0.0.0.0', port=8000)

if __name__ == '__main__':
    main()