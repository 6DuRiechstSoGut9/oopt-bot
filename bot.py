import os
import logging
import threading
import telebot
from flask import Flask
from waitress import serve
from dotenv import load_dotenv
import requests
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

@app.route('/')
def home():
    return "🤖 Бот ООПТ Вологодской области работает!"

TOKEN = os.getenv('TELEGRAM_TOKEN')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')

if not TOKEN:
    logger.error("❌ TELEGRAM_TOKEN не установлен!")
    raise ValueError("TELEGRAM_TOKEN не установлен")

bot = telebot.TeleBot(TOKEN)

class SimpleDocumentSearch:
    def __init__(self):
        self.documents = []
        self.loaded = False
        
    def extract_text_from_file(self, file_path):
        """Извлечение текста из разных форматов файлов"""
        text = ""
        try:
            if file_path.endswith('.pdf'):
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            elif file_path.endswith('.docx'):
                doc = docx.Document(file_path)
                for paragraph in doc.paragraphs:
                    if paragraph.text:
                        text += paragraph.text + "\n"
            elif file_path.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
        except Exception as e:
            logger.error(f"Ошибка чтения файла {file_path}: {e}")
        return text
    
    def load_documents(self):
        """Загружаем все документы"""
        documents_dir = 'documents'
        if not os.path.exists(documents_dir):
            logger.warning(f"❌ Папка {documents_dir} не найдена")
            return
        
        self.documents = []
        file_count = 0
        
        for root, dirs, files in os.walk(documents_dir):
            for file in files:
                if file.endswith(('.pdf', '.docx', '.txt')):
                    file_path = os.path.join(root, file)
                    logger.info(f"📄 Обрабатываем файл: {file}")
                    
                    text = self.extract_text_from_file(file_path)
                    if text and len(text.strip()) > 10:
                        self.documents.append({
                            'file': file,
                            'text': text
                        })
                        file_count += 1
        
        self.loaded = True
        logger.info(f"✅ Загружено документов: {file_count}")
    
    def search_documents(self, query):
        """Простой поиск по документам"""
        if not self.loaded or not self.documents:
            return []
        
        query_lower = query.lower()
        results = []
        
        for doc in self.documents:
            text_lower = doc['text'].lower()
            
            # Простой поиск по ключевым словам
            query_words = [word for word in query_lower.split() if len(word) > 2]
            matches = sum(1 for word in query_words if word in text_lower)
            
            if matches > 0:
                # Находим фрагмент с совпадением
                for word in query_words:
                    if word in text_lower:
                        index = text_lower.find(word)
                        start = max(0, index - 50)
                        end = min(len(doc['text']), index + 150)
                        snippet = doc['text'][start:end]
                        
                        results.append({
                            'file': doc['file'],
                            'snippet': snippet,
                            'score': matches
                        })
                        break
        
        # Сортируем по релевантности
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:2]
    
    def ask_deepseek(self, query, context):
        """Запрос к DeepSeek API"""
        if not DEEPSEEK_API_KEY:
            return "❌ API ключ DeepSeek не настроен. Добавьте DEEPSEEK_API_KEY в переменные окружения."
        
        prompt = f"""На основе информации об ООПТ Вологодской области ответь на вопрос.

ИНФОРМАЦИЯ ИЗ ДОКУМЕНТОВ:
{context}

ВОПРОС: {query}

Ответь на основе предоставленных данных. Если информации недостаточно, сообщи об этом."""

        try:
            headers = {
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "deepseek-chat",
                "messages": [
                    {
                        "role": "system", 
                        "content": "Ты помощник по ООПТ Вологодской области. Отвечай точно на основе предоставленных данных."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "max_tokens": 800,
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
doc_search = SimpleDocumentSearch()

def initialize_documents():
    """Инициализация документов"""
    logger.info("🔄 Загрузка документов...")
    doc_search.load_documents()
    logger.info(f"✅ Загружено: {len(doc_search.documents)} документов")

# Запускаем инициализацию
threading.Thread(target=initialize_documents, daemon=True).start()

@bot.message_handler(commands=['start'])
def send_welcome(message):
    welcome_text = f"""🤖 Бот ООПТ Вологодской области

📚 Документов загружено: {len(doc_search.documents)}
💡 Задавайте вопросы об ООПТ!

Примеры:
• "ООПТ Вытегорского района"
• "Заказник Модно" 
• "Памятники природы"

/status - статус системы"""
    
    bot.reply_to(message, welcome_text)

@bot.message_handler(commands=['status'])
def send_status(message):
    status_info = f"""📊 Статус:

• Документы: {len(doc_search.documents)}
• DeepSeek: {'✅' if DEEPSEEK_API_KEY else '❌'}
• Python: 3.13.4

Готов к работе!"""
    
    bot.reply_to(message, status_info)

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    bot.send_chat_action(message.chat.id, 'typing')
    
    if not doc_search.loaded:
        bot.reply_to(message, "🔄 Загрузка документов...")
        return
    
    user_query = message.text.strip()
    
    # Ищем в документах
    search_results = doc_search.search_documents(user_query)
    
    if search_results:
        # Собираем контекст
        context = "\n\n".join([
            f"Из {result['file']}:\n{result['snippet']}..." 
            for result in search_results
        ])
        
        # Генерируем ответ
        answer = doc_search.ask_deepseek(user_query, context)
        
        # Добавляем источники
        sources = ", ".join(set(result['file'] for result in search_results))
        full_answer = f"{answer}\n\n📚 Источники: {sources}"
        
    else:
        full_answer = f"❌ По запросу '{user_query}' не найдено информации в документах."
    
    bot.reply_to(message, full_answer)

def main():
    logger.info("🚀 Запуск бота на Python 3.13.4...")
    threading.Thread(target=bot.infinity_polling, daemon=True).start()
    serve(app, host='0.0.0.0', port=8000)

if __name__ == '__main__':
    main()
