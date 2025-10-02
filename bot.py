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
app.route('/')(lambda: "🤖 Бот ООПТ работает!")

TOKEN = os.getenv('TELEGRAM_TOKEN')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')

if not TOKEN:
    logger.error("❌ TELEGRAM_TOKEN не установлен!")
if not DEEPSEEK_API_KEY:
    logger.warning("⚠️ DEEPSEEK_API_KEY не установлен")

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
            # Создаем папку для тестирования
            os.makedirs(documents_dir, exist_ok=True)
            logger.info(f"📁 Создана папка {documents_dir}")
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
                            'text': text,
                            'size': len(text)
                        })
                        file_count += 1
        
        self.loaded = True
        logger.info(f"✅ Загружено документов: {file_count}")
        
        # Если документов нет, создаем тестовый
        if file_count == 0:
            self.create_sample_document()
    
    def create_sample_document(self):
        """Создаем тестовый документ если нет документов"""
        sample_text = """ООПТ Вологодской области

Заказники:
1. Вытегорский район - Верхне-Андомский (4014 га)
2. Ежозерский (3013 га)
3. Шимозерский (8553 га)

Памятники природы:
- Геологические памятники в Бабушкинском районе
- Ботанические памятники в Великоустюгском районе

Всего в области более 100 ООПТ регионального значения."""
        
        sample_path = 'documents/образец_оопт.txt'
        with open(sample_path, 'w', encoding='utf-8') as f:
            f.write(sample_text)
        
        self.documents.append({
            'file': 'образец_оопт.txt',
            'text': sample_text,
            'size': len(sample_text)
        })
        logger.info("📝 Создан образец документа")
    
    def search_documents(self, query):
        """Простой поиск по документам"""
        if not self.loaded or not self.documents:
            return []
        
        query_lower = query.lower()
        results = []
        
        for doc in self.documents:
            text_lower = doc['text'].lower()
            
            # Считаем релевантность по количеству совпадающих слов
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
                            'text': doc['text'],
                            'snippet': snippet,
                            'score': matches,
                            'matched_word': word
                        })
                        break
        
        # Сортируем по релевантности
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:2]  # Возвращаем топ-2 результата
    
    def ask_deepseek(self, query, context):
        """Запрос к DeepSeek API"""
        if not DEEPSEEK_API_KEY:
            return "❌ API ключ DeepSeek не настроен. Добавьте DEEPSEEK_API_KEY в переменные окружения."
        
        prompt = f"""На основе предоставленной информации об ООПТ (Особо Охраняемых Природных Территориях) Вологодской области ответь на вопрос.

ИНФОРМАЦИЯ ИЗ ДОКУМЕНТОВ:
{context}

ВОПРОС: {query}

Ответь максимально информативно на основе предоставленных данных. Если информации недостаточно, скажи: "В предоставленных документах нет полной информации по этому вопросу."

ОТВЕТ:"""

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
                error_msg = f"Ошибка API ({response.status_code})"
                logger.error(f"DeepSeek API error: {response.text}")
                return f"❌ {error_msg}"
                
        except Exception as e:
            logger.error(f"DeepSeek request exception: {e}")
            return f"❌ Ошибка соединения: {str(e)}"

# Инициализация системы поиска
doc_search = SimpleDocumentSearch()

def initialize_documents():
    """Инициализация документов в отдельном потоке"""
    logger.info("🔄 Загрузка документов...")
    doc_search.load_documents()
    logger.info(f"✅ Загружено документов: {len(doc_search.documents)}")

# Запускаем инициализацию
threading.Thread(target=initialize_documents, daemon=True).start()

@bot.message_handler(commands=['start'])
def send_welcome(message):
    status = "✅ Загружено" if doc_search.loaded else "🔄 Загрузка..."
    doc_count = len(doc_search.documents)
    
    welcome_text = f"""🤖 Бот ООПТ Вологодской области

📚 Поиск по документам об Особо Охраняемых Природных Территориях
{status} документов: {doc_count}

💡 **Примеры запросов:**
• "ООПТ Вытегорского района"
• "Заказник Модно"
• "Памятники природы"
• "Сколько всего ООПТ"

🔍 Бот ищет в документах и генерирует ответы с помощью DeepSeek AI.

📊 Для проверки статуса: /status
🆘 Помощь: /help"""
    
    bot.reply_to(message, welcome_text)

@bot.message_handler(commands=['status'])
def send_status(message):
    doc_count = len(doc_search.documents)
    status_info = f"""📊 **Статус системы:**

• Python версия: 3.13.4
• Документы загружены: {'✅ Да' if doc_search.loaded else '🔄 Нет'}
• Количество документов: {doc_count}
• DeepSeek API: {'✅ Настроен' if DEEPSEEK_API_KEY else '❌ Не настроен'}

💡 **Готов к работе!** Задавайте вопросы об ООПТ."""
    
    bot.reply_to(message, status_info)

@bot.message_handler(commands=['help'])
def send_help(message):
    help_text = """🆘 **Помощь:**

**Команды:**
/start - начать работу
/status - статус системы  
/help - эта справка

**Как работать:**
1. Задавайте вопросы на русском
2. Указывайте конкретные названия ООПТ
3. Используйте ключевые слова

**Примеры:**
"Какие ООПТ в Вытегорском районе?"
"Информация о Шимозерском заказнике"
"Список памятников природы Вологодской области"

📚 Бот работает с документами в папке 'documents'"""
    
    bot.reply_to(message, help_text)

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    bot.send_chat_action(message.chat.id, 'typing')
    
    if not doc_search.loaded:
        bot.reply_to(message, "🔄 Документы загружаются... Попробуйте через 10 секунд.")
        return
    
    user_query = message.text.strip()
    
    if len(user_query) < 2:
        bot.reply_to(message, "❌ Слишком короткий запрос.")
        return
    
    # Ищем в документах
    search_results = doc_search.search_documents(user_query)
    
    if search_results:
        # Собираем контекст
        context = "\n\n".join([
            f"Документ: {result['file']}\nФрагмент: {result['snippet']}..." 
            for result in search_results
        ])
        
        # Генерируем ответ
        answer = doc_search.ask_deepseek(user_query, context)
        
        # Добавляем источники
        sources = ", ".join(set(result['file'] for result in search_results))
        full_answer = f"{answer}\n\n📚 Источники: {sources}"
        
    else:
        full_answer = f"""❌ По запросу "{user_query}" не найдено информации.

💡 **Попробуйте:**
• Другие формулировки
• Конкретные названия ООПТ
• Районы Вологодской области

📋 Используйте /help для справки."""
    
    # Отправляем ответ
    try:
        bot.reply_to(message, full_answer)
    except Exception as e:
        logger.error(f"Error sending message: {e}")
        bot.reply_to(message, "❌ Ошибка при отправке ответа.")

def main():
    """Основная функция запуска"""
    logger.info("🚀 Запуск бота на Python 3.13.4...")
    
    # Запускаем бота в отдельном потоке
    bot_thread = threading.Thread(target=bot.infinity_polling, daemon=True)
    bot_thread.start()
    
    logger.info("✅ Бот запущен. Запускаем веб-сервер...")
    
    # Запускаем веб-сервер
    serve(app, host='0.0.0.0', port=8000)

if __name__ == '__main__':
    main()