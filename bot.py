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
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
from chromadb.config import Settings

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

class SemanticDocumentSearch:
    def __init__(self):
        self.documents = []
        self.chunks = []  # Текстовые фрагменты
        self.embeddings = None  # Эмбеддинги фрагментов
        self.loaded = False
        self.model = None
        
        # Инициализируем модель для эмбеддингов
        try:
            logger.info("🔄 Загрузка модели для семантического поиска...")
            self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            logger.info("✅ Модель для эмбеддингов загружена")
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели: {e}")
    
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
    
    def split_text_into_chunks(self, text, chunk_size=500, overlap=50):
        """Разбиваем текст на перекрывающиеся фрагменты"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
            if i + chunk_size >= len(words):
                break
                
        return chunks
    
    def load_documents(self):
        """Загружаем все документы и создаем эмбеддинги"""
        documents_dir = 'documents'
        if not os.path.exists(documents_dir):
            logger.warning(f"❌ Папка {documents_dir} не найдена")
            os.makedirs(documents_dir, exist_ok=True)
            logger.info(f"📁 Создана папка {documents_dir}")
            self.create_sample_document()
            return
        
        self.documents = []
        self.chunks = []
        file_count = 0
        chunk_count = 0
        
        for root, dirs, files in os.walk(documents_dir):
            for file in files:
                if file.endswith(('.pdf', '.docx', '.txt')):
                    file_path = os.path.join(root, file)
                    logger.info(f"📄 Обрабатываем файл: {file}")
                    
                    text = self.extract_text_from_file(file_path)
                    if text and len(text.strip()) > 10:
                        # Разбиваем текст на фрагменты
                        text_chunks = self.split_text_into_chunks(text)
                        
                        for i, chunk in enumerate(text_chunks):
                            self.chunks.append({
                                'file': file,
                                'chunk_id': i,
                                'text': chunk,
                                'full_text': text[:500] + "..."  # Для отладки
                            })
                            chunk_count += 1
                        
                        self.documents.append({
                            'file': file,
                            'text': text,
                            'size': len(text),
                            'chunks': len(text_chunks)
                        })
                        file_count += 1
        
        # Создаем эмбеддинги для всех фрагментов
        if self.chunks and self.model:
            logger.info(f"🔄 Создаем эмбеддинги для {chunk_count} фрагментов...")
            chunk_texts = [chunk['text'] for chunk in self.chunks]
            self.embeddings = self.model.encode(chunk_texts, show_progress_bar=False)
            logger.info("✅ Эмбеддинги созданы")
        
        self.loaded = True
        logger.info(f"✅ Загружено документов: {file_count}, фрагментов: {chunk_count}")
        
        if file_count == 0:
            self.create_sample_document()
    
    def create_sample_document(self):
        """Создаем тестовый документ если нет документов"""
        sample_text = """ООПТ Вологодской области

Особо охраняемые природные территории (ООПТ) - это участки земли, которые имеют особое природоохранное значение.

Заказники Вологодской области:
1. Верхне-Андомский заказник - расположен в Вытегорском районе, площадь 4014 гектаров
2. Ежозерский заказник - находится в Бабаевском районе, площадь 3013 гектаров  
3. Шимозерский заказник - расположен в Вытегорском районе, площадь 8553 гектара

Памятники природы:
- Геологические памятники в Бабушкинском районе
- Ботанические памятники в Великоустюгском районе
- Ландшафтные памятники в Череповецком районе

Всего в Вологодской области насчитывается более 100 особо охраняемых природных территорий регионального значения. Основные категории: заказники, памятники природы, охраняемые природные ландшафты."""
        
        sample_path = 'documents/образец_оопт.txt'
        os.makedirs('documents', exist_ok=True)
        with open(sample_path, 'w', encoding='utf-8') as f:
            f.write(sample_text)
        
        # Перезагружаем документы
        self.load_documents()
        logger.info("📝 Создан образец документа")
    
    def semantic_search(self, query, top_k=3):
        """Семантический поиск по смыслу, а не по ключевым словам"""
        if not self.loaded or not self.chunks or self.embeddings is None:
            return []
        
        try:
            # Создаем эмбеддинг для запроса
            query_embedding = self.model.encode([query])
            
            # Вычисляем косинусное сходство
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            
            # Получаем топ-K наиболее релевантных фрагментов
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.3:  # Порог релевантности
                    chunk = self.chunks[idx]
                    results.append({
                        'file': chunk['file'],
                        'text': chunk['text'],
                        'snippet': chunk['text'][:200] + "...",
                        'score': float(similarities[idx]),
                        'type': 'semantic'
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Ошибка семантического поиска: {e}")
            return []
    
    def search_documents(self, query):
        """Основной поиск - использует семантический поиск"""
        return self.semantic_search(query)
    
    def ask_deepseek(self, query, context):
        """Запрос к DeepSeek API"""
        if not DEEPSEEK_API_KEY:
            return "❌ API ключ DeepSeek не настроен. Добавьте DEEPSEEK_API_KEY в переменные окружения."
        
        prompt = f"""На основе предоставленной информации об ООПТ (Особо Охраняемых Природных Территориях) Вологодской области ответь на вопрос.

ИНФОРМАЦИЯ ИЗ ДОКУМЕНТОВ:
{context}

ВОПРОС: {query}

Ответь максимально информативно ТОЛЬКО на основе предоставленных данных. Если в документах нет информации для ответа, скажи: "В предоставленных документах нет информации по этому вопросу."

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
                        "content": "Ты помощник по ООПТ Вологодской области. Отвечай ТОЧНО на основе предоставленных данных. Не придумывай информацию."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "max_tokens": 800,
                "temperature": 0.1  # Снижаем температуру для более точных ответов
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
doc_search = SemanticDocumentSearch()

def initialize_documents():
    """Инициализация документов в отдельном потоке"""
    logger.info("🔄 Загрузка документов и создание эмбеддингов...")
    doc_search.load_documents()
    logger.info(f"✅ Загружено документов: {len(doc_search.documents)}, фрагментов: {len(doc_search.chunks)}")

# Запускаем инициализацию
threading.Thread(target=initialize_documents, daemon=True).start()

@bot.message_handler(commands=['start'])
def send_welcome(message):
    status = "✅ Загружено" if doc_search.loaded else "🔄 Загрузка..."
    doc_count = len(doc_search.documents)
    
    welcome_text = f"""🤖 Бот ООПТ Вологодской области

📚 **Семантический поиск** по документам об Особо Охраняемых Природных Территориях
{status} документов: {doc_count}

🎯 **Теперь бот понимает смысл запросов!**

💡 **Примеры запросов:**
• "Какие заказники есть в Вытегорском районе?"
• "Площадь Шимозерского заказника"  
• "Сколько всего ООПТ в области?"
• "Памятники природы Бабушкинского района"

🔍 Бот использует семантический поиск и DeepSeek AI.

📊 Для проверки статуса: /status
🆘 Помощь: /help"""
    
    bot.reply_to(message, welcome_text)

@bot.message_handler(commands=['status'])
def send_status(message):
    doc_count = len(doc_search.documents)
    chunk_count = len(doc_search.chunks) if doc_search.chunks else 0
    has_embeddings = doc_search.embeddings is not None
    
    status_info = f"""📊 **Статус системы:**

• Документы загружены: {'✅ Да' if doc_search.loaded else '🔄 Нет'}
• Количество документов: {doc_count}
• Текстовых фрагментов: {chunk_count}
• Семантический поиск: {'✅ Активен' if has_embeddings else '❌ Не готов'}
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
2. Бот понимает смысл, а не только ключевые слова
3. Используйте естественные формулировки

**Примеры:**
"Какие ООПТ в Вытегорском районе?"
"Информация о Шимозерском заказнике"  
"Список памятников природы"
"Сколько всего охраняемых территорий?"

📚 Бот работает с документами в папке 'documents'"""
    
    bot.reply_to(message, help_text)

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    bot.send_chat_action(message.chat.id, 'typing')
    
    if not doc_search.loaded:
        bot.reply_to(message, "🔄 Документы загружаются... Это может занять несколько минут.")
        return
    
    user_query = message.text.strip()
    
    if len(user_query) < 2:
        bot.reply_to(message, "❌ Слишком короткий запрос.")
        return
    
    # Ищем в документах с помощью семантического поиска
    search_results = doc_search.search_documents(user_query)
    
    if search_results:
        # Собираем контекст из наиболее релевантных фрагментов
        context = "\n\n".join([
            f"[Релевантность: {result['score']:.2f}]\nФрагмент: {result['text']}" 
            for result in search_results
        ])
        
        # Генерируем ответ
        answer = doc_search.ask_deepseek(user_query, context)
        
        # Добавляем источники
        sources = ", ".join(set(result['file'] for result in search_results))
        full_answer = f"{answer}\n\n📚 Источники: {sources}"
        
    else:
        full_answer = f"""❌ По запросу "{user_query}" не найдено релевантной информации.

💡 **Попробуйте:**
• Переформулировать вопрос
• Использовать более общие запросы
• Указать конкретные районы или названия

📋 Используйте /help для справки."""
    
    # Отправляем ответ
    try:
        bot.reply_to(message, full_answer)
    except Exception as e:
        logger.error(f"Error sending message: {e}")
        bot.reply_to(message, "❌ Ошибка при отправке ответа.")

def main():
    """Основная функция запуска"""
    logger.info("🚀 Запуск бота с семантическим поиском...")
    
    # Запускаем бота в отдельном потоке
    bot_thread = threading.Thread(target=bot.infinity_polling, daemon=True)
    bot_thread.start()
    
    logger.info("✅ Бот запущен. Запускаем веб-сервер...")
    
    # Запускаем веб-сервер
    serve(app, host='0.0.0.0', port=8000)

if __name__ == '__main__':
    main()
