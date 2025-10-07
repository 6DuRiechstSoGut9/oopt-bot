import os
import logging
import threading
import telebot
from flask import Flask, jsonify
from waitress import serve
from dotenv import load_dotenv
import requests
import PyPDF2
import docx
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import uuid

# Загружаем переменные окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Получаем порт из переменных окружения Render
PORT = int(os.environ.get('PORT', 8000))

@app.route('/')
def home():
    return "🤖 Бот ООПТ работает!"

@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy", 
        "documents_loaded": len(doc_search.documents),
        "bot_ready": doc_search.loaded
    })

TOKEN = os.getenv('TELEGRAM_TOKEN')
GIGACHAT_CREDENTIALS = os.getenv('GIGACHAT_CREDENTIALS')

if not TOKEN:
    logger.error("❌ TELEGRAM_TOKEN не установлен!")
    raise ValueError("TELEGRAM_TOKEN обязателен для работы бота")

bot = telebot.TeleBot(TOKEN)

class GigaChatAPI:
    def __init__(self, credentials):
        self.credentials = credentials
        self.base_url = "https://gigachat.devices.sberbank.ru/api/v1"
        self.access_token = None
        self.token_expires = 0
        
    def get_access_token(self):
        """Получаем access token для GigaChat API"""
        try:
            headers = {
                'Authorization': f'Bearer {self.credentials}',
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            data = {
                'scope': 'GIGACHAT_API_PERS'
            }
            
            # Отключаем проверку SSL для совместимости
            response = requests.post(
                'https://ngw.devices.sberbank.ru:9443/api/v2/oauth',
                headers=headers,
                data=data,
                verify=False,
                timeout=10
            )
            
            if response.status_code == 200:
                token_data = response.json()
                self.access_token = token_data['access_token']
                self.token_expires = threading.get_ident() + 1800
                logger.info("✅ GigaChat токен получен")
                return self.access_token
            else:
                logger.error(f"❌ Ошибка получения токена: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Исключение при получении токена: {e}")
            return None
    
    def get_valid_token(self):
        """Получаем валидный токен (обновляем если истек)"""
        if not self.access_token or threading.get_ident() > self.token_expires:
            return self.get_access_token()
        return self.access_token
    
    def chat_completion(self, messages, temperature=0.1, max_tokens=500):
        """Отправляем запрос к GigaChat API"""
        token = self.get_valid_token()
        if not token:
            return None
            
        try:
            headers = {
                'Authorization': f'Bearer {token}',
                'Content-Type': 'application/json'
            }
            
            data = {
                "model": "GigaChat",
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            response = requests.post(
                f'{self.base_url}/chat/completions',
                headers=headers,
                json=data,
                verify=False,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                logger.error(f"❌ GigaChat API error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ GigaChat request exception: {e}")
            return None

class ProfessionalDocumentSearch:
    def __init__(self):
        self.documents = []
        self.chunks = []
        self.embeddings = None
        self.model = None
        self.loaded = False
        self.gigachat = None
        
        # Инициализируем GigaChat если есть credentials
        if GIGACHAT_CREDENTIALS:
            self.gigachat = GigaChatAPI(GIGACHAT_CREDENTIALS)
    
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
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks
    
    def load_documents(self):
        """Загружаем все документы и создаем векторный индекс"""
        documents_dir = 'documents'
        if not os.path.exists(documents_dir):
            logger.warning(f"❌ Папка {documents_dir} не найдена")
            os.makedirs(documents_dir, exist_ok=True)
            logger.info(f"📁 Создана папка {documents_dir}")
            self.create_sample_document()
            return
        
        # Загружаем модель для эмбеддингов
        try:
            logger.info("🔄 Загрузка модели для эмбеддингов...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Модель загружена")
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели: {e}")
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
                                'file_path': file_path
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
            self.embeddings = self.model.encode(chunk_texts)
            logger.info(f"✅ Эмбеддинги созданы. Размерность: {self.embeddings.shape[1]}")
        
        self.loaded = True
        logger.info(f"✅ Загружено документов: {file_count}, фрагментов: {chunk_count}")
    
    def create_sample_document(self):
        """Создаем тестовый документ если нет документов"""
        sample_text = """Верховинский лес - особо охраняемая природная территория в Вологодской области.
        
Орнитофауна Верховинского леса включает следующие виды птиц:
- Певчие птицы: зяблик, пеночка-теньковка, зарянка, певчий дрозд
- Тетеревиные: глухарь, рябчик, тетерев
- Хищные птицы: ястреб-перепелятник, канюк
- Дятлы: большой пестрый дятел, белоспинный дятел
- Совы: серая неясыть, ушастая сова

В период миграции в лесу останавливаются стаи дроздов, зябликов и других перелетных птиц. Лес характеризуется богатым видовым разнообразием птиц благодаря разнообразным биотопам."""
        
        sample_path = 'documents/Верховинский лес.txt'
        os.makedirs('documents', exist_ok=True)
        with open(sample_path, 'w', encoding='utf-8') as f:
            f.write(sample_text)
        
        # Перезагружаем документы
        self.load_documents()
        logger.info("📝 Создан образец документа")
    
    def semantic_search(self, query, top_k=3):
        """Семантический поиск с использованием косинусного сходства"""
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
                        'score': float(similarities[idx]),
                        'file_path': chunk['file_path']
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Ошибка семантического поиска: {e}")
            return []
    
    def search_documents(self, query):
        """Основной поиск"""
        return self.semantic_search(query)
    
    def generate_intelligent_answer(self, query, search_results):
        """Генерирует интеллектуальный ответ на основе найденной информации"""
        if not search_results:
            return "❌ В документах не найдено информации по вашему запросу.\n\n💡 Попробуйте переформулировать вопрос или уточнить название ООПТ."
        
        # Собираем всю релевантную информацию
        context_parts = []
        for result in search_results:
            context_parts.append(f"Документ: {result['file']}\nИнформация: {result['text']}")
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""На основе предоставленной информации из документов об ООПТ Вологодской области ответь на вопрос пользователя.

ВОПРОС: {query}

ИНФОРМАЦИЯ ИЗ ДОКУМЕНТОВ:
{context}

Инструкции:
1. Ответь ТОЛЬКО на основе предоставленной информации
2. Будь конкретен и точен
3. Если информации недостаточно, укажи это
4. Не добавляй информацию, которой нет в документах
5. Ответ должен быть кратким и информативным

ОТВЕТ:"""
        
        if not self.gigachat:
            # Если GigaChat не настроен, формируем простой ответ
            files = ", ".join(set(r['file'] for r in search_results))
            best_match = search_results[0]['text']
            return f"📄 Найдено в документах: {files}\n\n📝 Релевантная информация: {best_match}"
        
        try:
            messages = [
                {
                    "role": "system", 
                    "content": "Ты - помощник по ООПТ Вологодской области. Отвечай точно на основе предоставленных документов. Не придумывай информацию."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ]
            
            answer = self.gigachat.chat_completion(messages)
            
            if answer:
                # Добавляем источники
                sources = ", ".join(set(result['file'] for result in search_results))
                return f"{answer}\n\n📚 Источники: {sources}"
            else:
                logger.error("GigaChat вернул пустой ответ")
                # Возвращаем ответ без нейросети
                files = ", ".join(set(r['file'] for r in search_results))
                best_info = "\n".join([f"• {r['file']}: {r['text'][:200]}..." for r in search_results])
                return f"📄 По вашему запросу найдена информация:\n\n{best_info}\n\n📚 Файлы: {files}"
                
        except Exception as e:
            logger.error(f"GigaChat request exception: {e}")
            # Возвращаем ответ без нейросети
            files = ", ".join(set(r['file'] for r in search_results))
            best_info = search_results[0]['text']
            return f"📄 Найдено в документах: {files}\n\n📝 Информация: {best_info}"

# Инициализация системы поиска
doc_search = ProfessionalDocumentSearch()

def initialize_documents():
    """Инициализация документов в отдельном потоке"""
    logger.info("🔄 Загрузка документов и создание векторного индекса...")
    doc_search.load_documents()
    logger.info(f"✅ Загружено документов: {len(doc_search.documents)}")

# Запускаем инициализацию в фоне
threading.Thread(target=initialize_documents, daemon=True).start()

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    status = "✅ Загружено" if doc_search.loaded else "🔄 Загрузка..."
    doc_count = len(doc_search.documents)
    has_gigachat = doc_search.gigachat is not None
    
    welcome_text = f"""🤖 Бот ООПТ Вологодской области

📚 **Профессиональный семантический поиск** по документам
{status} документов: {doc_count}
🤖 GigaChat: {'✅ Подключен' if has_gigachat else '❌ Не настроен'}

💡 **Задавайте вопросы естественным языком:**
• "Какие птицы живут в Верховинском лесу?"
• "ООПТ Вытегорского района"  
• "Площадь Шимозерского заказника"

🎯 Бот понимает смысл и находит релевантную информацию!

📊 Статус системы: /status"""
    
    bot.reply_to(message, welcome_text)

@bot.message_handler(commands=['status'])
def send_status(message):
    doc_count = len(doc_search.documents)
    chunk_count = len(doc_search.chunks) if doc_search.chunks else 0
    has_embeddings = doc_search.embeddings is not None
    has_gigachat = doc_search.gigachat is not None
    
    status_info = f"""📊 **Статус системы:**

• Документов загружено: {doc_count}
• Фрагментов проиндексировано: {chunk_count}
• Семантический поиск: {'✅ Активен' if has_embeddings else '❌ Не готов'}
• GigaChat API: {'✅ Настроен' if has_gigachat else '❌ Не настроен'}

💡 Система готова к работе!"""
    
    bot.reply_to(message, status_info)

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    bot.send_chat_action(message.chat.id, 'typing')
    
    if not doc_search.loaded:
        bot.reply_to(message, "🔄 Документы загружаются... Это может занять несколько минут.")
        return
    
    user_query = message.text.strip()
    
    if len(user_query) < 3:
        bot.reply_to(message, "❌ Слишком короткий запрос. Уточните вопрос.")
        return
    
    # Ищем в документах
    search_results = doc_search.search_documents(user_query)
    
    # Генерируем интеллектуальный ответ
    answer = doc_search.generate_intelligent_answer(user_query, search_results)
    
    # Отправляем ответ
    try:
        bot.reply_to(message, answer)
    except Exception as e:
        logger.error(f"Error sending message: {e}")
        bot.reply_to(message, "❌ Ошибка при отправке ответа.")

def run_bot():
    """Запуск бота в отдельном потоке"""
    logger.info("🤖 Запуск Telegram бота...")
    try:
        bot.infinity_polling(timeout=60, long_polling_timeout=60)
    except Exception as e:
        logger.error(f"❌ Ошибка бота: {e}")
        # Перезапуск через 10 секунд
        threading.Timer(10.0, run_bot).start()

def main():
    """Основная функция запуска"""
    logger.info(f"🚀 Запуск приложения на порту {PORT}...")
    
    # Запускаем бота в отдельном потоке
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()
    
    # Запускаем Flask приложение
    app.run(host='0.0.0.0', port=PORT, debug=False)

if __name__ == '__main__':
    main()
