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

class ImprovedDocumentSearch:
    def __init__(self):
        self.documents = []
        self.chunks = []
        self.embeddings = None
        self.loaded = False
        self.model = None
        
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
    
    def split_text_into_chunks(self, text, chunk_size=800, overlap=100):
        """Увеличиваем размер фрагментов для лучшего контекста"""
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
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
            self.embeddings = self.model.encode(chunk_texts, show_progress_bar=False)
            logger.info("✅ Эмбеддинги созданы")
        
        self.loaded = True
        logger.info(f"✅ Загружено документов: {file_count}, фрагментов: {chunk_count}")
    
    def create_sample_document(self):
        """Создаем тестовый документ если нет документов"""
        sample_text = """Верховинский лес - особо охраняемая природная территория. 
        
Орнитофауна Верховинского леса включает следующие виды птиц:
- Певчие птицы: зяблик, пеночка-теньковка, зарянка, певчий дрозд
- Тетеревиные: глухарь, рябчик, тетерев
- Хищные птицы: ястреб-перепелятник, канюк
- Дятлы: большой пестрый дятел, белоспинный дятел
- Совы: серая неясыть, ушастая сова

В период миграции в лесу останавливаются стаи дроздов, зябликов и других перелетных птиц."""
        
        sample_path = 'documents/Верховинский лес.docx'
        os.makedirs('documents', exist_ok=True)
        
        # Создаем простой txt файл вместо docx для простоты
        sample_path = 'documents/Верховинский лес.txt'
        with open(sample_path, 'w', encoding='utf-8') as f:
            f.write(sample_text)
        
        # Перезагружаем документы
        self.load_documents()
        logger.info("📝 Создан образец документа")
    
    def semantic_search(self, query, top_k=5):
        """Улучшенный семантический поиск"""
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
                if similarities[idx] > 0.2:  # Понижаем порог для большего охвата
                    chunk = self.chunks[idx]
                    results.append({
                        'file': chunk['file'],
                        'text': chunk['text'],
                        'score': float(similarities[idx]),
                        'file_path': chunk['file_path']
                    })
            
            # Группируем по файлам и выбираем лучшие фрагменты из каждого
            file_results = {}
            for result in results:
                filename = result['file']
                if filename not in file_results or result['score'] > file_results[filename]['score']:
                    file_results[filename] = result
            
            return list(file_results.values())[:3]  # Возвращаем топ-3 из разных файлов
            
        except Exception as e:
            logger.error(f"Ошибка семантического поиска: {e}")
            return []
    
    def search_documents(self, query):
        """Основной поиск"""
        return self.semantic_search(query)
    
    def generate_intelligent_answer(self, query, search_results):
        """Генерирует интеллектуальный ответ на основе найденной информации"""
        if not search_results:
            return "К сожалению, в документах не найдено конкретной информации по вашему запросу."
        
        # Собираем всю релевантную информацию
        context_parts = []
        for result in search_results:
            context_parts.append(f"Из документа '{result['file']}': {result['text']}")
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""Пользователь спрашивает: "{query}"

На основе следующей информации из документов составь КРАТКИЙ и ИНФОРМАТИВНЫЙ ответ. Если информации недостаточно, так и скажи.

ИНФОРМАЦИЯ ИЗ ДОКУМЕНТОВ:
{context}

Требования к ответу:
- Будь краток и точен
- Перечисли конкретные факты из документов
- Не добавляй информацию, которой нет в документах
- Если в документах мало информации, так и скажи

ОТВЕТ:"""
        
        if not DEEPSEEK_API_KEY:
            # Если API нет, формируем простой ответ самостоятельно
            files = ", ".join(set(r['file'] for r in search_results))
            best_text = search_results[0]['text'][:300] + "..." if len(search_results[0]['text']) > 300 else search_results[0]['text']
            return f"Найдена информация в документах: {files}\n\nОсновные сведения: {best_text}"
        
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
                        "content": "Ты помощник, который точно отвечает на основе предоставленных документов. Не придумывай информацию."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "max_tokens": 500,
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
                answer = result['choices'][0]['message']['content']
                
                # Добавляем источники
                sources = ", ".join(set(result['file'] for result in search_results))
                return f"{answer}\n\n📚 Источники: {sources}"
            else:
                return f"❌ Ошибка при обработке запроса. Статус: {response.status_code}"
                
        except Exception as e:
            logger.error(f"DeepSeek request exception: {e}")
            # Возвращаем ответ без нейросети
            files = ", ".join(set(r['file'] for r in search_results))
            return f"По вашему запросу найдена информация в документах: {files}\n\nДля получения точного ответа проверьте указанные документы."

# Инициализация системы поиска
doc_search = ImprovedDocumentSearch()

def initialize_documents():
    """Инициализация документов в отдельном потоке"""
    logger.info("🔄 Загрузка документов и создание эмбеддингов...")
    doc_search.load_documents()
    logger.info(f"✅ Загружено документов: {len(doc_search.documents)}")

# Запускаем инициализацию
threading.Thread(target=initialize_documents, daemon=True).start()

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    welcome_text = """🤖 Бот ООПТ Вологодской области

📚 **Умный поиск** по документам об ООПТ

💡 **Просто задайте вопрос**, например:
• "Какие птицы живут в Верховинском лесу?"
• "ООПТ Вытегорского района"  
• "Площадь Шимозерского заказника"

🎯 Бот понимает смысл вопросов и ищет по всем документам!

📊 Статус: /status"""
    
    bot.reply_to(message, welcome_text)

@bot.message_handler(commands=['status'])
def send_status(message):
    doc_count = len(doc_search.documents)
    status_info = f"""📊 **Статус системы:**

• Документов загружено: {doc_count}
• Семантический поиск: {'✅ Активен' if doc_search.loaded else '🔄 Загрузка'}
• Система готова к работе!

💡 Задавайте вопросы об ООПТ Вологодской области"""
    
    bot.reply_to(message, status_info)

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    bot.send_chat_action(message.chat.id, 'typing')
    
    if not doc_search.loaded:
        bot.reply_to(message, "🔄 Документы еще загружаются... Попробуйте через минуту.")
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

def main():
    """Основная функция запуска"""
    logger.info("🚀 Запуск улучшенного бота...")
    
    # Запускаем бота в отдельном потоке
    bot_thread = threading.Thread(target=bot.infinity_polling, daemon=True)
    bot_thread.start()
    
    logger.info("✅ Бот запущен. Запускаем веб-сервер...")
    serve(app, host='0.0.0.0', port=8000)

if __name__ == '__main__':
    main()
