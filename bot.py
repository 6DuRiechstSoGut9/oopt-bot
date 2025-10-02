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
from sentence_transformers import SentenceTransformer

# Загружаем переменные окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.route('/')(lambda: "🤖 Бот ООПТ с RAG (DeepSeek) работает!")

TOKEN = os.getenv('TELEGRAM_TOKEN')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')

bot = telebot.TeleBot(TOKEN)

class RAGSystem:
    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.chunks = []
        # Используем локальную модель для эмбеддингов (бесплатно)
        self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        
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
    
    def split_into_chunks(self, text, chunk_size=500):
        """Разбиваем текст на чанки"""
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
    
    def get_embedding(self, text):
        """Получаем embedding через локальную модель (бесплатно)"""
        try:
            embedding = self.embedding_model.encode(text)
            return embedding
        except Exception as e:
            logger.error(f"Ошибка получения embedding: {e}")
            return None
    
    def load_documents(self):
        """Загружаем и обрабатываем все документы"""
        if not os.path.exists('documents'):
            logger.warning("Папка documents не найдена")
            return
        
        self.documents = []
        self.chunks = []
        self.embeddings = []
        
        processed_files = 0
        
        for root, dirs, files in os.walk('documents'):
            for file in files:
                if file.endswith(('.pdf', '.docx', '.txt')):
                    file_path = os.path.join(root, file)
                    logger.info(f"Обрабатываем файл: {file}")
                    
                    text = self.extract_text_from_file(file_path)
                    if text and len(text.strip()) > 50:  # Игнорируем слишком короткие тексты
                        file_chunks = self.split_into_chunks(text)
                        for chunk in file_chunks:
                            embedding = self.get_embedding(chunk)
                            if embedding is not None:
                                self.chunks.append(chunk)
                                self.embeddings.append(embedding)
                                self.documents.append({
                                    'file': file,
                                    'chunk_preview': chunk[:100] + '...' if len(chunk) > 100 else chunk
                                })
                        processed_files += 1
        
        logger.info(f"Обработано файлов: {processed_files}")
        logger.info(f"Загружено чанков: {len(self.chunks)}")
        logger.info(f"Размер векторной базы: {len(self.embeddings)}")
    
    def search_similar_chunks(self, query, top_k=3):
        """Поиск наиболее релевантных чанков"""
        if not self.embeddings:
            return []
        
        query_embedding = self.get_embedding(query)
        if query_embedding is None:
            return []
        
        # Вычисляем косинусное сходство
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Получаем топ-K наиболее релевантных чанков
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.3:  # Понижаем порог для лучшего покрытия
                results.append({
                    'chunk': self.chunks[idx],
                    'similarity': float(similarities[idx]),
                    'source': f"Из документа: {self.documents[idx]['file']}"
                })
        
        return results
    
    def ask_deepseek(self, query, context):
        """Запрос к DeepSeek API"""
        if not DEEPSEEK_API_KEY:
            return "❌ API ключ DeepSeek не настроен. Добавьте DEEPSEEK_API_KEY в переменные окружения."
        
        prompt = f"""Ты - помощник по ООПТ (Особо Охраняемым Природным Территориям) Вологодской области. 
Отвечай ТОЛЬКО на основе предоставленного контекста. Будь точным и информативным.

КОНТЕКСТ:
{context}

ВОПРОС: {query}

Если в контексте нет информации для ответа, скажи: "В предоставленных документах нет информации по этому вопросу."

ОТВЕТ:"""
        
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
                logger.error(f"Ошибка DeepSeek API: {response.status_code} - {response.text}")
                return f"❌ Ошибка API: {response.status_code}"
                
        except Exception as e:
            logger.error(f"Исключение при запросе к DeepSeek: {e}")
            return f"❌ Ошибка соединения: {str(e)}"
    
    def generate_answer(self, query, context_chunks):
        """Генерация ответа с использованием контекста"""
        if not context_chunks:
            return "❌ В документах не найдено релевантной информации для ответа на ваш вопрос."
        
        # Объединяем контекст
        context = "\n\n".join([f"{chunk['source']}\n{chunk['chunk']}" for chunk in context_chunks])
        
        # Используем DeepSeek для генерации ответа
        answer = self.ask_deepseek(query, context)
        
        return answer

# Инициализация RAG системы
rag_system = RAGSystem()

def initialize_rag():
    """Инициализация RAG системы в отдельном потоке"""
    logger.info("Начинаем загрузку и обработку документов...")
    rag_system.load_documents()
    if rag_system.chunks:
        logger.info("✅ RAG система готова к работе!")
    else:
        logger.warning("⚠️ RAG система загружена, но документы не обработаны")

# Запускаем инициализацию в отдельном потоке
threading.Thread(target=initialize_rag, daemon=True).start()

@bot.message_handler(commands=['start'])
def send_welcome(message):
    welcome_text = f"""🤖 Бот ООПТ Вологодской области с RAG

📚 Использует DeepSeek для интеллектуального поиска по документам
💡 Задавайте вопросы на естественном языке!

Примеры запросов:
• "Какие ООПТ есть в Вытегорском районе?"
• "Расскажи о заказнике Модно"  
• "Какие самые крупные охраняемые территории?"
• "Сколько всего ООПТ в области?"

Статус: {'✅ Документы загружены' if rag_system.chunks else '🔄 Загрузка документов...'}
Чанков: {len(rag_system.chunks)}
"""
    bot.reply_to(message, welcome_text)

@bot.message_handler(commands=['status'])
def send_status(message):
    status_text = f"""📊 Статус RAG системы:

• Обработано чанков: {len(rag_system.chunks)}
• Размер векторной базы: {len(rag_system.embeddings)}
• Модель эмбеддингов: локальная (бесплатно)
• DeepSeek API: {'✅ Настроен' if DEEPSEEK_API_KEY else '❌ Не настроен'}

Система готова к вопросам!"""
    bot.reply_to(message, status_text)

@bot.message_handler(commands=['debug'])
def send_debug(message):
    """Отладочная информация"""
    if not rag_system.chunks:
        bot.reply_to(message, "❌ Документы еще не загружены")
        return
    
    # Показываем примеры чанков
    debug_info = "📋 Примеры загруженных чанков:\n\n"
    for i, chunk in enumerate(rag_system.chunks[:3]):  # Первые 3 чанка
        debug_info += f"{i+1}. {chunk[:100]}...\n\n"
    
    debug_info += f"\nВсего чанков: {len(rag_system.chunks)}"
    bot.reply_to(message, debug_info)

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    # Показываем, что бот думает
    bot.send_chat_action(message.chat.id, 'typing')
    
    if not rag_system.chunks:
        bot.reply_to(message, "🔄 Документы еще загружаются. Попробуйте через минуту.")
        return
    
    # Поиск релевантных чанков
    similar_chunks = rag_system.search_similar_chunks(message.text, top_k=3)
    
    # Генерация ответа
    if similar_chunks:
        answer = rag_system.generate_answer(message.text, similar_chunks)
    else:
        answer = "❌ В документах не найдено релевантной информации для ответа на ваш вопрос."
    
    bot.reply_to(message, answer)

def main():
    """Основная функция запуска"""
    logger.info("Запуск бота...")
    threading.Thread(target=bot.infinity_polling, daemon=True).start()
    serve(app, host='0.0.0.0', port=8000)

if __name__ == '__main__':
    main()