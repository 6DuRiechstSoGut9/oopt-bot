import os
import logging
import threading
import time
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
        self.loading = False
        self.error = None
        
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
        """Загружаем все документы из всех папок"""
        self.loading = True
        self.error = None
        
        try:
            self.documents = []
            file_count = 0
            
            # Ищем все файлы во всех папках (кроме служебных)
            ignored_dirs = {'.git', '__pycache__', '.venv', 'venv'}
            allowed_extensions = ('.pdf', '.docx', '.txt')
            
            logger.info("🔍 Поиск документов во всех папках...")
            
            for root, dirs, files in os.walk('.'):
                # Пропускаем служебные папки
                dirs[:] = [d for d in dirs if d not in ignored_dirs and not d.startswith('.')]
                
                for file in files:
                    if file.endswith(allowed_extensions):
                        file_path = os.path.join(root, file)
                        # Пропускаем файлы в служебных путях
                        if any(ignored in root for ignored in ignored_dirs):
                            continue
                            
                        logger.info(f"📄 Обрабатываем файл: {file_path}")
                        
                        text = self.extract_text_from_file(file_path)
                        if text and len(text.strip()) > 10:
                            self.documents.append({
                                'file': file,
                                'path': file_path,
                                'text': text,
                                'size': len(text)
                            })
                            file_count += 1
                            logger.info(f"✅ Обработан: {file} ({len(text)} символов)")
                        else:
                            logger.warning(f"⚠️ Пропущен (мало текста): {file}")
            
            self.loaded = True
            logger.info(f"🎉 Загрузка завершена! Найдено документов: {file_count}")
            
        except Exception as e:
            self.error = str(e)
            logger.error(f"❌ Ошибка загрузки документов: {e}")
        finally:
            self.loading = False
    
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
                        start = max(0, index - 100)
                        end = min(len(doc['text']), index + 300)
                        snippet = doc['text'][start:end]
                        
                        results.append({
                            'file': doc['file'],
                            'path': doc['path'],
                            'snippet': snippet,
                            'full_text': doc['text'],
                            'score': matches
                        })
                        break
        
        # Сортируем по релевантности
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:3]
    
    def ask_deepseek(self, query, context):
        """Запрос к DeepSeek API"""
        if not DEEPSEEK_API_KEY:
            return None, "❌ API ключ DeepSeek не настроен."
        
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
                return result['choices'][0]['message']['content'], None
            else:
                return None, f"❌ Ошибка API: {response.status_code}"
                
        except Exception as e:
            return None, f"❌ Ошибка соединения: {str(e)}"
    
    def generate_simple_answer(self, query, search_results):
        """Генерация простого ответа без DeepSeek"""
        if not search_results:
            return "❌ Не найдено информации по вашему запросу."
        
        # Собираем наиболее релевантные фрагменты
        answer_parts = []
        
        for result in search_results:
            # Находим наиболее релевантный фрагмент
            text_lower = result['full_text'].lower()
            query_words = [word for word in query.lower().split() if len(word) > 2]
            
            # Ищем предложения с ключевыми словами
            sentences = result['full_text'].split('. ')
            relevant_sentences = []
            
            for sentence in sentences:
                if any(word in sentence.lower() for word in query_words):
                    relevant_sentences.append(sentence.strip())
            
            if relevant_sentences:
                file_info = f"**Из {result['file']}:**"
                content = ". ".join(relevant_sentences[:3]) + "."
                answer_parts.append(f"{file_info}\n{content}")
        
        if answer_parts:
            answer = "\n\n".join(answer_parts)
            return f"📚 Найдена информация по запросу '{query}':\n\n{answer}"
        else:
            return f"❌ В документах есть информация по теме '{query}', но не удалось выделить конкретный ответ."

# Инициализация системы
doc_search = SimpleDocumentSearch()

def initialize_documents():
    """Инициализация документов"""
    logger.info("🔄 Начинаем поиск документов во всех папках...")
    doc_search.load_documents()
    
    if doc_search.loaded:
        logger.info(f"🎉 Загрузка завершена! Документов: {len(doc_search.documents)}")
    elif doc_search.error:
        logger.error(f"❌ Ошибка загрузки: {doc_search.error}")
    else:
        logger.warning("⚠️ Загрузка не завершена по неизвестной причине")

# Запускаем инициализацию
logger.info("🚀 Запуск инициализации документов...")
doc_thread = threading.Thread(target=initialize_documents, daemon=True)
doc_thread.start()

@bot.message_handler(commands=['start'])
def send_welcome(message):
    if doc_search.loading:
        status_msg = "🔄 Поиск документов..."
    elif doc_search.loaded:
        status_msg = f"✅ Найдено документов: {len(doc_search.documents)}"
    elif doc_search.error:
        status_msg = f"❌ Ошибка: {doc_search.error}"
    else:
        status_msg = "🔄 Статус неизвестен"
    
    deepseek_status = "✅ DeepSeek доступен" if DEEPSEEK_API_KEY else "❌ DeepSeek не настроен"
    
    welcome_text = f"""🤖 Бот ООПТ Вологодской области

{status_msg}
{deepseek_status}

💡 **Режимы работы:**
• С DeepSeek - интеллектуальные ответы
• Без DeepSeek - поиск по документам

**Примеры запросов:**
• "Заповедники Вологодской области"
• "ООПТ Вытегорского района" 
• "Памятники природы"

/status - подробный статус
/files - список документов
/mode - текущий режим работы"""
    
    bot.reply_to(message, welcome_text)

@bot.message_handler(commands=['status'])
def send_status(message):
    if doc_search.loading:
        status_text = "🔄 **Идет поиск документов...**\nПожалуйста, подождите."
    elif doc_search.loaded:
        status_text = f"""✅ **Система готова к работе!**

• Найдено документов: {len(doc_search.documents)}
• Общий объем текста: {sum(doc['size'] for doc in doc_search.documents)} символов
• DeepSeek API: {'✅ Настроен' if DEEPSEEK_API_KEY else '❌ Не настроен'}
• Режим работы: {'🤖 С DeepSeek' if DEEPSEEK_API_KEY else '📚 Только поиск'}"""
    elif doc_search.error:
        status_text = f"""❌ **Ошибка загрузки**

• Ошибка: {doc_search.error}"""
    else:
        status_text = "⚪ **Статус неизвестен**\nПопробуйте перезапустить бота."
    
    bot.reply_to(message, status_text)

@bot.message_handler(commands=['mode'])
def show_mode(message):
    """Показать текущий режим работы"""
    if DEEPSEEK_API_KEY:
        mode_text = """🤖 **Режим: С DeepSeek**

Бот использует DeepSeek AI для генерации интеллектуальных ответов на основе найденных документов."""
    else:
        mode_text = """📚 **Режим: Только поиск**

Бот ищет информацию в документах и показывает найденные фрагменты."""
    
    bot.reply_to(message, mode_text)

@bot.message_handler(commands=['files'])
def list_files(message):
    """Показать список найденных файлов"""
    if not doc_search.loaded:
        bot.reply_to(message, "❌ Документы еще не загружены.")
        return
    
    if not doc_search.documents:
        bot.reply_to(message, "❌ Документы не найдены.")
        return
    
    files_text = "📁 **Найденные документы:**\n\n"
    
    # Показываем только первые 10 файлов чтобы не перегружать
    for doc in doc_search.documents[:10]:
        files_text += f"• {doc['file']} ({doc['size']} символов)\n"
    
    if len(doc_search.documents) > 10:
        files_text += f"\n... и еще {len(doc_search.documents) - 10} файлов"
    
    files_text += f"\n\nВсего: {len(doc_search.documents)} документов"
    
    bot.reply_to(message, files_text)

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    bot.send_chat_action(message.chat.id, 'typing')
    
    # Проверяем статус загрузки
    if doc_search.loading:
        bot.reply_to(message, "🔄 Документы еще загружаются. Пожалуйста, подождите и попробуйте через 30 секунд.")
        return
    
    if not doc_search.loaded:
        if doc_search.error:
            bot.reply_to(message, f"❌ Ошибка загрузки документов: {doc_search.error}")
        else:
            bot.reply_to(message, "❌ Документы не загружены.")
        return
    
    user_query = message.text.strip()
    
    if len(user_query) < 2:
        bot.reply_to(message, "❌ Слишком короткий запрос.")
        return
    
    # Ищем в документах
    search_results = doc_search.search_documents(user_query)
    
    if not search_results:
        answer = f"❌ По запросу '{user_query}' не найдено информации в документах."
        bot.reply_to(message, answer)
        return
    
    # Пробуем использовать DeepSeek если доступен
    if DEEPSEEK_API_KEY:
        # Собираем контекст
        context = "\n\n".join([
            f"Из {result['file']}:\n{result['snippet']}..." 
            for result in search_results
        ])
        
        # Генерируем ответ через DeepSeek
        deepseek_answer, error = doc_search.ask_deepseek(user_query, context)
        
        if deepseek_answer:
            # Успешный ответ от DeepSeek
            sources = ", ".join(set(result['file'] for result in search_results))
            full_answer = f"{deepseek_answer}\n\n📚 Источники: {sources}"
        else:
            # Ошибка DeepSeek - используем простой режим
            logger.warning(f"DeepSeek ошибка: {error}")
            simple_answer = doc_search.generate_simple_answer(user_query, search_results)
            sources = ", ".join(set(result['file'] for result in search_results))
            full_answer = f"🤖 Не удалось получить ответ от AI\n\n{simple_answer}\n\n📚 Источники: {sources}"
    else:
        # Режим без DeepSeek
        simple_answer = doc_search.generate_simple_answer(user_query, search_results)
        sources = ", ".join(set(result['file'] for result in search_results))
        full_answer = f"{simple_answer}\n\n📚 Источники: {sources}"
    
    # Отправляем ответ
    try:
        if len(full_answer) > 4000:
            parts = [full_answer[i:i+4000] for i in range(0, len(full_answer), 4000)]
            for part in parts:
                bot.reply_to(message, part)
        else:
            bot.reply_to(message, full_answer)
    except Exception as e:
        logger.error(f"Ошибка отправки сообщения: {e}")
        bot.reply_to(message, "❌ Ошибка при отправке ответа.")

def main():
    logger.info("🚀 Запуск бота на Python 3.13.4...")
    
    # Даем время на начальную загрузку документов
    logger.info("⏳ Ожидаем завершения начальной загрузки документов...")
    for i in range(30):
        if doc_search.loaded or doc_search.error:
            break
        time.sleep(1)
        if i % 10 == 0:
            logger.info(f"⏰ Ожидание загрузки... {i} сек")
    
    if doc_search.loaded:
        logger.info(f"🎉 Начальная загрузка завершена! Документов: {len(doc_search.documents)}")
    elif doc_search.error:
        logger.error(f"❌ Ошибка начальной загрузки: {doc_search.error}")
    else:
        logger.warning("⚠️ Начальная загрузка не завершена, продолжаем в фоне")
    
    # Запускаем бота
    threading.Thread(target=bot.infinity_polling, daemon=True).start()
    logger.info("✅ Бот запущен. Запускаем веб-сервер...")
    
    # Запускаем веб-сервер
    serve(app, host='0.0.0.0', port=8000)

if __name__ == '__main__':
    main()
