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
            
            # Логируем найденные папки и файлы
            if self.documents:
                folders = set(os.path.dirname(doc['path']) for doc in self.documents)
                logger.info(f"📁 Найдены папки с документами: {list(folders)}")
            
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
                        start = max(0, index - 50)
                        end = min(len(doc['text']), index + 150)
                        snippet = doc['text'][start:end]
                        
                        results.append({
                            'file': doc['file'],
                            'path': doc['path'],
                            'snippet': snippet,
                            'score': matches
                        })
                        break
        
        # Сортируем по релевантности
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:3]
    
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
    
    welcome_text = f"""🤖 Бот ООПТ Вологодской области

{status_msg}

💡 Задавайте вопросы об ООПТ!

Примеры:
• "ООПТ Вытегорского района"
• "Заказник Модно" 
• "Памятники природы"
• "Сколько всего ООПТ"

/status - подробный статус
/files - список найденных файлов"""
    
    bot.reply_to(message, welcome_text)

@bot.message_handler(commands=['status'])
def send_status(message):
    if doc_search.loading:
        status_text = "🔄 **Идет поиск документов...**\nПожалуйста, подождите."
    elif doc_search.loaded:
        # Собираем информацию о папках
        folders = set(os.path.dirname(doc['path']) for doc in doc_search.documents)
        
        status_text = f"""✅ **Система готова к работе!**

• Найдено документов: {len(doc_search.documents)}
• Папки с документами: {len(folders)}
• Общий объем текста: {sum(doc['size'] for doc in doc_search.documents)} символов
• DeepSeek API: {'✅ Настроен' if DEEPSEEK_API_KEY else '❌ Не настроен'}"""
    elif doc_search.error:
        status_text = f"""❌ **Ошибка загрузки**

• Ошибка: {doc_search.error}"""
    else:
        status_text = "⚪ **Статус неизвестен**\nПопробуйте перезапустить бота."
    
    bot.reply_to(message, status_text)

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
    
    # Группируем по папкам
    folders = {}
    for doc in doc_search.documents:
        folder = os.path.dirname(doc['path'])
        if folder not in folders:
            folders[folder] = []
        folders[folder].append(doc)
    
    for folder, docs in folders.items():
        files_text += f"📂 **{folder or 'Корневая папка'}**\n"
        for doc in docs:
            files_text += f"• {doc['file']} ({doc['size']} символов)\n"
        files_text += "\n"
    
    files_text += f"Всего: {len(doc_search.documents)} документов"
    
    # Если слишком длинное сообщение, разбиваем
    if len(files_text) > 4000:
        parts = [files_text[i:i+4000] for i in range(0, len(files_text), 4000)]
        for part in parts:
            bot.reply_to(message, part)
    else:
        bot.reply_to(message, files_text)

@bot.message_handler(commands=['reload'])
def reload_documents(message):
    """Команда для перезагрузки документов"""
    if doc_search.loading:
        bot.reply_to(message, "🔄 Документы уже загружаются. Дождитесь завершения.")
        return
    
    bot.reply_to(message, "🔄 Перезагружаем документы...")
    
    # Запускаем перезагрузку в отдельном потоке
    def reload():
        doc_search.load_documents()
        if doc_search.loaded:
            bot.reply_to(message, f"✅ Перезагрузка завершена! Найдено документов: {len(doc_search.documents)}")
        else:
            bot.reply_to(message, f"❌ Ошибка перезагрузки: {doc_search.error}")
    
    threading.Thread(target=reload, daemon=True).start()

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    bot.send_chat_action(message.chat.id, 'typing')
    
    # Проверяем статус загрузки
    if doc_search.loading:
        bot.reply_to(message, "🔄 Документы еще загружаются. Пожалуйста, подождите и попробуйте через 30 секунд.")
        return
    
    if not doc_search.loaded:
        if doc_search.error:
            bot.reply_to(message, f"❌ Ошибка загрузки документов: {doc_search.error}\nИспользуйте /reload для повторной попытки.")
        else:
            bot.reply_to(message, "❌ Документы не загружены. Используйте /reload для загрузки.")
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
            f"Из {result['file']}:\n{result['snippet']}..." 
            for result in search_results
        ])
        
        # Генерируем ответ
        answer = doc_search.ask_deepseek(user_query, context)
        
        # Добавляем источники
        sources = ", ".join(set(result['file'] for result in search_results))
        full_answer = f"{answer}\n\n📚 Источники: {sources}"
        
    else:
        full_answer = f"""❌ По запросу '{user_query}' не найдено информации в документах.

💡 Попробуйте:
• Использовать другие ключевые слова
• Указать конкретное название ООПТ
• Уточнить район расположения

📋 Используйте /files чтобы посмотреть список доступных документов"""
    
    # Отправляем ответ (разбиваем если слишком длинный)
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
    for i in range(30):  # Ждем до 30 секунд
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
