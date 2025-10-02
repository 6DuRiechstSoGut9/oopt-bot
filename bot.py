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
import re

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
        """Умный поиск по документам"""
        if not self.loaded or not self.documents:
            return []
        
        query_lower = query.lower()
        results = []
        
        # Ключевые слова для поиска
        keywords = [word for word in query_lower.split() if len(word) > 2]
        
        for doc in self.documents:
            text_lower = doc['text'].lower()
            
            # Ищем совпадения ключевых слов
            matches = sum(1 for word in keywords if word in text_lower)
            
            if matches > 0:
                # Находим наиболее релевантный фрагмент
                best_snippet = self.extract_best_snippet(doc['text'], keywords)
                
                results.append({
                    'file': doc['file'],
                    'path': doc['path'],
                    'snippet': best_snippet,
                    'full_text': doc['text'],
                    'score': matches
                })
        
        # Сортируем по релевантности
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:3]
    
    def extract_best_snippet(self, text, keywords):
        """Извлекает наиболее релевантный фрагмент текста"""
        sentences = re.split(r'[.!?]+', text)
        relevant_sentences = []
        
        for sentence in sentences:
            sentence_clean = sentence.strip()
            if len(sentence_clean) < 10:
                continue
                
            # Проверяем релевантность предложения
            sentence_lower = sentence_clean.lower()
            relevance = sum(1 for keyword in keywords if keyword in sentence_lower)
            
            if relevance > 0:
                relevant_sentences.append((sentence_clean, relevance))
        
        # Сортируем по релевантности и берем топ-3 предложения
        relevant_sentences.sort(key=lambda x: x[1], reverse=True)
        best_sentences = [sentence for sentence, _ in relevant_sentences[:3]]
        
        return ". ".join(best_sentences) + "."
    
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
        """Генерация структурированного ответа без DeepSeek"""
        if not search_results:
            return self.get_no_results_message(query)
        
        # Анализируем тип запроса
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['заповедник', 'заказник', 'оопт', 'территори']):
            return self.generate_oopts_info(query, search_results)
        elif any(word in query_lower for word in ['район', 'местоположен', 'где']):
            return self.generate_location_info(query, search_results)
        elif any(word in query_lower for word in ['сколько', 'количество', 'число']):
            return self.generate_count_info(query, search_results)
        else:
            return self.generate_general_info(query, search_results)
    
    def generate_oopts_info(self, query, search_results):
        """Генерация информации об ООПТ"""
        answer_parts = ["🌿 **Информация об ООПТ Вологодской области:**\n"]
        
        for result in search_results:
            file_name = result['file'].replace('.docx', '').replace('.pdf', '').replace('.txt', '')
            
            # Извлекаем основную информацию из текста
            text = result['full_text']
            
            # Ищем профиль
            profile_match = re.search(r'Профиль\s*[—–:-]?\s*(.*?)(?:\n|$)', text, re.IGNORECASE)
            profile = profile_match.group(1).strip() if profile_match else "не указан"
            
            # Ищем цели создания
            goals_match = re.search(r'Цели? создания?.*?[—–:-]?\s*(.*?)(?:\n|\.|$)', text, re.IGNORECASE)
            goals = goals_match.group(1).strip() if goals_match else "не указаны"
            
            # Ищем ограничения
            restrictions_match = re.search(r'запрещаются.*?:(.*?)(?:\n\n|\n\s*\n|$)', text, re.DOTALL)
            restrictions = restrictions_match.group(1).strip()[:200] + "..." if restrictions_match else "стандартный режим охраны"
            
            info = f"""**📋 {file_name}**
• **Профиль:** {profile}
• **Цели:** {goals[:100]}...
• **Ограничения:** {restrictions}"""

            answer_parts.append(info)
        
        answer_parts.append(f"\n📚 *На основе анализа {len(search_results)} документов*")
        return "\n\n".join(answer_parts)
    
    def generate_location_info(self, query, search_results):
        """Генерация информации о расположении"""
        answer_parts = ["🗺️ **Расположение ООПТ:**\n"]
        
        for result in search_results:
            file_name = result['file'].replace('.docx', '').replace('.pdf', '').replace('.txt', '')
            text = result['full_text']
            
            # Ищем упоминания районов
            districts = []
            common_districts = ['Вытегорский', 'Вологодский', 'Череповецкий', 'Великоустюгский', 
                              'Бабаевский', 'Кирилловский', 'Шекснинский', 'Устюженский']
            
            for district in common_districts:
                if district.lower() in text.lower():
                    districts.append(district)
            
            location_info = f"**{file_name}**"
            if districts:
                location_info += f" - расположен в {', '.join(districts)} районе"
            else:
                location_info += " - район расположения не указан"
            
            answer_parts.append(location_info)
        
        return "\n• ".join(answer_parts)
    
    def generate_count_info(self, query, search_results):
        """Генерация информации о количестве"""
        total_docs = len(self.documents)
        return f"""📊 **Статистика ООПТ:**

• Обработано документов: {total_docs}
• Найдено упоминаний по вашему запросу: {len(search_results)}
• Общий объем информации: {sum(doc['size'] for doc in self.documents):,} символов

💡 *Для точной статистики используйте конкретные названия ООПТ*"""
    
    def generate_general_info(self, query, search_results):
        """Генерация общей информации"""
        answer_parts = [f"🔍 **Результаты по запросу '{query}':**\n"]
        
        for i, result in enumerate(search_results, 1):
            file_name = result['file'].replace('.docx', '').replace('.pdf', '').replace('.txt', '')
            snippet = result['snippet']
            
            # Обрезаем слишком длинные сниппеты
            if len(snippet) > 300:
                snippet = snippet[:300] + "..."
            
            answer_parts.append(f"{i}. **{file_name}**\n{snippet}")
        
        return "\n\n".join(answer_parts)
    
    def get_no_results_message(self, query):
        """Сообщение когда ничего не найдено"""
        return f"""❌ По запросу "{query}" не найдено информации.

💡 **Советы для поиска:**
• Используйте конкретные названия ООПТ
• Указывайте районы Вологодской области  
• Попробуйте: "заказники", "памятники природы", "Вытегорский район"

📋 **Доступные команды:**
/files - список всех документов
/help - справка по использованию"""

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

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    welcome_text = """🤖 Бот ООПТ Вологодской области

📚 **Интеллектуальный поиск по документам об Особо Охраняемых Природных Территориях**

🔍 **Как работать с ботом:**
• Задавайте вопросы на естественном языке
• Используйте конкретные названия ООПТ
• Указывайте районы Вологодской области

💡 **Примеры запросов:**
• "Заповедники и заказники"
• "ООПТ Вытегорского района" 
• "Памятники природы"
• "Сколько всего охраняемых территорий"

📋 **Команды:**
/start или /help - эта справка
/status - статус системы
/files - список документов
/mode - режим работы"""

    bot.reply_to(message, welcome_text)

@bot.message_handler(commands=['status'])
def send_status(message):
    if doc_search.loading:
        status_text = "🔄 **Идет поиск документов...**\nПожалуйста, подождите."
    elif doc_search.loaded:
        status_text = f"""✅ **Система готова к работе!**

• Найдено документов: {len(doc_search.documents)}
• Общий объем текста: {sum(doc['size'] for doc in doc_search.documents):,} символов
• DeepSeek API: {'✅ Настроен' if DEEPSEEK_API_KEY else '❌ Не настроен'}
• Режим работы: {'🤖 С DeepSeek' if DEEPSEEK_API_KEY else '📚 Умный поиск'}"""
    elif doc_search.error:
        status_text = f"""❌ **Ошибка загрузки**

• Ошибка: {doc_search.error}"""
    else:
        status_text = "⚪ **Статус неизвестен**"
    
    bot.reply_to(message, status_text)

@bot.message_handler(commands=['mode'])
def show_mode(message):
    """Показать текущий режим работы"""
    if DEEPSEEK_API_KEY:
        mode_text = """🤖 **Режим: С DeepSeek**

Бот использует DeepSeek AI для генерации интеллектуальных ответов на основе найденных документов."""
    else:
        mode_text = """📚 **Режим: Умный поиск**

Бот анализирует документы и предоставляет структурированную информацию:
• Основные сведения об ООПТ
• Расположение и районы
• Статистику и общую информацию"""
    
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
    
    # Группируем по первым цифрам (предполагая что это номера ООПТ)
    grouped_files = {}
    for doc in doc_search.documents:
        # Извлекаем номер из названия файла
        match = re.match(r'(\d+)', doc['file'])
        if match:
            prefix = match.group(1)
        else:
            prefix = '其他'
        
        if prefix not in grouped_files:
            grouped_files[prefix] = []
        grouped_files[prefix].append(doc)
    
    # Сортируем по номерам
    for prefix in sorted(grouped_files.keys(), key=lambda x: int(x) if x.isdigit() else 999):
        files = grouped_files[prefix]
        files_text += f"**{prefix}xx:**\n"
        for doc in files[:5]:  # Показываем первые 5 файлов каждой группы
            files_text += f"• {doc['file']} ({doc['size']:,} символов)\n"
        if len(files) > 5:
            files_text += f"  ... и еще {len(files) - 5} файлов\n"
        files_text += "\n"
    
    files_text += f"📊 Всего: {len(doc_search.documents)} документов"
    
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
        answer = doc_search.get_no_results_message(user_query)
        bot.reply_to(message, answer)
        return
    
    # Пробуем использовать DeepSeek если доступен
    if DEEPSEEK_API_KEY:
        # Собираем контекст
        context = "\n\n".join([
            f"Из {result['file']}:\n{result['snippet']}" 
            for result in search_results
        ])
        
        # Генерируем ответ через DeepSeek
        deepseek_answer, error = doc_search.ask_deepseek(user_query, context)
        
        if deepseek_answer:
            # Успешный ответ от DeepSeek
            sources = ", ".join(set(result['file'] for result in search_results))
            full_answer = f"{deepseek_answer}\n\n📚 Источники: {sources}"
        else:
            # Ошибка DeepSeek - используем умный поиск
            logger.warning(f"DeepSeek ошибка: {error}")
            simple_answer = doc_search.generate_simple_answer(user_query, search_results)
            sources = ", ".join(set(result['file'] for result in search_results))
            full_answer = f"{simple_answer}\n\n📚 Источники: {sources}"
    else:
        # Режим умного поиска
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
    
    if doc_search.loaded:
        logger.info(f"🎉 Начальная загрузка завершена! Документов: {len(doc_search.documents)}")
    elif doc_search.error:
        logger.error(f"❌ Ошибка начальной загрузки: {doc_search.error}")
    
    # Запускаем бота
    threading.Thread(target=bot.infinity_polling, daemon=True).start()
    logger.info("✅ Бот запущен. Запускаем веб-сервер...")
    
    # Запускаем веб-сервер
    serve(app, host='0.0.0.0', port=8000)

if __name__ == '__main__':
    main()
