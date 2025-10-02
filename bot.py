import os
import logging
import threading
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
from flask import Flask
from waitress import serve

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Flask app для health checks
app = Flask(__name__)

@app.route('/')
def health_check():
    return "🤖 Бот ООПТ Вологодской области работает!"

@app.route('/health')
def health():
    return "OK"

# Telegram бот
class OOPTBot:
    def __init__(self):
        self.token = os.getenv('TELEGRAM_TOKEN')
        self.application = Application.builder().token(self.token).build()
        self.setup_handlers()
    
    def setup_handlers(self):
        """Настройка обработчиков команд"""
        self.application.add_handler(CommandHandler("start", self.start))
        self.application.add_handler(CommandHandler("help", self.help))
        self.application.add_handler(CommandHandler("search", self.search))
        self.application.add_handler(CommandHandler("report", self.report))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
    
    async def start(self, update: Update, context: CallbackContext):
        """Обработчик команды /start"""
        welcome_text = """
🤖 **Бот ООПТ Вологодской области**

Я помогу вам:
• Найти информацию об ООПТ
• Ответить на вопросы о природных территориях  
• Принять жалобу о нарушениях

📋 **Команды:**
/search - поиск по ООПТ
/report - сообщить о нарушении
/help - помощь

Задайте вопрос об ООПТ Вологодской области!
        """
        await update.message.reply_text(welcome_text)
    
    async def help(self, update: Update, context: CallbackContext):
        """Обработчик команды /help"""
        help_text = """
📖 **Помощь по боту:**

🔍 **Поиск информации:**
• Задайте вопрос об ООПТ
• Укажите название, район, площадь
• Спросите о режиме охраны

📝 **Жалобы о нарушениях:**
• Используйте /report для подачи жалобы
• Опишите проблему подробно
• Приложите фото если есть

Примеры вопросов:
• "Какие ООПТ в Вытегорском районе?"
• "Расскажи о заказнике Модно" 
• "Как подать жалобу на нарушение?"
        """
        await update.message.reply_text(help_text)
    
    async def search(self, update: Update, context: CallbackContext):
        """Обработчик команды /search"""
        await update.message.reply_text(
            "🔍 **Поиск по ООПТ**\n\n"
            "Задайте ваш вопрос об особо охраняемых природных территориях:\n"
            "• Название ООПТ\n"
            "• Район расположения\n" 
            "• Площадь территории\n"
            "• Режим охраны"
        )
    
    async def report(self, update: Update, context: CallbackContext):
        """Обработчик команды /report"""
        report_text = """
📝 **Подача жалобы о нарушении**

Опишите подробно:
1. **Место нарушения** (какая ООПТ, район)
2. **Суть нарушения** (что произошло)
3. **Дата и время**
4. **Фото/доказательства** (если есть)

Ваше обращение будет передано в соответствующие органы.
        """
        await update.message.reply_text(report_text)
    
    async def handle_message(self, update: Update, context: CallbackContext):
        """Обработчик текстовых сообщений"""
        user_question = update.message.text
        
        # Здесь будет интеграция с DeepSeek и вашими документами
        answer = await self.generate_answer(user_question)
        
        await update.message.reply_text(answer)
    
    async def generate_answer(self, question: str) -> str:
        """Генерация ответа на вопрос"""
        # Временная заглушка - потом подключим DeepSeek
        return f"""
🔍 **Ваш вопрос:** {question}

📚 **Ответ:** Я анализирую базу данных ООПТ Вологодской области и готовлю подробный ответ на ваш запрос.

💡 *Режим интеллектуального поиска активирован*
"""
    
    def run(self):
        """Запуск бота"""
        print("🚀 Запуск Telegram бота...")
        self.application.run_polling()

def run_telegram_bot():
    """Запуск Telegram бота в отдельном потоке"""
    bot = OOPTBot()
    bot.run()

if __name__ == '__main__':
    # Запускаем Telegram бота в фоне
    bot_thread = threading.Thread(target=run_telegram_bot)
    bot_thread.daemon = True
    bot_thread.start()
    
    # Запускаем Flask сервер для Render
    print("🌐 Запуск веб-сервера для Render...")
    serve(app, host='0.0.0.0', port=8000)