import os
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# Настройки
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

async def start(update: Update, context: CallbackContext):
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

async def help_command(update: Update, context: CallbackContext):
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

async def handle_message(update: Update, context: CallbackContext):
    """Обработчик текстовых сообщений"""
    user_question = update.message.text
    
    # Здесь будет интеграция с DeepSeek
    answer = f"🔍 **Ваш вопрос:** {user_question}\n\n📚 Я анализирую документы об ООПТ Вологодской области и готовлю ответ..."
    
    await update.message.reply_text(answer)

async def report_command(update: Update, context: CallbackContext):
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

def main():
    """Основная функция"""
    # Создаем приложение
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Добавляем обработчики
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("report", report_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Запускаем бота
    application.run_polling()
    print("Бот запущен!")

if __name__ == '__main__':
    main()