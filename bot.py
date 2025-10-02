import os
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
import aiohttp

# Настройки
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')

class OOPTBot:
    def __init__(self):
        self.app = Application.builder().token(TELEGRAM_TOKEN).build()
        self.setup_handlers()
    
    def setup_handlers(self):
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CommandHandler("help", self.help))
        self.app.add_handler(CommandHandler("search", self.search))
        self.app.add_handler(CommandHandler("report", self.report))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
    
    async def start(self, update: Update, context: CallbackContext):
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
    
    async def handle_message(self, update: Update, context: CallbackContext):
        user_question = update.message.text
        answer = await self.get_deepseek_answer(user_question)
        await update.message.reply_text(answer)
    
    async def get_deepseek_answer(self, question: str) -> str:
        # Интеграция с DeepSeek API будет здесь
        # Пока заглушка с моими ответами на основе ваших документов
        return f"🔍 Ответ на вопрос: '{question}'\n\n(Интеграция с DeepSeek настраивается...)"