# app.py — запуск Telegram-бота с RAG + минимальный heartbeat для HF Space
import threading
import gradio as gr
from bot import main as start_bot  # импортируем main() из bot.py

# -------------------- Функция запуска бота --------------------
def run_bot():
    try:
        print("Запуск Telegram-бота с GigaChat...")
        start_bot()  # polling внутри bot.py
    except Exception as e:
        print("Ошибка при запуске бота:", e)

# Запуск бота в отдельном потоке
bot_thread = threading.Thread(target=run_bot, daemon=True)
bot_thread.start()

# -------------------- Gradio heartbeat --------------------
def heartbeat():
    return "Бот запущен и активен. Проверь Telegram!"

with gr.Blocks() as demo:
    gr.Markdown("### Telegram-бот с RAG и GigaChat")
    gr.Textbox(label="Heartbeat", value=heartbeat(), interactive=False)

demo.launch(share=False)
