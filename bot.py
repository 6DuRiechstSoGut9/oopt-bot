# bot_webhook.py
# Работает с python-telegram-bot v20 (async style)
# Использует transformers AutoTokenizer (use_fast=False) и AutoModel для эмбеддингов
# Simple mean pooling -> L2-normalize -> cosine similarity
# Webhook для Render

import os
import logging
import glob
import numpy as np
from pathlib import Path
from typing import List

import torch
from transformers import AutoTokenizer, AutoModel

from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters

# ---------- CONFIG ----------
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DOCS_DIR = Path("documents")
CHUNK_SIZE = 400
EMBED_DEVICE = "cpu"  # Render free — CPU
TOP_K = 3
WEBHOOK_URL = os.environ.get("WEBHOOK_URL")  # пример: https://your-service.onrender.com
PORT = int(os.environ.get("PORT", 8443))
# ----------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Loader / Index ----------
class SimpleEmbeddingIndex:
    def __init__(self, model_name: str, device: str = "cpu"):
        logger.info("Loading tokenizer and model (use_fast=False)...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.device = device

        self.texts: List[str] = []
        self.embs: np.ndarray = np.zeros((0, self.model.config.hidden_size), dtype=np.float32)

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        all_embs = []
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                enc = self.tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt", return_attention_mask=True)
                input_ids = enc["input_ids"].to(self.device)
                attention_mask = enc["attention_mask"].to(self.device)
                out = self.model(input_ids=input_ids, attention_mask=attention_mask)
                pooled = self.mean_pooling(out, attention_mask)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                all_embs.append(pooled.cpu().numpy())
        return np.vstack(all_embs) if all_embs else np.zeros((0, self.model.config.hidden_size), dtype=np.float32)

    def build_from_texts(self, texts: List[str]):
        logger.info(f"Building index for {len(texts)} texts...")
        self.texts = texts
        self.embs = self.encode(texts, batch_size=8)

    def search(self, query: str, top_k: int = 3):
        q_emb = self.encode([query])
        if q_emb.shape[0] == 0 or self.embs.shape[0] == 0:
            return []
        sims = self.embs @ q_emb[0]
        topk_idx = np.argsort(-sims)[:top_k]
        results = [(int(idx), float(sims[idx])) for idx in topk_idx]
        return results

# ---------- Document utils ----------
def load_documents_chunks(directory: Path, chunk_size: int = 400) -> List[str]:
    directory.mkdir(parents=True, exist_ok=True)
    texts = []
    for path in sorted(glob.glob(str(directory / "*.txt"))):
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()
        except Exception:
            with open(path, "r", encoding="latin-1") as f:
                text = f.read().strip()
        if not text:
            continue
        start = 0
        n = len(text)
        while start < n:
            end = min(n, start + chunk_size)
            chunk = text[start:end].strip()
            texts.append(chunk)
            start = end
    if not texts:
        texts.append("Пример содержимого. Добавьте файлы .txt в папку documents для индексации.")
    return texts

# ---------- Bot logic ----------
class BotApp:
    def __init__(self):
        self.index = SimpleEmbeddingIndex(MODEL_NAME, device=EMBED_DEVICE)
        self.texts = load_documents_chunks(DOCS_DIR, CHUNK_SIZE)
        self.index.build_from_texts(self.texts)

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("Бот запущен. Отправь вопрос — я найду похожие фрагменты.")

    async def reload_docs(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.texts = load_documents_chunks(DOCS_DIR, CHUNK_SIZE)
        self.index.build_from_texts(self.texts)
        await update.message.reply_text(f"Документы переиндексированы. Фрагментов: {len(self.texts)}")

    async def handle_msg(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        text = (update.message.text or "").strip()
        if not text:
            await update.message.reply_text("Пустое сообщение — пришли текст вопроса.")
            return
        results = self.index.search(text, top_k=TOP_K)
        if not results:
            await update.message.reply_text("Ничего не найдено.")
            return
        reply_lines = []
        for idx, score in results:
            snippet = self.texts[idx]
            score_s = f"{score:.4f}"
            snippet_short = snippet if len(snippet) <= 800 else snippet[:800] + "..."
            reply_lines.append(f"#{idx} (sim={score_s}):\n{snippet_short}\n")
        await update.message.reply_text("\n\n".join(reply_lines))

# ---------- Entrypoint ----------
def main():
    if not TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN не установлен в environment. Выход.")
        return
    if not WEBHOOK_URL:
        logger.error("WEBHOOK_URL не установлен. Выход.")
        return

    app_logic = BotApp()
    application = ApplicationBuilder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", app_logic.start))
    application.add_handler(CommandHandler("reload", app_logic.reload_docs))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, app_logic.handle_msg))

    logger.info(f"Запуск webhook на порту {PORT} с URL {WEBHOOK_URL}...")
    application.run_webhook(
        listen="0.0.0.0",
        port=PORT,
        webhook_url=WEBHOOK_URL
    )

if __name__ == "__main__":
    main()
