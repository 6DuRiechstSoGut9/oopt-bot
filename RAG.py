import os
import json
import numpy as np
from pathlib import Path
from docx import Document
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ========== Параметры ==========
DOCS_DIR = ""        # папка с docx
INDEX_PATH = "faiss.index"
META_PATH = "chunks_meta.json"  # сохраняет список метаданных и текста
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_CHARS = 1000           # размер чанка в символах (примерно)
CHUNK_OVERLAP = 100
TOP_K = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ========== 1) Конвертация DOCX -> текст ==========
def docx_to_text(path: str) -> str:
    doc = Document(path)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n".join(paragraphs)

# ========== 2) Чанкинг ==========
def chunk_text(text: str, chunk_size=CHUNK_CHARS, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append((start, end, chunk))
        if end == L:
            break
        start = max(0, end - overlap)
    return chunks

# ========== 3) Индексация (конвертируем все docx, считаем эмбеддинги, строим FAISS) ==========
def build_index(docs_dir=DOCS_DIR, emb_model_name=EMBED_MODEL, index_path=INDEX_PATH, meta_path=META_PATH):
    sbert = SentenceTransformer(emb_model_name)
    all_texts = []
    meta = []  # list of dicts: {file, start, end, text}
    #for p in Path(docs_dir).glob("*.docx"):
    for p in Path(docs_dir).rglob("*.docx"):
        txt = docx_to_text(str(p))
        for start, end, chunk in chunk_text(txt):
            '''
            meta.append({"file": str(p.name), "start": start, "end": end})
            all_texts.append(chunk)
            '''
            prefix = f"Документ: {p.stem}. "
            chunk = prefix + chunk
            meta.append({"file": str(p.name), "start": start, "end": end, "idx": len(all_texts)})
            all_texts.append(chunk)

    if not all_texts:
        raise RuntimeError("Нет docx или они пусты в папке: " + docs_dir)

    # вычисляем эмбеддинги
    embeddings = sbert.encode(all_texts, show_progress_bar=True, convert_to_numpy=True)
    # нормализация для косинусного поиска
    faiss.normalize_L2(embeddings)
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)  # inner product on normalized vectors = cosine
    index.add(embeddings)
    try:
        faiss.write_index(index, index_path)
    except:
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump({}, f)
        faiss.write_index(index, index_path)

    # после получения эмбеддингов в build_index
    #embeddings = sbert.encode(all_texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = np.array(embeddings, dtype='float32')
    faiss.normalize_L2(embeddings)

    # сохраним мета и тексты для восстановления retrieved chunks
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "texts": all_texts}, f, ensure_ascii=False)
    #print(f"Indexed {len(all_texts)} chunks. Index saved to {index_path}, meta to {meta_path}.")

# ========== 4) Ретривал ==========
def load_index(index_path=INDEX_PATH, meta_path=META_PATH):
    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        j = json.load(f)
    return index, j["meta"], j["texts"]

def retrieve(query: str, sbert_model: SentenceTransformer, index, texts, meta, top_k=TOP_K):
    #q_emb = sbert_model.encode([query], convert_to_numpy=True)
    q_emb = sbert_model.encode([query], convert_to_numpy=True)
    q_emb = np.array(q_emb, dtype='float32')
    faiss.normalize_L2(q_emb)
    #faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or score<0.5: continue
        results.append({"score": float(score), "text": texts[idx], "meta": meta[idx]})



    return results

# ========== 5) Интеграция с вашим transformers pipeline ==========
def rag_answer(query: str,
               tokenizer_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
               model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
               index_path=INDEX_PATH, meta_path=META_PATH,
               top_k=TOP_K,
               max_new_tokens=150):

    # загрузка индекс/мета и sbert (можно кешировать между вызовами)
    sbert = SentenceTransformer(EMBED_MODEL)
    index, meta, texts = load_index(index_path, meta_path)

    """
    res = retrieve("Цели создания ООПТ – Сохранение и восстановление", sbert, index,
                   texts,
                   meta, top_k=top_k)
    for r in res[:10]:
        print(r["score"], r["meta"]["file"], r["text"][:200].replace("\n", " "))
    

    res = retrieve("Почвенный покров", sbert, index, texts, meta)
    for r in res:
        print(r["meta"]["file"], round(r["score"], 3))

    return
    """

    # retrieve

    # точные совпадения
    exact_hits = []
    for m, t in zip(meta, texts):
        if query.lower() in t.lower():
            exact_hits.append({"meta": m, "text": t, "rerank_score": 999.0})  # выше всех

    # смысловой поиск
    retrieved = retrieve(query, sbert, index, texts, meta, top_k=top_k)

    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    pairs = [(query, r["text"]) for r in retrieved]
    scores = reranker.predict(pairs)

    for i, s in enumerate(scores):
        retrieved[i]["rerank_score"] = float(s)

    # объединяем
    combined = exact_hits + retrieved
    combined = sorted(combined, key=lambda x: x["rerank_score"], reverse=True)

    combined = combined[:top_k]
    
    """
    retrieved = retrieve(query, sbert, index, texts, meta, top_k=top_k)
    
    from sentence_transformers import CrossEncoder

    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    pairs = [(query, r["text"]) for r in retrieved]
    scores = reranker.predict(pairs)

    for i, s in enumerate(scores):
        retrieved[i]["rerank_score"] = float(s)

    retrieved = sorted(retrieved, key=lambda x: x["rerank_score"], reverse=False)
    
    # составим контекст с источниками
    context_blocks = []
    for i, r in enumerate(retrieved, 1):
        m = r["meta"]
        context_blocks.append(f"---Источник {i}: {m['file']} [{m['start']}:{m['end']}], score={r['score']:.3f}---\n{r['text']}")
    context = "\n\n".join(context_blocks)
    print(context)
    """
    return combined
    """
    # --- формируем контекст ---
    context_blocks = []
    for i, r in enumerate(combined, 1):
        m = r["meta"]
        score = r.get("score", 0.0)
        rerank = r.get("rerank_score", 0.0)
        context_blocks.append(
            f"---Источник {i}: {m['file']} [{m['start']}:{m['end']}], score={score:.3f}, rerank={rerank:.3f}---\n{r['text']}"
        )

    context = "\n\n".join(context_blocks)
    print(context)
    
    return combined
    # подготовим messages так, чтобы вписаться в ваш apply_chat_template
    # system: ограничиваем модель использовать только контекст
    messages = [
        {"role": "system", "content": "Ты помощник. Используй только предоставленный контекст при ответе."},
        {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {query}\n\nИнструкция: Ответь быстро и по-русски. Если информации нет в контексте, скажи 'Не найдено в документах'."}
    ]

    # загрузка токенайзера/модели (совместимы с вашим кодом)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)

    # подготовка inputs через ваш apply_chat_template (как в примере)
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    # декодируем только сгенерированную часть
    gen = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return gen, retrieved
    """

"""
if __name__ == "__main__":
    # 1) Построить индекс (один раз) — раскомментируйте при первом запуске
    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        print("Строим индекс из папки:", DOCS_DIR)
        build_index()
    else:
        print("Индекс найден, пропускаем сборку.")

    # 2) Запрос — демонстрация
    q = input("Запрос!:")
    answer, retrieved = rag_answer(q)
    
    print("Ответ:\n", answer)
    print("\nRetrieved chunks (files and scores):")
    for r in retrieved:
        print(r["meta"]["file"], r["meta"]["start"], r["meta"]["end"], f"score={r['score']:.3f}")
"""
