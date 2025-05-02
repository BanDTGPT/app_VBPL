import os
import re
import logging
import json
import threading
import uuid
import numpy as np
import faiss
from flask import Flask, request, jsonify, render_template_string, session
from openai import OpenAI
import os
import logging
from flask import Flask
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)

# Load biến môi trường từ file .env
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "default-secret")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not OPENAI_API_KEY or not TELEGRAM_BOT_TOKEN:
    logging.error("Thiếu OPENAI_API_KEY hoặc TELEGRAM_BOT_TOKEN trong biến môi trường!")
    exit(1)

# Hàm xử lý lệnh /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Chào mừng đến bot tra cứu pháp luật AI!")

# Hàm xử lý tin nhắn văn bản
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    # Tại đây bạn tích hợp gọi retriever + AI trả lời thực tế
    response_text = f"Bạn nói: {user_text}\n(Phần trả lời AI sẽ được tích hợp ở đây)"
    await update.message.reply_text(response_text)

async def main():
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))

    await application.run_polling()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

class LawRetriever:
    def __init__(self, api_key, embedding_dim=1536):
        self.client = OpenAI(api_key=api_key)
        self.embedding_dim = embedding_dim
        self.index = None
        self.records = []

    def load_data(self, records):
        self.records = records

    def create_index(self):
        vectors = []
        valid_records = []
        for rec in self.records:
            try:
                resp = self.client.embeddings.create(
                    input=rec['text'],
                    model="text-embedding-ada-002"
                )
                emb = np.array(resp.data[0].embedding, dtype=np.float32)
                vectors.append(emb)
                valid_records.append(rec)
            except Exception as e:
                logging.error(f"Embedding error: {e}")
        self.records = valid_records
        if not vectors:
            logging.warning("Không có embedding để tạo index.")
            self.index = None
            return
        mat = np.vstack(vectors)
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(mat)
        logging.info(f"Tạo index Faiss với {len(valid_records)} records.")

    def query(self, question, top_k=5):
        if not self.index or not self.records:
            logging.warning("Index hoặc dữ liệu chưa được load.")
            return []

        try:
            resp = self.client.embeddings.create(
                input=question,
                model="text-embedding-ada-002"
            )
            q_emb = np.array(resp.data[0].embedding, dtype=np.float32).reshape(1, -1)
        except Exception as e:
            logging.error(f"Lỗi tạo embedding cho câu hỏi: {e}")
            return []

        D, I = self.index.search(q_emb, top_k)
        results = []
        for idx in I[0]:
            if 0 <= idx < len(self.records):
                results.append(self.records[idx])
        return results

    def format_results(self, results):
        formatted = []
        for r in results:
            ref = f"{r.get('type', '')} {r.get('number', '')} ({r.get('url', '')})"
            content = r.get('text', '')
            formatted.append(f"{ref}\n{content}")
        return "\n\n---\n\n".join(formatted)

law_retriever = LawRetriever(OPENAI_API_KEY)

MEMORY_FOLDER = "memory"
os.makedirs(MEMORY_FOLDER, exist_ok=True)

def get_memory_path(session_id):
    return os.path.join(MEMORY_FOLDER, f"memory_{session_id}.json")

def load_memory(session_id):
    path = get_memory_path(session_id)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_memory(session_id, memory):
    path = get_memory_path(session_id)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)

# Telegram bot setup
bot = Bot(token=TELEGRAM_BOT_TOKEN, request=Request(con_pool_size=8))
dispatcher = Dispatcher(bot, None, workers=4, use_context=True)

def start(update: Update, context: CallbackContext):
    update.message.reply_text("Chào mừng đến với bot tra cứu pháp luật. Hãy gửi câu hỏi của bạn!")

def handle_message(update: Update, context: CallbackContext):
    user_id = update.message.from_user.id
    text = update.message.text
    session_id = str(user_id)

    memory = load_memory(session_id)

    results = law_retriever.query(text)
    context_text = law_retriever.format_results(results)
    prompt = f"Dựa trên các đoạn luật:\n{context_text}\n\nHỏi: {text}\nTrả lời chi tiết."

    try:
        response = law_retriever.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=2000,
        )
        answer = response.choices[0].message.content
    except Exception as e:
        answer = f"Lỗi khi gọi AI: {e}"

    memory.append({"user": text, "ai": answer})
    save_memory(session_id, memory)
    update.message.reply_text(answer)

dispatcher.add_handler(CommandHandler("start", start))
dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))

from flask import session
import uuid

INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Tra cứu Pháp luật AI</title>
    <style>
        body {
            font-family: Arial;
            margin: 40px;
            background: #f9f9f9;
        }
        #container {
            max-width: 700px;
            margin: auto;
            background: white;
            padding: 20px 30px;
            border-radius: 12px;
            box-shadow: 0 5px 15px rgba(0,0,0,.1);
        }
        textarea {
            width: 100%;
            height: 80px;
            font-size: 16px;
            margin-bottom: 15px;
            border-radius: 8px;
            border: 1px solid #ccc;
            padding: 10px;
            resize: vertical;
            font-family: Arial, sans-serif;
        }
        button {
            background: #0078D4;
            border: none;
            color: white;
            padding: 12px 26px;
            font-size: 18px;
            border-radius: 8px;
            cursor: pointer;
            font-weight: bold;
        }
        button:hover {
            background: #005ea2;
        }
        #answer {
            white-space: pre-wrap;
            background: #f0f0f0;
            border-radius: 8px;
            padding: 15px;
            min-height: 150px;
            font-size: 16px;
            margin-top: 20px;
            border: 1px solid #ccc;
        }
        h1 {
            text-align: center;
            font-weight: 700;
            margin-bottom: 20px;
            color: #333;
        }
    </style>
</head>
<body>
<div id="container">
    <h1>Tra cứu văn bản pháp luật AI</h1>
    <textarea id="question" placeholder="Nhập câu hỏi pháp luật..."></textarea>
    <button onclick="sendQuestion()">Gửi câu hỏi</button>
    <div id="answer"></div>
</div>

<script>
async function sendQuestion() {
    const question = document.getElementById('question').value.trim();
    if(!question) {
        alert("Vui lòng nhập câu hỏi");
        return;
    }
    const answerDiv = document.getElementById('answer');
    answerDiv.textContent = "Đang xử lý, vui lòng chờ...";

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({question: question})
        });
        const data = await response.json();
        if(data.answer) {
            answerDiv.innerHTML = data.answer.replace(/\n/g, '<br>');
        } else if(data.error) {
            answerDiv.textContent = "Lỗi: " + data.error;
        }
    } catch(e) {
        answerDiv.textContent = "Lỗi kết nối: " + e.message;
    }
}
</script>
</body>
</html>
"""

@app.before_request
def ensure_session():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())

@app.route('/', methods=['GET'])
def index():
    return render_template_string(INDEX_HTML)

@app.route('/chat', methods=['POST'])
def chat_route():
    data = request.get_json()
    question = data.get('question','').strip()
    if not question:
        return jsonify({"error": "Vui lòng nhập câu hỏi"}), 400

    session_id = session.get('session_id')
    memory = load_memory(session_id)

    try:
        results = law_retriever.query(question, top_k=5)
        context = law_retriever.format_results(results)

        history_text = "\n".join([f"Bạn: {m['user']}\nAI: {m['ai']}" for m in memory[-10:]])

        prompt = f"{history_text}\n\nDựa trên các đoạn luật sau đây:\n{context}\n\nHỏi: {question}\nTrả lời chi tiết."

        response = law_retriever.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=2000
        )
        answer = response.choices[0].message.content

        # Lưu ký ức
        memory.append({"user": question, "ai": answer})
        save_memory(session_id, memory)

        return jsonify({"answer": answer})
    except Exception as e:
        logging.error(f"Lỗi trong chat: {e}")
        return jsonify({"error": "Có lỗi xảy ra, vui lòng thử lại."})
from flask import send_from_directory
import threading
import webbrowser
import time

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(RESULT_FOLDER, filename, as_attachment=True)

import threading
import asyncio

def run_bot():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())

import asyncio

async def start_bot():
    await application.run_polling()

if __name__ == "__main__":
    import threading
    import webbrowser
    import time

    # Mở browser sau 1 giây
    port = int(os.environ.get("PORT", 8000))
    url = f"http://localhost:{port}/"

    def open_browser():
        time.sleep(1)
        webbrowser.open(url)

    threading.Thread(target=open_browser).start()

    # Chạy Flask app và Telegram bot đồng thời
    loop = asyncio.get_event_loop()

    # Chạy bot trong executor để không gây lỗi loop đang chạy
    loop.run_in_executor(None, lambda: asyncio.run(start_bot()))

    # Chạy flask trên main thread event loop
    app.run(host="0.0.0.0", port=port, debug=True)