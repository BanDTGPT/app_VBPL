import os
import json
import logging
import numpy as np
import faiss
from flask import Flask, request, jsonify, render_template_string, session
from openai import OpenAI
from dotenv import load_dotenv
import asyncio
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
import threading
import webbrowser
import time
import uuid

logging.basicConfig(level=logging.INFO)
load_dotenv()

# --- Flask app ---
app = Flask(__name__)

# Cấu hình từ biến môi trường
app.secret_key = os.getenv("FLASK_SECRET_KEY", "some-secret")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not OPENAI_API_KEY or not TELEGRAM_BOT_TOKEN:
    logging.error("Bạn chưa thiết lập OPENAI_API_KEY hoặc TELEGRAM_BOT_TOKEN trong biến môi trường.")
    exit(1)

# --- Module LawRetriever ---
class LawRetriever:
    def __init__(self, api_key, embed_dim=1536):
        self.client = OpenAI(api_key=api_key)
        self.embedding_dim = embed_dim
        self.index = None
        self.records = []

    def load_data(self, records):
        self.records = records

    def create_index(self):
        vectors = []
        valid_records = []
        for rec in self.records:
            try:
                response = self.client.embeddings.create(
                    input=rec.get('text', ''),
                    model="text-embedding-ada-002"
                )
                emb = np.array(response.data[0].embedding, dtype=np.float32)
                vectors.append(emb)
                valid_records.append(rec)
            except Exception as e:
                logging.error(f"Lỗi tạo embedding: {e}")

        self.records = valid_records

        if not vectors:
            logging.warning("Không có embedding để tạo chỉ mục.")
            self.index = None
            return

        matrix = np.vstack(vectors)
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(matrix)
        logging.info(f"Chỉ mục Faiss tạo thành công với {len(valid_records)} bản ghi.")

    def query(self, text, top_k=5):
        if not self.index or not self.records:
            raise RuntimeError("Chưa có chỉ mục hoặc dữ liệu.")

        try:
            response = self.client.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            q_emb = np.array(response.data[0].embedding, dtype=np.float32).reshape(1, -1)
        except Exception as e:
            logging.error(f"Lỗi tạo embeddings câu hỏi: {e}")
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

# --- Module Memory quản lý trạng thái chat ---
MEMORY_DIR = "memory"
os.makedirs(MEMORY_DIR, exist_ok=True)

def get_memory_path(session_id):
    return os.path.join(MEMORY_DIR, f"memory_{session_id}.json")

def load_memory(session_id):
    path = get_memory_path(session_id)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return []

def save_memory(session_id, mem_data):
    path = get_memory_path(session_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mem_data, f, ensure_ascii=False, indent=2)
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
            answerDiv.innerHTML = data.answer.replace(/\\n/g, '<br>');
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

import threading
import webbrowser
import time
import asyncio

async def start_bot():
    # Khởi tạo và chạy bot Telegram (hàm main async bạn đã định nghĩa)
    await main()

def run_bot():
    # Tạo event loop mới cho thread bot, chạy main async
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(start_bot())

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    url = f"http://localhost:{port}/"

    # Mở trình duyệt sau 1 giây để server sẵn sàng
    def open_browser():
        time.sleep(1)
        webbrowser.open(url)

    threading.Thread(target=open_browser).start()
    threading.Thread(target=run_bot, daemon=True).start()  # Chạy bot Telegram song song

    # Chạy Flask app chính trên luồng chính
    app.run(host="0.0.0.0", port=port, debug=True)