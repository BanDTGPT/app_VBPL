import os
import json
import uuid
import logging
import threading
import webbrowser
import time
import numpy as np
import faiss
from flask import Flask, request, jsonify, render_template_string, session
from openai import OpenAI
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your_secret_key")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logging.error("Bạn chưa thiết lập OPENAI_API_KEY trong môi trường.")
    exit(1)

# LawRetriever đơn giản
class LawRetriever:
    def __init__(self, api_key, embed_dim=1536):
        self.client = OpenAI(api_key=api_key)
        self.embedding_dim = embed_dim
        self.index = None
        self.records = []

    def load_data(self):
        # Demo sample, bạn thay bằng đọc file thực tế hoặc DB
        self.records = [
            {"text":"Đoạn luật mẫu 1", "type":"Luật", "number":"31/2024", "url":"local"},
            {"text":"Đoạn luật mẫu 2", "type":"Nghị định", "number":"71/2024", "url":"http://hethongphapluat.vn"}
        ]

    def create_index(self):
        vectors = []
        valid_records = []
        for rec in self.records:
            try:
                rsp = self.client.embeddings.create(input=rec["text"], model="text-embedding-ada-002")
                vec = np.array(rsp.data[0].embedding, dtype=np.float32)
                vectors.append(vec)
                valid_records.append(rec)
            except Exception as e:
                logging.error("Lỗi tạo embedding: %s", e)
        self.records = valid_records
        if not vectors:
            self.index = None
            return
        matrix = np.vstack(vectors)
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(matrix)
        logging.info("Tạo index Faiss với %d vectors", len(valid_records))

    def query(self, question, top_k=3):
        if not self.index:
            raise RuntimeError("Chưa có index")
        try:
            rsp = self.client.embeddings.create(input=question, model="text-embedding-ada-002")
            q_vec = np.array(rsp.data[0].embedding, dtype=np.float32).reshape(1, -1)
        except Exception as e:
            logging.error("Embedding lỗi câu hỏi: %s", e)
            return []
        distances, indices = self.index.search(q_vec, top_k)
        results = [self.records[i] for i in indices[0] if i < len(self.records)]
        return results

    def format_results(self, results):
        texts = []
        for r in results:
            ref = f"{r.get('type','')} {r.get('number','')} ({r.get('url','')})"
            texts.append(f"{ref}\n{r.get('text','')}")
        return "\n\n---\n\n".join(texts)

law_retriever = LawRetriever(OPENAI_API_KEY)

# Quản lý lịch sử chat local
MEMORY_DIR = "history"
os.makedirs(MEMORY_DIR, exist_ok=True)

def get_history_path(session_id):
    return os.path.join(MEMORY_DIR, f"history_{session_id}.json")

def load_history(session_id):
    path = get_history_path(session_id)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_history(session_id, history):
    path = get_history_path(session_id)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

@app.before_request
def setup_session():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())

from flask import render_template_string

INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Tra cứu Pháp luật AI</title>
    <style>
        body {{font-family: Arial; background: #f4f6f8; padding: 20px;}}
        #container {{
            background: white; max-width: 700px; margin: auto;
            padding: 25px; border-radius: 10px;
            box-shadow: 0 0 15px rgba(0,0,0,0.1);
        }}
        label {{font-weight: bold; margin-top: 15px; display: block;}}
        select, input[type=text], textarea, button {{
            width: 100%; padding: 8px; margin-top: 5px;
            border: 1px solid #bbb; border-radius: 6px; box-sizing: border-box;
        }}
        textarea#chatInput {{resize: vertical; height: 80px; font-size: 16px;}}
        button {{
            margin-top: 20px; padding: 10px 22px;
            background-color: #0078D4; color: white; border: none;
            border-radius: 6px; font-size: 16px; cursor: pointer; font-weight: bold;
        }}
        button:hover {{
            background-color: #005a9e;
        }}
        #chatOutput {{
            margin-top: 25px; border: 1px solid #ccc; background:#fff;
            height: 250px; overflow-y: auto; padding: 12px; border-radius: 6px;
            white-space: pre-wrap; font-family: monospace; font-size: 14px;
        }}
    </style>
</head>
<body>
<div id="container">
    <h2>Tra cứu Pháp luật AI</h2>

    <label for="aiVersion">Chọn phiên bản AI:</label>
    <select id="aiVersion" required>
        <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
        <option value="gpt-4">GPT-4</option>
        <option value="gpt-4o-mini">GPT-4 Turbo (Mini)</option>
    </select>

    <label for="localPath">Đường dẫn dữ liệu Local (folder hoặc file):</label>
    <input type="text" id="localPath" placeholder="VD: C:/data/laws">

    <label for="googleDriveId">Folder ID Google Drive:</label>
    <input type="text" id="googleDriveId" placeholder="Nhập folder ID Google Drive">

    <label for="webUrls">URL web tin cậy (phân cách dấu phẩy hoặc xuống dòng):</label>
    <textarea id="webUrls" rows="4" placeholder="https://hethongphapluat.vn, https://vbpl.vn"></textarea>

    <button onclick="saveConfig()">Lưu cấu hình</button>

    <hr/>

    <label for="chatInput">Nhập câu hỏi:</label>
    <textarea id="chatInput" rows="4" placeholder="Nhập câu hỏi luật..."></textarea>
    <button onclick="sendChat()">Gửi câu hỏi</button>

    <div id="chatOutput"></div>
</div>

<script>
async function saveConfig() {{
    const ai_model = document.getElementById('aiVersion').value;
    const local_path = document.getElementById('localPath').value.trim();
    const google_drive_id = document.getElementById('googleDriveId').value.trim();
    const web_urls_raw = document.getElementById('webUrls').value.trim();
    const web_urls = web_urls_raw ? web_urls_raw.split(/[\\n,]+/).map(u => u.trim()).filter(u => u) : [];
    const resp = await fetch('/save_config', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{
            ai_model,
            data_sources: {{
                local_path,
                google_drive_id,
                web_urls
            }}
        }})
    }});
    const data = await resp.json();
    alert(data.status || data.error || 'Cấu hình được lưu');
}}

async function sendChat() {{
    const question = document.getElementById('chatInput').value.trim();
    if (!question) {{
        alert('Vui lòng nhập câu hỏi!');
        return;
    }}
    const chatOutput = document.getElementById('chatOutput');
    chatOutput.innerHTML += '<div><b>Bạn:</b> ' + question + '</div>';
    chatOutput.innerHTML += '<div><i>Đang xử lý...</i></div>';
    chatOutput.scrollTop = chatOutput.scrollHeight;

    const resp = await fetch('/chat', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{question}})
    }});

    const data = await resp.json();

    chatOutput.innerHTML = chatOutput.innerHTML.replace('<div><i>Đang xử lý...</i></div>', '');
    if (data.answer) {{
        chatOutput.innerHTML += '<div><b>AI:</b> ' + data.answer.replace(/\\n/g, '<br>') + '</div>';
    }} else if (data.error) {{
        chatOutput.innerHTML += '<div><b>Lỗi:</b> ' + data.error + '</div>';
    }}
    chatOutput.scrollTop = chatOutput.scrollHeight;
}}
</script>
</body>
</html>
"""

@app.route('/save_config', methods=['POST'])
def save_config():
    global config
    data = request.get_json()
    if not data:
        return jsonify({"error": "Dữ liệu không hợp lệ"}), 400
    config.update({
        "ai_model": data.get("ai_model", config.get("ai_model", "gpt-3.5-turbo")),
        "data_sources": data.get("data_sources", config.get("data_sources", {}))
    })
    return jsonify({"status": "Cấu hình đã lưu"})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Bạn chưa nhập câu hỏi"}), 400

    try:
        law_retriever.update_config(config.get("data_sources", {}))
        law_retriever.load_data()
        law_retriever.create_index()
        results = law_retriever.query(question)

        context = law_retriever.format_results(results)

        prompt = f"Dựa trên các đoạn luật dưới đây:\n{context}\n\nHỏi: {question}\nTrả lời chi tiết."

        response = law_retriever.client.chat.completions.create(
            model=config.get("ai_model", "gpt-3.5-turbo"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1500
        )
        answer = response.choices[0].message.content

        return jsonify({"answer": answer})
    except Exception as e:
        logging.error(f"Lỗi chat: {e}")
        return jsonify({"error": "Lỗi trong quá trình xử lý, thử lại sau."})

import threading
import webbrowser
import time

def open_browser(url):
    time.sleep(1)  # đợi server khởi chạy ổn định
    webbrowser.open(url)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    url = f"http://localhost:{port}/"
    # Tự động mở trình duyệt sau 1 giây
    threading.Thread(target=open_browser, args=(url,), daemon=True).start()

    # Chạy Flask app
    app.run(host="0.0.0.0", port=port, debug=True)