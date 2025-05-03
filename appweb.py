import os
import json
import logging
import numpy as np
import faiss
from flask import Flask, request, jsonify, session
from openai import OpenAI
from dotenv import load_dotenv

from flask import render_template_string

INDEX_HTML = """
<!DOCTYPE html>
<html>
<head><title>Trang Chủ Tra cứu Pháp luật AI</title></head>
<body>
<h1>Chào mừng bạn đến Tra cứu Pháp luật AI</h1>
<p>Vào <a href="/setting">Cài đặt</a> để bắt đầu.</p>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

logging.basicConfig(level=logging.INFO)
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your-secret-key")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logging.error("OPENAI_API_KEY chưa được thiết lập")
    exit(1)

# Lớp LawRetriever - chuyên tra cứu embedded search
class LawRetriever:
    def __init__(self, api_key, embedding_dim=1536):
        self.client = OpenAI(api_key=api_key)
        self.embedding_dim = embedding_dim
        self.index = None
        self.records = []
        self.config = {
            "local_path": "",
            "google_drive_folder_id": "",
            "web_urls": []
        }

    def update_config(self, config_data):
        self.config.update(config_data)

    def load_data(self):
        # Cần mở rộng: đọc file từ local_path, Google Drive, URL, parse chuẩn hóa
        # Hiện mẫu giả định data sẵn
        self.records = [
            {"text": "Nội dung pháp luật ví dụ 1", "type": "Luật", "number": "31/2024", "url": "local"},
            {"text": "Nội dung pháp luật ví dụ 2", "type": "Nghị định", "number": "71/2024", "url": "https://hethongphapluat.vn"},
        ]

    def create_index(self):
        if not self.records:
            logging.warning("Không có dữ liệu để tạo index")
            self.index = None
            return
        vectors = []
        valid_records = []
        for rec in self.records:
            try:
                rsp = self.client.embeddings.create(input=rec["text"], model="text-embedding-ada-002")
                emb = np.array(rsp.data[0].embedding).astype("float32")
                vectors.append(emb)
                valid_records.append(rec)
            except Exception as e:
                logging.error(f"Lỗi tạo embedding: {e}")
        if not vectors:
            logging.warning("Không tạo được vector embedding")
            self.index = None
            return
        self.records = valid_records
        matrix = np.stack(vectors)
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(matrix)
        logging.info(f"Tạo index FAISS với {len(valid_records)} bản ghi")

    def query(self, question, top_k=3):
        if not self.index or not self.records:
            raise RuntimeError("Chưa tạo index hoặc dữ liệu rỗng")
        try:
            rsp = self.client.embeddings.create(input=question, model="text-embedding-ada-002")
            q_emb = np.array(rsp.data[0].embedding).astype("float32").reshape(1, -1)
        except Exception as e:
            logging.error(f"Lỗi tạo embedding câu hỏi: {e}")
            return []
        D, I = self.index.search(q_emb, top_k)
        return [self.records[i] for i in I[0] if i < len(self.records)]

    def format_results(self, results):
        return "\n\n---\n\n".join(
            [f"{r['type']} {r['number']} ({r['url']})\n{r['text']}" for r in results]
        )

law_retriever = LawRetriever(OPENAI_API_KEY)

from flask import render_template_string

INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>App Tra cứu Pháp luật AI</title>
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
    <button onclick="sendQuestion()">Gửi câu hỏi</button>

    <div id="chatOutput"></div>
</div>

<script>
async function saveConfig() {{
    const ai_version = document.getElementById('aiVersion').value;
    const local_path = document.getElementById('localPath').value.trim();
    const gdrive_id = document.getElementById('googleDriveId').value.trim();
    const web_urls_raw = document.getElementById('webUrls').value;
    const web_urls = web_urls_raw.split(/[\\n,]+/).map(x => x.trim()).filter(x => x.length > 0);

    const resp = await fetch('/save_config', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{
            ai_model: ai_version,
            data_sources: {{
                local_path: local_path,
                google_drive_id: gdrive_id,
                web_urls: web_urls
            }}
        }})
    }});
    const data = await resp.json();
    alert(data.status || data.error || 'Đã lưu cấu hình');
}}

async function sendQuestion() {{
    const question = document.getElementById('chatInput').value.trim();
    if(!question) {{
        alert('Vui lòng nhập câu hỏi');
        return;
    }}
    const output = document.getElementById('chatOutput');
    output.innerHTML += `<p><strong>Bạn:</strong> ${question}</p><p><em>Đang xử lý...</em></p>`;

    const resp = await fetch('/chat', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{question}}),
    }});
    const data = await resp.json();

    output.innerHTML = output.innerHTML.replace('<p><em>Đang xử lý...</em></p>', '');
    if(data.answer) {{
        output.innerHTML += `<p><strong>AI:</strong> ${data.answer.replace(/\n/g, '<br>')}</p>`;
    }} else {{
        output.innerHTML += `<p><strong>AI:</strong> Có lỗi xảy ra, vui lòng thử lại.</p>`;
    }}
    output.scrollTop = output.scrollHeight;
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
        return jsonify({"error": "Nhập câu hỏi"}), 400

    try:
        law_retriever.update_config(config.get("data_sources", {}))
        law_retriever.load_data()
        law_retriever.create_index()
        results = law_retriever.query(question)

        context = law_retriever.format_results(results)

        prompt = f"Dựa trên dữ liệu dưới đây:\n{context}\n\nHỏi: {question}\nTrả lời chi tiết."

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
        return jsonify({"error": "Lỗi xử lý, vui lòng thử lại."})


import threading
import webbrowser
import time

def open_browser(url):
    time.sleep(1)
    webbrowser.open(url)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    url = f"http://localhost:{port}/"
    threading.Thread(target=open_browser, args=(url,), daemon=True).start()
    app.run(host="0.0.0.0", port=port, debug=True)