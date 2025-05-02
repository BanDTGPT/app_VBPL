import os
import re
import json
import logging
import threading
import webbrowser
import time
import numpy as np
import faiss
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, render_template_string, send_from_directory
from werkzeug.utils import secure_filename
from openai import OpenAI

from dotenv import load_dotenv
import os

# Load các biến môi trường từ file .env
load_dotenv()

# Ví dụ lấy biến môi trường
openai_api_key = os.getenv("OPENAI_API_KEY")
port = int(os.getenv("PORT", 8000))
telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")

# Khởi tạo app Flask hoặc OpenAI client với các biến này
# Ví dụ:
from flask import Flask

app = Flask(__name__)
# Khởi tạo OpenAI client với openai_api_key

# Cài đặt logger
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
MEMORY_FOLDER = "memory"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(MEMORY_FOLDER, exist_ok=True)

app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB limit
ALLOWED_EXTENSIONS = {'docx', 'pdf', 'xlsx', 'html'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Module lưu trữ memory local ---
def get_memory_filepath(session_id):
    return os.path.join(MEMORY_FOLDER, f"memory_{session_id}.json")

def load_memory(session_id):
    path = get_memory_filepath(session_id)
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_memory(session_id, memory_data):
    path = get_memory_filepath(session_id)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(memory_data, f, ensure_ascii=False, indent=2)

# --- Module chuẩn hóa và tách văn bản ---
def parse_tiet_in_khoan(text_khoan):
    tiet_list = []
    lines = text_khoan.strip().split('\n')
    current_tiet = None
    current_noidung = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if re.match(r'^[-+]\s+', line):
            if current_tiet or current_noidung:
                tiet_list.append({'tiet': current_tiet if current_tiet else '-', 
                                  'noidung': '\n'.join(current_noidung).strip()})
            current_tiet = '-'
            current_noidung = [re.sub(r'^[-+]\s+', '', line)]
            continue
        m = re.match(r'^([a-zA-Z])\)\s*(.*)', line)
        if m:
            if current_tiet or current_noidung:
                tiet_list.append({'tiet': current_tiet if current_tiet else m.group(1), 
                                  'noidung': '\n'.join(current_noidung).strip()})
            current_tiet = m.group(1)
            current_noidung = [m.group(2).strip()]
            continue
        current_noidung.append(line)

    if current_tiet or current_noidung:
        tiet_list.append({'tiet': current_tiet if current_tiet else '', 
                          'noidung': '\n'.join(current_noidung).strip()})
    return tiet_list


def parse_docx_file(filepath, use_ai=False):
    import docx
    doc = docx.Document(filepath)
    records = []

    chapter_pat = re.compile(r'^CHƯƠNG\s+([IVXLCDM]+)(?:\s*-\s*(.*))?$', re.IGNORECASE)
    article_pat = re.compile(r'^Điều\s+(\d+)[\.\:]?(.*)$', re.IGNORECASE)
    clause_pat = re.compile(r'^(Khoản)\s+(\d+)[\.\:]?(.*)$', re.IGNORECASE)

    current_chap = ""
    current_art = ""
    current_clause = ""
    buffer_clause = ""

    def flush_clause():
        nonlocal buffer_clause
        if buffer_clause.strip():
            tiet_list = parse_tiet_in_khoan(buffer_clause)
            for tiet in tiet_list:
                records.append({
                    "Chương": current_chap,
                    "Điều": current_art,
                    "Khoản": current_clause,
                    "Tiết": tiet['tiet'],
                    "Nội dung": tiet['noidung'],
                })
            buffer_clause = ""

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        m = chapter_pat.match(text)
        if m:
            flush_clause()
            current_chap = "Chương " + m.group(1)
            current_art = current_clause = ""
            continue

        m = article_pat.match(text)
        if m:
            flush_clause()
            current_art = "Điều " + m.group(1)
            current_clause = ""
            buffer_clause = m.group(2).strip() if m.group(2) else ""
            continue

        m = clause_pat.match(text)
        if m:
            flush_clause()
            current_clause = "Khoản " + m.group(2)
            buffer_clause = m.group(3).strip() if m.group(3) else ""
            continue

        if buffer_clause:
            buffer_clause += "\n" + text
        else:
            buffer_clause = text
    flush_clause()
    return records


def extract_text_pdf(filepath):
    import pdfplumber
    texts = []
    try:
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                texts.append(page.extract_text() or "")
        return "\n".join(texts)
    except Exception as e:
        print(f"Lỗi đọc PDF: {e}")
        return ""


def extract_text_html(url_or_content):
    import requests
    from bs4 import BeautifulSoup
    if url_or_content.lower().startswith("http"):
        try:
            r = requests.get(url_or_content)
            r.raise_for_status()
            content = r.text
        except Exception as e:
            print(f"Lỗi lấy HTML: {e}")
            return ""
    else:
        content = url_or_content
    soup = BeautifulSoup(content, 'html.parser')
    content_div = soup.find("div", class_="content") or soup.body
    if not content_div:
        return ""
    paras = content_div.find_all(['p','div'])
    texts = [p.get_text(separator=' ', strip=True) for p in paras if p.get_text(strip=True)]
    return "\n".join(texts)

# --- Module Law Retriever tích hợp Embedding và FAISS Index ---
class LawRetriever:
    def __init__(self, api_key, embedding_dim=1536):
        self.client = OpenAI(api_key=api_key)
        self.embedding_dim = embedding_dim
        self.index = None
        self.records = []

    def load_data(self, records):
        self.records = records

    def create_index(self):
        # ... (tạo embedding và build index tương tự sample đã gửi)...

    def query(self, question, top_k=5):
        # ... (tra cứu embedded search)...

    def format_results(self, results):
        # ... (format kết quả để tạo prompt)...

# Khởi tạo retriever
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "your_api_key_here"
law_retriever = LawRetriever(OPENAI_API_KEY)
INDEX_HTML = """ 
<!DOCTYPE html>
<html>
<head>
  <title>App Tra cứu Luật AI</title>
  <style>
    /* Giao diện đẹp, responsive, logo, khu vực chat và cấu hình */
    /* Các style như mình gửi mẫu trước */
  </style>
</head>
<body>
  <div id="container">
    <img src="https://cdn-icons-png.flaticon.com/512/2972/2972315.png" alt="Logo" id="logo">
    <h1>Tra cứu văn bản pháp luật AI</h1>

    <!-- Khung cấu hình nguồn dữ liệu -->
    <div>
      <h2>Cấu hình nguồn dữ liệu</h2>
      <label>Đường dẫn folder Google Drive hoặc ổ cứng:</label>
      <input type="text" id="localData" />
      <br/>
      <label>Danh sách URL Web tin cậy (mỗi URL một dòng):</label>
      <textarea id="webUrls" rows="5"></textarea>
      <br/>
      <button onclick="saveConfig()">Lưu cấu hình</button>
    </div>

    <!-- Khung chat -->
    <div>
      <h2>Đặt câu hỏi</h2>
      <textarea id="question" rows="4" placeholder="Nhập câu hỏi tại đây..."></textarea><br/>
      <button onclick="sendQuestion()">Gửi câu hỏi</button>
    </div>

    <div id="chatHistory" style="border: 1px solid #ccc; height: 250px; overflow-y: auto; margin-top:15px; padding:8px;"></div>

    <div id="message"></div>
  </div>

<script>
  async function saveConfig() {
    const localData = document.getElementById('localData').value.trim();
    const webUrls = document.getElementById('webUrls').value.trim().split('\\n').map(u => u.trim()).filter(u => u);
    const resp = await fetch('/config', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({localData, webUrls})
    });
    const data = await resp.json();
    document.getElementById('message').textContent = data.status || data.error;
  }

  async function sendQuestion() {
    const question = document.getElementById('question').value.trim();
    if (!question) {
      alert("Vui lòng nhập câu hỏi!");
      return;
    }
    addChat("Bạn", question);
    document.getElementById('question').value = '';
    addChat("AI", "Đang xử lý, vui lòng đợi...");

    const resp = await fetch('/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question})
    });
    const data = await resp.json();
    if(data.answer) {
      updateLastChat("AI", data.answer);
    } else {
      updateLastChat("AI", "Không có phản hồi.");
    }
  }

  function addChat(speaker, text) {
    const div = document.createElement('div');
    div.innerHTML = `<b>${speaker}:</b> ${text.replace(/\\n/g, '<br/>')}`;
    document.getElementById('chatHistory').appendChild(div);
    document.getElementById('chatHistory').scrollTop = document.getElementById('chatHistory').scrollHeight;
  }
  function updateLastChat(speaker, text) {
    const chat = document.getElementById('chatHistory');
    const last = chat.lastChild;
    if(last) {
      last.innerHTML = `<b>${speaker}:</b> ${text.replace(/\\n/g, '<br/>')}`;
    }
  }

  // Load cấu hình lúc mở trang
  window.onload = async () => {
    const resp = await fetch('/config');
    if(resp.ok) {
      const data = await resp.json();
      document.getElementById('localData').value = data.localData || '';
      document.getElementById('webUrls').value = (data.webUrls || []).join('\\n');
    }
  }
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

@app.route('/config', methods=['GET', 'POST'])
def config_route():
    if request.method == 'POST':
        data = request.get_json()
        localData = data.get('localData', '').strip()
        webUrls = data.get('webUrls', [])
        if not isinstance(webUrls, list):
            return jsonify({"error": "Danh sách URL không hợp lệ"}), 400
        config_data = {"localData": localData, "webUrls": webUrls}
        with open('config.json', 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)
        return jsonify({"status": "Lưu cấu hình thành công"})
    else:
        if os.path.exists('config.json'):
            with open('config.json', 'r', encoding='utf-8') as f:
                return jsonify(json.load(f))
        else:
            return jsonify({"localData": "", "webUrls": []})

@app.route('/chat', methods=['POST'])
def chat_route():
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return
@app.route('/chat', methods=['POST'])
def chat_route():
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Vui lòng nhập câu hỏi"}), 400

    try:
        # Load cấu hình
        if os.path.exists('config.json'):
            with open('config.json', 'r', encoding='utf-8') as f:
                config_data = json.load(f)
        else:
            config_data = {"localData": "", "webUrls": []}

        # Chuẩn bị dữ liệu từ config (vd: load từ localData hoặc webUrls)
        records = law_retriever.prepare_data(question)
        if records:
            law_retriever.load_data(records)
            law_retriever.create_embeddings()
        else:
            records = []

        # Truy vấn dữ liệu liên quan
        results = law_retriever.query(question) if records else []

        # Tạo prompt để gọi OpenAI
        context = law_retriever.format_for_prompt(results)
        prompt = f"Dựa trên các đoạn văn bản luật sau đây:\n{context}\n\nHỏi: {question}\nTrả lời chi tiết và trích dẫn."

        # Gọi OpenAI ChatCompletion
        response = law_retriever.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=2000
        )
        answer = response.choices[0].message.content
        return jsonify({"answer": answer})

    except Exception as e:
        logging.error(f"Lỗi khi xử lý chat: {e}")
        return jsonify({"error": "Đã xảy ra lỗi khi xử lý câu hỏi. Vui lòng thử lại sau."})


if __name__ == '__main__':
    import threading
    import webbrowser
    import time

    port = int(os.environ.get('PORT', 8000))
    url = f'http://localhost:{port}/'

    def open_browser():
        time.sleep(1)
        webbrowser.open(url)

    threading.Thread(target=open_browser).start()
    app.run(host='0.0.0.0', port=port, debug=True)