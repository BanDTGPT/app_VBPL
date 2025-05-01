import os
import threading
import time
import queue
import hashlib
import json

from flask import Flask, request, jsonify, render_template_string
import openai
from cryptography.fernet import Fernet

app = Flask(__name__)

# --- Mã hóa config ---
FERNET_KEY_FILE = 'fernet.key'
CONFIG_FILE = 'config.enc'

def load_or_create_fernet_key():
    if os.path.exists(FERNET_KEY_FILE):
        with open(FERNET_KEY_FILE, 'rb') as f:
            return f.read()
    key = Fernet.generate_key()
    with open(FERNET_KEY_FILE, 'wb') as f:
        f.write(key)
    return key

fernet_key = load_or_create_fernet_key()
fernet = Fernet(fernet_key)

def save_config(data):
    raw = json.dumps(data).encode()
    enc = fernet.encrypt(raw)
    with open(CONFIG_FILE, 'wb') as f:
        f.write(enc)

def load_config():
    if not os.path.exists(CONFIG_FILE):
        return {}
    with open(CONFIG_FILE, 'rb') as f:
        enc = f.read()
    raw = fernet.decrypt(enc)
    return json.loads(raw)

config = load_config()

# Load API keys từ biến môi trường nếu có (ưu tiên env vars)
updated = False
if os.environ.get("OPENAI_API_KEY"):
    config['openai_api_key'] = os.environ["OPENAI_API_KEY"]
    updated = True
if os.environ.get("TELEGRAM_BOT_TOKEN"):
    config['telegram_bot_token'] = os.environ["TELEGRAM_BOT_TOKEN"]
    updated = True
if updated:
    save_config(config)

# Token rate limit quản lý
token_usage = {'count': 0, 'timestamp': time.time()}
token_lock = threading.Lock()

# Cache câu hỏi trả lời
cache = {}

# Queue lưu request chat
api_queue = queue.Queue()

def wait_token_available(estimated_tokens):
    while True:
        with token_lock:
            now = time.time()
            if now - token_usage['timestamp'] > 60:
                token_usage['timestamp'] = now
                token_usage['count'] = 0
            token_limit = config.get('token_limit_per_minute', 30000)
            if token_usage['count'] + estimated_tokens <= token_limit:
                token_usage['count'] += estimated_tokens
                return
        time.sleep(0.5)

def call_openai_api(question, model):
    est_tokens = len(question)//4 + 256
    wait_token_available(est_tokens)
    openai.api_key = config.get('openai_api_key', '')
    try:
        rsp = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": question}],
            temperature=0.2,
            max_tokens=2000,
        )
        return rsp['choices'][0]['message']['content']
    except Exception as e:
        return f"Lỗi API OpenAI: {e}"

def get_default_source_data():
    src_cfg = config.get("tg_default_data_source", {})
    src_type = src_cfg.get("type", "")
    path = src_cfg.get("path", "")
    if src_type == "gdrive":
        # TODO: tích hợp thực lấy dữ liệu Google Drive
        return f"[Dữ liệu giả định từ Google Drive folder ID: {path}]"
    elif src_type == "web":
        # TODO: lấy dữ liệu web thật
        return f"[Dữ liệu giả định từ web url: {path}]"
    else:
        return "[Chưa cấu hình nguồn dữ liệu mặc định cho bot]"

def api_worker():
    while True:
        item = api_queue.get()
        if item is None:
            break
        question, model, use_default_source, result = item
        key = hashlib.sha256(
            (question + model + ("default" if use_default_source else "")).encode()
        ).hexdigest()
        if key in cache:
            result["answer"] = cache[key]
        else:
            context = ""
            if use_default_source:
                context = get_default_source_data()
            prompt = f"Dựa trên dữ liệu dưới đây:\n{context}\n\nHỏi: {question}\nTrả lời chi tiết và trích dẫn."
            answer = call_openai_api(prompt, model)
            cache[key] = answer
            result["answer"] = answer
        api_queue.task_done()

worker_thread = threading.Thread(target=api_worker, daemon=True)
worker_thread.start()

INDEX_HTML = """
<!DOCTYPE html>
<html>
<head><title>AI Pháp luật</title></head>
<body style="font-family: Arial">
<h1>Ứng dụng Pháp luật AI</h1>
<h3>Cấu hình API & Model</h3>
<form id="configForm">
    OpenAI API Key:<br>
    <input type="password" style="width:400px" id="apiKey" required><br>
    Telegram Bot Token:<br>
    <input type="password" style="width:400px" id="telegramToken"><br>
    Chọn model:<br>
    <select id="model">
        <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
        <option value="gpt-4">GPT-4</option>
        <option value="gpt-4-32k">GPT-4 (32k)</option>
        <option value="gpt-4o-mini">GPT-4 Turbo (Mini)</option>
    </select><br>
    Giới hạn token / phút:<br>
    <input type="number" id="tokenLimit" value="30000"><br><br>
    <h4>Cấu hình nguồn dữ liệu mặc định cho Telegram Bot</h4>
    Loại nguồn:
    <select id="tgSourceType">
        <option value="gdrive">Google Drive</option>
        <option value="web">Web URL</option>
        <option value="">None</option>
    </select><br>
    Path (folder ID hoặc URL):<br>
    <input type="text" id="tgSourcePath" style="width:400px"><br><br>
    <button type="submit">Lưu cấu hình</button>
</form>
<hr>
<h3>Đặt câu hỏi (Web)</h3>
<textarea id="question" rows="4" cols="80" placeholder="Nhập câu hỏi ở đây"></textarea><br>
<button onclick="sendQuestion(false)">Gửi câu hỏi (Chọn nguồn dữ liệu)</button>
<button onclick="sendQuestion(true)">Gửi câu hỏi (Mặc định Telegram Bot)</button>
<h3>Trả lời:</h3>
<pre id="answer" style="white-space: pre-wrap; border:1px solid #ccc; width:90%; height:200px;"></pre>

<script>
document.getElementById('configForm').onsubmit = async function(e){
    e.preventDefault();
    let apiKey = document.getElementById('apiKey').value.trim();
    if (!apiKey) { alert("Vui lòng nhập OpenAI API Key!"); return; }
    let tgToken = document.getElementById('telegramToken').value.trim();
    let model = document.getElementById('model').value;
    let tokenLimit = parseInt(document.getElementById('tokenLimit').value);
    let tgSourceType = document.getElementById('tgSourceType').value;
    let tgSourcePath = document.getElementById('tgSourcePath').value.trim();

    let resp = await fetch('/config', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({
            openai_api_key: apiKey,
            telegram_bot_token: tgToken,
            model: model,
            token_limit_per_minute: tokenLimit,
            tg_default_data_source: {type: tgSourceType, path: tgSourcePath}
        })
    });
    let data = await resp.json();
    alert(data.status || "Đã lưu cấu hình!");
}

async function sendQuestion(useDefaultSource){
    let q = document.getElementById('question').value.trim();
    if(!q){
        alert("Nhập câu hỏi!");
        return;
    }
    document.getElementById('answer').textContent = "Đang xử lý, vui lòng chờ...";
    let resp = await fetch('/ask', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({question: q, use_default_source: useDefaultSource})
    });
    let data = await resp.json();
    if(data.answer){
        document.getElementById('answer').textContent = data.answer;
    }else{
        document.getElementById('answer').textContent = "Lỗi: không nhận được câu trả lời.";
    }
}
</script>
</body></html>
"""

@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

@app.route('/config', methods=['POST'])
def config_route():
    data = request.get_json()
    if not data or 'openai_api_key' not in data or not data
