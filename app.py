import os
import threading
import time
import queue
import hashlib
import json
import logging
import signal
import sys

from flask import Flask, request, jsonify, render_template_string
import openai
from cryptography.fernet import Fernet

app = Flask(__name__)

# --- Logger ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

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
    logging.info("Tạo mới fernet key")
    return key

fernet_key = load_or_create_fernet_key()
fernet = Fernet(fernet_key)

def save_config(data):
    raw = json.dumps(data).encode()
    enc = fernet.encrypt(raw)
    with open(CONFIG_FILE, 'wb') as f:
        f.write(enc)
    logging.info("Đã lưu config mã hóa")

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

# Token quản lý rate limit
token_usage = {'count': 0, 'timestamp': time.time()}
token_lock = threading.Lock()

cache = {}
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
        logging.error(f"Lỗi gọi OpenAI API: {e}")
        return f"Lỗi API OpenAI: {e}"

def get_default_source_data():
    src_cfg = config.get("tg_default_data_source", {})
    src_type = src_cfg.get("type", "")
    path = src_cfg.get("path", "")
    if src_type == "gdrive":
        return f"[Dữ liệu giả định từ Google Drive folder ID: {path}]"
    elif src_type == "web":
        return f"[Dữ liệu giả định từ web url: {path}]"
    else:
        return "[Chưa cấu hình nguồn dữ liệu mặc định cho bot]"

def api_worker():
    while True:
        item = api_queue.get()
        if item is None:
            logging.info("Worker thread dừng")
            break
        question, model, use_default_source, result = item
        key = hashlib.sha256(
            (question + model + ("default" if use_default_source else "")).encode()
        ).hexdigest()
        if key in cache:
            logging.info("Trả lời từ cache")
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
    # Cung cấp cấu hình hiện hành để hiển thị mặc định form cấu hình
    # Cách đơn giản là embed cấu hình vào javascript hoặc load khi load trang (ở đây để đơn giản không đổ hiện config sẵn)
    return render_template_string(INDEX_HTML)

@app.route('/config', methods=['POST'])
def config_route():
    data = request.get_json()
    if not data or 'openai_api_key' not in data or not data['openai_api_key']:
        return jsonify({'status': 'Bạn phải nhập OpenAI API Key!'}), 400
    # Cập nhật config và lưu
    config.update(data)
    save_config(config)
    logging.info("Cấu hình mới đã được lưu")
    return jsonify({'status': 'Lưu cấu hình thành công!'})

@app.route('/ask', methods=['POST'])
def ask_route():
    data = request.get_json()
    question = data.get('question', '').strip()
    use_default_source = data.get('use_default_source', False)
    if not question:
        return jsonify({'answer': 'Vui lòng nhập câu hỏi!'}), 400
    model = config.get('model', 'gpt-3.5-turbo')
    result = {}
    api_queue.put((question, model, use_default_source, result))
    timeout = 20
    poll_interval = 0.5
    waited = 0
    while waited < timeout:
        if 'answer' in result:
            return jsonify({'answer': result['answer']})
        time.sleep(poll_interval)
        waited += poll_interval
    return jsonify({'answer': 'Hết thời gian chờ xử lý, vui lòng thử lại sau.'}), 504
    
def signal_handler(sig, frame):
    logging.info('Nhận tín hiệu dừng, kết thúc worker thread...')
    api_queue.put(None)  # Đưa None để worker thread nhận và thoát
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logging.info(f"Khởi chạy app trên cổng {port}")
    app.run(host="0.0.0.0", port=port)