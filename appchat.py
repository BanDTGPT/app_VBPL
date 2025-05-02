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
from openai import OpenAI
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
env_api_key = os.environ.get("OPENAI_API_KEY")
env_tg_token = os.environ.get("TELEGRAM_BOT_TOKEN")
if env_api_key:
    config['openai_api_key'] = env_api_key
    updated = True
if env_tg_token:
    config['telegram_bot_token'] = env_tg_token
    updated = True
if updated:
    save_config(config)

api_key_for_client = config.get("openai_api_key")
if not api_key_for_client:
    logging.error("Không tìm thấy OpenAI API Key trong biến môi trường hoặc config! Ứng dụng sẽ không hoạt động đúng.")

# Tạo client OpenAI với api_key lấy từ config hoặc env
client = OpenAI(api_key=api_key_for_client)

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
            token_limit = config.get('token_limit_per_minute', config.get('web_token_limit_per_minute', 30000))
            if token_usage['count'] + estimated_tokens <= token_limit:
                token_usage['count'] += estimated_tokens
                return
        time.sleep(0.5)

def call_openai_api(question, model):
    est_tokens = len(question)//4 + 256
    wait_token_available(est_tokens)
    try:
        rsp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": question}],
            temperature=0.2,
            max_tokens=2000,
        )
        return rsp.choices[0].message.content
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

# Xử lý tín hiệu dừng app an toàn, đặt luôn ở đây (phần 1)
def signal_handler(sig, frame):
    logging.info('Nhận tín hiệu dừng, kết thúc worker thread...')
    api_queue.put(None)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Pháp luật</title>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background: url('https://images.unsplash.com/photo-1509027577630-36f6b166db44?auto=format&fit=crop&w=1470&q=80') no-repeat center center fixed;
            background-size: cover;
            color: #333;
            padding: 20px;
        }
        #container {
            max-width: 900px;
            margin: auto;
            background: rgba(255,255,255,0.95);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        h1, h3 {
            text-align: center;
            font-weight: 700;
            margin-bottom: 15px;
        }
        form {
            margin-bottom: 25px;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 8px;
            background: #fafafa;
        }
        label {
            font-weight: 600;
        }
        input[type="text"], input[type="password"], input[type="number"], select, textarea {
            width: 100%;
            padding: 8px 10px;
            margin: 6px 0 12px 0;
            border: 1px solid #bbb;
            border-radius: 5px;
            font-size: 16px;
            box-sizing: border-box;
            resize: vertical;
        }
        button {
            background-color: #0078D4;
            color: white;
            padding: 10px 22px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
            transition: background-color 0.3s ease;
        }
        button:hover {
            background-color: #005ea1;
        }
        #chatHistory {
            height: 250px;
            overflow-y: auto;
            border: 1px solid #ccc;
            padding: 10px;
            background: #fff;
            border-radius: 5px;
            white-space: pre-wrap;
            font-family: monospace;
            font-size: 14px;
        }
        #question {
            min-height: 70px;
            font-size: 16px;
        }
        .section-title {
            margin-bottom: 10px;
            font-size: 18px;
            font-weight: 700;
            border-bottom: 2px solid #0078D4;
            padding-bottom: 5px;
        }
    </style>
</head>
<body>
<div id="container">
    <img src="https://cdn-icons-png.flaticon.com/512/2972/2972315.png" alt="Logo" style="display:block; margin:auto; width:80px; margin-bottom:15px;">
    <h1>Ứng dụng Pháp luật AI</h1>
    
    <div class="section-title">1. Cấu hình chung</div>
    <form id="commonConfigForm">
        <label for="apiKey">OpenAI API Key:</label>
        <input type="password" id="apiKey" required>
        
        <label for="telegramToken">Telegram Bot Token:</label>
        <input type="password" id="telegramToken">
        
        <button type="submit">Lưu cấu hình chung</button>
    </form>
    
    <div class="section-title">2. Cấu hình Web App</div>
    <form id="webConfigForm">
        <label for="webModel">Model AI:</label>
        <select id="webModel">
            <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
            <option value="gpt-4">GPT-4</option>
            <option value="gpt-4-32k">GPT-4 (32k)</option>
            <option value="gpt-4o-mini">GPT-4 Turbo (Mini)</option>
        </select>
        
        <label for="webTokenLimit">Giới hạn token / phút:</label>
        <input type="number" id="webTokenLimit" value="30000" min="1000" max="120000">
        
        <label for="webDataSource">Nguồn dữ liệu Web (ví dụ: Google Drive folder/URL web):</label>
        <input type="text" id="webDataSource">
        
        <button type="submit">Lưu cấu hình Web</button>
    </form>
    
    <div class="section-title">3. Cấu hình Telegram Bot</div>
    <form id="botConfigForm">
        <label for="botSourceType">Loại nguồn dữ liệu bot:</label>
        <select id="botSourceType">
            <option value="gdrive">Google Drive</option>
            <option value="web">Web URL</option>
            <option value="">Không có</option>
        </select>
        
        <label for="botSourcePath">Path (folder ID hoặc URL):</label>
        <input type="text" id="botSourcePath">
        
        <button type="submit">Lưu cấu hình Telegram Bot</button>
    </form>
    
    <hr>
    
    <div class="section-title">4. Hỏi & Đáp</div>
    <textarea id="question" placeholder="Nhập câu hỏi ở đây"></textarea><br>
    <button onclick="sendQuestion(false)">Gửi câu hỏi (Nguồn Web App)</button>
    <button onclick="sendQuestion(true)">Gửi câu hỏi (Nguồn Telegram Bot)</button>
    
    <h3>Lịch sử hỏi đáp</h3>
    <div id="chatHistory"></div>
</div>

<script>
const chatHistory = document.getElementById("chatHistory");

// Append tin nhắn hỏi/đáp vào lịch sử
function appendChat(role, message) {
    const entry = document.createElement("div");
    entry.style.marginBottom = "10px";
    entry.innerHTML = `<b>${role}:</b> ${message.replace(/\\n/g, "<br>")}`;
    chatHistory.appendChild(entry);
    chatHistory.scrollTop = chatHistory.scrollHeight; // Cuộn xuống cuối
}

// Lấy config hiện hành từ server, đổ vào form
async function loadConfig() {
    try {
        const resp = await fetch("/get_config");
        const cfg = await resp.json();
        
        document.getElementById("apiKey").value = cfg.openai_api_key || "";
        document.getElementById("telegramToken").value = cfg.telegram_bot_token || "";

        document.getElementById("webModel").value = cfg.web_model || "gpt-3.5-turbo";
        document.getElementById("webTokenLimit").value = cfg.web_token_limit_per_minute || 30000;
        document.getElementById("webDataSource").value = cfg.web_data_source || "";

        document.getElementById("botSourceType").value = (cfg.tg_default_data_source && cfg.tg_default_data_source.type) || "";
        document.getElementById("botSourcePath").value = (cfg.tg_default_data_source && cfg.tg_default_data_source.path) || "";
    } catch (e) {
        console.error("Không thể lấy cấu hình:", e);
    }
}

// Lưu cấu hình chung
document.getElementById("commonConfigForm").onsubmit = async function(e){
    e.preventDefault();
    const data = {
        openai_api_key: document.getElementById("apiKey").value.trim(),
        telegram_bot_token: document.getElementById("telegramToken").value.trim()
    };
    if(!data.openai_api_key){
        alert("OpenAI API Key là bắt buộc!");
        return;
    }
    try {
        const resp = await fetch("/config_common", {
            method: "POST",
            headers: {'Content-Type':'application/json'},
            body: JSON.stringify(data)
        });
        const result = await resp.json();
        alert(result.status);
    } catch(err) {
        alert("Lỗi lưu cấu hình chung");
    }
};

// Lưu cấu hình Web App
document.getElementById("webConfigForm").onsubmit = async function(e){
    e.preventDefault();
    const data = {
        web_model: document.getElementById("webModel").value,
        web_token_limit_per_minute: parseInt(document.getElementById("webTokenLimit").value),
        web_data_source: document.getElementById("webDataSource").value.trim()
    };
    try {
        const resp = await fetch("/config_web", {
            method: "POST",
            headers: {'Content-Type':'application/json'},
            body: JSON.stringify(data)
        });
        const result = await resp.json();
        alert(result.status);
    } catch(err) {
        alert("Lỗi lưu cấu hình Web");
    }
};

// Lưu cấu hình Telegram Bot
document.getElementBy
document.getElementById("botConfigForm").onsubmit = async function(e){
    e.preventDefault();
    const data = {
        tg_default_data_source: {
            type: document.getElementById("botSourceType").value,
            path: document.getElementById("botSourcePath").value.trim()
        }
    };
    try {
        const resp = await fetch("/config_bot", {
            method: "POST",
            headers: {'Content-Type':'application/json'},
            body: JSON.stringify(data)
        });
        const result = await resp.json();
        alert(result.status);
    } catch(err) {
        alert("Lỗi lưu cấu hình Telegram Bot");
    }
};

// Gửi câu hỏi tới server
async function sendQuestion(useDefaultSource){
    let q = document.getElementById("question").value.trim();
    if(!q){
        alert("Nhập câu hỏi!");
        return;
    }
    appendChat("Bạn", q);
    document.getElementById("question").value = "";
    let answerArea = document.getElementById("chatHistory");
    appendChat("AI Pháp luật", "Đang xử lý, vui lòng chờ...");
    try {
        let resp = await fetch("/ask", {
            method: "POST",
            headers: {'Content-Type':'application/json'},
            body: JSON.stringify({question: q, use_default_source: useDefaultSource})
        });
        let data = await resp.json();
        if(data.answer){
            // Xóa dòng chờ
            removeLastAIWaiting();
            appendChat("AI Pháp luật", data.answer);
        } else {
            removeLastAIWaiting();
            appendChat("Lỗi", "Không nhận được câu trả lời.");
        }
    } catch(e){
        removeLastAIWaiting();
        appendChat("Lỗi", "Lỗi kết nối hoặc phản hồi.");
    }
}

// Hàm xóa dòng "đang xử lý"
function removeLastAIWaiting(){
    const nodes = chatHistory.childNodes;
    for(let i=nodes.length-1; i>=0; i--){
        if(nodes[i].innerHTML && nodes[i].innerHTML.includes("Đang xử lý, vui lòng chờ")){
            chatHistory.removeChild(nodes[i]);
            break;
        }
    }
}

// Gọi loadConfig khi trang được tải
window.onload = loadConfig;
</script>
</body>
</html>
"""
@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

@app.route('/get_config', methods=['GET'])
def get_config_route():
    # Trả về config hiện tại cho frontend load
    resp = {
        "openai_api_key": config.get("openai_api_key", ""),
        "telegram_bot_token": config.get("telegram_bot_token", ""),
        "web_model": config.get("web_model", "gpt-3.5-turbo"),
        "web_token_limit_per_minute": config.get("web_token_limit_per_minute", 30000),
        "web_data_source": config.get("web_data_source", ""),
        "tg_default_data_source": config.get("tg_default_data_source", {"type": "", "path": ""}),
    }
    return jsonify(resp)

@app.route('/config_common', methods=['POST'])
def config_common_route():
    data = request.get_json()
    if not data or 'openai_api_key' not in data or not data['openai_api_key']:
        return jsonify({'status': 'Bạn phải nhập OpenAI API Key!'}), 400
    config['openai_api_key'] = data['openai_api_key']
    config['telegram_bot_token'] = data.get('telegram_bot_token', '')
    save_config(config)
    logging.info("Cấu hình chung đã được cập nhật")
    return jsonify({'status': 'Đã lưu cấu hình chung'})

@app.route('/config_web', methods=['POST'])
def config_web_route():
    data = request.get_json()
    if not data:
        return jsonify({'status': 'Dữ liệu không hợp lệ'}), 400
    config['web_model'] = data.get('web_model', config.get('web_model', 'gpt-3.5-turbo'))
    config['web_token_limit_per_minute'] = data.get('web_token_limit_per_minute', config.get('web_token_limit_per_minute', 30000))
    config['web_data_source'] = data.get('web_data_source', config.get('web_data_source', ''))
    save_config(config)
    logging.info("Cấu hình Web App đã được cập nhật")
    return jsonify({'status': 'Đã lưu cấu hình Web App'})

@app.route('/config_bot', methods=['POST'])
def config_bot_route():
    data = request.get_json()
    if not data or 'tg_default_data_source' not in data:
        return jsonify({'status': 'Dữ liệu không hợp lệ'}), 400
    config['tg_default_data_source'] = data.get('tg_default_data_source', {})
    save_config(config)
    logging.info("Cấu hình Telegram Bot đã được cập nhật")
    return jsonify({'status': 'Đã lưu cấu hình Telegram Bot'})

@app.route('/ask', methods=['POST'])
def ask_route():
    data = request.get_json()
    question = (data.get('question', '') or '').strip()
    use_default_source = data.get('use_default_source', False)
    if not question:
        return jsonify({'answer': 'Vui lòng nhập câu hỏi!'}), 400
    model = config.get("web_model", "gpt-3.5-turbo") if not use_default_source else config.get("web_model", "gpt-3.5-turbo")
    # Dùng model Web App cho hết, bạn có thể tùy chỉnh nếu muốn dùng model riêng Telegram Bot
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

# Xử lý tín hiệu tắt app an toàn
def signal_handler(sig, frame):
    logging.info('Nhận tín hiệu dừng, kết thúc worker thread...')
    api_queue.put(None)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logging.info(f"Khởi chạy app trên cổng {port}")
    app.run(host="0.0.0.0", port=port)