import os
import json
import logging
import numpy as np
import faiss
from openai import OpenAI
from google.oauth2 import service_account
from googleapiclient.discovery import build
import io
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload

logging.basicConfig(level=logging.INFO)

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
                logging.error(f"Embedding lỗi: {e}")

        self.records = valid_records
        if not vectors:
            logging.error("Không có vector embedding hợp lệ.")
            self.index = None
            return

        matrix = np.vstack(vectors)
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(matrix)
        logging.info(f"Tạo chỉ mục embedding với {len(valid_records)} records.")

    def query(self, question, top_k=5):
        if not self.index or not self.records:
            raise RuntimeError("Chưa tạo index hoặc chưa load dữ liệu.")
        try:
            emb_rsp = self.client.embeddings.create(input=question, model="text-embedding-ada-002")
            q_emb = np.array(emb_rsp.data[0].embedding, dtype=np.float32).reshape(1, -1)
        except Exception as e:
            logging.error(f"Lỗi tạo embedding câu hỏi: {e}")
            return []

        D, I = self.index.search(q_emb, top_k)
        results = []
        for idx in I[0]:
            if idx < len(self.records):
                results.append(self.records[idx])
        return results

    def format_results(self, results):
        formatted = []
        for r in results:
            ref = f"{r.get('type', '')} {r.get('number', '')} ({r.get('url', '')})"
            text = r.get('text', '')
            formatted.append(f"{ref}\n{text}")
        return "\n\n---\n\n".join(formatted)


class GoogleDriveSync:
    def __init__(self, service_account_file, folder_id):
        self.folder_id = folder_id
        self.scopes = ['https://www.googleapis.com/auth/drive.file']
        self.creds = service_account.Credentials.from_service_account_file(
            service_account_file, scopes=self.scopes)
        self.service = build('drive', 'v3', credentials=self.creds)

    def upload_file(self, filename, filepath):
        file_metadata = {'name': filename, 'parents': [self.folder_id]}
        media = MediaIoBaseUpload(io.FileIO(filepath, 'rb'), mimetype='application/json')
        file = self.service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        logging.info(f"Uploaded file {filename} to Google Drive with ID: {file.get('id')}")
        return file.get('id')

    def download_file(self, file_id, local_path):
        request = self.service.files().get_media(fileId=file_id)
        fh = io.FileIO(local_path, 'wb')
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            logging.info(f"Download {int(status.progress() * 100)}% completed.")
        fh.close()


class MemoryManager:
    def __init__(self, local_folder='memory', drive_sync=None):
        self.local_folder = local_folder
        self.drive_sync = drive_sync
        os.makedirs(self.local_folder, exist_ok=True)

    def get_local_path(self, session_id):
        return os.path.join(self.local_folder, f"memory_{session_id}.json")

    def load_memory(self, session_id):
        path = self.get_local_path(session_id)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def save_memory(self, session_id, memory):
        path = self.get_local_path(session_id)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(memory, f, ensure_ascii=False, indent=2)
        if self.drive_sync:
            self.drive_sync.upload_file(os.path.basename(path), path)

# Khởi tạo với config biến môi trường
SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")

drive_sync = GoogleDriveSync(SERVICE_ACCOUNT_FILE, DRIVE_FOLDER_ID) if SERVICE_ACCOUNT_FILE and DRIVE_FOLDER_ID else None

memory_manager = MemoryManager(drive_sync=drive_sync)

# Exports đầy đủ LawRetriever và MemoryManager để app web và bot dùng chung:
__all__ = ['LawRetriever', 'memory_manager', 'law_retriever']


from flask import render_template_string, session
import uuid

INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>Tra cứu Pháp luật AI</title>
  <style>
    body { font-family: Arial; margin:40px; background:#fafafa; }
    #container { max-width: 700px; margin:auto; background:#fff; padding:20px 30px; border-radius:12px; box-shadow:0 5px 15px rgba(0,0,0,0.1); }
    textarea { width: 100%; height: 90px; font-size:16px; margin-bottom:15px; border-radius:8px; border:1px solid #ccc; padding:10px; resize: vertical;}
    button { background:#0078D4; border:none; color:#fff; padding:12px 26px; font-size:18px; border-radius:8px; cursor:pointer; font-weight:bold;}
    button:hover { background:#005ea2; }
    #answer { white-space: pre-wrap; background:#eaeaea; border-radius:8px; padding:15px; min-height:200px; font-size:16px; margin-top:20px; border:1px solid #aaa;}
    h1 { text-align:center; font-weight:700; margin-bottom:25px; color:#333;}
  </style>
</head>
<body>
  <div id="container">
    <h1>Tra cứu Pháp luật AI</h1>
    <textarea id="question" placeholder="Nhập câu hỏi pháp luật..."></textarea>
    <br>
    <button onclick="sendQuestion()">Gửi câu hỏi</button>
    <div id="answer"></div>
  </div>
<script>
async function sendQuestion() {
  const q = document.getElementById('question').value.trim();
  if (!q) { alert('Vui lòng nhập câu hỏi'); return; }
  const ans = document.getElementById('answer');
  ans.textContent = 'Đang xử lý, xin chờ...';
  try {
    let resp = await fetch('/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question: q})
    });
    let data = await resp.json();
    if (data.answer) {
      ans.innerHTML = data.answer.replace(/\n/g,'<br>');
    } else if (data.error) {
      ans.textContent = 'Lỗi: ' + data.error;
    }
  } catch(e) {
    ans.textContent = 'Lỗi kết nối: ' + e.message;
  }
}
</script>
</body>
</html>
"""

@app.before_request
def prepare_session():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())

@app.route('/', methods=['GET'])
def index_page():
    return render_template_string(INDEX_HTML)

@app.route('/chat', methods=['POST'])
def chat_route():
    data = request.get_json()
    question = data.get('question','').strip()
    if not question:
        return jsonify({"error": "Vui lòng nhập câu hỏi"}), 400

    session_id = session.get('session_id')
    memory = memory_manager.load_memory(session_id)

    try:
        results = law_retriever.query(question, top_k=5)
        context = law_retriever.format_results(results)
        history_str = '\n'.join([f"Bạn: {m['user']}\nAI: {m['ai']}" for m in memory[-10:]])
        prompt = f"{history_str}\n\nDựa trên các đoạn luật dưới đây:\n{context}\n\nHỏi: {question}\nTrả lời chi tiết."

        resp = law_retriever.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content": prompt}],
            temperature=0.2,
            max_tokens=2000
        )
        answer = resp.choices[0].message.content

        memory.append({"user": question, "ai": answer})
        memory_manager.save_memory(session_id, memory)

        return jsonify({"answer": answer})
    except Exception as err:
        logging.error(f"Lỗi chat: {err}")
        return jsonify({"error": "Xảy ra lỗi trong quá trình xử lý. Vui lòng thử lại."})

import threading
import asyncio
import webbrowser
import time

# Hàm async chạy bot Telegram
async def main():
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    # Các handler bot
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    # Chạy bot Telegram
    await application.run_polling()

# Hàm chạy bot trong thread với event loop riêng
def run_bot():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    url = f"http://localhost:{port}/"

    # Tự động mở trình duyệt web sau 1 giây server chạy
    def open_browser():
        time.sleep(1)
        webbrowser.open(url)

    threading.Thread(target=open_browser, daemon=True).start()
    threading.Thread(target=run_bot, daemon=True).start()

    # Chạy Flask app chính trên luồng chính
    app.run(host="0.0.0.0", port=port, debug=True)