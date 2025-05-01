import os
import re
import json
import tempfile
import logging
from flask import Flask, request, jsonify, render_template_string, send_from_directory
from werkzeug.utils import secure_filename

import docx
import pandas as pd
import pdfplumber
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MB max upload

ALLOWED_EXTENSIONS = {'docx', 'pdf', 'xlsx', 'html'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ----------- CÁC HÀM XỬ LÝ FILE --------------

def extract_text_pdf(path):
    texts = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                texts.append(page.extract_text() or "")
        return "\n".join(texts)
    except Exception as e:
        logging.error(f"Error reading PDF {path}: {e}")
        return ""

def extract_text_html(url):
    try:
        r = requests.get(url)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')
        # Lấy toàn bộ text trong body hoặc div.content tùy website
        content_div = soup.find('div', class_='content') or soup.body
        if not content_div:
            return ""
        return "\n".join(p.get_text(separator=' ', strip=True) for p in content_div.find_all(['p', 'div']))
    except Exception as e:
        logging.error(f"Error fetching html {url}: {e}")
        return ""

def parse_tiet_in_khoan(text_khoan):
    tiet_list = []
    lines = text_khoan.strip().split('\n')
    current_tiet = None
    current_noidung = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if re.match(r'^[-+]\s+', line):  # gạch đầu dòng
            if current_tiet or current_noidung:
                tiet_list.append({'tiet': current_tiet if current_tiet else '-', 
                                  'noidung': '\n'.join(current_noidung).strip()})
            current_tiet = '-'
            current_noidung = [re.sub(r'^[-+]\s+', '', line)]
            continue
        m = re.match(r'^([a-zA-Z])\)\s*(.*)', line)  # a), b), ...
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

def parse_docx_text(text_block):
    """
    Tách text block theo chương, điều, khoản, tiết theo quy tắc.
    text_block là một chuỗi nhiều dòng văn bản, ví dụ trích xuất từ DOCX hoặc PDF.
    """
    # Regex pattern tương tự mô tả ở trên bạn có thể chỉnh sửa
    chapter_pat = re.compile(r'^CHƯƠNG\s+([IVXLCDM]+)(?:\s*-\s*(.*))?$', re.IGNORECASE)
    article_pat = re.compile(r'^Điều\s+(\d+)[\.\:]?\s*(.*)$', re.IGNORECASE)
    clause_pat = re.compile(r'^(Khoản)\s+(\d+)[\.\:]?\s*(.*)$', re.IGNORECASE)

    current_chap = ""
    current_art = ""
    current_clause = ""
    results = []
    lines = text_block.split('\n')
    buffer_clause_text = ""

    def flush_clause_text():
        nonlocal buffer_clause_text
        if buffer_clause_text.strip():
            tiet_list = parse_tiet_in_khoan(buffer_clause_text)
            for tiet in tiet_list:
                results.append({
                    "Chương": current_chap,
                    "Điều": current_art,
                    "Khoản": current_clause,
                    "Tiết": tiet["tiet"],
                    "Nội dung": tiet["noidung"]
                })
            buffer_clause_text = ""

    for line in lines:
        line = line.strip()
        if not line:
            continue
        m_chap = chapter_pat.match(line)
        if m_chap:
            flush_clause_text()
            current_chap = "Chương " + m_chap.group(1)
            current_art = ""
            current_clause = ""
            continue
        m_art = article_pat.match(line)
        if m_art:
            flush_clause_text()
            current_art = f"Điều {m_art.group(1)}"
            current_clause = ""
            # Nếu có phần nội dung sau số điều, thêm vào clause text
            if m_art.group(2).strip():
                buffer_clause_text = m_art.group(2).strip()
            else:
                buffer_clause_text = ""
            continue
        m_clause = clause_pat.match(line)
        if m_clause:
            flush_clause_text()
            current_clause = f"Khoản {m_clause.group(2)}"
            # Nội dung trong khoản
            buffer_clause_text = m_clause.group(3).strip() if m_clause.group(3) else ""
            continue
        # Dòng bình thường nối vào clause hiện tại
        if buffer_clause_text:
            buffer_clause_text += "\n" + line
        else:
            buffer_clause_text = line
    flush_clause_text()
    return results
from flask import redirect, url_for, send_from_directory

@app.route('/', methods=['GET'])
def index():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>App Đọc Lọc Văn Bản Pháp Luật</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 40px; background: #f6f8fa;
            }
            .container {
                max-width: 700px; margin: auto;
                background: white; padding: 20px; border-radius: 8px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.15);
            }
            h2 { text-align: center; margin-bottom: 20px;}
            input[type="file"] {
                width: 100%; padding: 10px;
                border: 1px solid #ccc; border-radius: 5px;
                background: #f9f9f9;
            }
            label {font-weight: bold; display: block; margin-top:15px;}
            button {
                margin-top: 20px; padding: 10px 20px; font-size: 16px;
                background: #0078D4; color: white; border:none; border-radius: 5px;
                cursor: pointer;
            }
            button:hover {background:#005a9e;}
            .result-links { margin-top: 20px;}
            .result-links a { display: block; margin-bottom: 10px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Upload nhiều file văn bản pháp luật</h2>
            <form action="/upload_files" method="post" enctype="multipart/form-data">
                <label>Chọn file (docx, pdf, xlsx, html):</label>
                <input type="file" name="files" multiple required>
                <label><input type="checkbox" name="use_ai_support"> Phân tích nâng cao bằng AI (dự phòng)</label>
                <button type="submit">Tải lên và phân tích</button>
            </form>
            <div class="result-links">
                %s
            </div>
        </div>
    </body>
    </html>
    """
    # Lấy danh sách file kết quả đã lưu để hiện liên kết download
    files = []
    if os.path.exists(RESULT_FOLDER):
        files = os.listdir(RESULT_FOLDER)
    links = ""
    for f in files:
        links += f'<a href="/download/{f}">{f}</a>'
    return html % links

@app.route('/upload_files', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return "Không có file được gửi.", 400
    files = request.files.getlist('files')
    use_ai = 'use_ai_support' in request.form

    result_links = []
    for file in files:
        if file.filename == '':
            continue
        if not allowed_file(file.filename):
            return f"File không được hỗ trợ: {file.filename}", 400
        filename = secure_filename(file.filename)
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(save_path)

        extension = filename.rsplit('.', 1)[1].lower()

        records = []
        if extension == 'docx':
            records = parse_docx_file(save_path, use_ai=use_ai)
        elif extension == 'pdf':
            text = extract_text_pdf(save_path)
            records = parse_docx_text(text)
        elif extension == 'xlsx':
            # Đọc XLSX trực tiếp vào records rồi kiểm tra cấu trúc
            df = pd.read_excel(save_path)
            # Kiểm tra cột bắt buộc để làm file
            expected_cols = {'Chương','Điều','Khoản','Tiết','Nội dung'}
            if expected_cols.issubset(set(df.columns)):
                records = df.to_dict(orient='records')
            else:
                return f"File {filename} thiếu cột bắt buộc.", 400
        elif extension == 'html':
            # Lấy url từ tên file (hoặc bổ sung trường hợp bạn upload link HTML)
            url = file.filename  # Giả sử file tên là url.html -> lấy url từ tên file user nhập khác được
            html_text = extract_text_html(url)
            records = parse_docx_text(html_text)

        else:
            return f"Không hỗ trợ định dạng file {filename}", 400

        base_name = os.path.splitext(filename)[0]
        out_xlsx = os.path.join(RESULT_FOLDER, f"{base_name}_processed.xlsx")
        out_json = os.path.join(RESULT_FOLDER, f"{base_name}_processed.json")

        # Xuất ra xlsx
        df_out = pd.DataFrame(records)
        df_out.to_excel(out_xlsx, index=False, encoding='utf-8')

        # Xuất ra json
        with open(out_json, 'w', encoding='utf-8') as fjson:
            json.dump(records, fjson, ensure_ascii=False, indent=4)

        result_links.append(out_xlsx)
        result_links.append(out_json)

    # Trả về trang HTML có link tải file
    links_html = "<h3>File kết quả:</h3>"
    for f in result_links:
        fname = os.path.basename(f)
        links_html += f'<a href="/download/{fname}" target="_blank">{fname}</a><br>'
    return """
    <html><body>
    %s
    <br><a href="/">Quay lại trang chính</a>
    </body></html>
    """ % links_html

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(RESULT_FOLDER, filename, as_attachment=True)

# Bạn nên bổ sung các hàm parse_docx_file, parse_docx_text, extract_text_pdf, extract_text_html và hàm parse_tiet_in_khoan như phần 1 đã gửi.

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8001))
    app.run(host="0.0.0.0", port=port)