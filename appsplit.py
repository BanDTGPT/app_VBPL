import re
import docx
import pdfplumber
import requests
from bs4 import BeautifulSoup
import logging

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
    """
    Đọc file DOCX và tách văn bản pháp luật theo cấu trúc: Chương, Điều, Khoản, Tiết
    Dùng quy tắc regex đơn giản. use_ai hiện là placeholder có thể mở rộng.
    """
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

def parse_docx_text(text):
    """Tách text thô thành các đoạn nhỏ."""
    chunks = text.split('\n\n')
    records = []
    for c in chunks:
        if len(c.strip()) > 10:
            records.append({"Nội dung": c.strip()})
    return records

def extract_text_pdf(filepath):
    """Trích xuất văn bản thô từ file PDF."""
    texts = []
    try:
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                texts.append(page.extract_text() or "")
        return "\n".join(texts)
    except Exception as e:
        logging.error(f"Lỗi đọc PDF: {e}")
        return ""

def extract_text_html(url_or_content):
    """Lấy văn bản từ URL hoặc HTML thô."""
    if url_or_content.lower().startswith("http"):
        try:
            r = requests.get(url_or_content)
            r.raise_for_status()
            content = r.text
        except Exception as e:
            logging.error(f"Lỗi lấy HTML: {e}")
            return ""
    else:
        content = url_or_content

    soup = BeautifulSoup(content, 'html.parser')
    content_div = soup.find("div", class_="content") or soup.body
    if not content_div:
        return ""
    paras = content_div.find_all(['p', 'div'])
    texts = []
    for p in paras:
        t = p.get_text(separator=' ', strip=True)
        if t:
            texts.append(t)
    return "\n".join(texts)
    from flask import send_from_directory

@app.route('/', methods=['GET'])
def index():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>App Đọc Lọc Văn Bản Pháp Luật</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px; background: #f6f8fa;
            }}
            .container {{
                max-width: 700px; margin: auto;
                background: white; padding: 20px; border-radius: 8px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.15);
            }}
            h2 {{ text-align: center; margin-bottom: 20px;}}
            input[type="file"] {{
                width: 100%;
                padding: 10px;
                border: 1px solid #ccc;
                border-radius: 5px;
                background: #f9f9f9;
            }}
            label {{
                font-weight: bold;
                display: block;
                margin-top: 15px;
            }}
            button {{
                margin-top: 20px;
                padding: 10px 20px;
                font-size: 16px;
                background: #0078D4;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
            }}
            button:hover {{
                background:#005a9e;
            }}
            .result-links {{
                margin-top: 20px;
            }}
            .result-links a {{
                display: block;
                margin-bottom: 10px;
                color: #0078D4;
                text-decoration: none;
            }}
            .result-links a:hover {{
                text-decoration: underline;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Upload nhiều file văn bản pháp luật</h2>
            <form action="/upload_files" method="post" enctype="multipart/form-data">
                <input type="file" name="files" multiple required><br>
                <label><input type="checkbox" name="use_ai_support"> Sử dụng AI hỗ trợ phân tích</label><br>
                <button type="submit">Upload và phân tích</button>
            </form>
            <div class="result-links">
                {links}
            </div>
        </div>
    </body>
    </html>
    """
    files = []
    if os.path.exists(RESULT_FOLDER):
        files = sorted(os.listdir(RESULT_FOLDER))
    links = ""
    for f in files:
        links += f'<a href="/download/{f}" target="_blank">{f}</a><br>'
    return html.format(links=links)

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
            df = pd.read_excel(save_path)
            expected_cols = {'Chương', 'Điều', 'Khoản', 'Tiết', 'Nội dung'}
            if expected_cols.issubset(set(df.columns)):
                records = df.to_dict(orient='records')
            else:
                return f"File {filename} thiếu cột bắt buộc.", 400
        elif extension == 'html':
            url = filename  # Bạn có thể thay đổi lấy URL khác tùy thực tế
            html_text = extract_text_html(url)
            records = parse_docx_text(html_text)
        else:
            return f"Không hỗ trợ định dạng file {filename}", 400

        base_name = os.path.splitext(filename)[0]
        out_xlsx = os.path.join(RESULT_FOLDER, f"{base_name}_processed.xlsx")
        out_json = os.path.join(RESULT_FOLDER, f"{base_name}_processed.json")

        df_out = pd.DataFrame(records)
        df_out.to_excel(out_xlsx, index=False, encoding='utf-8')

        with open(out_json, 'w', encoding='utf-8') as fjson:
            json.dump(records, fjson, ensure_ascii=False, indent=4)

        result_links.append(out_xlsx)
        result_links.append(out_json)

    links_html = "<h3>File kết quả:</h3>"
    for f in result_links:
        fname = os.path.basename(f)
        links_html += f'<a href="/download/{fname}" target="_blank">{fname}</a><br>'

    return """
    <html><body>
    {0}
    <br><a href="/">Quay lại trang chính</a>
    </body></html>
    """.format(links_html)
    @app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(RESULT_FOLDER, filename, as_attachment=True)


if __name__ == "__main__":
    import sys

    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULT_FOLDER, exist_ok=True)

    port = int(os.environ.get("PORT", 8001))
    logging.info(f"Starting appsplit on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=True)