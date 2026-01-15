import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from docx import Document
import google.genai
from tkinter import messagebox
import threading
import tempfile
import shutil
import pytesseract
from PIL import Image
import fitz  # PyMuPDF

# --- CẤU HÌNH CỤC BỘ VÀ TỐI ƯU ---

# Đặt đường dẫn Tesseract-OCR (Kiểm tra lại đường dẫn này!)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Lấy GOOGLE_API_KEY từ biến môi trường
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")  # Trường hợp dùng tên cũ

try:
    if GEMINI_API_KEY:
        client = google.genai.Client(api_key=GEMINI_API_KEY)
    else:
        # Nếu không có Key, ta vẫn chạy ứng dụng Tkinter nhưng không thể tạo câu trả lời
        client = None
        print("Cảnh báo: GOOGLE_API_KEY không được tìm thấy. Tính năng tạo câu trả lời (Gemini) sẽ bị vô hiệu hóa.")
except Exception as e:
    print(f"Lỗi khởi tạo Gemini Client: {e}")
    client = None

# --- FILE LƯU TRỮ INDEX VÀ CACHE (Sử dụng thư mục rag_cache) ---
INDEX_DIR = "rag_cache"
if not os.path.exists(INDEX_DIR):
    os.makedirs(INDEX_DIR)

# Cho Tab 1 (Văn bản tổng thể - General/Gemini)
INDEX_FILE = os.path.join(INDEX_DIR, "faiss_index_general.bin")
CHUNKS_FILE = os.path.join(INDEX_DIR, "chunks_general.pkl")
# Cho Tab 2 (Văn bản OCR của từng trang/ảnh - Visual/Page)
V_INDEX_FILE = os.path.join(INDEX_DIR, "faiss_index_visual.bin")
V_CHUNKS_FILE = os.path.join(INDEX_DIR, "chunks_visual.pkl")

DOC_PATH_FILE = os.path.join(INDEX_DIR, "doc_path.txt")
CONTEXT_DATA_FILE = os.path.join(INDEX_DIR, "context_data.pkl")


# ---- CÁC HÀM TIỆN ÍCH ----

def chunk_text(text, chunk_size=800):
    """Chia văn bản thành các chunks có kích thước cố định."""
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


# --- CÁC HÀM XỬ LÝ ĐỌC FILE VÀ THU THẬP DỮ LIỆU GỐC ---

def read_pdf(file_path, context_data):
    """Đọc văn bản từ file PDF, trích xuất văn bản và lưu trữ ảnh trang (có OCR)."""
    text = ""
    context_data['type'] = 'PDF'
    context_data['source'] = file_path
    context_data['visual_texts'] = []  # Văn bản OCR/Text của từng trang
    context_data['visual_source_map'] = []
    temp_dir = tempfile.mkdtemp()
    context_data['temp_img_dir'] = temp_dir

    try:
        doc = fitz.open(file_path)
        mat = fitz.Matrix(300 / 72, 300 / 72)

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text("text")

            if not page_text.strip():
                # Tối ưu: Chỉ tạo pixmap và chạy OCR nếu không có văn bản có thể chọn được
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                page_text = pytesseract.image_to_string(img, lang='vie+eng')

            if not page_text.strip():
                continue

            # --- LƯU TRỮ VĂN BẢN VÀ ÁNH XẠ NGUỒN (VISUAL INDEX) ---
            context_data['visual_texts'].append(page_text)

            # Lưu trữ thông tin trang/ảnh để truy xuất
            map_entry = {"type": "PDF_PAGE", "page": page_num + 1}

            # --- LƯU TRỮ ẢNH TRANG PDF ---
            img_path = os.path.join(temp_dir, f"page_{page_num + 1}.png")

            # Tối ưu: Chỉ tạo pixmap khi chưa có (tránh gọi quá nhiều lần)
            pix = page.get_pixmap(matrix=mat, alpha=False)

            pix.save(img_path)

            map_entry['img_path'] = img_path

            context_data['visual_source_map'].append(map_entry)

            text += page_text + "\n\n"

        doc.close()
        context_data['text_content'] = text
        return text
    except Exception as e:
        print(f"Lỗi khi đọc PDF/OCR: {e}")
        if 'temp_dir' in context_data and os.path.exists(context_data['temp_img_dir']):
            shutil.rmtree(context_data['temp_img_dir'])
        return ""


def read_docx(file_path, context_data):
    """Đọc văn bản từ file DOCX."""
    doc = Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    context_data['type'] = 'DOCX'
    context_data['source'] = file_path
    context_data['text_content'] = text
    # Cho DOCX, Visual Index là các chunks văn bản
    context_data['visual_texts'] = chunk_text(text, chunk_size=800)
    context_data['visual_source_map'] = [{"type": "DOCX_CHUNK", "index": i} for i in
                                         range(len(context_data['visual_texts']))]
    return text


def read_image(file_path, context_data):
    """Sử dụng OCR để trích xuất văn bản từ file hình ảnh (JPG, PNG)."""
    try:
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img, lang='vie+eng')

        context_data['type'] = 'Image'
        context_data['source'] = file_path
        context_data['text_content'] = text
        context_data['visual_texts'] = [text]
        context_data['visual_source_map'] = [{"type": "IMAGE_FILE", "path": file_path}]
        return text
    except Exception as e:
        print(f"Lỗi khi OCR hình ảnh: {e}")
        return ""


# ---- Lưu và Tải Index/Context ----
def save_dual_index(index_gen, chunks_gen, index_vis, chunks_vis, doc_path, context_data):
    """Lưu cả hai index FAISS và metadata."""

    # ... (Logic dọn dẹp thư mục tạm thời cũ và lưu trữ index/metadata) ...
    old_context = {}
    if os.path.exists(CONTEXT_DATA_FILE):
        with open(CONTEXT_DATA_FILE, "rb") as f:
            old_context = pickle.load(f)
            if 'temp_img_dir' in old_context and os.path.exists(old_context['temp_img_dir']):
                try:
                    shutil.rmtree(old_context['temp_img_dir'])
                except Exception as e:
                    print(f"Lỗi khi xóa thư mục tạm thời cũ: {e}")

    # Ghi General Index
    faiss.write_index(index_gen, INDEX_FILE)
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks_gen, f)

    # Ghi Visual Index
    faiss.write_index(index_vis, V_INDEX_FILE)
    with open(V_CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks_vis, f)

    # Ghi metadata
    with open(DOC_PATH_FILE, "w", encoding="utf-8") as f:
        f.write(doc_path)
    with open(CONTEXT_DATA_FILE, "wb") as f:
        pickle.dump(context_data, f)


def load_dual_index():
    """Tải cả hai index FAISS và metadata."""
    required_files = [INDEX_FILE, CHUNKS_FILE, V_INDEX_FILE, V_CHUNKS_FILE, DOC_PATH_FILE, CONTEXT_DATA_FILE]

    if not all(os.path.exists(f) for f in required_files):
        return None, None, None, None, None, {}

    try:
        # Load Index và Chunks
        index_gen = faiss.read_index(INDEX_FILE)
        with open(CHUNKS_FILE, "rb") as f:
            chunks_gen = pickle.load(f)

        index_vis = faiss.read_index(V_INDEX_FILE)
        with open(V_CHUNKS_FILE, "rb") as f:
            chunks_vis = pickle.load(f)

        with open(DOC_PATH_FILE, "r", encoding="utf-8") as f:
            doc_path = f.read().strip()
        with open(CONTEXT_DATA_FILE, "rb") as f:
            context_data = pickle.load(f)

        return index_gen, chunks_gen, index_vis, chunks_vis, doc_path, context_data
    except Exception as e:
        print(f"Lỗi khi tải index/chunks/context: {e}")
        # Xóa cache nếu bị lỗi tải
        for f in required_files:
            if os.path.exists(f):
                os.remove(f)
        return None, None, None, None, None, {}


# ---- Build FAISS từ file bất kỳ ----
def build_faiss_from_file(file_path, model):
    """
    Xây dựng Dual Index (General và Visual) từ file nguồn (PDF/DOCX/Image).
    """
    if client is None:
        messagebox.showerror("Lỗi API Key", "Gemini Client không được khởi tạo. Vui lòng kiểm tra GOOGLE_API_KEY.")
        return None, None, None, None, {}, None

    file_extension = os.path.splitext(file_path)[1].lower()
    context_data = {}

    # 1. Trích xuất Text tổng thể và Visual Texts
    text_general = ""
    if file_extension == ".docx":
        text_general = read_docx(file_path, context_data)
        source_type = "DOCX"
    elif file_extension == ".pdf":
        text_general = read_pdf(file_path, context_data)
        source_type = "PDF (có OCR)"
    elif file_extension in [".jpg", ".jpeg", ".png"]:
        text_general = read_image(file_path, context_data)
        source_type = "Hình ảnh (OCR)"
    else:
        messagebox.showerror("Lỗi", "Định dạng file không được hỗ trợ.")
        return None, None, None, None, {}, None

    if not text_general.strip():
        messagebox.showerror("Lỗi", f"Không thể trích xuất văn bản từ file {file_path}.")
        return None, None, None, None, {}, None

    # 2. Xử lý General Index (Tab 1)
    chunks_general = chunk_text(text_general)
    if not chunks_general: return None, None, None, None, {}, None

    try:
        # Tối ưu: Mã hóa một lần cho tất cả chunks
        visual_texts = context_data.get('visual_texts', [])

        # Kiểm tra nếu visual_texts bị rỗng (ví dụ: docx không có visual text)
        if not visual_texts:
            embeddings_all = model.encode(chunks_general)
        else:
            embeddings_all = model.encode(chunks_general + visual_texts)

        # Tách lại embeddings cho từng index
        len_gen = len(chunks_general)
        embeddings_gen = embeddings_all[:len_gen]
        embeddings_vis = embeddings_all[len_gen:]

        dim = embeddings_gen.shape[1]

        # General Index
        index_gen = faiss.IndexFlatL2(dim)
        index_gen.add(np.array(embeddings_gen))

        # Visual Index
        index_vis = faiss.IndexFlatL2(dim)
        if embeddings_vis.size > 0:
            index_vis.add(np.array(embeddings_vis))

    except Exception as e:
        messagebox.showerror("Lỗi Embedding/FAISS", f"Lỗi khi mã hóa hoặc tạo Index FAISS: {e}")
        return None, None, None, None, {}, None

    # 4. Lưu cả hai index
    save_dual_index(index_gen, chunks_general, index_vis, visual_texts, file_path, context_data)

    return index_gen, chunks_general, index_vis, visual_texts, context_data, source_type


# ---- Tìm kiếm ----
def search(query, model, index, chunks, top_k=3):
    """Tìm kiếm vector FAISS và trả về các chunks liên quan."""
    if index is None or chunks is None or not query.strip():
        return [], []

    try:
        query_vec = model.encode([query])
        query_vec = np.array(query_vec, dtype="float32")

        # Tối ưu: Chỉ tìm kiếm top_k nếu index.ntotal > 0
        top_k = min(top_k, index.ntotal)
        if top_k == 0: return [], []

        distances, indices = index.search(query_vec, top_k)

        valid_indices = indices[0][indices[0] >= 0]

        if valid_indices.size == 0:
            return [], []

        valid_indices_list = valid_indices.tolist()

        return [chunks[i] for i in valid_indices_list], valid_indices_list

    except Exception as e:
        print(f"Lỗi tìm kiếm FAISS/Embedding: {e}")
        return [], []

    # ---- Hỏi Gemini ----


def answer_question(query, context_chunks):
    """Sử dụng Gemini để tạo câu trả lời dựa trên ngữ cảnh RAG."""
    if client is None:
        return "Lỗi: Gemini API Client chưa được khởi tạo. Vui lòng kiểm tra GOOGLE_API_KEY."

    context = "\n".join(context_chunks)

    prompt = f"""
Bạn là một trợ lý thông minh, có nhiệm vụ trả lời câu hỏi: {query} dựa trên tài liệu: {context}.
Hãy trả lời trực tiếp câu hỏi và luôn ghi rõ nguồn thông tin TÓM TẮT từ tài liệu đã cho.
Nếu thông tin về **số thông báo, ngày, tên file** có trong tài liệu, hãy trích xuất chúng.
Nếu thông tin không có trong tài liệu, trả lời "Không tìm thấy thông tin phù hợp trong dữ liệu nguồn hiện tại."
"""
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        return f"Lỗi gọi Gemini API: {e}. Có thể đã vượt quá giới hạn RPM (Requests Per Minute)."