import os
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk
import threading
from PIL import Image, ImageTk
from sentence_transformers import SentenceTransformer
import customtkinter as ctk
import time

# Thay đổi import để sử dụng tên file logic mới
from rag_logic import (
    load_dual_index, build_faiss_from_file, search, answer_question,
    client, GEMINI_API_KEY,  # Import client và key để kiểm tra trạng thái
)

# --- CẤU HÌNH BẢO MẬT VÀ CTK ---
ctk.set_appearance_mode("System")  # Chế độ Dark/Light theo hệ thống
ctk.set_default_color_theme("blue")  # Thiết lập chủ đề màu sắc

# Tên đăng nhập và Mật khẩu Mẫu
TEACHER_USERNAME = "giaovien"
TEACHER_PASSWORD = "giaovien"


# ---- Hàm căn giữa cửa sổ (Tiện ích) ----
def center_window(win, width, height):
    screen_w = win.winfo_screenwidth()
    screen_h = win.winfo_screenheight()
    x = (screen_w // 2) - (width // 2)
    y = (screen_h // 2) - (height // 2)
    win.geometry(f"{width}x{height}+{x}+{y}")


# ---- Giao diện tra cứu chính ----
class ChatbotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Chatbot Tra Cứu Tài Liệu Thông Minh")
        center_window(self.root, 1280, 720)

        self._initialize_variables()
        self._load_images()  # TẢI ẢNH LÊN TRƯỚC
        self._setup_ui()

        initial_status = "Đang tải mô hình nhúng và index, vui lòng chờ...\n"
        if GEMINI_API_KEY is None or client is None:
            initial_status = "❌ GOOGLE_API_KEY không được đặt. Tính năng tạo câu trả lời bị vô hiệu hóa.\nĐang tải mô hình nhúng cục bộ..."

        self.answer_box.insert(tk.END, initial_status)
        threading.Thread(target=self._load_model_thread, daemon=True).start()

    def _initialize_variables(self):
        # ... (Giữ nguyên các biến khởi tạo) ...
        self.model_loaded = False
        self.model = None
        self.index_gen = None
        self.chunks_gen = None
        self.index_vis = None
        self.chunks_vis = None
        self.context_data = {}
        self.doc_path = ""
        self.source_type = ""
        self.last_chunk_indices_vis = []
        self.tk_images_list = []

        # Thêm biến để giữ tham chiếu ảnh logo/robot
        self.robot_image = None
        self.school_logo_image = None

        # --- BIẾN TRẠNG THÁI ĐĂNG NHẬP ---
        self.is_teacher_logged_in = False

    def _load_images(self, size=(60, 60)):
        """Tải và resize ảnh Robot và Logo Trường."""
        try:
            #
            size = (170, 130)
            assets_dir = os.path.join(os.path.dirname(__file__), "assets")

            # --- Tải ảnh Robot ---
            robot_path = os.path.join(assets_dir, "robot_icon.png")
            original_robot_img = Image.open(robot_path)
            resized_robot_img = original_robot_img.resize(size, Image.Resampling.LANCZOS if hasattr(Image,
                                                                                                    'Resampling') else Image.LANCZOS)
            self.robot_image = ctk.CTkImage(light_image=resized_robot_img, dark_image=resized_robot_img, size=size)

            # --- Tải Logo Trường ---
            logo_path = os.path.join(assets_dir, "school_logo.png")
            original_school_logo = Image.open(logo_path)
            resized_school_logo = original_school_logo.resize(size, Image.Resampling.LANCZOS if hasattr(Image,
                                                                                                        'Resampling') else Image.LANCZOS)
            self.school_logo_image = ctk.CTkImage(light_image=resized_school_logo, dark_image=resized_school_logo,
                                                  size=size)

        except FileNotFoundError as e:
            print(
                f"LỖI: Không tìm thấy file ảnh trong thư mục assets/ (Chi tiết: {e}). Đảm bảo có thư mục 'assets' ngang hàng với app.py.")
        except Exception as e:
            print(f"LỖI khi xử lý ảnh logo/robot: {e}")

    def _setup_ui(self):
        # Thiết lập Frame chính cho bố cục hiện đại
        main_frame = ctk.CTkFrame(self.root, corner_radius=10)
        main_frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

        # --- Frame Đăng nhập và Tiêu đề ---
        top_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        top_frame.pack(pady=(15, 5))

        # Khung chứa tiêu đề và logo
        title_frame = ctk.CTkFrame(top_frame, fg_color="transparent")
        title_frame.pack(side=tk.LEFT, padx=(0, 50))

        # 1. Hiển thị Robot (Bên trái)
        if self.robot_image:
            ctk.CTkLabel(title_frame, image=self.robot_image, text="").pack(side=tk.LEFT, padx=(0, 15), pady=(0, 0))

            # Tiêu đề chính (ở giữa)
        ctk.CTkLabel(title_frame, text="HỆ THỐNG TRA CỨU TÀI LIỆU THÔNG MINH ",
                     font=ctk.CTkFont(family="Arial", size=24, weight="bold"),
                     text_color="#1F6AA5").pack(side=tk.LEFT, padx=10)

        # 2. Hiển thị Logo Trường (Bên phải)
        if self.school_logo_image:
            ctk.CTkLabel(title_frame, image=self.school_logo_image, text="").pack(side=tk.RIGHT, padx=(15, 0),
                                                                                  pady=(0, 0))

            # --- Khung Đăng nhập (Bên phải của Top Frame) ---
        login_frame = ctk.CTkFrame(top_frame, border_width=1, corner_radius=8, fg_color="transparent")
        login_frame.pack(side=tk.RIGHT, padx=(50, 0))

        ctk.CTkLabel(login_frame, text="Khu vực Giáo viên", font=ctk.CTkFont(weight="bold")).pack(pady=5, padx=10)

        # Thêm các trường nhập liệu
        self.username_entry = ctk.CTkEntry(login_frame, placeholder_text="Tên đăng nhập", width=150)
        self.username_entry.pack(pady=5, padx=10)

        self.password_entry = ctk.CTkEntry(login_frame, placeholder_text="Mật khẩu", show="*", width=150)
        self.password_entry.pack(pady=5, padx=10)

        self.login_button = ctk.CTkButton(login_frame, text="Đăng nhập", command=self._handle_login, corner_radius=6)
        self.login_button.pack(pady=5, padx=10)

        self.logout_button = ctk.CTkButton(login_frame, text="Đăng xuất", command=self._handle_logout, fg_color="red",
                                           hover_color="darkred")
        self.logged_in_label = ctk.CTkLabel(login_frame, text=f"Đã đăng nhập: {TEACHER_USERNAME}",
                                            font=ctk.CTkFont(weight="bold"), text_color="green")

        # --- Phần còn lại của UI (Dưới khung Login/Title) ---

        # Frame Upload và Thông báo Nguồn
        upload_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        upload_frame.pack(pady=5)

        # Nút Upload (Phải được tạo trước khi gọi _update_ui_on_login_status)
        self.upload_button = ctk.CTkButton(upload_frame, text="🛠️ Cập nhật File nguồn ",
                                           command=self.choose_file,
                                           font=ctk.CTkFont(size=14, weight="bold"),
                                           height=35, corner_radius=8, state=tk.DISABLED)
        self.upload_button.pack(side=tk.LEFT, padx=10)

        self.source_label = ctk.CTkLabel(upload_frame, text="Nguồn dữ liệu: Đang tải...",
                                         font=ctk.CTkFont(size=12, slant="italic"),
                                         text_color="#4A90E2")
        self.source_label.pack(side=tk.LEFT, padx=10)

        # Frame nhập và nút Tra cứu
        search_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        search_frame.pack(pady=10)

        self.entry = ctk.CTkEntry(search_frame, font=ctk.CTkFont(size=14),
                                  width=700, height=35, placeholder_text="Nhập câu hỏi của bạn.")
        self.entry.pack(side=tk.LEFT, padx=5)

        ctk.CTkButton(search_frame, text="🔍 Tra cứu", command=self.query,
                      font=ctk.CTkFont(size=14, weight="bold"),
                      height=35, corner_radius=8).pack(side=tk.LEFT, padx=5)

        # KHU VỰC TAB HIỂN THỊ KẾT QUẢ
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.tab_answer = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_answer, text="1. Gemini hỗ trợ tìm kiếm")
        self.answer_box = scrolledtext.ScrolledText(self.tab_answer, wrap=tk.WORD, font=("Arial", 12), padx=10, pady=10)
        self.answer_box.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

        self.tab_context = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_context, text="2. Ảnh liên quan ")
        self.context_text_box = scrolledtext.ScrolledText(self.tab_context, wrap=tk.WORD, font=("Arial", 10), padx=10,
                                                          pady=10)
        self.context_text_box.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

        # --- GỌI HÀM CẬP NHẬT TRẠNG THÁI CUỐI CÙNG (Đã di chuyển) ---
        self._update_ui_on_login_status()

        # --- LOGIC ĐĂNG NHẬP/ĐĂNG XUẤT ---

    def _handle_login(self):
        username = self.username_entry.get()
        password = self.password_entry.get()

        if username == TEACHER_USERNAME and password == TEACHER_PASSWORD:
            self.is_teacher_logged_in = True
            messagebox.showinfo("Đăng nhập thành công", "Chào mừng Giáo viên. Bạn có thể cập nhật file nguồn.")
            self._update_ui_on_login_status()
            self.password_entry.delete(0, tk.END)
        else:
            messagebox.showerror("Lỗi đăng nhập", "Tên đăng nhập hoặc mật khẩu không đúng.")
            self.password_entry.delete(0, tk.END)

    def _handle_logout(self):
        self.is_teacher_logged_in = False
        messagebox.showinfo("Đăng xuất", "Đã đăng xuất khỏi khu vực Giáo viên.")
        self._update_ui_on_login_status()

    def _update_ui_on_login_status(self):
        """Cập nhật trạng thái của các nút dựa trên self.is_teacher_logged_in."""
        if self.is_teacher_logged_in:
            self.upload_button.configure(state=tk.NORMAL)

            # Ẩn các trường nhập và nút login
            self.login_button.pack_forget()
            self.username_entry.pack_forget()
            self.password_entry.pack_forget()

            # Hiện trạng thái đăng nhập và nút logout
            self.logged_in_label.configure(text=f"Đã đăng nhập: {TEACHER_USERNAME}")
            self.logged_in_label.pack(pady=5, padx=10)
            self.logout_button.pack(pady=5, padx=10)
        else:
            self.upload_button.configure(state=tk.DISABLED)

            # Hiện lại các trường nhập liệu
            self.logged_in_label.pack_forget()
            self.username_entry.pack(pady=5, padx=10)
            self.password_entry.pack(pady=5, padx=10)
            self.login_button.pack(pady=5, padx=10)
            self.logout_button.pack_forget()

            # --- CÁC HÀM CÒN LẠI (GIỮ NGUYÊN LOGIC) ---

    def _load_model_thread(self):
        try:
            self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            results = load_dual_index()
            if results and len(results) == 6:
                self.index_gen, self.chunks_gen, self.index_vis, self.chunks_vis, self.doc_path, self.context_data = results
            self.model_loaded = True
            status_text = "Mình có thể giúp gì cho bạn ?"
            if client is None:
                status_text += "\n⚠️ Tính năng trả lời (Gemini) bị vô hiệu hóa do thiếu API Key."
            if self.doc_path:
                self.source_label.configure(text=f"Nguồn dữ liệu đang dùng: {os.path.basename(self.doc_path)}")
            else:
                self.source_label.configure(text="Nguồn dữ liệu đang dùng: CHƯA CÓ. Vui lòng chọn file nguồn.")
            self.answer_box.delete("1.0", tk.END)
            self.answer_box.insert(tk.END, f"{status_text}\n")
        except Exception as e:
            self.answer_box.delete("1.0", tk.END)
            self.answer_box.insert(tk.END, f"❌ Lỗi tải mô hình/index: {e}. Kiểm tra kết nối mạng/thư viện.\n")

    def choose_file(self):
        if not self.is_teacher_logged_in:
            messagebox.showwarning("Cần Đăng nhập",
                                   "Vui lòng đăng nhập với tài khoản Giáo viên để cập nhật file nguồn.")
            return

        if not self.model_loaded:
            messagebox.showerror("Lỗi", "Vui lòng đợi mô hình tải xong trước khi chọn file.")
            return

        file_path = filedialog.askopenfilename(
            title="Chọn file nguồn ",
            filetypes=[("Supported Files", "*.docx;*.pdf;*.jpg;*.jpeg;*.png")],
        )
        if not file_path: return

        self.answer_box.delete("1.0", tk.END)
        self.answer_box.insert(tk.END, f"⏳ Đang xử lý file {os.path.basename(file_path)}... (Tạo 2 Index)\n")
        self.source_label.configure(text=f"Nguồn dữ liệu đang dùng: Đang xử lý {os.path.basename(file_path)}...")

        def build_task():
            try:
                from rag_logic import build_faiss_from_file
                results = build_faiss_from_file(file_path, self.model)

                if results and len(results) == 6:
                    self.index_gen, self.chunks_gen, self.index_vis, self.chunks_vis, self.context_data, self.source_type = results
                    self.doc_path = file_path
                    self.source_label.configure(
                        text=f"Nguồn dữ liệu đang dùng: {os.path.basename(self.doc_path)} ({self.source_type})")
                    messagebox.showinfo("Thành công",
                                        f"Đã cập nhật dữ liệu từ file mới: {os.path.basename(file_path)}!")
                    self.answer_box.delete("1.0", tk.END)
                    self.answer_box.insert(tk.END, "Dữ liệu mới đã sẵn sàng. Bạn có thể bắt đầu tra cứu.\n")
                else:
                    self.source_label.configure(text="Nguồn dữ liệu đang dùng: Xử lý thất bại")
                    self.answer_box.delete("1.0", tk.END)
                    self.answer_box.insert(tk.END, "Xử lý file thất bại. Vui lòng kiểm tra console/log lỗi.\n")
            except Exception as e:
                self.source_label.configure(text="Nguồn dữ liệu đang dùng: Lỗi hệ thống")
                messagebox.showerror("Lỗi hệ thống", f"Lỗi không xác định khi xử lý file: {e}")

        threading.Thread(target=build_task, daemon=True).start()

    def query(self):
        if not self.model_loaded or not self.index_gen or not self.chunks_gen:
            messagebox.showerror("Lỗi", "Hệ thống chưa sẵn sàng, vui lòng đợi Index/Model tải xong.")
            return

        query = self.entry.get().strip()
        if not query: return

        self.answer_box.delete("1.0", tk.END)
        self.answer_box.insert(tk.END, "⏳ Đợi chút nha, mình đang suy nghĩ câu trả lời....\n")
        self.context_text_box.delete("1.0", tk.END)
        self.context_text_box.insert(tk.END, "Đang xử lý ảnh liên quan và OCR text...\n")
        self.notebook.select(self.tab_answer)

        def task():
            try:
                from rag_logic import search, answer_question
                context_chunks_gen, _ = search(query, self.model, self.index_gen, self.chunks_gen, top_k=5)
                _, chunk_indices_vis = search(query, self.model, self.index_vis, self.chunks_vis, top_k=3)
                self.last_chunk_indices_vis = chunk_indices_vis

                if client:
                    answer = answer_question(query, context_chunks_gen)
                else:
                    answer = "Gemini API Client không khả dụng. Ngữ cảnh tìm được: \n" + "\n---\n".join(
                        context_chunks_gen)

                self.answer_box.delete("1.0", tk.END)
                self.answer_box.insert(tk.END, answer)
                self._display_context()

            except Exception as e:
                self.answer_box.delete("1.0", tk.END)
                self.answer_box.insert(tk.END, f"⚠️ Lỗi xử lý/gọi API: {e}\n")
                self.context_text_box.delete("1.0", tk.END)
                self.context_text_box.insert(tk.END, "Lỗi xử lý ngữ cảnh/ảnh.")

        threading.Thread(target=task, daemon=True).start()

    def _display_context(self):
        self.context_text_box.delete("1.0", tk.END)
        self.tk_images_list = []
        if not self.last_chunk_indices_vis:
            self.context_text_box.insert(tk.END, "Không tìm thấy hình ảnh liên quan.\n")
            return

        source_type = self.context_data.get('type')

        for i, visual_index in enumerate(self.last_chunk_indices_vis):
            if visual_index >= len(self.chunks_vis): continue

            map_data = self.context_data['visual_source_map'][visual_index]
            image_to_display = None
            image_info_text = ""
            found_text_to_display = self.chunks_vis[visual_index]

            self.context_text_box.insert(tk.END, f"\n\n================================\n")
            self.context_text_box.insert(tk.END, f"🔎 KẾT QUẢ LIÊN QUAN THỨ {i + 1}\n")
            self.context_text_box.insert(tk.END, f"================================\n\n")

            if source_type in ['Image', 'PDF']:
                img_path = map_data.get('img_path')

                if source_type == 'PDF':
                    page_num = map_data.get('page')
                    image_info_text = f"Ảnh liên quan: Trang PDF {page_num}"
                else:
                    image_info_text = "Ảnh liên quan: File ảnh gốc"
                    img_path = map_data.get('path')

                if img_path and os.path.exists(img_path):
                    try:
                        image_to_display = Image.open(img_path)
                    except Exception:
                        self.context_text_box.insert(tk.END, f"Không thể tải ảnh: {img_path}\n")

            elif source_type == 'DOCX':
                self.context_text_box.insert(tk.END, "Dữ liệu nguồn là DOCX. Chỉ hiển thị đoạn văn bản liên quan:\n\n")
                self.context_text_box.insert(tk.END,
                                             f"--- ĐOẠN VĂN BẢN LIÊN QUAN (Chunk {visual_index + 1}) ---\n\n{found_text_to_display}")
                continue

            if image_to_display:
                self.context_text_box.insert(tk.END, image_info_text + "\n\n")
                self._insert_image_into_text_widget(image_to_display)
                self.context_text_box.insert(tk.END,
                                             f"\n--- VĂN BẢN TƯƠNG ỨNG (OCR/TEXT) ---\n\n{found_text_to_display}")
            else:
                self.context_text_box.insert(tk.END, "Không tìm thấy dữ liệu hình ảnh liên quan để hiển thị.\n")

    def _insert_image_into_text_widget(self, img):
        max_width = self.root.winfo_width() * 80 // 100
        if max_width <= 0: max_width = 800

        img_width, img_height = img.size

        if img_width > max_width:
            ratio = max_width / img_width
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
            img = img.resize((new_width, new_height),
                             Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)

        tk_img = ImageTk.PhotoImage(img)
        self.tk_images_list.append(tk_img)

        self.context_text_box.image_create(tk.END, image=tk_img)
        self.context_text_box.insert(tk.END, "\n\n")


if __name__ == "__main__":
    root = ctk.CTk()
    ChatbotApp(root)
    root.mainloop()