# Chat With Your Data – Data Analysis Chatbot

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
  <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Đây là một **ứng dụng chatbot phân tích dữ liệu** được xây dựng bằng **Streamlit + LangChain Agent**.  
Người dùng có thể **tải file CSV**, đặt câu hỏi bằng **ngôn ngữ tự nhiên**, và chatbot sẽ tự động:

- Phân tích dữ liệu
- Sinh và thực thi **code Python trên DataFrame**
- Vẽ **biểu đồ phù hợp**
- Trả lời **dựa trên kết quả thực thi code** (data-driven)

---

## Mục tiêu Project
- Xây dựng chatbot có khả năng **phân tích dữ liệu tự động**
- Áp dụng tư duy **LLM Agent cho Data Analysis**
- Mọi câu trả lời đều dựa trên **kết quả code**
- Phù hợp làm **đồ án / project CV cho Data – AI – LLM**

---

## Cách Chatbot Hoạt Động

1. Người dùng upload file **CSV**
2. Dữ liệu được load vào `DataFrame df`
3. Người dùng nhập câu hỏi bằng ngôn ngữ tự nhiên
4. LLM Agent:
   - Phân tích yêu cầu
   - Sinh code Python để xử lý dữ liệu
   - Thực thi code thông qua `python_repl_ast`
   - Tự động vẽ biểu đồ phù hợp
5. Chatbot trả về:
   - Nhận xét bằng tiếng Việt
   - Biểu đồ
   - Code đã chạy
   - Kết quả dữ liệu (DataFrame / Series)

---

## Quy Trình Phân Tích (Agent Logic)

Agent được điều khiển bằng prompt cố định với **3 bước bắt buộc**:

### 1️. Khám phá dữ liệu
- `df.info()`
- `df.describe()`


### 2️. Xử lý & tính toán
- `groupby`
- lọc dữ liệu
- tính tổng, tỷ lệ, thống kê


### 3️. Trực quan hóa
- Tự động chọn biểu đồ:
  - Bar
  - Line
  - Pie
  - Boxplot
- Luôn vẽ **ít nhất 1 biểu đồ**
- Biểu đồ được hiển thị trực tiếp trong Streamlit

⚠️ Chatbot **không được trả lời nếu chưa chạy code**.

---


## 🛠️ Công Nghệ Sử Dụng
- Python
- Streamlit
- Pandas
- Matplotlib
- Seaborn
- LangChain (Pandas DataFrame Agent)
- HuggingFace / Gemini API

---

## 📂 Cấu Trúc Project

DA-For-Everyone/
├── src/
│ ├── models/ # Load LLM
│ └── utils/ # Hàm vẽ biểu đồ
├── notebooks/ # Notebook thử nghiệm
├── docs/ # Tài liệu
├── Chat_With_Your_Data.py # File Streamlit chính
├── requirements.txt
└── README.md

yaml
Sao chép mã

---

## Cách Chạy Project

### 1️. Clone repository
git clone https://github.com/HoangKhang226/DA-For-Everyone.git
cd DA-For-Everyone

### 2️. Cài thư viện
bash
Sao chép mã
pip install -r requirements.txt

### 3️. Thiết lập biến môi trường
Tạo file .env:

env
Sao chép mã
GOOGLE_API_KEY=your_api_key_here

### 4️. Chạy ứng dụng
bash
Sao chép mã
streamlit run Chat_With_Your_Data.py
💡 Tính Năng Nổi Bật
Chat với dữ liệu CSV

Tự sinh & thực thi code Pandas

Hiển thị:

Câu trả lời

Code đã chạy

Bảng dữ liệu

Biểu đồ

Lưu lịch sử chat trong session

👤 Author
Hoàng Khang
Data Science Student

GitHub: https://github.com/HoangKhang226
