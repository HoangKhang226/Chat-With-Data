# Chat With Data – Data Analysis Chatbot
https://github.com/user-attachments/assets/f0dfceb7-8d16-4cac-8535-17961161e7ff
**Chat With Your Data** là một ứng dụng **chatbot phân tích dữ liệu** được xây dựng bằng **Streamlit + LangChain Agent**.  
Người dùng có thể:

- Upload file **CSV**  
- Đặt câu hỏi bằng **ngôn ngữ tự nhiên**  
- Chatbot sẽ tự động:  
  - Phân tích dữ liệu  
  - Sinh và thực thi **code Python trên DataFrame**  
  - Vẽ **biểu đồ phù hợp**  
  - Trả lời dựa trên **kết quả thực thi code**

---

## Mục tiêu Project

- Xây dựng chatbot **phân tích dữ liệu tự động**  
- Áp dụng **LLM Agent** cho Data Analysis, Data Science  
- Mọi câu trả lời dựa trên **kết quả code** và **nội dung dataset**

---

## Cách Chatbot Hoạt Động

1. Người dùng upload file **CSV**.    
   Ngoài ra có thể upload thêm nội dung dataset thông qua các file **DOCX**, **TXT**, **PDF**.    
2. Dữ liệu được load vào `DataFrame df`.  
3. Người dùng nhập câu hỏi bằng **ngôn ngữ tự nhiên**.  
4. **LLM Agent** thực hiện:  
   - Phân tích yêu cầu  
   - Sinh code Python để xử lý dữ liệu  
   - Thực thi code bằng `python_repl_ast`  
   - Vẽ **biểu đồ phù hợp**  
5. Chatbot trả về:  
   - Nhận xét  
   - Biểu đồ  
   - Code đã chạy  
   - Kết quả dữ liệu (`DataFrame` / `Series`)

---

## Cấu trúc dự án

├── src  
│   ├── agents  
│   │   ├── action.py          # Chức năng classify_intent, planner, executor  
│   │   ├── load_data.py       # Hàm load dữ liệu văn bản  
│   │   └── summary.py         # Hàm tóm tắt nội dung văn bản  
│   ├── models  
│   │   └── llm.py             # Hàm load_llm để load LLM  
│   ├── ui  
│   │   └── chat_history.py    # Hiển thị lịch sử trò chuyện  
│   └── utils  
│       └── data_visualize.py  # Hàm print_chart (visualize dữ liệu)  
├── prompt  
│   └── prompting.py           # prompt_executor, prompt_planner  
├── app.py                     # File chính chạy Streamlit  
└── README.md

##  Công Nghệ Sử Dụng
- Python  
- Streamlit  
- Pandas  
- Matplotlib   
- Seaborn  
- LangChain (Pandas DataFrame Agent)  
- ...  
---

##  Cách Chạy Project

1. Clone repository    
   git clone https://github.com/HoangKhang226/DA-For-Everyone.git    
   cd Chat-With-Data    
2. Cài thư viện  
   pip install -r requirements.txt    
3. Thiết lập biến môi trường    
   Tạo file .env: GOOGLE_API_KEY=your_api_key_here    
4. Chạy ứng dụng    
   streamlit run Chat_With_Your_Data.py    

## Tính Năng Nổi Bật  
  - Chat với dữ liệu CSV  
  - Tự sinh & thực thi code Pandas  
## Hiển thị:  
  - Câu trả lời  
  - Code đã chạy  
  - Bảng dữ liệu  
  - Biểu đồ  
  - Lưu lịch sử chat trong session và có thể xem lại  
 
👤 Author  
Hoàng Khang – Data Science Student  
GitHub: https://github.com/HoangKhang226  
