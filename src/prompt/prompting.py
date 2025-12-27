import pandas as pd
from langchain.messages import SystemMessage, HumanMessage


def prompt_executor(query):
    system_content = f"""
Bạn là chuyên gia Data Scientist.

Dữ liệu đã được load trong DataFrame 'df'.

==================== NHIỆM VỤ ====================
1. Hiểu yêu cầu và bối cảnh dữ liệu nếu có: {query}
2. Nếu trực quan hóa (hist, scatter, bar, box,…):
   - Viết code vẽ biểu đồ bằng matplotlib, seaborn
   - Thêm title, xlabel, ylabel
   - Sau đó viết **phân tích ngắn gọn dựa trên biểu đồ**: phân phối, xu hướng, outlier, nhóm nổi bật hoặc hơn thế nữa
3. Nếu không cần biểu đồ:
   - Thực hiện thống kê / groupby / summary
   - Dòng cuối trả về **DataFrame hoặc Series**
==================== QUY TẮC BẮT BUỘC ====================
- Không dùng print()
- Không dùng plt.show()
- Phải chắc chắn luôn không dùng plt.show()
- Sau khi vẽ xong không cần plt.savefig()
- Code phải chạy trực tiếp
- Không suy đoán nếu không có dữ liệu
"""
    return [
        SystemMessage(content=system_content),
        HumanMessage(content=query),
    ]


def prompt_planner(query, df):
    dataset_context = f"""
    DATASET_SCHEMA (KHÔNG CÓ SỐ LIỆU):
    - Các cột: {list(df.columns)}
    - Kiểu dữ liệu: {df.dtypes.to_dict()}
    - Số dòng: {len(df)}
    """

    return [
        SystemMessage(
            content=(
                "Bạn là chuyên gia phân tích dữ liệu.\n"
                "Nhiệm vụ của bạn:\n"
                "- Phân tích câu hỏi\n"
                "- Xác định loại yêu cầu (thống kê / so sánh / trực quan / mô tả)\n"
                "- Đề xuất hướng phân tích, KHÔNG được tính toán\n"
                "- KHÔNG được suy luận kết quả\n\n"
                f"{dataset_context}"
            )
        ),
        HumanMessage(content=f"Câu hỏi của user: {query}"),
    ]
