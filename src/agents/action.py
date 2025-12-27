import pandas as pd
from src.utils.data_visualize import print_chart
from src.prompt.prompting import prompt_executor, prompt_planner


def classify_intent(llm, query: str) -> str:
    """
    Phân loại câu hỏi người dùng
    Trả về: "PLANNER" hoặc "EXECUTOR"
    """
    try:
        prompt = f"""
Bạn là hệ thống phân loại câu hỏi trong chatbot phân tích dữ liệu.
Chỉ trả về 1 từ duy nhất: PLANNER hoặc EXECUTOR
Câu hỏi: "{query}"
"""
        return llm.invoke(prompt).content.strip().upper()
    except Exception as e:
        print(f"Error classify_intent: {e}")
        return "EXECUTOR"


def executor(agent, df, query):
    """
    Thực hiện phân tích dữ liệu với agent đã có sẵn.
    - Tự check null / missing
    - Tự chọn chart hợp lý
    - Robust với dataset lớn
    Trả về dict chuẩn:
        output_text, data_table, executed_code, has_plot, fig
    """
    try:
        # Giới hạn dữ liệu lớn: sample 1000 row max, giữ column quan trọng
        df_sample = (
            df.sample(n=min(1000, len(df)), random_state=42)
            if len(df) > 1000
            else df.copy()
        )
        prompt = prompt_executor(query)
        response = agent.invoke(prompt)
    except Exception as e:
        return {
            "output_text": f"Error khi gọi executor: {e}",
            "data_table": None,
            "executed_code": "",
            "has_plot": False,
            "fig": None,
        }

    output_text = response.get("output", "")
    steps = response.get("intermediate_steps", [])

    executed_code = ""
    fig = None
    data_result = None
    has_plot = False

    if steps:
        last_action, observation = steps[-1]
        executed_code = last_action.tool_input.get("query", "")

        # Lấy dữ liệu kết quả
        if isinstance(observation, (pd.DataFrame, pd.Series)):
            data_result = observation
        elif isinstance(observation, str) and observation.strip():
            data_result = observation

        # Kiểm tra plot
        if "plt." in executed_code:
            has_plot = True
            try:
                fig = print_chart(executed_code, df)
            except Exception as e:
                print(f"Error khi vẽ chart: {e}")
                fig = None

    return {
        "output_text": output_text,
        "data_table": data_result,
        "executed_code": executed_code,
        "has_plot": has_plot,
        "fig": fig,
    }


def planner(llm, df, query, context=None):
    """
    Trả về gợi ý chiến lược phân tích dữ liệu
    - df: DataFrame của dataset
    - query: câu hỏi người dùng
    - context: nội dung dataset chi tiết (từ file docx, pdf, txt) - không bắt buộc
    """
    try:
        #  Lấy sample dataset nhỏ để AI nắm schema
        df_sample = df.sample(3, random_state=42)
        dataset_summary = f"""
DATASET SUMMARY (schema + sample):
- Các cột: {list(df.columns)}
- Kiểu dữ liệu: {df.dtypes.to_dict()}
- 5 dòng sample:
{df_sample.to_dict(orient='records')}
"""

        #  Chuẩn bị prompt
        full_prompt = "Bạn là chuyên gia phân tích dữ liệu.\n"
        if context:
            full_prompt += f"Mô tả dataset chi tiết:\n{context}\n\n"
        full_prompt += f"{dataset_summary}\nCâu hỏi của user: {query}\n"
        full_prompt += (
            "Yêu cầu: Gợi ý hướng phân tích (thống kê, trực quan, so sánh, mô tả) "
            "KHÔNG sinh code, chỉ đưa ra chiến lược, insight tiềm năng."
        )

        #  Gọi LLM
        response = llm.invoke(full_prompt)

        return response.content

    except Exception as e:
        print(f"Error planner: {e}")
        return "Không thể tạo gợi ý chiến lược."
