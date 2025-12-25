import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from langchain_experimental.agents.agent_toolkits.pandas.base import (
    create_pandas_dataframe_agent,
)
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpointEmbeddings, ChatHuggingFace
from src.models.llm import load_llm
from src.utils.data_visualize import print_chart

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = "gemini-2.0-flash"

from langchain_core.prompts import PromptTemplate


def process_query(da_agent, query):
    # Tạo Template
    template = """
    Bạn là một Chuyên gia Phân tích Dữ liệu (Data Scientist) cao cấp. 
    Dữ liệu đã được tải sẵn vào biến DataFrame 'df'.

    NHIỆM VỤ CỦA BẠN:
    1. Phân tích yêu cầu của người dùng: {user_query}.
    2. Thực hiện quy trình phân tích 3 bước thông qua 'python_repl_ast':
    - Bước 1 (Khám phá): Kiểm tra cấu trúc df (df.info()) và các thống kê cơ bản (df.describe()) để hiểu dữ liệu liên quan đến câu hỏi.
    - Bước 2 (Xử lý & Tính toán): Thực hiện các phép gom nhóm (groupby), lọc, hoặc tính toán tỷ lệ cần thiết để làm rõ yêu cầu.
    - Bước 3 (Trực quan hóa): Luôn chủ động vẽ ít nhất 1 biểu đồ (Bar, Line, Pie, hoặc Boxplot) phù hợp nhất với loại dữ liệu đang phân tích.

    QUY TẮC THỰC THI MÃ:
    - Luôn nhập thư viện cần thiết: import pandas as pd, import matplotlib.pyplot as plt, import seaborn as sns.
    - Khi vẽ biểu đồ:
        + Cài đặt kích thước hình ảnh: plt.figure(figsize=(10, 6)).
        + Thêm tiêu đề (Title), nhãn trục (X-label, Y-label) đầy đủ.
        + Kết thúc bằng lệnh: plt.savefig('temp_chart.png').
    - Phải trả về kết quả cuối cùng bằng văn bản tiếng Việt, nhận xét dựa trên số liệu thu được từ code.

    QUY TẮC NGHIÊM NGẶT:
    - KHÔNG giải thích suông. Mọi câu trả lời phải dựa trên kết quả thực thi code.
    - Nếu người dùng hỏi chung chung, hãy tự chọn góc nhìn phân tích sâu sắc nhất (ví dụ: xu hướng theo thời gian, phân phối dữ liệu, hoặc tương quan).

    YÊU CẦU: {user_query}
    """
    prompt_template = PromptTemplate(input_variables=["user_query"], template=template)

    # Format câu query trước khi đưa vào Agent
    final_query = prompt_template.format(user_query=query)

    response = da_agent.invoke(final_query)

    output_text = response.get("output", "")
    steps = response.get("intermediate_steps", [])

    st.write(output_text)
    has_plot = False
    executed_code = ""
    fig = None
    data_result = None
    if steps:
        last_action, observation = steps[-1]

        executed_code = last_action.tool_input.get("query", "")
        if isinstance(observation, (pd.DataFrame, pd.Series)):
            data_result = observation
        elif isinstance(observation, str) and len(observation.strip()) > 0:
            # Trường hợp Trả về String (do dùng print() hoặc kết quả text)
            data_result = observation

        if any(keyword in executed_code for keyword in ["plt.", "pd.", "df."]):
            has_plot = True

            if "plt." in executed_code:
                fig = print_chart(executed_code, st.session_state.df)
                if fig:
                    with st.expander("View code"):
                        st.code(executed_code, language="python")
                    st.pyplot(fig)
            else:
                with st.expander("View data code"):
                    st.code(executed_code, language="python")
                if data_result is not None:
                    if isinstance(data_result, (pd.DataFrame, pd.Series)):
                        st.dataframe(data_result)
                    else:
                        st.text(data_result)

    if "history" not in st.session_state:
        st.session_state.history = []

    st.session_state.history.append({"role": "user", "content": query})
    st.session_state.history.append(
        {
            "role": "assistant",
            "content": output_text,
            "is_plot": has_plot,
            "code": executed_code,
            "fig": fig,  # Lưu đối tượng figure để hiển thị lại
            "data_table": data_result,
        }
    )
    return response


def display_chat_history():
    st.markdown("---")
    st.markdown("**Chat History:**")

    for i, chat in enumerate(st.session_state.history):
        if chat["role"] == "user":
            st.info(f"**Query:** {chat['content']}")
        else:
            st.success(f"**Assistant:** {chat['content']}")

            # Hiển thị code
            if chat.get("code"):
                with st.expander(f"View code {i//2 + 1}"):
                    st.code(chat["code"], language="python")

            # Hiển thị bảng Pandas nếu có lưu
            if chat.get("data_table") is not None:
                with st.expander("View data:"):
                    res = chat["data_table"]
                    if isinstance(res, (pd.DataFrame, pd.Series)):
                        st.dataframe(res)
                    else:
                        st.text(res)

            # Hiển thị biểu đồ
            if chat.get("is_plot") and chat.get("fig"):
                with st.expander("View Chart: "):
                    st.pyplot(chat["fig"])


def main():
    # set up fronted end
    st.set_page_config(
        page_title="Chat with your data", page_icon="khang", layout="wide"
    )
    st.header("Chat with your data")

    if "df" not in st.session_state:
        st.session_state.df = None
    if "history" not in st.session_state:
        st.session_state.history = []

    # load llms
    llm = load_llm(MODEL_NAME, api_key)

    # load csv
    with st.sidebar:
        uploaded_file = st.file_uploader("Up your file csv here", type="csv")

        if uploaded_file is not None:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.success("Đã tải file thành công!")

    if st.session_state.df is not None:
        st.write("Dữ liệu của bạn: ", st.session_state.df.head())

        # create DA agent to query with our data
        da_agent = create_pandas_dataframe_agent(
            llm=llm,
            df=st.session_state.df,
            agent_type="tool-calling",  #   gọi tool nội bộ để xử lý DataFrame
            allow_dangerous_code=True,  # chạy những đoạn code nguy hiểm
            verbose=True,  # in ra log chi tiết quá trình suy nghĩ và thực thi.
            return_intermediate_steps=True,  # trả về các bước trung gian
        )

        query = st.text_input("Enter your  question: ")

        if st.button("Run query"):
            with st.spinner("Processing..."):
                process_query(da_agent, query)
        # input query and preprocess
        # display chat history
        if st.session_state.history:
            display_chat_history()


main()
