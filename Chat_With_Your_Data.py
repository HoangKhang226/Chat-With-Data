import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from langchain_experimental.agents.agent_toolkits.pandas.base import (
    create_pandas_dataframe_agent,
)

from src.models.llm import load_llm
from src.agents.action import classify_intent, planner, executor
from src.ui.chat_history import display_chat_history
from src.agents.load_data import load_data
from src.agents.summary import summary

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = "gemini-2.0-flash"


def process_query(llm, agent, query):

    intent = classify_intent(llm, query)
    if intent == "PLANNER":
        response_data = {
            "role": "assistant",
            "content": planner(
                llm, st.session_state.df, query, st.session_state.dataset_context
            ),
            "is_plot": False,
            "code": None,
            "fig": None,
            "data_table": None,
        }
    else:
        result = executor(agent, st.session_state.df, query)
        response_data = {
            "role": "assistant",
            "content": result["output_text"],
            "is_plot": result["has_plot"],
            "code": result["executed_code"],
            "fig": result["fig"],
            "data_table": result["data_table"],
        }

    st.write(response_data["content"])
    data = response_data["data_table"]
    if isinstance(data, (pd.DataFrame, pd.Series)):
        st.dataframe(data)
    if response_data["fig"] is not None:
        st.pyplot(response_data["fig"])
    if response_data["code"]:
        st.code(response_data["code"], language="python")

    if "history" not in st.session_state:
        st.session_state.history = []

    st.session_state.history.append({"role": "user", "content": query})
    st.session_state.history.append(response_data)


def main():
    # set up fronted end
    st.set_page_config(page_title="Chat with your data", layout="wide")
    st.header("Chat with Your Data using LLMs")

    if "df" not in st.session_state:
        st.session_state.df = None
    if "history" not in st.session_state:
        st.session_state.history = []
    if "dataset_context" not in st.session_state:
        st.session_state.dataset_context = ""

    # load llms
    llm = load_llm(MODEL_NAME, api_key)

    # load csv
    with st.sidebar:
        uploaded_file = st.file_uploader("Up your file csv here", type="csv")
        if uploaded_file is not None:
            st.session_state.df = pd.read_csv(uploaded_file)
        st.markdown("Load text context (optional)")

        text_source_type = st.selectbox(
            "Select text source",
            options=["none", "pdf", "docx", "txt"],
        )

        text_file = None

        if text_source_type in ["pdf", "docx", "txt"]:
            text_file = st.file_uploader(
                "Upload file text",
                type=[text_source_type],
                key="text_file",
            )

        # Load Text
        if st.button("Load data"):
            documents = None

            if text_source_type in ["pdf", "docx", "txt"] and text_file:
                file_path = f"temp.{text_source_type}"
                with open(file_path, "wb") as f:
                    f.write(text_file.read())

                documents = load_data(text_source_type, file_path)

            if documents:
                full_text = " ".join([doc.page_content for doc in documents])
                st.session_state.dataset_context = summary(full_text, llm)
                st.write(st.session_state.dataset_context)

    # create DA agent to query with our data
    if st.session_state.df is not None:
        st.write("Your Dataset ", st.session_state.df.head())

    if st.session_state.df is not None:
        agent = create_pandas_dataframe_agent(
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
                process_query(llm, agent, query)

    # display chat history
    if st.session_state.history:
        display_chat_history(st.session_state.history)


main()
