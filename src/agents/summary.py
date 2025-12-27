from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st


@st.cache_data
def summary(text: str, llm):
    if not text:
        return ""

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=200,
    )

    chunks = text_splitter.split_text(text)
    summaries = []

    for chunk in chunks:
        prompt = (
            "Bạn là một Data Analyst. "
            "Hãy tóm tắt đoạn văn sau trong khoảng 3–4 câu:\n\n"
            f"{chunk}"
        )
        result = llm.invoke(prompt)
        summaries.append(result.content)

    final_prompt = (
        "Bạn là một Data Analyst. "
        "Hãy hợp nhất các bản tóm tắt sau thành một đoạn 3–4 câu, "
        "giữ lại thông tin quan trọng, không lặp:\n\n" + "\n".join(summaries)
    )

    return llm.invoke(final_prompt).content
