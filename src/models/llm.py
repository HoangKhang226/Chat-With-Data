from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st


@st.cache_resource
def load_llm(model_name, api_key):

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,  # Truyền key vào đây
        temperature=0.3,
    )
