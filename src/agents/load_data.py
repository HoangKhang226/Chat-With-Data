from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
import streamlit as st


@st.cache_data
def load_data(source_type, source):
    if source_type == "web":
        loader = WebBaseLoader(source)

    elif source_type == "pdf":
        loader = PyPDFLoader(source)

    elif source_type == "docx":
        loader = Docx2txtLoader(source)

    elif source_type == "txt":
        loader = TextLoader(source, encoding="utf-8")

    else:
        raise ValueError("Source type not supported")

    return loader.load()
