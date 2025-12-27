import streamlit as st
import pandas as pd


def display_chat_history(history):
    st.markdown("---")
    st.markdown("**Chat History:**")

    for i, chat in enumerate(history):
        if chat["role"] == "user":
            st.info(f"**Query:** {chat['content']}")
        if chat["role"] == "assistant":
            with st.expander(f"Assistant Response {i//2 + 1}"):
                st.write(chat["content"])
                if chat.get("data_table") is not None:
                    res = chat["data_table"]
                    if isinstance(res, (pd.DataFrame, pd.Series)):
                        st.dataframe(res)
                    else:
                        st.text(res)
                if chat.get("is_plot") and chat.get("fig"):
                    st.pyplot(chat["fig"])
                if chat.get("code"):
                    st.code(chat["code"], language="python")
