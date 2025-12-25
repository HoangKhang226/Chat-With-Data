import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


def print_chart(code: str, df: pd.DataFrame):
    try:
        # 1. Làm sạch code: Loại bỏ dấu nháy ngược nếu AI bao quanh code bằng markdown
        clean_code = code.strip().replace("```python", "").replace("```", "")

        local_vars = {"plt": plt, "df": df}

        # 2. Biên dịch code đã làm sạch
        compiled_code = compile(clean_code, "<string>", "exec")

        exec(compiled_code, globals(), local_vars)
        return plt.gcf()

    except SyntaxError as se:
        st.error(f"Lỗi cú pháp trong code AI sinh ra: {se}")
        st.info(f"Đoạn code bị lỗi: {code}")  # In ra để bạn kiểm tra ký tự lạ
        return None
    except Exception as e:
        st.error(f"Lỗi thực thi: {e}")
        return None
