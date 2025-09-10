import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import codecs

from algorithms import apriori, naive_bayes, decision_tree, kmeans, roughset

# ===== Cấu hình trang =====
st.set_page_config(
    page_title="Ứng dụng khai thác dữ liệu",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== Hàm load CSS =====
def load_css(file_path):
    with codecs.open(file_path, "r", "utf-8", errors="ignore") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ===== Chọn giao diện sáng/tối =====
theme_mode = st.sidebar.radio("🎨 Giao diện", ["Light", "Dark"])
load_css("assets/base.css")
if theme_mode == "Dark":
    load_css("assets/dark.css")
else:
    load_css("assets/light.css")


# ===== Tiêu đề =====
st.title("📊 Ứng dụng khai thác dữ liệu")

# Sidebar chọn thuật toán
algorithms = ["Apriori","Rough Set", "Naive Bayes", "Decision Tree", "K-means"]
option = st.sidebar.selectbox("Chọn thuật toán", algorithms)

# Hiển thị thuật toán đang chọn
st.markdown(f"### 📌 **Thuật toán đang chọn:** :blue[{option}]", unsafe_allow_html=True)

# Upload file CSV
uploaded_file = st.file_uploader("📂 Import Data (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dữ liệu đầu vào")
    st.dataframe(df.head(), use_container_width=True)

    # ================= Apriori =================
    if option == "Apriori":
        st.subheader("📑 Luật kết hợp - Apriori")
        minsup = st.slider("Min Support", 0.1, 1.0, 0.5)
        minconf = st.slider("Min Confidence", 0.1, 1.0, 0.7)
        rules, fig = apriori.run_apriori(df, minsup, minconf)

        st.write("#### Luật sinh ra")
        st.dataframe(rules, use_container_width=True)

        if fig:
            st.write("#### Biểu đồ Support - Confidence - Lift")
            st.pyplot(fig)

    # ================= Naive Bayes =================
    elif option == "Naive Bayes":
        st.subheader("🎯 Phân lớp - Naive Bayes")
        target = st.selectbox("Chọn cột nhãn", df.columns)
        acc, preds, fig = naive_bayes.run_naive_bayes(df, target)

        st.write(f"**Độ chính xác:** {acc:.2f}")
        st.write("#### Kết quả dự đoán")
        st.dataframe(preds, use_container_width=True)

        if fig:
            st.write("#### Confusion Matrix")
            st.pyplot(fig)

    # ================= Decision Tree =================
    elif option == "Decision Tree":
        st.subheader("🌳 Cây quyết định - ID3")
        target = st.selectbox("Chọn cột nhãn", df.columns)
        acc, graph = decision_tree.run_decision_tree(df, target)
        st.write(f"**Độ chính xác:** {acc:.2f}")
        st.graphviz_chart(graph.source)

    # ================= K-means =================
    elif option == "K-means":
        st.subheader("📌 Gom cụm - K-means")
        k = st.slider("Số cụm k", 2, 10, 3)
        clustered, model, fig = kmeans.run_kmeans(df, k)

        st.write("#### Dữ liệu sau phân cụm")
        st.dataframe(clustered, use_container_width=True)

        if fig:
            st.write("#### Biểu đồ Scatter theo cụm")
            st.pyplot(fig)

    # ================= Rough Set =================
    elif option == "Rough Set":
        st.subheader("📐 Luật quyết định - Rough Set")
        target = st.selectbox("Chọn cột quyết định", df.columns)
        reduct, rules = roughset.run_roughset(df, target)

        st.write("#### Reduct tìm được")
        st.write(reduct)

        st.write("#### Các luật sinh ra")
        for r in rules:
            st.markdown(f"- {r}")

