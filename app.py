import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import codecs

from algorithms import apriori, decision_tree_c45, decision_tree_cart, decision_tree_id3, naive_bayes, kmeans, roughset

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
header = "Các thuật toán khai thác dữ liệu"
st.title("📊 " + header.upper())

# Sidebar chọn thuật toán
algorithms = ["Apriori",
    "Rough Set",
    "Naive Bayes",
    "Decision Tree - Quinlan (C4.5)",
    "Decision Tree - ID3 (Entropy)",
    "Decision Tree - CART (Gini)",
    "K-means"]
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
    # if option == "Apriori":
    #     st.subheader("📑 Luật kết hợp - Apriori")
    #     minsup = st.slider("Min Support", 0.1, 1.0, 0.5)
    #     minconf = st.slider("Min Confidence", 0.1, 1.0, 0.7)
    #     rules, fig = apriori.run_apriori(df, minsup, minconf)

    #     st.write("#### Luật sinh ra")
    #     st.dataframe(rules, use_container_width=True)

    #     if fig:
    #         st.write("#### Biểu đồ Support - Confidence - Lift")
    #         st.pyplot(fig)

    if option == "Apriori":
        st.subheader("📑 Luật kết hợp - Apriori")
        minsup = st.slider("Min Support", 0.1, 1.0, 0.5)
        minconf = st.slider("Min Confidence", 0.1, 1.0, 0.7)

        rules, fig1, fig2, log = apriori.run_apriori(df, minsup, minconf)

        st.write("#### Diễn giải chi tiết (Ck, Lk)")
        st.code(log, language="text")

        st.write("#### Luật sinh ra")
        st.dataframe(rules, use_container_width=True)

        if fig1:
            st.write("#### Biểu đồ Support – Confidence – Lift")
            st.pyplot(fig1)

        if fig2:
            st.write("#### Top 10 luật theo Confidence")
            st.pyplot(fig2)



    # ================= Naive Bayes =================
    # elif option == "Naive Bayes":
    #     st.subheader("🎯 Phân lớp - Naive Bayes")
    #     target = st.selectbox("Chọn cột nhãn", df.columns)
    #     acc, preds, fig = naive_bayes.run_naive_bayes(df, target)

    #     st.write(f"**Độ chính xác:** {acc:.2f}")
    #     st.write("#### Kết quả dự đoán")
    #     st.dataframe(preds, use_container_width=True)

    #     if fig:
    #         st.write("#### Confusion Matrix")
    #         st.pyplot(fig)
    elif option == "Naive Bayes":
        st.subheader("🤖 Phân lớp - Naive Bayes")
        target = st.selectbox("Chọn cột nhãn", df.columns)
        use_laplace = st.checkbox("Sử dụng Laplace smoothing (CategoricalNB)", value=False)

        acc, preds, fig = naive_bayes.run_naive_bayes(df, target, use_laplace)
        st.write(f"**Độ chính xác:** {acc:.2f}")
        st.dataframe(preds)
        st.pyplot(fig)
        


    # ================= Decision Tree - C4.5 =================
    # # elif option == "Decision Tree - Quinlan (C4.5)":
    # #     st.subheader("🌳 Cây quyết định - C4.5 (Gain Ratio)")
    # #     target = st.selectbox("Chọn cột nhãn", df.columns)
    # #     acc, graph = decision_tree_c45.run_decision_tree_c45(df, target)
    # #     st.write(f"**Độ chính xác:** {acc:.2f}")
    # #     st.graphviz_chart(graph.source)
    # elif option == "Decision Tree - Quinlan (C4.5)":
    #     st.subheader("🌳 Cây quyết định - C4.5 (Gain Ratio)")
    #     target = st.selectbox("Chọn cột nhãn", df.columns)
    #     acc, graph = decision_tree_c45.run_decision_tree_c45(df, target)
    #     st.write(f"**Độ chính xác:** {acc:.2f}")
    #     st.graphviz_chart(graph.source)
    # elif option == "Decision Tree - Quinlan (C4.5)":
    #     st.subheader("🌳 Cây quyết định - C4.5 (Gain Ratio)")
    #     target = st.selectbox("Chọn cột nhãn", df.columns)

    #     # Các tham số tùy chỉnh
    #     max_depth = st.slider("Giới hạn độ sâu của cây (max_depth)", min_value=1, max_value=10, value=5, step=1)
    #     min_samples_split = st.slider("Số mẫu tối thiểu để chia node (min_samples_split)", min_value=2, max_value=10, value=2, step=1)
    #     show_entropy = st.checkbox("Hiển thị Entropy", value=True)
    #     show_value = st.checkbox("Hiển thị Value (số lượng mẫu từng lớp)", value=True)

    #     # Chạy thuật toán C4.5 với tham số người dùng chọn
    #     acc, graph = decision_tree_c45.run_decision_tree_c45(
    #         df, 
    #         target,
    #         max_depth=max_depth, 
    #         min_samples_split=min_samples_split,
    #         show_entropy=show_entropy,
    #         show_value=show_value
    #     )

    #     st.write(f"**Độ chính xác:** {acc:.2f}")
    #     st.graphviz_chart(graph.source)
    elif option == "Decision Tree - Quinlan (C4.5)":
        st.subheader("🌳 Cây quyết định – C4.5 (Gain Ratio)")

        target = st.selectbox("Chọn cột nhãn", df.columns)
        max_depth = st.slider("Độ sâu tối đa (max_depth)", 1, 20, 5)
        min_samples_split = st.slider("Số mẫu tối thiểu để tách node", 2, 20, 2)
        show_entropy = st.checkbox("Hiển thị entropy", value=True)
        show_value = st.checkbox("Hiển thị giá trị (class counts)", value=True)

        acc, graph = decision_tree_c45.run_decision_tree_c45(
            df,
            target,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            show_entropy=show_entropy,
            show_value=show_value,
        )

        st.write(f"**Độ chính xác:** {acc:.2f}")
        st.graphviz_chart(graph.source)



    # ================= Decision Tree =================
    # elif option == "Decision Tree - ID3 (Entropy)":
    #     st.subheader("🌳 Cây quyết định - ID3 (Entropy)")
    #     target = st.selectbox("Chọn cột nhãn", df.columns)
    #     acc, graph = decision_tree_id3.run_decision_tree(df, target)
    #     st.write(f"**Độ chính xác:** {acc:.2f}")
    #     st.graphviz_chart(graph.source)
    elif option == "Decision Tree - ID3 (Entropy)":
        st.subheader("🌳 Cây quyết định – ID3 (Information Gain)")

        target = st.selectbox("Chọn cột nhãn", df.columns)

        splitter = st.selectbox("Splitter", ["best", "random"])
        max_depth = st.slider("Độ sâu tối đa (max_depth)", 1, 20, 5)
        min_samples_split = st.slider("Số mẫu tối thiểu để tách node", 2, 20, 2)
        min_samples_leaf = st.slider("Số mẫu tối thiểu ở node lá", 1, 20, 1)
        max_features = st.selectbox("Max features", [None, "sqrt", "log2"])
        random_state = st.number_input("Random state", value=42)

        acc, graph = decision_tree_id3.run_decision_tree_id3(
            df,
            target,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=None if max_features == "None" else max_features,
            random_state=random_state,
        )

        st.write(f"**Độ chính xác:** {acc:.2f}")
        st.graphviz_chart(graph.source)


    # ================= Decision Tree - CART =================
    # elif option == "Decision Tree - CART (Gini)":
    #     st.subheader("🌳 Cây quyết định - CART (Gini)")
    #     target = st.selectbox("Chọn cột nhãn", df.columns)
    #     acc, graph = decision_tree_cart.run_decision_tree_cart(df, target)
    #     st.write(f"**Độ chính xác:** {acc:.2f}")
    #     st.graphviz_chart(graph.source)
    elif option == "Decision Tree - CART (Gini)":
        st.subheader("🌳 Cây quyết định – CART (Gini Index)")

        target = st.selectbox("Chọn cột nhãn", df.columns)

        splitter = st.selectbox("Splitter", ["best", "random"])
        max_depth = st.slider("Độ sâu tối đa (max_depth)", 1, 20, 5)
        min_samples_split = st.slider("Số mẫu tối thiểu để tách node", 2, 20, 2)
        min_samples_leaf = st.slider("Số mẫu tối thiểu ở node lá", 1, 20, 1)
        max_features = st.selectbox("Max features", [None, "sqrt", "log2"])
        random_state = st.number_input("Random state", value=42)

        acc, graph = decision_tree_cart.run_decision_tree_cart(
            df,
            target,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=None if max_features == "None" else max_features,
            random_state=random_state,
        )

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
    # elif option == "Rough Set":
    #     st.subheader("📐 Phân tích Rough Set")

    #     # Người dùng chọn cột quyết định
    #     decision_attr = st.selectbox("Chọn cột quyết định", df.columns)

    #     # Người dùng chọn tập thuộc tính điều kiện B
    #     condition_attrs = st.multiselect(
    #         "Chọn tập thuộc tính điều kiện B",
    #         [c for c in df.columns if c != decision_attr],
    #         default=[c for c in df.columns if c != decision_attr]
    #     )

    #     # Tính hệ số phụ thuộc γ_B(C)
    #     if condition_attrs:
    #         gamma = roughset.dependency_degree(df, condition_attrs, decision_attr)
    #         st.write(f"**Độ phụ thuộc γ_B(C):** {gamma:.2f}")

    #     # Người dùng nhập tập đối tượng X để tính xấp xỉ
    #     indices_str = st.text_input("Nhập tập đối tượng X (ví dụ: 0,1,2)")
    #     if indices_str:
    #         try:
    #             X = {int(i.strip()) for i in indices_str.split(",")}
    #             lower = roughset.lower_approx(df, X, condition_attrs)
    #             upper = roughset.upper_approx(df, X, condition_attrs)

    #             st.write(f"**Lower Approx(X):** {sorted(lower)}")
    #             st.write(f"**Upper Approx(X):** {sorted(upper)}")
    #         except Exception as e:
    #             st.error(f"Lỗi nhập X: {e}")

    #     # Tính reducts
    #     reducts = roughset.find_reducts(df, decision_attr)
    #     st.write("#### Reducts tìm được:")
    #     for r in reducts:
    #         st.markdown(f"- {r}")

    #     # Sinh luật quyết định từ reducts
    #     st.write("#### Luật quyết định sinh ra:")
    #     all_rules = roughset.run_roughset(df, decision_attr)
    #     for reduct, rules in all_rules:
    #         st.markdown(f"**Từ reduct {reduct}:**")
    #         for r in rules:
    #             st.markdown(f"- {r}")
    elif option == "Rough Set":
        st.subheader("📘 Thuật toán Rough Set")

        target = st.selectbox("Chọn cột quyết định", df.columns)

        # mặc định chọn tất cả, user bỏ chọn để loại đi
        cond_attrs = st.multiselect(
            "Chọn thuộc tính điều kiện",
            [c for c in df.columns if c != target],
            default=[c for c in df.columns if c != target]
        )

        show_matrix = st.checkbox("Hiển thị ma trận phân biệt", value=False)
        max_reducts = st.slider("Số reducts tối đa hiển thị", 1, 10, 5)

        all_indices = list(range(len(df)))
        approx_set = st.multiselect("Chọn tập X (dòng) để tính xấp xỉ", all_indices)

        results, explanation, df_matrix = roughset.run_roughset(
            df,
            decision_attr=target,
            condition_attrs=cond_attrs,
            show_matrix=show_matrix,
            max_reducts=max_reducts,
            approx_set=set(approx_set) if approx_set else None
        )

        st.text(explanation)

        if show_matrix and df_matrix is not None:
            st.write("#### Ma trận phân biệt")
            st.dataframe(df_matrix, use_container_width=True)



else:
    st.info("⬆️ Vui lòng upload file CSV để bắt đầu.")

# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import codecs

# from algorithms import apriori, decision_tree_c45, decision_tree_cart, decision_tree_id3, naive_bayes, kmeans, roughset

# # ===== Cấu hình trang =====
# st.set_page_config(
#     page_title="Ứng dụng khai thác dữ liệu",
#     page_icon="📊",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # ===== Hàm load CSS =====
# def load_css(file_path):
#     with codecs.open(file_path, "r", "utf-8", errors="ignore") as f:
#         st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# # ===== Chọn giao diện sáng/tối =====
# theme_mode = st.sidebar.radio("🎨 Giao diện", ["Light", "Dark"])
# load_css("assets/base.css")
# if theme_mode == "Dark":
#     load_css("assets/dark.css")
# else:
#     load_css("assets/light.css")

# # ===== Tiêu đề =====
# header = "Các thuật toán khai thác dữ liệu"
# st.title("📊 " + header.upper())

# # Sidebar chọn thuật toán
# algorithms = [
#     "Apriori",
#     "Rough Set",
#     "Naive Bayes",
#     "Decision Tree",
#     "K-means"
# ]
# option = st.sidebar.selectbox("Chọn thuật toán", algorithms)

# # Hiển thị thuật toán đang chọn
# st.markdown(f"### 📌 **Thuật toán đang chọn:** :blue[{option}]", unsafe_allow_html=True)

# # Upload file CSV
# uploaded_file = st.file_uploader("📂 Import Data (CSV)", type="csv")

# if uploaded_file:
#     df = pd.read_csv(uploaded_file)
#     st.subheader("Dữ liệu đầu vào")
#     st.dataframe(df.head(), use_container_width=True)

#     # ================= Apriori =================
#     if option == "Apriori":
#         st.subheader("📑 Luật kết hợp - Apriori")
#         minsup = st.slider("Min Support", 0.1, 1.0, 0.5)
#         minconf = st.slider("Min Confidence", 0.1, 1.0, 0.7)
#         rules, fig = apriori.run_apriori(df, minsup, minconf)

#         st.write("#### Luật sinh ra")
#         st.dataframe(rules, use_container_width=True)

#         if fig:
#             st.write("#### Biểu đồ Support - Confidence - Lift")
#             st.pyplot(fig)

#     # ================= Naive Bayes =================
#     elif option == "Naive Bayes":
#         st.subheader("🎯 Phân lớp - Naive Bayes")
#         target = st.selectbox("Chọn cột nhãn", df.columns)
#         acc, preds, fig = naive_bayes.run_naive_bayes(df, target)

#         st.write(f"**Độ chính xác:** {acc:.2f}")
#         st.write("#### Kết quả dự đoán")
#         st.dataframe(preds, use_container_width=True)

#         if fig:
#             st.write("#### Confusion Matrix")
#             st.pyplot(fig)

#     # ================= Decision Tree =================
#     elif option == "Decision Tree":
#         st.subheader("🌳 Cây quyết định")
#         target = st.selectbox("Chọn cột nhãn", df.columns)
#         method = st.radio("Chọn thuật toán cây", ["ID3 (Entropy)", "C4.5 (Gain Ratio)", "CART (Gini)"])

#         if method.startswith("ID3"):
#             acc, graph = decision_tree_id3.run_decision_tree(df, target)
#         elif method.startswith("C4.5"):
#             acc, graph = decision_tree_c45.run_decision_tree_c45(df, target)
#         else:
#             acc, graph = decision_tree_cart.run_decision_tree_cart(df, target)

#         st.write(f"**Độ chính xác ({method}):** {acc:.2f}")
#         st.graphviz_chart(graph.source)

#     # ================= K-means =================
#     elif option == "K-means":
#         st.subheader("📌 Gom cụm - K-means")
#         k = st.slider("Số cụm k", 2, 10, 3)
#         clustered, model, fig = kmeans.run_kmeans(df, k)

#         st.write("#### Dữ liệu sau phân cụm")
#         st.dataframe(clustered, use_container_width=True)

#         if fig:
#             st.write("#### Biểu đồ Scatter theo cụm")
#             st.pyplot(fig)

#     # ================= Rough Set =================
#     # elif option == "Rough Set":
#     #     st.subheader("📐 Luật quyết định - Rough Set")
#     #     target = st.selectbox("Chọn cột quyết định", df.columns)
#     #     reduct, rules = roughset.run_roughset(df, target)

#     #     st.write("#### Reduct tìm được")
#     #     st.write(reduct)

#     #     st.write("#### Các luật sinh ra")
#     #     for r in rules:
#     #         st.markdown(f"- {r}")
#     elif option == "Rough Set":
#         st.subheader("📐 Phân tích Rough Set")

#         # Người dùng chọn cột quyết định
#         decision_attr = st.selectbox("Chọn cột quyết định", df.columns)

#         # Người dùng chọn tập thuộc tính điều kiện B
#         condition_attrs = st.multiselect(
#             "Chọn tập thuộc tính điều kiện B",
#             [c for c in df.columns if c != decision_attr],
#             default=[c for c in df.columns if c != decision_attr]
#         )

#         # Tính hệ số phụ thuộc γ_B(C)
#         if condition_attrs:
#             gamma = roughset.dependency_degree(df, condition_attrs, decision_attr)
#             st.write(f"**Độ phụ thuộc γ_B(C):** {gamma:.2f}")

#         # Người dùng nhập tập đối tượng X để tính xấp xỉ
#         indices_str = st.text_input("Nhập tập đối tượng X (ví dụ: 0,1,2)")
#         if indices_str:
#             try:
#                 X = {int(i.strip()) for i in indices_str.split(",")}
#                 lower = roughset.lower_approx(df, X, condition_attrs)
#                 upper = roughset.upper_approx(df, X, condition_attrs)

#                 st.write(f"**Lower Approx(X):** {sorted(lower)}")
#                 st.write(f"**Upper Approx(X):** {sorted(upper)}")
#             except Exception as e:
#                 st.error(f"Lỗi nhập X: {e}")

#         # Tính reducts
#         reducts = roughset.find_reducts(df, decision_attr)
#         st.write("#### Reducts tìm được:")
#         for r in reducts:
#             st.markdown(f"- {r}")

#         # Sinh luật quyết định từ reducts
#         st.write("#### Luật quyết định sinh ra:")
#         all_rules = roughset.run_roughset(df, decision_attr)
#         for reduct, rules in all_rules:
#             st.markdown(f"**Từ reduct {reduct}:**")
#             for r in rules:
#                 st.markdown(f"- {r}")

# else:
#     st.info("⬆️ Vui lòng upload file CSV để bắt đầu.")
