Data Mining Algorithms App

Ứng dụng minh họa các thuật toán khai thác dữ liệu bằng Python, có giao diện web trực quan nhờ Streamlit.

Chức năng:

Cài đặt & demo 5 thuật toán cơ bản:
- Apriori → sinh luật kết hợp từ dữ liệu giỏ hàng.
- Rough Set → sinh luật quyết định từ bảng quyết định.
- Naïve Bayes → phân lớp dữ liệu dựa vào xác suất.
- Decision Tree (ID3) → phân lớp bằng cây quyết định.
- K-means → gom cụm dữ liệu 2D/đa chiều.
- Hỗ trợ upload file CSV để thử nghiệm.
- Có thể chọn tham số đầu vào (minsup, minconf, số cụm k, cột nhãn…).
- Hiển thị kết quả trực quan (bảng, cây quyết định, đồ thị phân cụm).

⚙️ Cài đặt môi trường

1. Cài đặt thư viện
pip install -r requirements.txt

2. 🚀 Chạy ứng dụng
cd DataMining
streamlit run app.py

Mặc định ứng dụng chạy tại: http://localhost:8501