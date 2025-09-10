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

streamlit run app.py

Mặc định ứng dụng chạy tại: http://localhost:8501


📊 Demo giao diện
1. Chọn thuật toán

Trong sidebar, chọn Apriori / Naive Bayes / Decision Tree / K-means / Rough Set.

2. Upload file CSV

Nhấn 📂 Tải file CSV và chọn dữ liệu.

3. Tùy chỉnh tham số

Apriori → Min Support, Min Confidence.

Naïve Bayes → Chọn cột nhãn (target).

Decision Tree → Chọn cột nhãn, hiển thị cây.

K-means → Chọn số cụm k, hiển thị biểu đồ phân cụm.

Rough Set → Chọn cột quyết định, sinh luật quyết định.

4. Xem kết quả

Bảng luật kết hợp (Apriori).

Độ chính xác & bảng dự đoán (Naïve Bayes).

Cây quyết định (Decision Tree).

Đồ thị phân cụm (K-means).

Danh sách luật quyết định (Rough Set).

📚 Thư viện sử dụng:

pandas

numpy

scikit-learn

mlxtend
 (Apriori)

matplotlib

seaborn

graphviz
 (Decision Tree)

streamlit

👨‍💻 Thành viên thực hiện:

Lê Hùng Vũ Linh

Nguyễn Thị Phương Linh

Huỳnh Khánh An

Nguyễn Thị Mỹ Ánh
