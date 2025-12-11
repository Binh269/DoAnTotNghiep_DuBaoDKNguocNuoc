# 📊 Ứng dụng Dự báo Chuỗi Thời gian Mờ - K-Means + PSO + Giải mờ Gauss

## 🎯 Mục Đích

Ứng dụng này dự báo giá trị chuỗi thời gian sử dụng **Mô hình Chuỗi Thời gian Mờ (Fuzzy Time Series)** cải tiến:
- **K-means** kết hợp **PSO (Particle Swarm Optimization)** để tìm số cụm $K$ tối ưu.
- **Chỉ số Davies-Bouldin (DBI)** làm hàm mục tiêu để đánh giá chất lượng phân cụm.
- **Hàm Gauss** để mờ hóa dữ liệu (Fuzzification).
- **Mô hình Cao cấp (High-order)** hỗ trợ đa bậc (ví dụ: bậc 1, bậc 3).
- **Dự báo có trọng số thời gian** để tăng độ chính xác.

---

## ✨ Tính Năng Chính

### 1. **Tối ưu hóa K bằng PSO & Davies-Bouldin**
- Sử dụng thuật toán Bầy đàn (PSO) để tìm số cụm $K$.
- Hàm mục tiêu: **Tối thiểu hóa chỉ số Davies-Bouldin (DBI)**.
- DBI thấp nghĩa là các cụm phân tách tốt và gọn gàng hơn.

### 2. **Xử lý Dữ liệu Linh hoạt**
- Hỗ trợ file CSV và Excel.
- **Tự động tổng hợp dữ liệu (Resample)**: Tính trung bình theo Ngày, Tháng hoặc Năm ngay trên giao diện.

### 3. **Fuzzification (Mờ hóa Gauss)**
- Tự động tính toán các khoảng dựa trên tâm cụm K-means.
- Sử dụng hàm Gauss để tính độ thuộc ($\mu$), chuyển dữ liệu số sang tập mờ ($A_1, A_2, \dots$).

### 4. **Quan hệ Mờ (FLRs) & Nhóm (FLRGs)**
- Hỗ trợ chạy song song nhiều bậc quan hệ (Order) cùng lúc.
- Xây dựng nhóm quan hệ mờ phụ thuộc thời gian (Time-dependent FLRGs).

### 5. **Dự báo & Đánh giá**
- Giải mờ dựa trên trọng số thời gian (Time-weighted Defuzzification).
- Tự động tính toán sai số **MSE** và **MAPE**.
- Biểu đồ trực quan so sánh Thực tế vs Dự báo.

### 6. **Xuất Báo Cáo**
- Xuất toàn bộ kết quả (FLRs, FLRGs, Dự báo) ra file **Word (.docx)** chuyên nghiệp.

---

## 🚀 Bắt Đầu Nhanh

### Cài đặt thư viện
```bash
pip install streamlit pandas numpy matplotlib scikit-learn python-docx openpyxl