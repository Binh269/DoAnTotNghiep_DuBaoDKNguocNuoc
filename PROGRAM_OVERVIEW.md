# Tổng quan chương trình — Dự báo nhu cầu nước sinh hoạt (Fuzzy Time Series + PSO)

Tài liệu tóm tắt này mô tả cấu trúc, pipeline dữ liệu và thuật toán của hệ thống.

## Mục tiêu
- Xây hệ thống dự báo nhu cầu nước sinh hoạt dựa trên chuỗi thời gian mờ (Fuzzy Time Series).
- Tối ưu tham số mô hình (số khoảng, tỉ lệ chồng lấp) bằng Particle Swarm Optimization (PSO).
- Giao diện trực tuyến bằng Streamlit, dữ liệu lưu trong SQL Server (SSMS).

## Kiến trúc tổng quan
- `app.py`: giao diện Streamlit — lựa chọn nguồn dữ liệu, tiền xử lý, chạy PSO và hiển thị biểu đồ.
- `src/ket_noi_db.py`: kết nối tới SQL Server (pyodbc) và hàm `doc_du_lieu(table_name)` trả về `DataFrame` chuẩn.
- `src/xuly_du_lieu.py`: đọc dữ liệu, chuẩn hóa cột ngày/giá trị, resample (nếu cần) và moving average.
- `src/mohinh/chuoi_thoi_gian_mo.py`: triển khai mô hình chuỗi thời gian mờ (lag-1) với hàm membership (Gaussian / tam giác tùy phiên bản).
- `src/opt/pso.py`: triển khai PSO đơn giản để tối ưu tham số liên tục.
- `src/evaluate.py`: pipeline huấn luyện/kiểm thử; chia train/val/test; gọi PSO và đánh giá.
- `tao_du_lieu_ao.py`: sinh dữ liệu ảo (theo tham số: năm bắt đầu, số ngày) và insert vào bảng `DuLieuNuocAo`.

## Dữ liệu
- Lấy trực tiếp từ SQL Server (không dùng CSV trong pipeline chính).
- Bảng chính: `DuLieuNuoc` (ID, NgayThang, LuongNuoc).
- Bảng dữ liệu ảo: `DuLieuNuocAo` (ID, NgayThang, LuongNuoc) — dùng để test và demo.

## Pipeline dữ liệu & mô hình
1. Đọc dữ liệu từ SSMS bằng `doc_du_lieu(table)` → chuẩn hóa cột `date` và `value`.
2. Tiền xử lý (ở UI):
   - Chọn phân giải: Daily hoặc Monthly (resample/agg).
   - Làm mượt bằng Moving Average (cửa sổ do người dùng chọn).
3. Chuỗi thời gian mờ (lag-1):
   - Phân vùng giá trị thành `so_khoang` khoảng đều.
   - Mỗi giá trị được fuzzify thành membership vector (tam giác hoặc Gaussian).
   - Xây luật A_{t-1} → A_t từ tần suất kết hợp membership.
   - Dự báo 1-step bằng defuzzify trung bình trọng số các trung tâm.
4. Tối ưu tham số bằng PSO (trên tập validation):
   - PSO tối ưu `so_khoang` (số khoảng); có thể tinh chỉnh `overlap` bằng grid search.
   - Lưu kết quả tốt nhất và đánh giá trên tập test (RMSE).

## Thuật toán chính (tóm tắt kỹ thuật)
- Fuzzification:
  - Với mỗi khoảng có trung tâm c_i.
  - Tính membership μ_i(x) (tam giác hoặc Gaussian với sigma phụ thuộc `overlap`).
- Rule building:
  - Với từng cặp (x_{t-1}, x_t), cộng trọng số μ_{t-1}(i) * μ_t(j) cho luật i→j.
  - Chuẩn hóa hàng luật i thành phân phối xác suất hậu quả.
- Forecast:
  - Từ trạng thái hiện tại (membership tại giá trị cuối), kết hợp luật để có phân phối hậu quả.
  - Defuzzify: y = Σ p_j * center_j.
- PSO (tối ưu liên tục):
  - Mỗi hạt biểu diễn vector tham số (ví dụ `[so_khoang, overlap]` hoặc chỉ `[so_khoang]`).
  - Hàm mục tiêu: RMSE trên tập validation (giảm là tốt).
  - PSO: cập nhật vận tốc và vị trí theo w, c1, c2 với ngẫu nhiên r1,r2.

## Thực thi & tham số chính
- Chạy UI: `streamlit run app.py` → chỉnh tham số PSO ở sidebar:
  - `Số hạt` (Particles): 5–30
  - `Số vòng` (Iterations): 5–100
- Tiền xử lý:
  - `Phân giải`: Daily/Monthly
  - `Moving Average window`: 1–60
- Sinh dữ liệu ảo: ở sidebar App hoặc `python tao_du_lieu_ao.py` (script cũng còn để chạy độc lập).

## Hạn chế & ghi chú
- Mô hình mờ lag-1 đơn giản, phù hợp cho dữ liệu có tính Markov bậc 1; không bắt mọi loại chuỗi thời gian.
- Overlap có thể không cải thiện RMSE cho một số dữ liệu — đã bổ sung grid-search để đánh giá.
- Kết nối DB dùng `pyodbc` — pandas cảnh báo khuyến nghị dùng SQLAlchemy, nhưng kết nối hiện tại hoạt động.
- Tốc độ: PSO có thể chậm trên nhiều hạt/số vòng; chạy trên sample nhỏ khi test.

## Muốn mở rộng
- Thay PSO bằng Differential Evolution hoặc Bayesian Optimization để tối ưu hiệu quả hơn.
- Mở rộng mô hình mờ sang mô hình bậc cao (lag > 1) hoặc mô hình có trạng thái ẩn.
- Lưu model đã huấn luyện và dự báo định kỳ vào DB.

---

Xem hướng dẫn nhanh và ví dụ trong `README.md`.
