# Dự án: Dự báo nhu cầu nước sinh hoạt bằng chuỗi thời gian mờ + PSO

Tài liệu hướng dẫn (Tiếng Việt).

## Mục tiêu
- Kết nối SSMS (SQL Server) để lấy dữ liệu sử dụng nước từ database.
- Xây mô hình chuỗi thời gian mờ xử lý dữ liệu.
- Dùng PSO tối ưu tham số khoảng mờ (số khoảng, overlap).
- Hiển thị kết quả trên hệ thống trực tuyến (Streamlit).

## Yêu cầu

### 1. Cấu hình SSMS
- **Server:** `BOSS\SQLEXPRESS` (local)
- **Database:** `DuDoanSuDungNuoc`
- **Bảng dữ liệu thực tế:** `DuLieuNuoc` (cột: `ID` (int), `NgayThang` (datetime), `LuongNuoc` (float))
- **Bảng dữ liệu ảo:** `DuLieuNuocAo` (cột: `ID` (int), `NgayThang` (datetime), `LuongNuoc` (float))
- **Tài khoản:** `sa` (Windows Authentication)

### 2. Cài đặt ODBC Driver
Đảm bảo máy đã cài **ODBC Driver 17 for SQL Server**. Tải tại:
https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server

## Hướng dẫn cài đặt & chạy

### 1. Tạo môi trường & cài dependencies:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Khởi tạo dữ liệu SSMS:

#### a) Dữ liệu thực tế (DuLieuNuoc)
Nếu bảng `DuLieuNuoc` trống, chạy script:
```powershell
python insert_du_lieu_mau.py
```
Điều này sẽ thêm 3 năm dữ liệu mẫu (từ 2021-01-01 đến 2023-12-31).

#### b) Dữ liệu ảo (DuLieuNuocAo) - **Tùy chọn**
Để tạo dữ liệu ảo trong bảng `DuLieuNuocAo`:
```powershell
python tao_du_lieu_ao.py
```
Hoặc sử dụng nút **"Sinh/cập nhật dữ liệu ảo"** trong giao diện Streamlit.

### 3. Chạy giao diện Streamlit:
```powershell
streamlit run app.py
```
Ứng dụng sẽ:
- Hiển thị tùy chọn chọn nguồn dữ liệu: "Dữ liệu thực tế" hoặc "Dữ liệu ảo"
- Tải dữ liệu từ bảng tương ứng trong SSMS
- Hiển thị 3 biểu đồ (chuỗi thời gian, phân bố, dữ liệu sau xử lý)
- Cho phép tiền xử lý dữ liệu (resample daily/monthly, moving average)
- Chạy tối ưu PSO để tìm tham số tốt nhất

### 4. Chạy demo từ dòng lệnh:
```powershell
python run_demo.py
```

## Cấu trúc thư mục
```
DưBaoDKNguocNuoc/
├── src/
│   ├── ket_noi_db.py           (kết nối SSMS)
│   ├── xuly_du_lieu.py         (đọc dữ liệu, tiền xử lý)
│   ├── mohinh/
│   │   └── chuoi_thoi_gian_mo.py  (mô hình fuzzy lag-1)
│   ├── opt/
│   │   └── pso.py              (PSO optimizer)
│   ├── evaluate.py             (pipeline evaluate)
│   └── utils.py                (tiện ích)
├── app.py                      (giao diện Streamlit)
├── run_demo.py                 (script demo)
├── insert_du_lieu_mau.py       (script thêm dữ liệu mẫu vào DuLieuNuoc)
├── tao_du_lieu_ao.py           (script sinh dữ liệu ảo vào DuLieuNuocAo)
├── requirements.txt            (dependencies)
└── README.md                   (file này)
```

## Ghi chú
- Ứng dụng sử dụng tiếng Việt trong biến, chú thích, và giao diện.
- **Tất cả dữ liệu được quản lý trong SSMS** — không còn dùng file CSV.
- Để thêm/sửa dữ liệu, hãy quản lý trực tiếp trong SSMS (SQL Management Studio).
- Dữ liệu ảo có thể được tạo lại bất kỳ lúc nào bằng cách chạy `tao_du_lieu_ao.py` hoặc dùng nút trong Streamlit.

## Xử lý sự cố

### Lỗi kết nối SSMS
Kiểm tra:
1. SSMS (SQL Server Express) đã bật chưa?
2. ODBC Driver 17 đã cài đặt?
3. Database `DuDoanSuDungNuoc` tồn tại?
4. Các bảng `DuLieuNuoc` và `DuLieuNuocAo` tồn tại và có dữ liệu?

### Bảng trống
Chạy script tương ứng:
- Dữ liệu thực tế: `python insert_du_lieu_mau.py`
- Dữ liệu ảo: `python tao_du_lieu_ao.py`

# Dự án Dự báo Nhu cầu Nước Sinh hoạt

Hệ thống này xây dựng mô hình dự báo nhu cầu nước sinh hoạt dựa trên Chuỗi Thời Gian Mờ (Fuzzy Time Series) kết hợp tối ưu tham số bằng Particle Swarm Optimization (PSO). Giao diện tương tác bằng Streamlit và dữ liệu lưu trong SQL Server (SSMS) — không dùng CSV trong pipeline chính.

Xem tóm tắt chi tiết: `PROGRAM_OVERVIEW.md`

Các file và thư mục chính:
- `app.py`: Streamlit app (UI, tiền xử lý, chạy PSO, hiển thị kết quả).
- `PROGRAM_OVERVIEW.md`: Tóm tắt kiến trúc, thuật toán, cách chạy.
- `src/ket_noi_db.py`: Kết nối SQL Server và hàm đọc/ghi dữ liệu.
- `src/xuly_du_lieu.py`: Tiền xử lý dữ liệu.
- `src/mohinh/chuoi_thoi_gian_mo.py`: Cài đặt mô hình fuzzy.
- `src/opt/pso.py`: Cài đặt PSO.
- `src/evaluate.py`: Pipeline huấn luyện/đánh giá.
- `tao_du_lieu_ao.py`: Sinh dữ liệu ảo và insert vào `DuLieuNuocAo`.

Hướng dẫn nhanh:
1. Kết nối tới SQL Server: sửa thông tin kết nối trong `src/ket_noi_db.py` nếu cần.
2. Cài đặt môi trường Python và thư viện cần thiết (ví dụ `pandas`, `numpy`, `pyodbc`, `streamlit`).
3. Chạy giao diện: `streamlit run app.py` (PowerShell: `streamlit run app.py`).
4. Dùng sidebar để chọn nguồn dữ liệu, tiền xử lý, hoặc sinh dữ liệu ảo và insert vào DB.

Ghi chú:
- Nếu pandas báo cảnh báo về `read_sql`, cân nhắc cài thêm `SQLAlchemy` để loại bỏ warning.
- Một số tệp tạm có thể đang được hệ thống khóa; khởi động lại Windows nếu cần để xóa tệp tạm.

Nếu cần, tôi có thể:
- Thêm hướng dẫn cài môi trường (requirements.txt / venv).
- Chạy kiểm thử nhanh hoặc commit thay đổi.
