# 📂 Cấu Trúc Dự Án Dự Báo Nhu Cầu Nước

## 🎯 Tổng Quan

Dự án sử dụng **Chuỗi Thời Gian Mờ (Fuzzy Time Series)** với **K-Means** và **Differential Evolution (DE)** để dự báo nhu cầu nước.

---

## 📦 Cấu Trúc Thư Mục

```
src/
├── tinhtoans.py                 # Hàm tinh toán chung (RMSE, chia tập)
├── toi_uu_hoa.py               # Tối ưu hóa khoảng bằng DE
├── chinh_dieu_phoi_hop.py       # Chương trình chính điều phối pipeline
│
├── mohinh/
│   ├── chuoi_thoi_gian_mo.py    # Model FTS chính (Markov lag-1)
│   └── __pycache__/
│
├── opt/
│   ├── pso.py                   # Particle Swarm Optimization
│   ├── de.py                    # Differential Evolution
│   └── __pycache__/
│
├── pipeline/                     # 7 bước xử lý chính
│   ├── __init__.py
│   ├── buoc_1_xac_dinh_u.py     # Bước 1: Xác định tập nền U
│   ├── buoc_2_phan_cum.py       # Bước 2: Phân cụm K-Means (auto-select k)
│   ├── buoc_3_tao_khoang.py     # Bước 3: Tạo các khoảng mờ
│   ├── buoc_4_toi_uu_de.py      # Bước 4: Tối ưu khoảng DE
│   ├── buoc_5_mo_hoa.py         # Bước 5: Mờ hóa & Luật Markov
│   ├── buoc_6_du_bao.py         # Bước 6: Dự báo trên test
│   ├── buoc_7_danh_gia.py       # Bước 7: Đánh giá kết quả
│   └── __pycache__/
│
├── ket_noi_db.py                # Kết nối SQL Server, tải/lưu dữ liệu
├── xuly_du_lieu.py              # Tiền xử lý dữ liệu (không thay đổi)
├── tao_du_lieu_ao.py            # Tạo dữ liệu ảo (không thay đổi)
├── utils.py                     # Hàm tiện ích cũ (không dùng)
├── evaluate.py                  # Pipeline cũ (không dùng, thay thế bởi chinh_dieu_phoi_hop.py)
└── __pycache__/
```

---

## 🚀 Luồng Xử Lý (7 Bước)

### **Bước 1: Xác Định Tập Nền U** (`buoc_1_xac_dinh_u.py`)
- **Input**: DataFrame tập huấn luyện
- **Output**: vmin, vmax
- **Mục đích**: Xác định phạm vi giá trị

### **Bước 2: Phân Cụm K-Means** (`buoc_2_phan_cum.py`)
- **Input**: DataFrame tập huấn luyện
- **Output**: Số cụm (k), tâm các cụm
- **Mục đích**: Tự động chọn k tối ưu bằng Silhouette Score

### **Bước 3: Tạo Các Khoảng Mờ** (`buoc_3_tao_khoang.py`)
- **Input**: vmin, vmax, tâm cụm, n_khoang
- **Output**: Ranh giới ban đầu (initial_edges), danh sách khoảng mờ
- **Mục đích**: Khởi tạo ranh giới từ trung điểm giữa tâm cụm

### **Bước 4: Tối Ưu Khoảng (DE)** (`buoc_4_toi_uu_de.py`)
- **Input**: Ranh giới ban đầu, tập train/val
- **Output**: Ranh giới tối ưu (best_edges), lịch sử tối ưu
- **Mục đích**: Cải thiện MSE trên tập validation

### **Bước 5: Mờ Hóa & Luật Markov** (`buoc_5_mo_hoa.py`)
- **Input**: Mô hình FTS đã huấn luyện
- **Output**: Membership samples, luật Markov với xác suất
- **Mục đích**: Hiển thị chi tiết fuzzification và quy tắc chuyển tiếp

### **Bước 6: Dự Báo Trên Test** (`buoc_6_du_bao.py`)
- **Input**: Mô hình, tập test
- **Output**: Bảng dự báo (date, actual, forecast)
- **Mục đích**: Sinh ra các dự báo

### **Bước 7: Đánh Giá Kết Quả** (`buoc_7_danh_gia.py`)
- **Input**: Dự báo, giá trị thực tế
- **Output**: MSE, RMSE, MAE, MAPE
- **Mục đích**: Đánh giá hiệu quả mô hình

---

## 📌 Các File Chính

| File | Mục Đích | Người Dùng |
|------|---------|-----------|
| `chinh_dieu_phoi_hop.py` | Điều phối pipeline 7 bước | Streamlit app |
| `toi_uu_hoa.py` | Tối ưu DE | Pipeline |
| `tinhtoans.py` | Công thức chung | Tất cả |
| `mohinh/chuoi_thoi_gian_mo.py` | Model FTS | Pipeline, tối ưu |
| `opt/{pso,de}.py` | Optimizer | Pipeline |
| `ket_noi_db.py` | SQL Server | Streamlit app |
| `xuly_du_lieu.py` | Tiền xử lý | Streamlit app |

---

## 💾 Không Thay Đổi

Những file sau vẫn giữ nguyên chức năng:
- `ket_noi_db.py` - Kết nối cơ sở dữ liệu
- `xuly_du_lieu.py` - Tiền xử lý dữ liệu
- `tao_du_lieu_ao.py` - Dữ liệu ảo
- `opt/pso.py`, `opt/de.py` - Optimizer

---

## 🎬 Cách Sử Dụng

### Chạy Pipeline Trực Tiếp
```python
from chinh_dieu_phoi_hop import chay_pipeline_7_buoc
import pandas as pd

df = pd.read_csv('data/dulieu.csv', parse_dates=['date'])
result = chay_pipeline_7_buoc(df, n_khoang=None)

# Hiển thị kết quả
for step in result['steps']:
    print(f"{step['ten']}: {step['mo_ta']}")
print(f"RMSE: {result['test_rmse']:.4f}")
```

### Chạy Streamlit
```bash
streamlit run app.py
```

---

## 📊 Dòng Dữ Liệu

```
DataFrame Input
    ↓
[Bước 1] Xác định U (vmin, vmax)
    ↓
[Bước 2] K-Means (auto k selection)
    ↓
[Bước 3] Tạo khoảng (initial_edges)
    ↓
[Bước 4] Tối ưu DE (best_edges)
    ↓
[Bước 5] Mờ hóa & Markov
    ↓
[Bước 6] Dự báo Test
    ↓
[Bước 7] Đánh giá (MSE, RMSE, MAE, MAPE)
    ↓
Kết quả Output
```

---

## 🔧 Thay Đổi Chính Từ Phiên Bản Cũ

| Tính Năng | Cũ | Mới |
|----------|-----|-----|
| **Import** | `from evaluate import chay_de_pipeline` | `from chinh_dieu_phoi_hop import chay_pipeline_7_buoc` |
| **Pipeline** | `chay_de_pipeline(df)` | `chay_pipeline_7_buoc(df)` |
| **Tổ Chức** | evaluate.py monolith | 7 file nhỏ riêng lẻ |
| **Tinh toán** | trong utils.py | trong tinhtoans.py |
| **Tối ưu DE** | trong evaluate.py | trong toi_uu_hoa.py |

---

## 📚 Tài Liệu Chi Tiết

Xem `QUIN_TRINH_CHI_TIET.md` để hiểu rõ các công thức toán học và ý nghĩa của mỗi bước.

---

## ✨ Tính Năng

✅ Tự động chọn số khoảng K bằng Silhouette Score  
✅ Lag-1 Markov model cho dự báo  
✅ Gaussian membership functions  
✅ Differential Evolution tối ưu khoảng  
✅ 30% fuzzy + 70% persistence blending  
✅ Hiển thị luật Markov với xác suất chuyển tiếp  
✅ Lịch sử tối ưu DE chi tiết  
✅ Đánh giá đầy đủ (MSE, RMSE, MAE, MAPE)

