# Hướng Dẫn Cấu Trúc Code Mới

## Mục Đích Tổ Chức Lại

Tách biệt các chức năng thành các file nhỏ, dễ duy trì, dễ tìm theo các bước:
- **trước**: `evaluate.py` chứa hết tất cả (>400 dòng)
- **sau**: Chia thành 7 file tương ứng 7 bước, mỗi file ~30-50 dòng

---

##  Các File Mới Được Tạo

### 1. **`src/tinhtoans.py`**  NEW
```python
# Công thức tinh toán chung
- rmse(y_true, y_pred)           # Tính RMSE
- chia_train_val_test(df)        # Chia 70/15/15
```
**Thay thế**: `utils.py` (những hàm chính)

---

### 2. **`src/toi_uu_hoa.py`**  NEW
```python
# Tối ưu hóa các khoảng mờ
- toi_uu_khoang_de(n_khoang, df_train, df_val, ...)
    └─ Chứa hàm eval_boundaries() + gọi differential_evolution()
```
**Thay thế**: Phần tối ưu DE trong `evaluate.py`

---

### 3. **`src/pipeline/`** ✨ NEW (Package 7 bước)

#### **`buoc_1_xac_dinh_u.py`**
```python
def buoc_1_xac_dinh_tap_nen_u(df_train):
    # Tìm vmin, vmax
    return {'ten': '...', 'ket_qua': [...], 'chi_tiet': {...}}
```

#### **`buoc_2_phan_cum.py`**
```python
def buoc_2_phan_cum_kmeans(df_train, seed=42):
    # K-Means + Silhouette auto-select k
    return {'ten': '...', 'ket_qua': {...}, 'chi_tiet': {...}}
```

#### **`buoc_3_tao_khoang.py`**
```python
def buoc_3_tao_khoang_ban_dau(vmin, vmax, centers, n_khoang):
    # Tạo initial_edges + fuzzy_intervals
    return {'ten': '...', 'ket_qua': [...], 'chi_tiet': {...}}
```

#### **`buoc_4_toi_uu_de.py`**
```python
def buoc_4_toi_uu_khoang_de(best_edges, iter_history, n_khoang):
    # Tóm tắt kết quả DE
    return {'ten': '...', 'ket_qua': [...], 'chi_tiet': {...}}
```

#### **`buoc_5_mo_hoa.py`**
```python
def buoc_5_mo_hoa_va_markov(model_final, df_train, df_val):
    # Membership + Markov transitions với xác suất
    return {'ten': '...', 'ket_qua': {...}, 'chi_tiet': {...}}
```

#### **`buoc_6_du_bao.py`**
```python
def buoc_6_du_bao(model_final, df_train, df_val, df_test):
    # Dự báo + bảng forecast_table
    return {'ten': '...', 'ket_qua': [...], 'chi_tiet': {...}}
```

#### **`buoc_7_danh_gia.py`**
```python
def buoc_7_danh_gia(preds_test, df_test):
    # Tính MSE, RMSE, MAE, MAPE
    return {'ten': '...', 'ket_qua': {...}, 'chi_tiet': {...}}
```

#### **`__init__.py`** ✨ NEW
```python
# Import tất cả 7 hàm
from .buoc_1_xac_dinh_u import buoc_1_xac_dinh_tap_nen_u
from .buoc_2_phan_cum import buoc_2_phan_cum_kmeans
...
```

---

### 4. **`src/chinh_dieu_phoi_hop.py`** ✨ NEW (Main Orchestrator)
```python
def chay_pipeline_7_buoc(df, n_khoang=None, seed=42):
    """
    Thực hiện pipeline đầy đủ 7 bước.
    
    Flow:
    1. Chia train/val/test
    2. Gọi buoc_1() → lấy vmin, vmax
    3. Gọi buoc_2() → lấy k, centers
    4. Gọi buoc_3() → lấy initial_edges
    5. Gọi toi_uu_khoang_de() → lấy best_edges, iter_history
    6. Huấn luyện mô hình cuối
    7. Gọi buoc_5() → membership, transitions
    8. Gọi buoc_6() → dự báo
    9. Gọi buoc_7() → đánh giá
    10. Return tất cả 'steps' + 'model' + 'preds' + 'test_rmse'
    """
```
**Thay thế**: `evaluate.py` → `chay_de_pipeline()`

---

##  Thay Đổi Trong `app.py`

**Trước:**
```python
from evaluate import chay_de_pipeline
res = chay_de_pipeline(df_input, n_khoang=None)
```

**Sau:**
```python
from chinh_dieu_phoi_hop import chay_pipeline_7_buoc
res = chay_pipeline_7_buoc(df_input, n_khoang=None)
```

 **Lợi ích**: Import từ một hàm duy nhất, dễ hiểu

---

##  Cấu Trúc Kết Quả (Output)

Mỗi bước trả về dict với cấu trúc:
```python
{
    'ten': 'Tên bước',
    'mo_ta': 'Mô tả chi tiết',
    'ket_qua': <dữ liệu hiển thị trên UI>,
    'chi_tiet': {<dữ liệu nội bộ>}
}
```

`chinh_dieu_phoi_hop.py` thu thập tất cả vào `steps = [step1, step2, ..., step7]` và trả về:
```python
{
    'steps': [7 step dicts],
    'best_edges': [...],
    'iter_history': [...],
    'preds': [dự báo test],
    'test_rmse': <số>,
    'model': <mô hình đã huấn luyện>
}
```

---

##  Cách Chạy

### Option 1: Chạy Streamlit (Giao diện web)
```bash
cd e:\DoAnTotNghiep\DuBaoDKNguocNuoc
streamlit run app.py
```

### Option 2: Chạy Python Script Trực Tiếp
```bash
cd e:\DoAnTotNghiep\DuBaoDKNguocNuoc\src
python chinh_dieu_phoi_hop.py
```

---

## Cac File Khong Thay Doi

| File | Ly Do | Tinh Trang |
|------|-------|----------|
| `ket_noi_db.py` | Xu ly SQL Server | Giu nguyen |
| `xuly_du_lieu.py` | Tien xu ly du lieu | Giu nguyen |
| `tao_du_lieu_ao.py` | Du lieu ao | Giu nguyen |
| `mohinh/chuoi_thoi_gian_mo.py` | Model FTS | Giu nguyen |
| `opt/pso.py`, `opt/de.py` | Optimizer | Giu nguyen |

---

## Uu Diem Cau Truc Moi

| Tieu Chi | Truoc | Sau |
|----------|-------|-----|
| So dong/file | 410 dong (1 file) | 30-50 dong (7 file) |
| De tim code | Kho (tat ca trong evaluate.py) | De (file tuong ung buoc) |
| Tai su dung | Kho (monolith) | De (tung ham doc lap) |
| Test Unit | Kho | De (moi buoc rieng) |
| Bao tri | Kho (doc 410 dong) | De (doc ~40 dong/buoc) |
| Hieu luong | Kho nham | De (co chu thich) |

---

## Tai Lieu Them

- `CAU_TRUC_MOI.md` - Tong quan cau truc & cach dung
- `QUIN_TRINH_CHI_TIET.md` - Cong thuc toan, y nghia moi buoc
- `src/pipeline/` moi file co docstring chi tiet

---

## Luoc Do Import

```
app.py
  chinh_dieu_phoi_hop.chay_pipeline_7_buoc()
      tinhtoans: rmse, chia_train_val_test
      toi_uu_hoa: toi_uu_khoang_de()
      mohinh.chuoi_thoi_gian_mo: ChuoiThoiGianMo
      pipeline.*: 7 ham buoc_N()
          (moi pipeline.buocN.py tu import can thiet)
```

---

**Ngày cập nhật**: 2025-12-08  
Trang thai: Hoan thanh & kiem tra syntax OK
