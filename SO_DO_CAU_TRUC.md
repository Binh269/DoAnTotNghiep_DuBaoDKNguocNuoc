# So Do Cau Truc Du An

## Cau Truc Thu Muc

Cau truc du an theo phuong phap modular:

DuBaoDKNguocNuoc/
  app.py                          [ENTRY POINT - Streamlit UI]
  src/                            [SOURCE CODE]
    chinh_dieu_phoi_hop.py        [ORCHESTRATOR - Main Pipeline]
    tinhtoans.py                  [UTILS - Tinh Toan]
    toi_uu_hoa.py                 [OPTIMIZER - DE Wrapper]
    ket_noi_db.py                 [DB Connection - Giu nguyen]
    xuly_du_lieu.py               [Data Processing - Giu nguyen]
    tao_du_lieu_ao.py             [Synthetic Data - Giu nguyen]
    pipeline/                     [STEPS PACKAGE - 7 Buoc]
      __init__.py
      buoc_1_xac_dinh_u.py        [Step 1: Xac dinh U]
      buoc_2_phan_cum.py          [Step 2: K-Means]
      buoc_3_tao_khoang.py        [Step 3: Tao Khoang]
      buoc_4_toi_uu_de.py         [Step 4: Toi Uu DE]
      buoc_5_mo_hoa.py            [Step 5: Mo Hoa & Markov]
      buoc_6_du_bao.py            [Step 6: Du Bao]
      buoc_7_danh_gia.py          [Step 7: Danh Gia]
      __pycache__/
    mohinh/                       [MODEL PACKAGE]
      chuoi_thoi_gian_mo.py       [FTS Model - Giu nguyen]
      __pycache__/
    opt/                          [OPTIMIZER PACKAGE]
      pso.py                      [PSO - Giu nguyen]
      de.py                       [DE - Giu nguyen]
      __pycache__/
    __pycache__/
  data/                           [DATA FOLDER]
    dulieu.csv
  README.md
  PROGRAM_OVERVIEW.md
  QUY_TRINH_CHI_TIET.md
  CAU_TRUC_MOI.md
  HUONG_DAN_CAU_TRUC.md
  requirements.txt

```
┌─────────────────────────────┐
│   app.py (Streamlit UI)     │
└──────────────┬──────────────┘
               │
               ↓ (DataFrame)
┌──────────────────────────────────────┐
│  chinh_dieu_phoi_hop.py              │
│  chay_pipeline_7_buoc(df)            │
└──────────────┬───────────────────────┘
               │
       ┌───────┼────────────────┐
       ↓       ↓                ↓
   [SPLIT]  [MODEL]        [EXECUTE 7 STEPS]
       │       │                │
       ↓       ↓                ↓
    Train   Mô hình      Bước 1→2→...→7
    Val     FTS final
    Test    (lag-1)
       │       │                │
       ├───────┴────────────────┤
       │
       ↓
    [Bước 1] Tập nền U
       ↓ (vmin, vmax)
    [Bước 2] K-Means auto-select
       ↓ (k, centers)
    [Bước 3] Tạo khoảng
       ↓ (initial_edges)
    [Bước 4] Tối ưu DE
       ↓ (best_edges)
    [Huấn luyện] Mô hình cuối
       ↓ (model_final)
    [Bước 5] Mờ hóa & Markov
       ↓ (transitions)
    [Bước 6] Dự báo test
       ↓ (preds)
    [Bước 7] Đánh giá
       ↓ (MSE, RMSE, MAE, MAPE)
       │
       ↓
┌──────────────────────────────┐
│  Kết quả: steps[] + model +  │
│  preds + test_rmse           │
└──────────────┬───────────────┘
               │
               ↓ (Display on UI)
       Streamlit st.subheader()
       st.dataframe()
       st.metric()
       st.line_chart()

```

---

## 📋 Dependency Graph

```
chinh_dieu_phoi_hop.py (Main)
  │
  ├─→ tinhtoans.py
  │     └─→ numpy, pandas
  │
  ├─→ toi_uu_hoa.py
  │     ├─→ tinhtoans.py
  │     ├─→ opt/de.py
  │     └─→ mohinh/chuoi_thoi_gian_mo.py
  │
  ├─→ mohinh/chuoi_thoi_gian_mo.py
  │     └─→ numpy
  │
  ├─→ pipeline/buoc_1_xac_dinh_u.py
  │     └─→ numpy
  │
  ├─→ pipeline/buoc_2_phan_cum.py
  │     ├─→ numpy
  │     └─→ sklearn (KMeans, silhouette_score)
  │
  ├─→ pipeline/buoc_3_tao_khoang.py
  │     └─→ numpy
  │
  ├─→ pipeline/buoc_4_toi_uu_de.py
  │     └─→ numpy
  │
  ├─→ pipeline/buoc_5_mo_hoa.py
  │     ├─→ numpy
  │     └─→ pandas
  │
  ├─→ pipeline/buoc_6_du_bao.py
  │     └─→ pandas
  │
  └─→ pipeline/buoc_7_danh_gia.py
        ├─→ numpy
        └─→ tinhtoans.py

```

---

## 🎬 Execution Timeline

```
Khi user click "Chạy mô hình" trong app.py:

1. app.py: Load data (CSV/Excel/DB)
2. app.py: Call chay_pipeline_7_buoc(df)
   │
   3. chinh_dieu_phoi_hop.py:
      ├─ chia_train_val_test(df)
      ├─ buoc_1(): vmin, vmax
      ├─ buoc_2(): KMeans → k, centers
      ├─ buoc_3(): initial_edges
      ├─ toi_uu_khoang_de(): 
      │  └─ DE loop → best_edges
      ├─ Huấn luyện model_final
      ├─ buoc_5(): membership, transitions
      ├─ buoc_6(): predictions
      ├─ buoc_7(): MSE/RMSE/MAE/MAPE
      │
      └─ Return {steps[], model, preds, test_rmse}
   │
4. app.py: Hiển thị kết quả trên UI
   ├─ Render 7 step (st.subheader, st.dataframe)
   ├─ Show metric (test_rmse)
   └─ Show chart (actual vs forecast)

```

---

## 📊 Thay Đổi So Với Phiên Bản Cũ

### Trước (Old evaluate.py)
```
evaluate.py (410 lines)
  ├─ objective_pso()
  ├─ chay_psu_toi_uu()
  └─ chay_de_pipeline()
       ├─ [step 1 inline] (10 lines)
       ├─ [step 2 inline] (30 lines)
       ├─ [step 3 inline] (20 lines)
       ├─ [step 4 inline] (25 lines)
       ├─ [step 5 inline] (15 lines)
       ├─ [step 6 inline] (20 lines)
       └─ [step 7 inline] (5 lines)
```

### Sau (New modular)
```
chinh_dieu_phoi_hop.py (100 lines)
  └─ chay_pipeline_7_buoc()
       ├─ buoc_1() ← pipeline/buoc_1_xac_dinh_u.py
       ├─ buoc_2() ← pipeline/buoc_2_phan_cum.py
       ├─ buoc_3() ← pipeline/buoc_3_tao_khoang.py
       ├─ buoc_4() ← pipeline/buoc_4_toi_uu_de.py (+ toi_uu_hoa.py)
       ├─ buoc_5() ← pipeline/buoc_5_mo_hoa.py
       ├─ buoc_6() ← pipeline/buoc_6_du_bao.py
       └─ buoc_7() ← pipeline/buoc_7_danh_gia.py

+ tinhtoans.py (20 lines)
+ toi_uu_hoa.py (50 lines)
+ pipeline/__init__.py (15 lines)

TOTAL: 100 + 20 + 50 + 15 + 7×40 = ~360 lines
       (distributed across 9 files instead of 1)
```


## 🔍 Quick Reference

| Tìm | Xem File |
|-----|----------|
| Công thức RMSE, chia tập | `tinhtoans.py` |
| Tối ưu DE | `toi_uu_hoa.py` |
| Tập nền U | `pipeline/buoc_1_xac_dinh_u.py` |
| K-Means | `pipeline/buoc_2_phan_cum.py` |
| Tạo khoảng | `pipeline/buoc_3_tao_khoang.py` |
| Tối ưu DE detail | `pipeline/buoc_4_toi_uu_de.py` |
| Membership + Markov | `pipeline/buoc_5_mo_hoa.py` |
| Dự báo | `pipeline/buoc_6_du_bao.py` |
| Đánh giá | `pipeline/buoc_7_danh_gia.py` |
| Điều phối chính | `chinh_dieu_phoi_hop.py` |
| FTS Model | `mohinh/chuoi_thoi_gian_mo.py` |
| DE Optimizer | `opt/de.py` |
| UI | `app.py` |

---


