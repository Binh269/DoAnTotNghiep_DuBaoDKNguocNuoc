# Quy Trinh Chi Tiet: Du Bao Nhu Cau Nuoc Bang Chuoi Thoi Gian Mo

PHIEN BAN CAP NHAT 2025-12-08: Dang theo cau truc moi voi pipeline 7 buoc

## Muc Luc
1. [Tong Quan](#tong-quan)
2. [Buoc 1: Xac Dinh Khoang U](#buoc-1-xac-dinh-khoang-u)
3. [Buoc 2: Fuzzify Du Lieu](#buoc-2-fuzzify-du-lieu)
4. [Buoc 3: Xay Dung Luat Markov](#buoc-3-xay-dung-luat-markov)
5. [Buoc 4: Du Bao Rang Sua](#buoc-4-du-bao-rang-sua)
6. [Buoc 5: Duoi Mo (Defuzzify)](#buoc-5-duoi-mo-defuzzify)
7. [Buoc 6: Hop Nhat Ket Qua](#buoc-6-hop-nhat-ket-qua)
8. [Buoc 7: Danh Gia Mo Hinh](#buoc-7-danh-gia-mo-hinh)

---

## Tong Quan

Muc tieu: Du bao gia tri nuoc (luong nuoc hoac ap luc) tai thoi diem t+1 dua tren chuoi thoi gian qua khu.

Phuong phap: Chuoi thoi gian mo (Fuzzy Time Series) ket hop voi:
- K-Means Clustering: Tu dong xac dinh so luong khoang mo toi uu
- Gaussian Membership Function: Ham khai trien mo dang chuong
- Markov Chain (Lag-1): Luat chuyen trang thai tu t-1 sang t
- Blending Strategy: Ket hop 30% du bao mo + 70% persistence (gia tri truoc)

Kien truc chinh:
```
Du lieu tho
[Tien xu ly] (Moving Average, Resampling)
[Train / Val / Test split]
[K-Means auto-select k]
[Tao khoang + Fuzzify]
[Xay luat Markov]
[Du bao & Defuzzify]
[Danh gia MSE]
```

---

## Buoc 1: Xac Dinh Khoang U

Duong dan tep: src/pipeline/buoc_1_xac_dinh_u.py

Muc tieu: Tao cac khoang U dua tren K-Means clustering

### 1.1 Qui Trinh Chi Tiet

Buoc 1.1: Tien xu ly (doc du lieu + tien xu ly)
- Goi ham lay_du_lieu_tu_db() tu ket_noi_db.py
- Tien xu ly: moving average, interpolate missing values
- Output: DataFrame voi cot ['date', 'value']

Buoc 1.2: Chia Train/Val/Test
- Train: 70%, Val: 15%, Test: 15%
- Khong shuffle - giu nguyen thu tu thoi gian
- Goi ham chia_train_val_test() tu src/tinhtoans.py

Buoc 1.3: K-Means auto-select k
- Dung Silhouette Score de chon k toi uu (k=2..12)
- X = df_train['value'].values.reshape(-1, 1)
- Lap qua cac k, tinh score, chon best_k
- Output: centers (tam cum) va best_k

Buoc 1.4: Tao ranh gioi (edges)
- Sap xep centers = sorted(centers)
- Tao midpoints giua cac tam cum
- Them min/max: edges = [vmin] + midpoints + [vmax]
- Output: edges dung de tao cac khoang

### 1.2 Vi Du Cu The

Neu centers = [1300, 1400, 1500]:
- vmin = 1200, vmax = 1600
- midpoints = [1350, 1450]
- edges = [1200, 1350, 1450, 1600]

Cac khoang:
- U_0: [1200, 1350)
- U_1: [1350, 1450)
- U_2: [1450, 1600]

Tams (centers): [1275, 1400, 1525]

### 1.3 Dau Ra

```python
{
  "ten": "Buoc 1: Xac Dinh Khoang U",
  "mo_ta": "Tu dong xac dinh so khoang va tao cac khoang mo theo K-Means",
  "ket_qua": {
    "chosen_k": 3,
    "edges": [1200.0, 1350.0, 1450.0, 1600.0],
    "centers": [1275.0, 1400.0, 1525.0],
    "silhouette_scores": {"2": 0.45, "3": 0.68, "4": 0.65}
  },
  "chi_tiet": {
    "train_size": 876,
    "val_size": 292,
    "test_size": 293
  }
}
```

---

## Buoc 2: Fuzzify Du Lieu

Duong dan tep: src/pipeline/buoc_2_fuzzify.py

Muc tieu: Chuyen du lieu thuc sang du lieu mo

### 2.1 Ham Membership (Gaussian)

Chuyen gia tri thuc x sang vector membership:

```
mu_i(x) = exp(-(x - c_i)^2 / (2 * sigma^2))

sigma = width * (0.3 + 0.7 * overlap)
width = (edges[-1] - edges[0]) / k
overlap = 0.2 (default)
```

Vi du: x = 1350
- mu_0(1350) = 0.85
- mu_1(1350) = 0.45
- mu_2(1350) = 0.12
- label = argmax([0.85, 0.45, 0.12]) = 0

### 2.2 Fuzzify Toan Bo Train Set

- Lap qua moi gia tri trong df_train
- Tinh membership vector cho moi gia tri
- Lay argmax de xac dinh nhan khoang
- Output: fuzzified_labels = [0, 0, 1, 2, 1, 0, ...]

### 2.3 Dau Ra

```python
{
  "ten": "Buoc 2: Fuzzify Du Lieu",
  "mo_ta": "Chuyen du lieu thuc sang du lieu mo dung Gaussian membership",
  "ket_qua": {
    "fuzzified_train": [0, 0, 1, 2, 1, 0, 1, 2, 2, 1],
    "n_fuzzified": 876,
    "unique_labels": [0, 1, 2]
  },
  "chi_tiet": {
    "sigma": 58.67,
    "membership_example": {"x": 1350, "mu": [0.85, 0.45, 0.12]}
  }
}
```

---

## Buoc 3: Xay Dung Luat Markov

Duong dan tep: src/pipeline/buoc_3_xay_luat_markov.py

Muc tieu: Xay dung luat Markov lag-1 tu du lieu mo

### 3.1 Khai Niem Markov Lag-1

Trang thai tiep theo (t) chi phu thuoc trang thai truoc (t-1):

A_0 -> A_1 -> A_2 -> A_1 -> A_0 -> ...
(fuzzified_labels tu buoc 2)

### 3.2 Xay Luat

Voi du lieu: fuzzified_labels = [0, 0, 1, 2, 1, 0, 1, 2, 2, 1]

Dem so chuyen tiep:
```
t=1: 0 -> 0 (1 lan)
t=2: 0 -> 1 (1 lan)
t=3: 1 -> 2 (1 lan)
t=4: 2 -> 1 (1 lan)
t=5: 1 -> 0 (1 lan)
...
```

Tuan so:
```python
quan_he = {
  0: {0: 1, 1: 1},      # Tu A_0: 1 lan -> A_0, 1 lan -> A_1
  1: {0: 1, 2: 2},      # Tu A_1: 1 lan -> A_0, 2 lan -> A_2
  2: {1: 2},            # Tu A_2: 2 lan -> A_1
}
```

Chuan hoa (xac suat):
```python
quan_he_prob = {
  0: {0: 0.5, 1: 0.5},         # P(A_0->A_0) = 50%, P(A_0->A_1) = 50%
  1: {0: 0.33, 2: 0.67},       # P(A_1->A_0) = 33%, P(A_1->A_2) = 67%
  2: {1: 1.0},                 # P(A_2->A_1) = 100%
}
```

### 3.3 Dau Ra

```python
{
  "ten": "Buoc 3: Xay Dung Luat Markov",
  "mo_ta": "Xay dung luat Markov lag-1 tu cac tran thuc hiep",
  "ket_qua": {
    "quan_he": {"0": {"0": 50, "1": 45}, "1": {"0": 30, "2": 60}, "2": {"1": 50}},
    "quan_he_prob": {"0": {"0": 0.526, "1": 0.474}, "1": {"0": 0.333, "2": 0.667}, "2": {"1": 1.0}}
  },
  "chi_tiet": {
    "n_transitions": 875
  }
}
```

---

## Buoc 4: Du Bao Rang Sua

Duong dan tep: src/pipeline/buoc_4_du_bao_rang_sua.py

Muc tieu: Du bao tren tep val/test dung luat Markov

### 4.1 Qui Trinh Du Bao Mot Gia Tri

Input: x_current (gia tri hien tai)
Output: y_pred (gia tri du bao)

Buoc 4.1.1: Xac dinh trang thai hien tai
- Tinh membership: mu = _membership(x_current)
- Lay nhan: prev_label = argmax(mu)

Buoc 4.1.2: Tra cuu luat
- prob_mapping = quan_he_prob.get(prev_label, {})

Buoc 4.1.3: Tao vector trong so
- result_weights = zeros(k)
- Gan trong so cho cac nhan tiep theo

Buoc 4.1.4: Defuzzify (lay weighted average)
- fuzzy_pred = dot(result_weights, centers)

### 4.2 Dau Ra

```python
{
  "ten": "Buoc 4: Du Bao Rang Sua",
  "mo_ta": "Du bao tren tep val dung luat Markov truoc khi toi uu",
  "ket_qua": {
    "pred_val": [1346.25, 1355.80, 1342.50, ...],
    "n_pred": 292,
    "rmse_val": 45.32
  },
  "chi_tiet": {
    "n_eval": 292
  }
}
```

---

## Buoc 5: Duoi Mo (Defuzzify)

Duong dan tep: src/pipeline/buoc_5_defuzzify.py

Muc tieu: Ap dung strategy hop nhat 30/70

### 5.1 Blending Strategy

Khi du bao, hop nhat:
- 30% tu du bao mo: fuzzy_pred
- 70% tu gia tri truoc: persistence (x_current)

Cong thuc:
```
blend_pred = 0.3 * fuzzy_pred + 0.7 * persistence
```

Vi du:
- fuzzy_pred = 1337.5
- x_current (persistence) = 1350
- blend_pred = 0.3*1337.5 + 0.7*1350 = 1346.25

Y nghia:
- 30%: Mo hinh hoc duoc (fuzzy)
- 70%: Giu gia tri truoc (tranh nhay lon)

### 5.2 Du Bao Toan Bo Test Set

- Su dung du lieu val + train lam lich su
- Lap qua moi gia tri trong test set
- Du bao va cap nhat lich su
- Output: preds = [1346.25, 1355.80, 1342.50, ...]

### 5.3 Dau Ra

```python
{
  "ten": "Buoc 5: Duoi Mo (Defuzzify)",
  "mo_ta": "Du bao tren tep test va ap dung blending 30/70",
  "ket_qua": {
    "pred_test": [1346.25, 1355.80, 1342.50, ...],
    "n_pred": 293,
    "blend_weights": {"fuzzy": 0.3, "persistence": 0.7}
  },
  "chi_tiet": {
    "n_eval": 293
  }
}
```

---

## Buoc 6: Hop Nhat Ket Qua

Duong dan tep: src/pipeline/buoc_6_hop_nhat.py

Muc tieu: Hop nhat du bao va tinh RMSE tren test set

### 6.1 Tinh RMSE

Cong thuc:
```
MSE = (1/n) * sum((y_true - y_pred)^2)
RMSE = sqrt(MSE)
```

Vi du:
- y_true = [1350, 1360, 1340, ...]
- y_pred = [1346, 1365, 1335, ...]
- errors = [4, -5, 5, ...]
- MSE = (16 + 25 + 25 + ...) / n
- RMSE = sqrt(MSE)

### 6.2 Dau Ra

```python
{
  "ten": "Buoc 6: Hop Nhat Ket Qua",
  "mo_ta": "Tong hop du bao va tinh cac chi so danh gia",
  "ket_qua": {
    "rmse": 8.6173,
    "mse": 74.2575,
    "mae": 6.234,
    "mape": 0.451
  },
  "chi_tiet": {
    "y_true": [1350, 1360, ...],
    "y_pred": [1346.25, 1355.80, ...]
  }
}
```

---

## Buoc 7: Danh Gia Mo Hinh

Duong dan tep: src/pipeline/buoc_7_danh_gia.py

Muc tieu: Tinh toan cac chi so danh gia cuoi cung va hien thi

### 7.1 Cac Metric Danh Gia

RMSE: Root Mean Squared Error
- Dao ham: sqrt(mean((y_true - y_pred)^2))
- Pham vi: [0, inf) (0 = toan hao)

MSE: Mean Squared Error
- Dao ham: mean((y_true - y_pred)^2)

MAE: Mean Absolute Error
- Dao ham: mean(|y_true - y_pred|)

MAPE: Mean Absolute Percentage Error
- Dao ham: mean(|y_true - y_pred| / |y_true|) * 100

### 7.2 Hien Thi Ket Qua

- Hien thi RMSE tren giao dien Streamlit
- Ve bieu do: thuc te vs du bao
- Ve bieu do sai so (residuals)
- Hien thi bang cac chi so danh gia

### 7.3 Dau Ra

```python
{
  "ten": "Buoc 7: Danh Gia Mo Hinh",
  "mo_ta": "Tinh toan va hien thi cac chi so danh gia cuoi cung",
  "ket_qua": {
    "rmse": 8.6173,
    "mse": 74.2575,
    "mae": 6.234,
    "mape": 0.451,
    "model_status": "Good"
  },
  "chi_tiet": {
    "evaluation_date": "2025-01-08",
    "test_size": 293
  }
}
```

---

## Tom Tat Qui Trinh 7 Buoc

| Buoc | Ten | Dau Vao | Dau Ra | File |
|------|-----|---------|--------|------|
| 1 | Xac Dinh Khoang U | df_train | edges, centers, k | buoc_1_xac_dinh_u.py |
| 2 | Fuzzify Du Lieu | values | fuzzified_labels | buoc_2_fuzzify.py |
| 3 | Xay Dung Luat Markov | labels | quan_he, quan_he_prob | buoc_3_xay_luat_markov.py |
| 4 | Du Bao Rang Sua | history | pred_val, rmse_val | buoc_4_du_bao_rang_sua.py |
| 5 | Duoi Mo (Defuzzify) | pred_fuzzy | pred_test, blend_weights | buoc_5_defuzzify.py |
| 6 | Hop Nhat Ket Qua | y_true, y_pred | rmse, mse, mae, mape | buoc_6_hop_nhat.py |
| 7 | Danh Gia Mo Hinh | rmse, chi_so | bieu_do, ket_luan | buoc_7_danh_gia.py |

---

## Cau Truc File Chinh

src/
  chinh_dieu_phoi_hop.py           # Main orchestrator - chay 7 buoc
  tinhtoans.py                     # Shared formulas (rmse, chia_train_val_test)
  toi_uu_hoa.py                    # DE optimization wrapper
  ket_noi_db.py                    # SQL Server connection
  xuly_du_lieu.py                  # Data preprocessing
  mohinh/
    chuoi_thoi_gian_mo.py          # Fuzzy time series model
  opt/
    de.py                          # Differential Evolution
  pipeline/
    __init__.py                    # Package init
    buoc_1_xac_dinh_u.py           # Step 1: Determine fuzzy intervals
    buoc_2_fuzzify.py              # Step 2: Fuzzify data
    buoc_3_xay_luat_markov.py      # Step 3: Build Markov rules
    buoc_4_du_bao_rang_sua.py      # Step 4: Rough forecasting
    buoc_5_defuzzify.py            # Step 5: Defuzzify & blend
    buoc_6_hop_nhat.py             # Step 6: Aggregate results
    buoc_7_danh_gia.py             # Step 7: Evaluate model

app.py                              # Streamlit UI
tao_du_lieu_ao.py                  # Synthetic data generator

---

## Tham So Mac Dinh

| Tham So | Gia Tri | Mo Ta |
|--------|--------|-------|
| overlap | 0.2 | Do chong lap Gaussian |
| lag | 1 | Markov order |
| fuzzy_weight | 0.3 | Trong luong mo hinh mo |
| persistence_weight | 0.7 | Trong luong tri hoan |
| train_ratio | 0.7 | Ty le train |
| val_ratio | 0.15 | Ty le val |
| test_ratio | 0.15 | Ty le test |
| k_max | 12 | K-Means max clusters |
| ma_window | 7 | Moving Average window |

---

## Ghi Chu

1. Khong shuffle du lieu thoi gian: Giu nguyen thu tu de duy tri tinh lien tuc
2. Blending 30/70 de tranh drift cao: 70% giu gia tri truoc, 30% mo hinh
---

## Ghi Chu

1. Khong shuffle du lieu thoi gian: Giu nguyen thu tu de duy tri tinh lien tuc
2. Blending 30/70 de tranh drift cao: 70% giu gia tri truoc, 30% mo hinh
3. Silhouette Score chon k: Tu dong tim so khoang toi uu
4. Lag-1 Markov: Don gian, nhung co the mo rong sang lag-p
5. Khong clipping bounds: Cho phep du bao linh hoat hon

---
