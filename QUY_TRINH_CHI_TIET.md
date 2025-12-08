# Quy Trình Chi Tiết: Dự Báo Nhu Cầu Nước Bằng Chuỗi Thời Gian Mờ

## Mục Lục
1. [Tổng Quan](#tổng-quan)
2. [Bước 1: Nạp Dữ Liệu](#bước-1-nạp-dữ-liệu)
3. [Bước 2: Tiền Xử Lý Dữ Liệu](#bước-2-tiền-xử-lý-dữ-liệu)
4. [Bước 3: Chia Tập Train/Val/Test](#bước-3-chia-tập-trainvaltest)
5. [Bước 4: K-Means Tự Chọn Số Khoảng](#bước-4-k-means-tự-chọn-số-khoảng)
6. [Bước 5: Tạo Khoảng Mờ](#bước-5-tạo-khoảng-mờ)
7. [Bước 6: Fuzzify Dữ Liệu](#bước-6-fuzzify-dữ-liệu)
8. [Bước 7: Xây Luật Markov Lag-1](#bước-7-xây-luật-markov-lag-1)
9. [Bước 8: Dự Báo & Defuzzify](#bước-8-dự-báo--defuzzify)
10. [Bước 9: Đánh Giá Mô Hình](#bước-9-đánh-giá-mô-hình)

---

## Tổng Quan

**Mục tiêu:** Dự báo giá trị nước (lượng nước hoặc áp lực) tại thời điểm t+1 dựa trên chuỗi thời gian quá khứ.

**Phương pháp:** Chuỗi thời gian mờ (Fuzzy Time Series) kết hợp với:
- **K-Means Clustering**: Tự động xác định số lượng khoảng mờ tối ưu
- **Gaussian Membership Function**: Hàm khai triển mờ dạng chuông
- **Markov Chain (Lag-1)**: Luật chuyển trạng thái từ t-1 → t
- **Blending Strategy**: Kết hợp 30% dự báo mờ + 70% persistence (giá trị trước)

**Kiến Trúc Chính:**
```
Dữ liệu thô
    ↓
[Tiền xử lý] (Moving Average, Resampling)
    ↓
[Train / Val / Test split]
    ↓
[K-Means auto-select k]
    ↓
[Tạo khoảng + Fuzzify]
    ↓
[Xây luật Markov]
    ↓
[Dự báo & Defuzzify]
    ↓
[Đánh giá MSE]
```

---

## Bước 1: Nạp Dữ Liệu

### 1.1 Nguồn Dữ Liệu

Hỗ trợ 3 nguồn:
1. **Dữ liệu thực tế (SSMS)** → Bảng `DuLieuNuoc`
2. **Dữ liệu import (CSV/Excel)** → Bảng `DuLieuNuocImport`
3. **Dữ liệu ảo (sinh tạo)** → Bảng `DuLieuNuocAo`

### 1.2 Quá Trình Nạp

**File:** `src/ket_noi_db.py` → `lay_du_lieu_tu_db()`

```python
# Input: Server, Database, Table name
# Output: DataFrame với cột ['date', 'value']

def lay_du_lieu_tu_db(
    server='BOSS\\SQLEXPRESS',
    database='DuDoanSuDungNuoc',
    table='DuLieuNuoc'
):
    # Bước 1.2.1: Kết nối SQL Server
    conn = pyodbc.connect(DRIVER + SERVER + DATABASE + UID + PWD)
    
    # Bước 1.2.2: Query dữ liệu
    query = f"SELECT [NgayThang], [LuongNuoc] FROM [{table}]"
    df = pd.read_sql(query, conn)  # Kiểu: DataFrame
    
    # Bước 1.2.3: Chuẩn hóa tên cột
    df.columns = ['date', 'value']
    
    # Bước 1.2.4: Chuyển kiểu dữ liệu
    df['date'] = pd.to_datetime(df['date'])        # Kiểu: datetime64[ns]
    df['value'] = pd.to_numeric(df['value'])       # Kiểu: float64
    
    # Bước 1.2.5: Loại bỏ giá trị thiếu
    df = df.dropna(subset=['value'])               # Kiểu: DataFrame (sạch)
    
    # Bước 1.2.6: Sắp xếp theo thời gian
    df = df.sort_values('date').reset_index()      # Kiểu: DataFrame (sorted)
    
    return df
```

### 1.3 Kiểu Dữ Liệu Đầu Ra

```
date (DatetimeIndex):  2020-01-13, 2020-01-14, ..., 2023-12-31
value (float64):       1234.5, 1245.3, ..., 1567.8
                       Shape: (n_samples,)
                       Example: n_samples = 1461 (4 năm hàng ngày)
```

---

## Bước 2: Tiền Xử Lý Dữ Liệu

### 2.1 Các Phép Xử Lý

**File:** `src/xuly_du_lieu.py` → `tien_xu_ly()`

Tùy chọn người dùng chọn từ UI sidebar:

| Tham số | Mô tả | Giá trị mặc định |
|--------|-------|-----------------|
| `phan_giai` | Phân giải: Daily (D) hoặc Monthly (M) | D |
| `cua_so_ma` | Kích thước Moving Average (ngày) | 7 |

### 2.2 Quá Trình Chi Tiết

#### 2.2.1 Xử Lý Giá Trị Thiếu
```python
# Interpolate missing dates
df = df.set_index('date')
df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq='D'))
df['value'] = df['value'].interpolate(method='linear')  # Kiểu: float64
df = df.reset_index()
```
**Output:** Chuỗi không có khoảng trống

#### 2.2.2 Resample (Nếu chọn M - Monthly)
```python
if phan_giai == 'M':
    df_monthly = df.set_index('date').resample('MS')['value'].mean()
    # Kiểu: Series với index DatetimeIndex (first day of month)
    # Giá trị: Trung bình lượng nước trong tháng
```

#### 2.2.3 Moving Average (Làm Mượt)
```python
ma_window = 7  # Cửa sổ 7 ngày
df['ma'] = df['value'].rolling(window=ma_window, center=True).mean()
# Kiểu: Series (float64) với NaN ở đầu/cuối
# Công thức: ma[t] = (value[t-3] + ... + value[t] + ... + value[t+3]) / 7
```

**Ví dụ MA (cửa sổ 3):**
```
value:  [100, 110, 120, 115, 105]
ma:     [NaN, 110, 115, 113.33, NaN]
        └─ (100+110+120)/3 = 110
```

### 2.3 Kiểu Dữ Liệu Đầu Ra Sau Tiền Xử Lý

```
Cột        Kiểu       Mô tả
─────────────────────────────────────────
date       datetime64 Ngày/Tháng
value      float64    Giá trị thực sau resample + interpolate
ma         float64    Moving Average
```

---

## Bước 3: Chia Tập Train/Val/Test

### 3.1 Tỷ Lệ Chia

**File:** `src/utils.py` → `chia_train_val_test()`

```python
def chia_train_val_test(df, train_ratio=0.6, val_ratio=0.2):
    """
    Chia chuỗi thời gian (không shuffle!):
    Train: 60%, Val: 20%, Test: 20%
    """
    n = len(df)
    n_train = int(n * 0.6)
    n_val = int(n * 0.2)
    
    df_train = df[:n_train]           # Dòng 0 → 876 (60%)
    df_val = df[n_train:n_train+n_val]    # Dòng 876 → 1168 (20%)
    df_test = df[n_train+n_val:]          # Dòng 1168 → 1461 (20%)
    
    return df_train, df_val, df_test
```

### 3.2 Kiểu Dữ Liệu

```
df_train:  DataFrame với shape (876, 2)  ← Dùng để train mô hình
df_val:    DataFrame với shape (292, 2)  ← Dùng để tune parameter
df_test:   DataFrame với shape (293, 2)  ← Dùng để test (không dùng khi train)

Mỗi DataFrame chứa cột ['date', 'value']
```

---

## Bước 4: K-Means Tự Chọn Số Khoảng

### 4.1 Mục Tiêu

Tìm số lượng khoảng mờ (k) **tối ưu** bằng **Silhouette Score**.

### 4.2 Quy Trình

**File:** `src/evaluate.py` → `chay_de_pipeline()`

#### 4.2.1 Tạo Tập Dữ Liệu K-Means
```python
X = df_train['value'].values.reshape(-1, 1)
# Kiểu: numpy.ndarray với shape (876, 1)
# Giá trị: [[1234.5], [1245.3], ..., [1567.8]]
```

#### 4.2.2 Lặp Qua k = 2..12
```python
for k in range(2, min(12, n_unique_values)):
    # 4.2.2a: Khởi tạo K-Means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    
    # 4.2.2b: Fit dữ liệu
    kmeans.fit(X)
    # Output:
    #   - kmeans.cluster_centers_: Shape (k, 1) → Tâm các cụm
    #   - kmeans.labels_: Shape (876,) → Nhãn cụm [0, 1, 2, ..., k-1]
    
    # 4.2.2c: Tính Silhouette Score
    score = silhouette_score(X, kmeans.labels_)
    # Công thức:
    #   s(i) = (b(i) - a(i)) / max(a(i), b(i))
    #   a(i) = khoảng cách trung bình tới các điểm cụm i
    #   b(i) = khoảng cách trung bình tới cụm gần nhất khác
    # Phạm vi: [-1, 1]  (1 = tốt, -1 = xấu, 0 = trùng lặp)
```

#### 4.2.3 Chọn k Tốt Nhất
```python
best_k = argmax(scores)  # k với silhouette score cao nhất
best_score = scores[best_k]

# Ví dụ: Nếu scores = {2: 0.45, 3: 0.68, 4: 0.65, 5: 0.62}
#        → best_k = 3, best_score = 0.68
```

### 4.3 Kiểu Dữ Liệu Đầu Ra

```python
chosen_k = 3 or 5 or 7  # int
silhouette_scores = {2: 0.45, 3: 0.68, 4: 0.65, ...}  # dict
```

**Output in Pipeline:**
```
Bước 2: Chọn số cụm K bằng K-Means (auto)
Mô tả: Tự động chọn k tối ưu bằng silhouette score
Kết quả: {
  'chosen_k': 3,
  'scores': {'2': 0.45, '3': 0.68, '4': 0.65, ...}
}
```

---

## Bước 5: Tạo Khoảng Mờ

### 5.1 Xác Định Biên (Edges)

**File:** `src/evaluate.py` → `chay_de_pipeline()`

Sau khi chọn được k (ví dụ k=3), chạy lại K-Means để lấy **tâm các cụm**:

```python
kmeans = KMeans(n_clusters=3, random_state=42)
centers = kmeans.fit(X).cluster_centers_.flatten()
# Output: centers = [1300, 1400, 1500]  (Shape: (3,))
#         Kiểu: numpy.ndarray (float64)

# Sắp xếp
centers_sorted = np.sort(centers)  # [1300, 1400, 1500]

# Tạo biên từ trung điểm giữa các tâm
if len(centers) > 1:
    midpoints = (centers_sorted[:-1] + centers_sorted[1:]) / 2.0
    # Công thức: midpoint[i] = (centers[i] + centers[i+1]) / 2
    # midpoints = [(1300+1400)/2, (1400+1500)/2]
    #           = [1350, 1450]
    
    # Thêm min/max
    vmin = df_train['value'].min()  # 1200
    vmax = df_train['value'].max()  # 1600
    
    edges = [vmin] + midpoints.tolist() + [vmax]
    # edges = [1200, 1350, 1450, 1600]
```

### 5.2 Công Thức Tạo Khoảng

```
Khoảng 1: [1200, 1350)  → Tâm c₁ = 1275
Khoảng 2: [1350, 1450)  → Tâm c₂ = 1400
Khoảng 3: [1450, 1600]  → Tâm c₃ = 1525
```

**Tâm khoảng (centers):**
```python
centers = (edges[:-1] + edges[1:]) / 2.0
# centers = [1275, 1400, 1525]
```

### 5.3 Kiểu Dữ Liệu

```
edges:   numpy.ndarray, shape (k+1,) = (4,)
         [1200, 1350, 1450, 1600]
         
centers: numpy.ndarray, shape (k,) = (3,)
         [1275, 1400, 1525]
```

---

## Bước 6: Fuzzify Dữ Liệu

### 6.1 Hàm Membership (Gaussian)

**File:** `src/mohinh/chuoi_thoi_gian_mo.py` → `_membership()`

Chuyển đổi giá trị thực thành độ thuộc chuỗi mờ bằng **hàm Gaussian**:

```python
def _membership(x):
    """
    Input:  x (float) = giá trị thực (ví dụ 1350)
    Output: mu (ndarray, shape (k,)) = độ thuộc mỗi khoảng
    
    Công thức:
    μᵢ(x) = exp(-(x - cᵢ)² / (2σ²))
    
    σ = width × (0.3 + 0.7 × overlap)
    width = (edges[-1] - edges[0]) / k = (1600-1200)/3 = 133.33
    overlap = 0.2 (default)
    σ = 133.33 × (0.3 + 0.7×0.2) = 133.33 × 0.44 = 58.67
    """
    
    sigma = 58.67
    centers = [1275, 1400, 1525]
    
    # Tính độ thuộc cho mỗi khoảng
    mu = np.zeros(3)
    for i in range(3):
        mu[i] = np.exp(-((x - centers[i])**2) / (2 * sigma**2))
    
    return mu
    # Output: [0.85, 0.45, 0.12]  ← x=1350 thuộc Khoảng 1 nhất
```

### 6.2 Ví Dụ Cụ Thể

```
x = 1350
μ₁(1350) = exp(-((1350-1275)²)/(2×58.67²)) = 0.85  ← Cao nhất
μ₂(1350) = exp(-((1350-1400)²)/(2×58.67²)) = 0.45
μ₃(1350) = exp(-((1350-1525)²)/(2×58.67²)) = 0.12

→ 1350 "thuộc" Khoảng 1 (A₁) nhiều nhất (85%)
```

### 6.3 Lấy Nhãn Khoảng (Argmax)

```python
label = np.argmax(mu)  # argmax([0.85, 0.45, 0.12]) = 0
# label = 0 ← Khoảng 1 (A₁)
```

### 6.4 Fuzzify Toàn Bộ Train Set

```python
for value in df_train['value']:
    mu = _membership(value)          # Shape (k,)
    label = np.argmax(mu)            # int [0, k-1]
    fuzzified_labels.append(label)

# Output: fuzzified_labels = [0, 0, 1, 2, 1, 0, ...]
#         Shape: (876,)
#         Kiểu: list of int
```

### 6.5 Kiểu Dữ Liệu

```
mu (membership):    numpy.ndarray, shape (k,)
                    [0.85, 0.45, 0.12]
                    
label (argmax):     int [0, k-1]
                    0, 1, 2, ...
                    
fuzzified_labels:   numpy.ndarray, shape (n_train,)
                    [0, 0, 1, 2, 1, 0, ...]
```

---

## Bước 7: Xây Luật Markov Lag-1

### 7.1 Khái Niệm

**Markov Chain bậc 1:** Trạng thái tiếp theo (t) chỉ phụ thuộc trạng thái trước (t-1).

```
A₀ → A₁ → A₂ → A₁ → A₀ → ...
└─────────────────────────┘
    fuzzified_labels từ bước 6
```

### 7.2 Xây Luật (Tần Suất Chuyển Tiếp)

**File:** `src/mohinh/chuoi_thoi_gian_mo.py` → `fit()`

```python
def fit(series):
    fuzzified_labels = [0, 0, 1, 2, 1, 0, 1, 2, 2, 1, ...]
    
    # Xây luật: đếm chuyển tiếp
    quan_he = {}  # Dictionary lưu luật
    
    for t in range(1, len(fuzzified_labels)):
        prev_label = fuzzified_labels[t-1]     # t-1
        curr_label = fuzzified_labels[t]       # t
        
        # Luật: prev_label → curr_label
        if prev_label not in quan_he:
            quan_he[prev_label] = {}
        
        if curr_label not in quan_he[prev_label]:
            quan_he[prev_label][curr_label] = 0
        
        quan_he[prev_label][curr_label] += 1
    
    return quan_he
```

### 7.3 Ví Dụ

**Dữ liệu:** `fuzzified_labels = [0, 0, 1, 2, 1, 0, 1, 2, 2, 1]`

**Chuyển tiếp:**
```
t=1: 0 → 0  (label[0] → label[1])
t=2: 0 → 1  (label[1] → label[2])
t=3: 1 → 2  (label[2] → label[3])
t=4: 2 → 1  (label[3] → label[4])
t=5: 1 → 0  (label[4] → label[5])
...
```

**Luật Markov (quan_he):**
```python
quan_he = {
    0: {0: 1, 1: 1},      # Từ A₀: 1 lần → A₀, 1 lần → A₁
    1: {0: 1, 2: 2},      # Từ A₁: 1 lần → A₀, 2 lần → A₂
    2: {1: 2},            # Từ A₂: 2 lần → A₁
}
```

**Chuẩn hóa (Xác Suất):**
```python
# Chuẩn hóa tần suất → xác suất
prob = {
    0: {0: 1/2, 1: 1/2},      # P(A₀→A₀) = 50%, P(A₀→A₁) = 50%
    1: {0: 1/3, 2: 2/3},      # P(A₁→A₀) = 33%, P(A₁→A₂) = 67%
    2: {1: 1},                # P(A₂→A₁) = 100%
}
```

### 7.4 Kiểu Dữ Liệu

```
quan_he: dict
  Key:   int (prev_label) [0, k-1]
  Value: dict
    Key:   int (curr_label) [0, k-1]
    Value: int (tần suất) hoặc float (xác suất)

Ví dụ:
  quan_he[0][1] = 1 (tần suất)
  quan_he[0][1] = 0.5 (xác suất)
```

---

## Bước 8: Dự Báo & Defuzzify

### 8.1 Quy Trình Dự Báo

**File:** `src/mohinh/chuoi_thoi_gian_mo.py` → `predict_next()`

```python
def predict_next(x_current):
    """
    Input:  x_current = giá trị thực tại t (hoặc [x_t-1, x_t])
    Output: y_pred = giá trị dự báo tại t+1
    """
```

#### 8.1.1 Xác Định Trạng Thái Hiện Tại
```python
# Tính membership và lấy nhãn
mu = _membership(x_current)         # [0.85, 0.45, 0.12]
prev_label = np.argmax(mu)          # 0
```

#### 8.1.2 Tra Cứu Luật
```python
# Từ quan_he, tìm luật từ prev_label
prob_mapping = quan_he.get(prev_label, {})
# prob_mapping = {0: 0.5, 1: 0.5}  (nếu prev_label=0)
```

#### 8.1.3 Defuzzify (Giải Mờ)

**Công thức trọng số:**
```python
# Tạo vector trọng số
result_weights = np.zeros(k)
for next_label, weight in prob_mapping.items():
    result_weights[next_label] = weight

# result_weights = [0.5, 0.5, 0.0]  (nếu prev_label=0)
```

**Lấy Weighted Average của Tâm Khoảng:**
```python
fuzzy_pred = np.dot(result_weights, centers)
# fuzzy_pred = [0.5, 0.5, 0.0] · [1275, 1400, 1525]
#            = 0.5×1275 + 0.5×1400 + 0.0×1525
#            = 637.5 + 700 + 0
#            = 1337.5
```

#### 8.1.4 Blending (Kết Hợp Fuzzy + Persistence)

**Chiến lược tránh drift cao:**
```python
persistence = x_current        # Giá trị trước
blend_pred = 0.3 × fuzzy_pred + 0.7 × persistence
#          = 0.3 × 1337.5 + 0.7 × 1350
#          = 401.25 + 945
#          = 1346.25
```

**Ý nghĩa:**
- **30% fuzzy_pred**: Dự báo từ mô hình học được
- **70% persistence**: Giữ giá trị trước (tránh nhảy quá lớn)

### 8.2 Ví Dụ Dự Báo Toàn Bộ Test Set

```python
history = [df_train['value'], df_val['value']]  # Lịch sử
preds = []

for true_val in df_test['value']:
    # Dự báo
    pred_val = model.predict_next(history[-1])
    preds.append(pred_val)
    
    # Cập nhật lịch sử
    history.append(true_val)

# Output:
# preds = [1346.25, 1355.80, 1342.50, ...]
# Shape: (293,)  ← 293 giá trị dự báo cho test set
```

### 8.3 Kiểu Dữ Liệu

```
x_current:        float
mu:               numpy.ndarray, shape (k,)
prev_label:       int [0, k-1]
prob_mapping:     dict {next_label: weight}
result_weights:   numpy.ndarray, shape (k,)
fuzzy_pred:       float
persistence:      float
blend_pred:       float  ← Dự báo cuối cùng
preds:            list of float, shape (n_test,)
```

---

## Bước 9: Đánh Giá Mô Hình

### 9.1 Metrics

**File:** `src/utils.py` → `rmse()`

```python
def rmse(y_true, y_pred):
    """
    Công thức:
    MSE = (1/n) × Σ(y_true - y_pred)²
    RMSE = √MSE = √((1/n) × Σ(y_true - y_pred)²)
    """
    errors = y_true - y_pred
    mse = np.mean(errors ** 2)
    rmse_val = np.sqrt(mse)
    return rmse_val
```

**Ví dụ:**
```
y_true = [1350, 1360, 1340, ...]
y_pred = [1346, 1365, 1335, ...]

errors = [4, -5, 5, ...]
errors² = [16, 25, 25, ...]
MSE = (16 + 25 + 25 + ...) / n
RMSE = √MSE
```

### 9.2 Quy Trình Đánh Giá

```python
# So sánh dự báo vs thực tế trên test set
rmse_test = rmse(df_test['value'].values, preds)
# Output: RMSE = 45.32  (đơn vị: giá trị nước)

# Hiện thị trên UI
st.metric('MSE trên tập Test', f'{rmse_test:.4f}')
```

### 9.3 Kiểu Dữ Liệu

```
y_true:     numpy.ndarray, shape (n_test,)
y_pred:     numpy.ndarray, shape (n_test,)
errors:     numpy.ndarray, shape (n_test,)
mse:        float
rmse:       float  ← Metric cuối cùng
```

---

## Tóm Tắt Quy Trình

| Bước | Tên | Input | Output | Kiểu Chính |
|------|-----|-------|--------|-----------|
| 1 | Nạp Dữ Liệu | Server | DataFrame | (n, 2) |
| 2 | Tiền Xử Lý | DataFrame | DataFrame (MA) | (n, 3) |
| 3 | Train/Val/Test | DataFrame | 3 × DataFrame | (n_train, 2), ... |
| 4 | K-Means | Train set | k, scores | int, dict |
| 5 | Tạo Khoảng | k | edges, centers | 2 × ndarray |
| 6 | Fuzzify | values | labels | ndarray (int) |
| 7 | Luật Markov | labels | quan_he | dict |
| 8 | Dự Báo | history | preds | list of float |
| 9 | Đánh Giá | y_true, preds | RMSE | float |

---

## Các Hàm & File Chính

```
src/
  ├── ket_noi_db.py
  │   ├── lay_du_lieu_tu_db()      # Nạp dữ liệu từ SQL
  │   └── nhap_du_lieu_vao_db()    # Import dữ liệu vào SQL
  ├── xuly_du_lieu.py
  │   └── tien_xu_ly()              # Tiền xử lý (MA, resample)
  ├── utils.py
  │   ├── chia_train_val_test()    # Chia tập dữ liệu
  │   └── rmse()                    # Tính RMSE
  ├── evaluate.py
  │   └── chay_de_pipeline()        # Orchestrate các bước
  └── mohinh/
      └── chuoi_thoi_gian_mo.py
          ├── __init__()            # Khởi tạo
          ├── _tao_khoang()         # Tạo khoảng
          ├── _membership()         # Gaussian membership
          ├── _fuzzify_value()      # Fuzzify một giá trị
          ├── fit()                 # Xây luật Markov
          └── predict_next()        # Dự báo + defuzzify

app.py
  ├── UI Streamlit
  ├── Nạp dữ liệu
  ├── Tiền xử lý
  ├── Chạy pipeline
  └── Hiển thị kết quả & biểu đồ
```

---

## Ví Dụ Hoàn Chỉnh

### Bước 1: Nạp Dữ Liệu
```
SQL: SELECT [NgayThang], [LuongNuoc] FROM [DuLieuNuoc]
↓
df = DataFrame({
  'date': [2020-01-13, 2020-01-14, ...],
  'value': [1234.5, 1245.3, ...]
})
Shape: (1461, 2)
```

### Bước 2: Tiền Xử Lý
```
Cửa sổ MA = 7 ngày
↓
df['ma'] = df['value'].rolling(7).mean()
↓
df sau MA:
  date         value    ma
  2020-01-13   1234.5   NaN
  2020-01-14   1245.3   NaN
  ...
  2020-01-20   1280.0   1255.3
  ...
```

### Bước 3: Chia Tập
```
n = 1461
n_train = 876 (60%)
n_val = 292 (20%)
n_test = 293 (20%)
```

### Bước 4: K-Means
```
Silhouette scores:
  k=2: 0.45
  k=3: 0.68 ← Best
  k=4: 0.65
  ...
→ chosen_k = 3
```

### Bước 5: Khoảng Mờ
```
centers = [1300, 1400, 1500] (từ K-Means)
edges = [1200, 1350, 1450, 1600]
centers (recalc) = [1275, 1400, 1525]
```

### Bước 6: Fuzzify Train Set
```
value=1350:
  mu = [0.85, 0.45, 0.12]
  label = 0

value=1410:
  mu = [0.40, 0.88, 0.25]
  label = 1

...
fuzzified_labels = [0, 0, 1, 2, 1, 0, ...]
```

### Bước 7: Luật Markov
```
quan_he = {
  0: {0: 50, 1: 45},      # 50 lần 0→0, 45 lần 0→1
  1: {0: 30, 2: 60},      # 30 lần 1→0, 60 lần 1→2
  2: {1: 50},             # 50 lần 2→1
}

Chuẩn hóa (xác suất):
quan_he = {
  0: {0: 0.526, 1: 0.474},
  1: {0: 0.333, 2: 0.667},
  2: {1: 1.0},
}
```

### Bước 8: Dự Báo Test Set
```
history = [train_vals, val_vals]

t=1: x_current=val_vals[-1]=1400
     → pred_1 = 1410 (ví dụ)
     
t=2: x_current=test_vals[0]=1405
     → pred_2 = 1408 (ví dụ)
     
...
preds = [1410, 1408, 1415, ...]
```

### Bước 9: Đánh Giá
```
y_true = [1405, 1410, 1415, ...]
y_pred = [1410, 1408, 1420, ...]
errors = [-5, 2, -5, ...]
MSE = 20.5
RMSE = 4.53
```

---

## Tham Số Mặc Định

| Tham số | Giá trị | Mô tả |
|--------|--------|-------|
| `overlap` | 0.2 | Độ chồng lấp Gaussian |
| `lag` | 1 | Markov order |
| `fuzzy_weight` | 0.3 | Trọng lượng mô hình mờ |
| `persistence_weight` | 0.7 | Trọng lượng trì hoãn |
| `train_ratio` | 0.6 | Tỷ lệ train |
| `val_ratio` | 0.2 | Tỷ lệ val |
| `test_ratio` | 0.2 | Tỷ lệ test |
| `k_max` | 12 | K-Means max clusters |
| `ma_window` | 7 | Moving Average window |

---

## Lưu Ý

1. **Không shuffle dữ liệu thời gian** → Giữ nguyên thứ tự để duy trì tính liên tục
2. **Blending 30/70 để tránh drift** → 70% giữ giá trị trước, 30% mô hình
3. **Silhouette Score chọn k** → Tự động tìm số khoảng tối ưu
4. **Lag-1 Markov** → Đơn giản, nhưng có thể mở rộng sang lag-p
5. **Không clipping bounds** → Cho phép dự báo linh hoạt hơn

---

**Tài liệu này cung cấp chi tiết hoàn chỉnh về từng bước xử lý dữ liệu, mô hình toán học, kiểu dữ liệu, và công thức tính toán trong hệ thống dự báo chuỗi thời gian mờ.**
