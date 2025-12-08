import numpy as np
import pandas as pd


class ChuoiThoiGianMo:
    """Mô hình chuỗi thời gian mờ đơn giản (lag-1) với phân đoạn đều và hàm membership Gaussian.

    - tham số:
      + so_khoang: số khoảng (int)
      + overlap: tỉ lệ chồng lấp giữa khoảng (0..0.5) - ảnh hưởng sigma của Gaussian

    Thuật toán:
    - Tạo các khoảng đều trên dải giá trị (min..max)
    - Hàm membership Gaussian với sigma phụ thuộc overlap
    - Fuzzify từng giá trị -> tập mờ
    - Xây luật Fuzzy (A_t-1 -> A_t) bằng tần suất
    - Dự báo: với trạng thái mờ hiện tại, lấy trung bình có trọng số các giá trị trung tâm của các hậu quả
    """

    def __init__(self, so_khoang=7, overlap=0.2, lag=1):
        self.so_khoang = so_khoang
        self.overlap = overlap
        self.lag = int(lag) if lag is not None else 1
        self.khoang = None
        self.trung_tam = None
        # quan_he: map previous-state -> dict(next_index -> weight)
        # previous-state is a tuple of length `lag` of integer labels (argmax memberships)
        # for lag=1 the key is (i,) as before
        self.quan_he = {}
        self.steps_info = {}  # lưu thông tin các bước tạo tập mờ để hiển thị
        self.custom_edges = None
        self.custom_centers = None
        self.training_mean = None
        self.training_std = None

    def set_partitions(self, edges=None, centers=None):
        """Cho phép thiết lập biên (edges) và trung tâm (centers) tùy chỉnh trước khi fit."""
        if edges is not None:
            self.custom_edges = np.array(edges, dtype=float)
            # derive centers if not provided
            if centers is None:
                self.custom_centers = (self.custom_edges[:-1] + self.custom_edges[1:]) / 2.0
            else:
                self.custom_centers = np.array(centers, dtype=float)

    def _tao_khoang(self, values):
        # Nếu user đã cung cấp custom_edges / centers thì dùng chúng
        if self.custom_edges is not None and self.custom_centers is not None:
            self.khoang = self.custom_edges
            self.trung_tam = self.custom_centers
            return

        vmin, vmax = np.min(values), np.max(values)
        r = vmax - vmin
        # thêm một chút đệm
        vmin -= 0.01 * r
        vmax += 0.01 * r
        edges = np.linspace(vmin, vmax, self.so_khoang + 1)
        centers = (edges[:-1] + edges[1:]) / 2.0
        self.khoang = edges
        self.trung_tam = centers

    def _membership(self, x):
        # trả về mảng membership cho mỗi khoang
        # sử dụng Gaussian membership function - overlap ảnh hưởng sigma
        mu = np.zeros(self.so_khoang)
        
        # độ lệch chuẩn của Gaussian phụ thuộc vào overlap
        # overlap=0: sigma nhỏ (chặt)
        # overlap=0.5: sigma lớn (rộng)
        width = (self.khoang[-1] - self.khoang[0]) / self.so_khoang
        sigma = width * (0.3 + 0.7 * self.overlap)  # sigma từ 0.3*width đến 1.0*width
        
        for i in range(self.so_khoang):
            center = self.trung_tam[i]
            mu[i] = np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
        
        return mu

    def fit(self, series: pd.Series):
        """Huấn luyện mô hình (xây luật từ chuỗi series). series là pandas Series hoặc numpy array.
        Xây luật Markov bậc-`lag` dựa trên nhãn argmax(membership) ở các thời điểm trước.
        """
        values = series.values if isinstance(series, pd.Series) else np.asarray(series)
        self._tao_khoang(values)
        # lưu trung bình và độ lệch chuẩn của dữ liệu training để chuẩn hóa dự báo
        self.training_mean = float(np.mean(values))
        self.training_std = float(np.std(values))
        # chuẩn bị info về tạo tập mờ
        vmin, vmax = float(np.min(values)), float(np.max(values))
        self.steps_info = {
            'vmin': vmin,
            'vmax': vmax,
            'so_khoang': int(self.so_khoang),
            'overlap': float(self.overlap),
            'lag': int(self.lag),
            'edges': list(map(float, self.khoang)),
            'centers': list(map(float, self.trung_tam))
        }
        # fuzzify các giá trị và lấy nhãn argmax để xây luật Markov bậc-lag
        memberships = [self._membership(x) for x in values]
        labels = [int(np.argmax(mu)) for mu in memberships]

        # Build counts for previous-state (tuple of lag labels) -> next label
        self.quan_he = {}
        for t in range(self.lag, len(values)):
            prev_state = tuple(labels[t - self.lag:t])
            next_label = labels[t]
            if prev_state not in self.quan_he:
                self.quan_he[prev_state] = {}
            self.quan_he[prev_state][next_label] = self.quan_he[prev_state].get(next_label, 0.0) + 1.0

        # normalize probabilities for each prev_state
        for prev_state in list(self.quan_he.keys()):
            total = sum(self.quan_he[prev_state].values())
            if total > 0:
                for j in list(self.quan_he[prev_state].keys()):
                    self.quan_he[prev_state][j] /= total
        # sau khi xây luật, tóm tắt quan hệ (rules) cho mục hiển thị
        # Tóm tắt các quy tắc theo trạng thái trước (prev_state)
        rules_summary = {}
        for prev_state, mapping in self.quan_he.items():
            items = list(mapping.items())
            items_sorted = sorted(items, key=lambda x: x[1], reverse=True)
            rules_summary[str(prev_state)] = [{'to': int(j), 'weight': float(w)} for j, w in items_sorted[:5]]
        self.steps_info['rules_summary'] = rules_summary
        # thêm sample memberships (vài giá trị đầu) để minh họa
        sample_memberships = []
        for idx, x in enumerate(values[:min(10, len(values))]):
            sample_memberships.append({'index': int(idx), 'value': float(x), 'membership': list(map(float, memberships[idx]))})
        self.steps_info['sample_memberships'] = sample_memberships

    def _fuzzify_value(self, x):
        mu = self._membership(x)
        return mu

    def predict_next(self, x_current):
        """Dự báo 1 bước tiếp theo từ giá trị hiện tại x_current.

        Trả về giá trị thực (defuzzified).
        Áp dụng chiến lược ổn định: kết hợp mô hình mờ (70%) với persistence (30%) để tránh drift.
        """
        # Nếu x_current là một iterable (lịch sử), dùng lag giá trị cuối để xác định prev_state
        try:
            is_sequence = hasattr(x_current, '__len__') and not isinstance(x_current, (str, bytes))
        except Exception:
            is_sequence = False

        # Nếu có đủ lag giá trị, xây prev_state từ nhãn argmax của mỗi phần tử
        if is_sequence and len(x_current) >= self.lag and self.lag > 0:
            last_vals = list(x_current)[-self.lag:]
            labels = [int(np.argmax(self._membership(float(v)))) for v in last_vals]
            prev_state = tuple(labels)
            probs = self.quan_he.get(prev_state)
            if probs:
                result_weights = np.zeros(self.so_khoang)
                for j, w in probs.items():
                    result_weights[int(j)] = float(w)
                # defuzzify
                fuzzy_pred = float(np.dot(result_weights, self.trung_tam))
                persistence = float(last_vals[-1])
                # Giảm trọng lượng mờ xuống 30%, tăng persistence (trì hoãn) lên 70% để tránh dự báo cao
                blend_pred = 0.3 * fuzzy_pred + 0.7 * persistence
                return float(blend_pred)

        # Fallback: sử dụng logic 1-lag dựa trên phân phối membership của giá trị hiện tại
        mu = self._fuzzify_value(float(x_current) if not is_sequence else float(x_current[-1]))
        result_weights = np.zeros(self.so_khoang)
        # sum contributions from all prev-states where prev_state contains each label
        # (approximate): for lag>1 we try to aggregate by the last label
        last_label = int(np.argmax(mu))
        # collect probabilities from states whose last element == last_label
        for prev_state, mapping in self.quan_he.items():
            if prev_state[-1] == last_label:
                for j, w in mapping.items():
                    result_weights[int(j)] += float(w) * mu[last_label]

        # if still empty, try any mapping using last_label as key in single-lag form
        if result_weights.sum() == 0:
            # try single-lag direct mapping if exists
            mapping = self.quan_he.get((last_label,), {})
            for j, w in mapping.items():
                result_weights[int(j)] += float(w) * mu[last_label]

        if result_weights.sum() == 0:
            # fallback: persistence
            return float(x_current[-1] if is_sequence else x_current)

        result_weights = result_weights / result_weights.sum()
        fuzzy_pred = float(np.dot(result_weights, self.trung_tam))
        persistence = float(x_current[-1] if is_sequence else x_current)
        blend_pred = 0.7 * fuzzy_pred + 0.3 * persistence
        if self.training_mean is not None and self.training_std is not None:
            lb = self.training_mean - 2 * self.training_std
            ub = self.training_mean + 2 * self.training_std
            blend_pred = np.clip(blend_pred, lb, ub)
        return float(blend_pred)

    def predict(self, series: pd.Series, steps: int = 1):
        """Dự báo tiếp theo theo lô: dùng giá trị cuối cùng của series làm điều kiện ban đầu."""
        last = float(series.values[-1])
        preds = []
        cur = last
        for _ in range(steps):
            nxt = self.predict_next(cur)
            preds.append(nxt)
            cur = nxt
        return np.array(preds)


if __name__ == '__main__':
    # ví dụ nhanh
    import pandas as pd
    from tao_du_lieu_ao import tao_du_lieu_ao

    tao_du_lieu_ao(dest='data/example_ao.csv')
    df = pd.read_csv('data/example_ao.csv', parse_dates=['date'])
    model = ChuoiThoiGianMo(so_khoang=7, overlap=0.2)
    model.fit(df['value'])
    print('Dự báo 7 bước:', model.predict(df['value'], steps=7))
