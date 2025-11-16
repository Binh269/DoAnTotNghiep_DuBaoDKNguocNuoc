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

    def __init__(self, so_khoang=7, overlap=0.2):
        self.so_khoang = so_khoang
        self.overlap = overlap
        self.khoang = None
        self.trung_tam = None
        self.quan_he = {}  # map index_khoang_t-1 -> list of (index_khoang_t, weight)

    def _tao_khoang(self, values):
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
        """Huấn luyện mô hình (xây luật từ chuỗi series). series là pandas Series index thời gian."""
        values = series.values
        self._tao_khoang(values)
        # fuzzify các giá trị sang chỉ số khoang (giữ membership)
        memberships = [self._membership(x) for x in values]
        # xây luật A_{t-1} -> A_t bằng cách cộng membership
        # dùng trung tâm làm đại diện
        self.quan_he = {i: {} for i in range(self.so_khoang)}
        for t in range(1, len(values)):
            mu_prev = memberships[t - 1]
            mu_cur = memberships[t]
            for i in range(self.so_khoang):
                if mu_prev[i] <= 0:
                    continue
                for j in range(self.so_khoang):
                    if mu_cur[j] <= 0:
                        continue
                    # trọng số cộng dồn - sử dụng cả membership để tạo sự khác biệt khi overlap khác nhau
                    weight = mu_prev[i] * mu_cur[j]
                    self.quan_he[i][j] = self.quan_he[i].get(j, 0.0) + weight
        # chuyển dict sang danh sách trọng số chuẩn hóa
        for i in range(self.so_khoang):
            totale = sum(self.quan_he[i].values())
            if totale > 0:
                for j in self.quan_he[i]:
                    self.quan_he[i][j] /= totale

    def _fuzzify_value(self, x):
        mu = self._membership(x)
        return mu

    def predict_next(self, x_current):
        """Dự báo 1 bước tiếp theo từ giá trị hiện tại x_current.

        Trả về giá trị thực (defuzzified).
        """
        mu = self._fuzzify_value(x_current)
        # tính phân phối hậu quả dựa trên quy tắc và phân phối hiện tại
        result_weights = np.zeros(self.so_khoang)
        for i in range(self.so_khoang):
            if mu[i] <= 0:
                continue
            # nếu không có luật, bỏ qua
            if not self.quan_he.get(i):
                continue
            for j, w in self.quan_he[i].items():
                result_weights[j] += mu[i] * w
        # defuzzify bằng trung bình trọng số trung tâm
        if result_weights.sum() == 0:
            # fallback: trả về giá trị hiện tại
            return x_current
        result_weights = result_weights / result_weights.sum()
        y = np.dot(result_weights, self.trung_tam)
        return float(y)

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
