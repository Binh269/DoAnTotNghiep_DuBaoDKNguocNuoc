import os
import sys
import numpy as np
import pandas as pd
from functools import partial

# thêm src vào sys.path để import module nội bộ
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mohinh.chuoi_thoi_gian_mo import ChuoiThoiGianMo
from opt.pso import PSO
from opt.de import differential_evolution
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from utils import rmse, chia_train_val_test


def objective_pso(candidate, df_train, df_val):
    """candidate: mảng [so_khoang_continuous, overlap]
    chuyển so_khoang thành int trong khoảng [3, 30]
    overlap trong khoảng [0, 0.5]
    trả về RMSE trên tập validation (nhỏ hơn là tốt)
    """
    so_khoang = int(round(candidate[0]))
    so_khoang = max(3, min(so_khoang, 30))
    overlap = float(candidate[1])
    overlap = max(0.0, min(overlap, 0.5))

    # huấn luyện mô hình trên df_train
    model = ChuoiThoiGianMo(so_khoang=so_khoang, overlap=overlap)
    model.fit(df_train['value'])
    # dự báo cho tập validation: mỗi bước 1 dùng giá trị thực tại
    preds = []
    # để đơn giản, dùng dự báo 1-step lăn
    series = pd.concat([df_train['value'], df_val['value']])
    window = len(df_train)
    # bắt đầu từ phần train, dự báo từng bước và so sánh với val
    history = list(df_train['value'].values)
    for true in df_val['value'].values:
        pred = model.predict_next(history[-model.lag:])
        preds.append(pred)
        history.append(true)
    return rmse(df_val['value'].values, preds)


def chay_psu_toi_uu(df, so_hat=10, vong=30):
    df_train, df_val, df_test = chia_train_val_test(df)
    steps = []
    
    # Bước 1: PSO tối ưu so_khoang với overlap cố định
    bounds_so_khoang = [(3, 30)]
    
    def obj_so_khoang(candidate):
        so_khoang = int(round(candidate[0]))
        so_khoang = max(3, min(so_khoang, 30))
        model = ChuoiThoiGianMo(so_khoang=so_khoang, overlap=0.2)
        model.fit(df_train['value'])
        history = list(df_train['value'].values)
        preds = []
        for true in df_val['value'].values:
            pred = model.predict_next(history[-model.lag:])
            preds.append(pred)
            history.append(true)
        return rmse(df_val['value'].values, preds)
    
    pso = PSO(obj_so_khoang, bounds_so_khoang, n_particles=so_hat, n_iter=vong)
    best_so_khoang, best_so_khoang_val = pso.run()
    so_khoang_opt = int(round(best_so_khoang[0]))
    # Thêm bước tóm tắt PSO
    steps.append({
        'ten': 'Tối ưu số khoảng mờ (PSO)',
        'mo_ta': f'PSO với số hạt={so_hat}, số vòng={vong}',
        'ket_qua': {
            'so_khoang_opt': so_khoang_opt,
            'val_rmse': float(best_so_khoang_val)
        }
    })

    # Tạo mô hình tạm với so_khoang_opt để lấy thông tin chi tiết các bước fuzzification
    try:
        temp_model = ChuoiThoiGianMo(so_khoang=so_khoang_opt, overlap=0.2)
        temp_model.fit(df_train['value'])
        si = getattr(temp_model, 'steps_info', {})
        # Chuẩn hóa tên bước theo 7 bước trong mô tả ảnh (hiển thị 6 bước trước huấn luyện cuối)
        # Bước 1: Xác định tập nền U
        steps.append({
            'ten': 'Bước 1: Xác định tập nền U',
            'mo_ta': 'Tìm giá trị min/max trên tập train và mở rộng để định nghĩa tập nền U (Universe).',
            'ket_qua': {'vmin': si.get('vmin'), 'vmax': si.get('vmax')}
        })
        # Bước 2: Chia tập U thành các khoảng
        steps.append({
            'ten': 'Bước 2: Chia tập U thành các khoảng',
            'mo_ta': f'Chia tập nền U thành {si.get("so_khoang")} khoảng đều (partition).',
            'ket_qua': {'edges': si.get('edges'), 'centers': si.get('centers')}
        })
        # Bước 3: Xác định các tập mờ (membership functions)
        width = None
        sigma = None
        try:
            edges_arr = np.array(si.get('edges', []), dtype=float)
            so_k = int(si.get('so_khoang', len(si.get('centers', []))))
            width = float((edges_arr[-1] - edges_arr[0]) / max(1, so_k))
            sigma = float(width * (0.3 + 0.7 * float(si.get('overlap', 0.2))))
        except Exception:
            pass
        steps.append({
            'ten': 'Bước 3: Xác định các tập mờ',
            'mo_ta': 'Chọn dạng hàm membership (ví dụ Gaussian) và tính sigma phụ thuộc overlap.',
            'ket_qua': {'overlap': si.get('overlap'), 'sigma': sigma}
        })
        # Bước 4: Fuzzify dữ liệu (gán membership cho từng quan sát)
        steps.append({
            'ten': 'Bước 4: Fuzzify dữ liệu',
            'mo_ta': 'Gán membership cho các giá trị quan sát (ví dụ hiển thị vài mẫu).',
            'ket_qua': {'sample_memberships': si.get('sample_memberships')}
        })
        # Bước 5: Mô hóa quan hệ mờ (xây dựng quy tắc A_{t-1} -> A_t)
        steps.append({
            'ten': 'Bước 5: Mô hình hóa quan hệ mờ',
            'mo_ta': 'Tổng hợp tần suất/trọng số của các quan hệ trạng thái trước -> sau và chuẩn hóa.',
            'ket_qua': {'rules_summary': si.get('rules_summary')}
        })
        # (Đã loại bỏ bước 6 theo yêu cầu; bước tối ưu sẽ được hiển thị tóm tắt ở phần PSO/Grid-search)
    except Exception:
        pass
    
    # Bước 2: Grid search overlap với so_khoang tối ưu
    overlap_candidates = np.linspace(0.0, 0.5, 11)  # [0, 0.05, 0.1, ..., 0.5]
    best_overlap = 0.0
    best_overlap_val = float('inf')
    
    overlap_results = []
    print(f'[INFO] Grid search overlap với so_khoang={so_khoang_opt}...')
    for overlap_test in overlap_candidates:
        model = ChuoiThoiGianMo(so_khoang=so_khoang_opt, overlap=overlap_test)
        model.fit(df_train['value'])
        history = list(df_train['value'].values)
        preds = []
        for true in df_val['value'].values:
            pred = model.predict_next(history[-model.lag:])
            preds.append(pred)
            history.append(true)
        val_rmse = rmse(df_val['value'].values, preds)
        print(f'  overlap={overlap_test:.2f}: RMSE={val_rmse:.4f}')
        overlap_results.append({'overlap': float(overlap_test), 'val_rmse': float(val_rmse)})
        if val_rmse < best_overlap_val:
            best_overlap_val = val_rmse
            best_overlap = overlap_test
    
    best = np.array([so_khoang_opt, best_overlap])
    best_val = best_overlap_val
    steps.append({
        'ten': 'Grid search tìm overlap',
        'mo_ta': f'Kiểm tra {len(overlap_candidates)} giá trị overlap với so_khoang={so_khoang_opt}',
        'ket_qua': overlap_results
    })
    # Thêm thông tin tạo tập mờ cho overlap tốt nhất (dựa trên train)
    try:
        best_model_train = ChuoiThoiGianMo(so_khoang=so_khoang_opt, overlap=best_overlap)
        best_model_train.fit(df_train['value'])
        steps.append({
            'ten': 'Tạo tập mờ (overlap tốt nhất trên train)',
            'mo_ta': f'Tạo tập mờ với overlap={best_overlap} trên tập train',
            'ket_qua': best_model_train.steps_info
        })
    except Exception:
        pass
    
    print('Tối ưu xong: best=', best, 'val=', best_val)
    # huấn luyện lại trên train+val
    so_khoang = int(round(best[0]))
    overlap = float(best[1])
    model = ChuoiThoiGianMo(so_khoang=so_khoang, overlap=overlap)
    model.fit(pd.concat([df_train['value'], df_val['value']]))
    # đánh giá trên test
    preds = []
    history = list(pd.concat([df_train['value'], df_val['value']]).values)
    for true in df_test['value'].values:
        pred = model.predict_next(history[-model.lag:])
        preds.append(pred)
        history.append(true)
    test_rmse = rmse(df_test['value'].values, preds)
    # Thêm bước huấn luyện lại và đánh giá lên danh sách steps
    steps.append({
        'ten': 'Huấn luyện cuối (train+val) và đánh giá trên test',
        'mo_ta': f'Thuật toán fuzzy với so_khoang={so_khoang}, overlap={overlap}',
        'ket_qua': {
            'test_rmse': float(test_rmse),
            'so_khoang': int(so_khoang),
            'overlap': float(overlap),
            'trung_tam': list(getattr(model, 'trung_tam', [])),
            'steps_info': getattr(model, 'steps_info', {})
        }
    })

    return {
        'best': best,
        'best_val': best_val,
        'test_rmse': test_rmse,
        'preds': preds,
        'trung_tam': model.trung_tam,
        'steps': steps
    }


def chay_de_pipeline(df, n_khoang=None, seed=42):
    """Thực hiện pipeline theo 7 bước yêu cầu với K-Means khởi tạo khoảng.

    Luôn sử dụng K-Means để tạo khoảng mờ và lag=1 cho mô hình Markov.
    Trả về dict giống cấu trúc để UI hiển thị.
    """
    df_train, df_val, df_test = chia_train_val_test(df)
    steps = []

    # Step 1: prepare U
    values_train = df_train['value'].values
    vmin = float(np.min(values_train))
    vmax = float(np.max(values_train))
    steps.append({'ten': 'Tập nền U', 'mo_ta': 'Xác định tập nền U từ dữ liệu train', 'ket_qua': [vmin, vmax]})

    # Step 2: KMeans to get initial centers
    # If n_khoang is None, auto-select k using silhouette score over a reasonable range
    if n_khoang is None:
        X = values_train.reshape(-1, 1)
        unique_vals = np.unique(X)
        max_k_try = min(12, max(2, len(unique_vals) - 1))
        best_k = 2
        best_score = -1.0
        scores = {}
        for k in range(2, max_k_try + 1):
            try:
                km_try = KMeans(n_clusters=k, random_state=seed).fit(X)
                lbls = km_try.labels_
                # silhouette requires at least 2 clusters and less than n_samples
                score = float(silhouette_score(X, lbls)) if len(np.unique(lbls)) > 1 else -1.0
            except Exception:
                score = -1.0
            scores[int(k)] = float(score)
            if score > best_score:
                best_score = score
                best_k = k
        n_khoang = int(best_k)
        steps.append({'ten': 'Bước 2: Chọn số cụm K bằng K-Means (auto)', 'mo_ta': 'Tự động chọn k tối ưu bằng silhouette score', 'ket_qua': {'chosen_k': int(n_khoang), 'scores': scores}})
    # now run KMeans with chosen n_khoang
    kmeans = KMeans(n_clusters=n_khoang, random_state=seed)
    centers = kmeans.fit(values_train.reshape(-1, 1)).cluster_centers_.flatten()
    centers_sorted = np.sort(centers)
    centers_dict = {f'Tâm cụm {i+1}': float(c) for i, c in enumerate(centers_sorted)}
    steps.append({'ten': 'Bước 2: Phân cụm bằng K-Means', 'mo_ta': f'Phân cụm thành {n_khoang} cụm trên tập train', 'ket_qua': centers_dict})

    # Step 3: initial partitions (always use K-Means)
    # build edges from KMeans centers: midpoints between sorted centers
    centers_sorted = np.sort(centers)
    # midpoints between adjacent centers form internal boundaries
    if len(centers_sorted) > 1:
        midpoints = (centers_sorted[:-1] + centers_sorted[1:]) / 2.0
        initial_edges = np.concatenate(([vmin], midpoints, [vmax]))
    else:
        initial_edges = np.array([vmin, vmax])
    steps.append({'ten': 'Bước 3: Tạo các khoảng ban đầu từ K-Means', 'mo_ta': 'Khởi tạo ranh giới bằng cách lấy trung điểm giữa các tâm cụm', 'ket_qua': list(map(float, initial_edges))})

    # Step 3.5: Display detailed fuzzy intervals
    fuzzy_intervals = []
    centers_final = (initial_edges[:-1] + initial_edges[1:]) / 2.0
    for i in range(n_khoang):
        left = float(initial_edges[i])
        right = float(initial_edges[i+1])
        center = float(centers_final[i])
        fuzzy_intervals.append({
            'khoang': f'Khoảng {i+1}',
            'range': f'[{left:.2f}, {right:.2f})',
            'tam': f'{center:.2f}',
            'A_label': f'A₀' if i == 0 else f'A_{i}'
        })
    steps.append({'ten': 'Bước 3.5: Chi tiết các khoảng mờ', 'mo_ta': 'Danh sách khoảng mờ với ranh giới (vmin-vmax) và tâm', 'ket_qua': fuzzy_intervals})

    # Prepare DE: optimize internal boundaries (exclude first and last)
    vmin, vmax = float(initial_edges[0]), float(initial_edges[-1])
    dim = n_khoang - 1  # number of internal boundaries

    # objective: given vector x (length dim) -> form edges = [vmin] + sorted(x) + [vmax]
    def eval_boundaries(x):
        # ensure sorted
        xs = np.sort(x)
        edges = np.concatenate(([vmin], xs, [vmax]))
        # build model
        model = ChuoiThoiGianMo(so_khoang=n_khoang, overlap=0.2, lag=1)
        model.set_partitions(edges=edges)
        model.fit(df_train['value'])
        # forecast on val
        history = list(df_train['value'].values)
        preds = []
        for true in df_val['value'].values:
            pred = model.predict_next(history[-model.lag:])
            preds.append(pred)
            history.append(true)
        return float(rmse(df_val['value'].values, preds))

    # bounds for each internal boundary
    bounds = [(vmin, vmax)] * dim if dim > 0 else []

    iter_history = []
    if dim == 0:
        # trivial case: one interval only
        best_edges = initial_edges
        iter_history = []
    else:
        # DE with default parameters
        de_pop = 10
        de_iter = 30
        best_x, best_val, history = differential_evolution(eval_boundaries, bounds, popsize=de_pop, iters=de_iter, seed=seed)
        # convert best_x to sorted edges
        best_edges = np.concatenate(([vmin], np.sort(best_x), [vmax]))
        # record iterations in readable format
        iter_history = []
        for it_idx, val, vec in history:
            edges_it = np.concatenate(([vmin], np.sort(vec), [vmax]))
            iter_history.append({'iter': int(it_idx), 'mse': float(val), 'boundaries': list(map(float, edges_it))})

    # Step 4: report DE iterations and final optimal partitions
    steps.append({'ten': 'Bước 4: Tối ưu các khoảng bằng Differential Evolution (DE)', 'mo_ta': 'DE tối ưu các ranh giới để tối thiểu MSE trên validation', 'ket_qua': iter_history})
    # report final optimal intervals
    optimal_intervals = []
    for i in range(n_khoang):
        left = float(best_edges[i])
        right = float(best_edges[i+1])
        optimal_intervals.append({'Khoảng': f'Khoảng {i+1}', 'left': left, 'right': right})
    steps.append({'ten': 'Các khoảng mờ tối ưu (sau DE)', 'mo_ta': '', 'ket_qua': optimal_intervals})

    # Step 5: Fuzzify data based on optimal intervals using Gaussian membership
    model_final = ChuoiThoiGianMo(so_khoang=n_khoang, overlap=0.2, lag=1)
    model_final.set_partitions(edges=best_edges)
    model_final.fit(pd.concat([df_train['value'], df_val['value']]))
    # create fuzzy labels (take argmax membership) for train+val
    combined = pd.concat([df_train[['date','value']], df_val[['date','value']]]).reset_index(drop=True)
    memberships = []
    for v in combined['value'].values[:min(20, len(combined))]:
        mu = model_final._fuzzify_value(v)  # returns numpy array
        memberships.append({'value': float(v), 'membership': list(map(float, mu)), 'label': int(np.argmax(mu))})
    steps.append({'ten': 'Bước 5: Mờ hóa dữ liệu dựa trên khoảng tối ưu', 'mo_ta': 'Hiển thị vài mẫu fuzzified (label: index của tập mờ có membership lớn nhất)', 'ket_qua': memberships})

    # Step 5.5: Display Markov Transitions & Probabilities
    if hasattr(model_final, 'quan_he') and model_final.quan_he:
        transitions = []
        for prev_state, next_mapping in model_final.quan_he.items():
            total_count = sum(next_mapping.values())
            for next_label, count in next_mapping.items():
                prob = float(count) / total_count if total_count > 0 else 0.0
                # Handle both tuple (lag-p) and int state keys
                if isinstance(prev_state, tuple):
                    from_label = f'A_({",".join(str(int(s)) for s in prev_state)})'
                else:
                    from_label = f'A_{int(prev_state)}'
                if isinstance(next_label, tuple):
                    to_label = f'A_({",".join(str(int(s)) for s in next_label)})'
                else:
                    to_label = f'A_{int(next_label)}'
                transitions.append({
                    'from': from_label,
                    'to': to_label,
                    'frequency': int(count),
                    'probability': f'{prob:.2%}',
                    'rule': f'{from_label} → {to_label} ({int(count)}x, {prob:.2%})'
                })
        steps.append({'ten': 'Bước 5.5: Luật Markov & Xác Suất Chuyển Tiếp', 'mo_ta': 'Hiển thị các chuyển tiếp từ trạng thái t-1 → t với tần suất và xác suất', 'ket_qua': transitions})

    # Step 6: Build fuzzy relations and forecast on test
    # relations already built in model_final.quan_he
    # Forecast on test using history = train+val
    history_vals = list(pd.concat([df_train['value'], df_val['value']]).values)
    preds_test = []
    for true in df_test['value'].values:
        pred = model_final.predict_next(history_vals[-model_final.lag:])
        preds_test.append(pred)
        history_vals.append(true)
    forecast_table = []
    test_dates = df_test['date'].reset_index(drop=True)
    for i, (d, actual, f) in enumerate(zip(test_dates, df_test['value'].values, preds_test)):
        forecast_table.append({'date': str(d), 'value': float(actual), 'forecast': float(f)})
    steps.append({'ten': 'Bước 6: Xây dựng quan hệ mờ và dự báo', 'mo_ta': 'Dự báo trên tập test bằng mô hình mờ với các khoảng tối ưu', 'ket_qua': forecast_table})

    # Step 7: Evaluate and visualize
    mse_final = float(rmse(df_test['value'].values, preds_test))
    steps.append({'ten': 'Bước 7: Đánh giá và trực quan hóa kết quả', 'mo_ta': 'Báo MSE cuối cùng trên test', 'ket_qua': {'mse_after_opt': mse_final}})

    return {
        'steps': steps,
        'best_edges': list(map(float, best_edges)),
        'iter_history': iter_history,
        'preds': preds_test,
        'test_rmse': mse_final,
        'model': model_final
    }


if __name__ == '__main__':
    if os.path.exists('data/dulieu.csv'):
        try:
            df = pd.read_csv('data/dulieu.csv', parse_dates=[0])
        except Exception:
            df = pd.read_csv('data/dulieu.csv')
    else:
        print('Không tìm thấy data/dulieu.csv — tạo dữ liệu ảo tạm thời data/du_lieu.csv')
        from xuly_du_lieu import sinh_du_lieu_mau
        sinh_du_lieu_mau(dest='data/du_lieu.csv')
        df = pd.read_csv('data/du_lieu.csv', parse_dates=['date'])

    res = chay_psu_toi_uu(df, so_hat=8, vong=20)
    print(res)
