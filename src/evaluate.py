import os
import sys
import numpy as np
import pandas as pd
from functools import partial

# thêm src vào sys.path để import module nội bộ
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mohinh.chuoi_thoi_gian_mo import ChuoiThoiGianMo
from opt.pso import PSO
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
        pred = model.predict_next(history[-1])
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
            pred = model.predict_next(history[-1])
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
            pred = model.predict_next(history[-1])
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
        pred = model.predict_next(history[-1])
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
