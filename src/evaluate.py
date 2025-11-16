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
    
    # Bước 2: Grid search overlap với so_khoang tối ưu
    overlap_candidates = np.linspace(0.0, 0.5, 11)  # [0, 0.05, 0.1, ..., 0.5]
    best_overlap = 0.0
    best_overlap_val = float('inf')
    
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
        if val_rmse < best_overlap_val:
            best_overlap_val = val_rmse
            best_overlap = overlap_test
    
    best = np.array([so_khoang_opt, best_overlap])
    best_val = best_overlap_val
    
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
    return {
        'best': best,
        'best_val': best_val,
        'test_rmse': test_rmse,
        'preds': preds,
        'trung_tam': model.trung_tam
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
