"""
Chương trình chính điều phối: Thực hiện pipeline 7 bước hoàn chỉnh.

Nhập dữ liệu → Tiền xử lý → Chia tập → 7 bước → Trả về kết quả.
"""

import os
import sys
import numpy as np
import pandas as pd

# Thêm src vào sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from mohinh.chuoi_thoi_gian_mo import ChuoiThoiGianMo
from tinhtoans import rmse, chia_train_val_test
from toi_uu_hoa import toi_uu_khoang_de

from pipeline.buoc_1_xac_dinh_u import buoc_1_xac_dinh_tap_nen_u
from pipeline.buoc_2_phan_cum import buoc_2_phan_cum_kmeans
from pipeline.buoc_3_tao_khoang import buoc_3_tao_khoang_ban_dau
from pipeline.buoc_4_toi_uu_de import buoc_4_toi_uu_khoang_de
from pipeline.buoc_5_mo_hoa import buoc_5_mo_hoa_va_markov
from pipeline.buoc_6_du_bao import buoc_6_du_bao
from pipeline.buoc_7_danh_gia import buoc_7_danh_gia


def chay_pipeline_7_buoc(df, n_khoang=None, seed=42):
    """
    Thực hiện pipeline đầy đủ 7 bước dự báo chuỗi thời gian mờ.
    
    Args:
        df: DataFrame chứa cột 'date' và 'value'
        n_khoang: Số khoảng mờ (nếu None thì tự động chọn)
        seed: Random seed
    
    Returns:
        dict chứa kết quả 7 bước, mô hình, dự báo, và RMSE
    """
    
    # Chia tập dữ liệu
    df_train, df_val, df_test = chia_train_val_test(df)
    steps = []
    
    print("[PIPELINE] Bắt đầu pipeline 7 bước...")
    
    # ==================== BƯỚC 1 ====================
    print("[BƯỚC 1] Xác định tập nền U...")
    step_1 = buoc_1_xac_dinh_tap_nen_u(df_train)
    steps.append(step_1)
    vmin, vmax = step_1['chi_tiet']['vmin'], step_1['chi_tiet']['vmax']
    
    # ==================== BƯỚC 2 ====================
    print("[BƯỚC 2] Phân cụm K-Means...")
    step_2 = buoc_2_phan_cum_kmeans(df_train, seed=seed)
    steps.append(step_2)
    n_khoang = step_2['chi_tiet']['chosen_k']
    centers = step_2['chi_tiet']['centers']
    
    # ==================== BƯỚC 3 ====================
    print("[BƯỚC 3] Tạo các khoảng mờ...")
    step_3 = buoc_3_tao_khoang_ban_dau(vmin, vmax, centers, n_khoang)
    steps.append(step_3)
    initial_edges = np.array(step_3['chi_tiet']['initial_edges'])
    
    # ==================== BƯỚC 4 ====================
    print("[BƯỚC 4] Tối ưu khoảng bằng DE...")
    de_result = toi_uu_khoang_de(n_khoang, df_train, df_val, initial_edges, 
                                  ChuoiThoiGianMo, seed=seed)
    best_edges = de_result['best_edges']
    iter_history = de_result['iter_history']
    step_4 = buoc_4_toi_uu_khoang_de(best_edges, iter_history, n_khoang)
    steps.append(step_4)
    
    # ==================== HUẤN LUYỆN MÔ HÌNH CUỐI ====================
    print("[MÔ HÌNH] Huấn luyện mô hình cuối (train+val)...")
    model_final = ChuoiThoiGianMo(so_khoang=n_khoang, overlap=0.2, lag=1)
    model_final.set_partitions(edges=best_edges)
    model_final.fit(pd.concat([df_train['value'], df_val['value']]))
    
    # ==================== BƯỚC 5 ====================
    print("[BƯỚC 5] Mờ hóa dữ liệu & Luật Markov...")
    step_5 = buoc_5_mo_hoa_va_markov(model_final, df_train, df_val)
    steps.append(step_5)
    
    # ==================== BƯỚC 6 ====================
    print("[BƯỚC 6] Dự báo trên tập test...")
    step_6_result = buoc_6_du_bao(model_final, df_train, df_val, df_test)
    steps.append(step_6_result)
    preds_test = step_6_result['chi_tiet']['preds']
    
    # ==================== BƯỚC 7 ====================
    print("[BƯỚC 7] Đánh giá kết quả...")
    step_7 = buoc_7_danh_gia(preds_test, df_test)
    steps.append(step_7)
    
    test_rmse = step_7['ket_qua']['rmse']
    
    print(f"[PIPELINE] Hoàn thành! RMSE = {test_rmse:.4f}")
    
    return {
        'steps': steps,
        'best_edges': list(map(float, best_edges)),
        'iter_history': iter_history,
        'preds': preds_test,
        'test_rmse': test_rmse,
        'model': model_final
    }


if __name__ == '__main__':
    # Test với dữ liệu ảo
    import pandas as pd
    from datetime import datetime, timedelta
    
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    values = np.sin(np.arange(100) * 0.1) * 100 + 1000 + np.random.normal(0, 20, 100)
    df_test = pd.DataFrame({'date': dates, 'value': values})
    
    result = chay_pipeline_7_buoc(df_test)
    
    for i, step in enumerate(result['steps'], 1):
        print(f"\n{'='*60}")
        print(f"{step['ten']}")
        print(f"Mô tả: {step['mo_ta']}")
        print(f"{'='*60}")
