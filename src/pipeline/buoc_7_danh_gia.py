"""
Bước 7: Đánh giá và trực quan hóa kết quả.

Tính toán các chỉ số đánh giá (MSE, RMSE, MAE, v.v.).
"""

import numpy as np
from tinhtoans import rmse


def buoc_7_danh_gia(preds_test, df_test):
    """
    Đánh giá kết quả dự báo.
    
    Args:
        preds_test: Danh sách dự báo trên tập test
        df_test: DataFrame tập kiểm tra
    
    Returns:
        dict với các chỉ số đánh giá
    """
    y_true = df_test['value'].values
    y_pred = np.array(preds_test)
    
    # Tính các chỉ số
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse_val = rmse(y_true, y_pred)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true))) * 100  # %
    
    return {
        'ten': 'Bước 7: Đánh giá kết quả',
        'mo_ta': 'Tính toán các chỉ số đánh giá mô hình',
        'ket_qua': {
            'mse': mse,
            'rmse': rmse_val,
            'mae': mae,
            'mape': mape
        },
        'chi_tiet': {
            'n_mau_test': len(y_true),
            'so_du_bao': len(y_pred),
            'chi_tiet_chi_so': {
                'MSE': f'{mse:.4f}',
                'RMSE': f'{rmse_val:.4f}',
                'MAE': f'{mae:.4f}',
                'MAPE': f'{mape:.2f}%'
            }
        }
    }
