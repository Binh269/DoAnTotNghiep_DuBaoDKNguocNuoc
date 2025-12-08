"""
Bước 6: Dự báo trên tập test.

Sử dụng mô hình mờ để dự báo các giá trị trên tập kiểm tra.
"""

import pandas as pd


def buoc_6_du_bao(model_final, df_train, df_val, df_test):
    """
    Dự báo trên tập test sử dụng mô hình mờ.
    
    Args:
        model_final: Mô hình FTS đã huấn luyện
        df_train: DataFrame tập huấn luyện
        df_val: DataFrame tập xác thực
        df_test: DataFrame tập kiểm tra
    
    Returns:
        dict với 'forecast_table', 'preds_test'
    """
    # Lịch sử = train + val
    history_vals = list(pd.concat([df_train['value'], df_val['value']]).values)
    preds_test = []
    
    for true in df_test['value'].values:
        pred = model_final.predict_next(history_vals[-model_final.lag:])
        preds_test.append(pred)
        history_vals.append(true)
    
    # Tạo bảng dự báo
    forecast_table = []
    test_dates = df_test['date'].reset_index(drop=True)
    for i, (d, actual, f) in enumerate(zip(test_dates, df_test['value'].values, preds_test)):
        forecast_table.append({
            'date': str(d),
            'value': float(actual),
            'forecast': float(f)
        })
    
    return {
        'ten': 'Bước 6: Dự báo trên tập test',
        'mo_ta': 'Dự báo các giá trị trên tập test bằng mô hình mờ',
        'ket_qua': forecast_table,
        'chi_tiet': {
            'n_du_bao': len(preds_test),
            'preds': list(map(float, preds_test))
        }
    }
