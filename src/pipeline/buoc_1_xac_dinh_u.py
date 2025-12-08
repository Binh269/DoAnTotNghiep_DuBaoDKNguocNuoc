"""
Bước 1: Xác định tập nền U.

Tìm giá trị min/max từ dữ liệu huấn luyện và thiết lập phạm vi khoảng.
"""

import numpy as np


def buoc_1_xac_dinh_tap_nen_u(df_train):
    """
    Xác định tập nền U từ dữ liệu tập huấn luyện.
    
    Args:
        df_train: DataFrame tập huấn luyện
    
    Returns:
        dict với 'vmin', 'vmax'
    """
    values_train = df_train['value'].values
    vmin = float(np.min(values_train))
    vmax = float(np.max(values_train))
    
    return {
        'ten': 'Bước 1: Tập nền U',
        'mo_ta': 'Xác định tập nền U từ dữ liệu train',
        'ket_qua': [vmin, vmax],
        'chi_tiet': {
            'vmin': vmin,
            'vmax': vmax,
            'so_mau_train': len(values_train)
        }
    }
