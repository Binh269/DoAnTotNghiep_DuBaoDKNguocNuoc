"""
Bước 3: Tạo các khoảng mờ từ tâm cụm K-Means.

Khởi tạo ranh giới các khoảng bằng trung điểm giữa các tâm cụm.
"""

import numpy as np


def buoc_3_tao_khoang_ban_dau(vmin, vmax, centers, n_khoang):
    """
    Tạo các khoảng mờ ban đầu từ tâm K-Means.
    
    Args:
        vmin: Giá trị min
        vmax: Giá trị max
        centers: Danh sách tâm cụm
        n_khoang: Số khoảng
    
    Returns:
        dict với 'initial_edges', 'fuzzy_intervals'
    """
    centers_sorted = np.sort(centers)
    
    if len(centers_sorted) > 1:
        midpoints = (centers_sorted[:-1] + centers_sorted[1:]) / 2.0
        initial_edges = np.concatenate(([vmin], midpoints, [vmax]))
    else:
        initial_edges = np.array([vmin, vmax])
    
    # Chi tiết các khoảng mờ
    fuzzy_intervals = []
    centers_final = (initial_edges[:-1] + initial_edges[1:]) / 2.0
    for i in range(n_khoang):
        left = float(initial_edges[i])
        right = float(initial_edges[i + 1])
        center = float(centers_final[i])
        fuzzy_intervals.append({
            'khoang': f'Khoảng {i+1}',
            'range': f'[{left:.2f}, {right:.2f})',
            'tam': f'{center:.2f}',
            'A_label': f'A₀' if i == 0 else f'A_{i}'
        })
    
    return {
        'ten': 'Bước 3: Tạo các khoảng mờ ban đầu',
        'mo_ta': 'Khởi tạo ranh giới từ trung điểm giữa các tâm cụm',
        'ket_qua': list(map(float, initial_edges)),
        'chi_tiet': {
            'n_khoang': int(n_khoang),
            'initial_edges': list(map(float, initial_edges)),
            'fuzzy_intervals': fuzzy_intervals
        }
    }
