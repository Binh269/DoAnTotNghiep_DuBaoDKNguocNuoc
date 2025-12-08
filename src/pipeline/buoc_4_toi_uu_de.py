"""
Bước 4: Tối ưu các khoảng mờ bằng Differential Evolution (DE).

Tối ưu các ranh giới để tối thiểu MSE trên tập xác thực.
"""

import numpy as np


def buoc_4_toi_uu_khoang_de(best_edges, iter_history, n_khoang):
    """
    Tổng hợp kết quả tối ưu DE.
    
    Args:
        best_edges: Ranh giới tối ưu
        iter_history: Lịch sử tối ưu DE
        n_khoang: Số khoảng
    
    Returns:
        dict với kết quả DE và khoảng tối ưu
    """
    optimal_intervals = []
    for i in range(n_khoang):
        left = float(best_edges[i])
        right = float(best_edges[i + 1])
        optimal_intervals.append({
            'Khoảng': f'Khoảng {i+1}',
            'left': left,
            'right': right
        })
    
    return {
        'ten': 'Bước 4: Tối ưu các khoảng bằng DE',
        'mo_ta': 'DE tối ưu ranh giới để tối thiểu MSE trên validation',
        'ket_qua': iter_history,
        'chi_tiet': {
            'n_iterations': len(iter_history),
            'best_edges': list(map(float, best_edges)),
            'optimal_intervals': optimal_intervals
        }
    }
