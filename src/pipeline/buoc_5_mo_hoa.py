"""
Bước 5: Mờ hóa dữ liệu và xây dựng luật Markov.

Gán membership cho các giá trị và tạo bảng chuyển tiếp Markov với xác suất.
"""

import numpy as np


def buoc_5_mo_hoa_va_markov(model_final, df_train, df_val):
    """
    Mờ hóa dữ liệu và tóm tắt luật Markov.
    
    Args:
        model_final: Mô hình FTS đã huấn luyện
        df_train: DataFrame tập huấn luyện
        df_val: DataFrame tập xác thực
    
    Returns:
        dict với 'memberships', 'transitions'
    """
    import pandas as pd
    
    # Hiển thị vài mẫu fuzzified
    combined = pd.concat([df_train[['date', 'value']], df_val[['date', 'value']]]).reset_index(drop=True)
    memberships = []
    for v in combined['value'].values[:min(20, len(combined))]:
        mu = model_final._fuzzify_value(v)
        memberships.append({
            'value': float(v),
            'membership': list(map(float, mu)),
            'label': int(np.argmax(mu))
        })
    
    # Luật Markov với xác suất
    transitions = []
    if hasattr(model_final, 'quan_he') and model_final.quan_he:
        for prev_state, next_mapping in model_final.quan_he.items():
            total_count = sum(next_mapping.values())
            for next_label, count in next_mapping.items():
                prob = float(count) / total_count if total_count > 0 else 0.0
                
                # Xử lý cả tuple (lag-p) và int
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
    
    return {
        'ten': 'Bước 5: Mờ hóa dữ liệu & Luật Markov',
        'mo_ta': 'Gán membership cho dữ liệu và xây dựng luật Markov với xác suất chuyển tiếp',
        'ket_qua': {
            'memberships': memberships,
            'transitions': transitions
        },
        'chi_tiet': {
            'n_mau_mo_hoa': len(memberships),
            'n_quy_tac': len(transitions)
        }
    }
