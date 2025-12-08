"""
Hàm tối ưu hóa các khoảng mờ bằng Differential Evolution (DE).
"""

import numpy as np
import pandas as pd
from tinhtoans import rmse


def toi_uu_khoang_de(n_khoang, df_train, df_val, initial_edges, model_class, seed=42):
    """Tối ưu các ranh giới khoảng mờ bằng DE.
    
    Args:
        n_khoang: Số khoảng mờ
        df_train: DataFrame tập huấn luyện
        df_val: DataFrame tập xác thực
        initial_edges: Ranh giới ban đầu
        model_class: Lớp mô hình (ChuoiThoiGianMo)
        seed: Random seed
    
    Returns:
        dict với keys: 'best_edges', 'iter_history'
    """
    from opt.de import differential_evolution
    
    vmin = float(initial_edges[0])
    vmax = float(initial_edges[-1])
    dim = n_khoang - 1
    
    def eval_boundaries(x):
        """Hàm mục tiêu: tối thiểu RMSE trên tập val."""
        xs = np.sort(x)
        edges = np.concatenate(([vmin], xs, [vmax]))
        
        model = model_class(so_khoang=n_khoang, overlap=0.2, lag=1)
        model.set_partitions(edges=edges)
        model.fit(df_train['value'])
        
        history = list(df_train['value'].values)
        preds = []
        for true in df_val['value'].values:
            pred = model.predict_next(history[-model.lag:])
            preds.append(pred)
            history.append(true)
        
        return float(rmse(df_val['value'].values, preds))
    
    bounds = [(vmin, vmax)] * dim if dim > 0 else []
    iter_history = []
    
    if dim == 0:
        # Trường hợp chỉ có 1 khoảng: không cần tối ưu
        best_edges = initial_edges
    else:
        # DE tối ưu
        de_pop = 10
        de_iter = 30
        best_x, best_val, history = differential_evolution(
            eval_boundaries, bounds, popsize=de_pop, iters=de_iter, seed=seed
        )
        best_edges = np.concatenate(([vmin], np.sort(best_x), [vmax]))
        
        # Lưu lịch sử tối ưu
        for it_idx, val, vec in history:
            edges_it = np.concatenate(([vmin], np.sort(vec), [vmax]))
            iter_history.append({
                'iter': int(it_idx),
                'mse': float(val),
                'boundaries': list(map(float, edges_it))
            })
    
    return {
        'best_edges': best_edges,
        'iter_history': iter_history
    }
