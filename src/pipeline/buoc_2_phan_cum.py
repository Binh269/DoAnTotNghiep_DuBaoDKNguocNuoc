"""
Bước 2: Phân cụm K-Means với tự động lựa chọn k.

Sử dụng K-Means để xác định số cụm tối ưu dựa trên silhouette score.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def buoc_2_phan_cum_kmeans(df_train, seed=42):
    """
    Phân cụm dữ liệu huấn luyện bằng K-Means với tự động lựa chọn k.
    
    Args:
        df_train: DataFrame tập huấn luyện
        seed: Random seed
    
    Returns:
        dict với 'chosen_k', 'centers', 'scores'
    """
    values_train = df_train['value'].values
    X = values_train.reshape(-1, 1)
    
    # Tự động chọn k bằng silhouette score
    unique_vals = np.unique(X)
    max_k_try = min(12, max(2, len(unique_vals) - 1))
    best_k = 2
    best_score = -1.0
    scores = {}
    
    for k in range(2, max_k_try + 1):
        try:
            km_try = KMeans(n_clusters=k, random_state=seed).fit(X)
            lbls = km_try.labels_
            score = float(silhouette_score(X, lbls)) if len(np.unique(lbls)) > 1 else -1.0
        except Exception:
            score = -1.0
        
        scores[int(k)] = float(score)
        if score > best_score:
            best_score = score
            best_k = k
    
    # Chạy K-Means với k tối ưu
    kmeans = KMeans(n_clusters=best_k, random_state=seed)
    centers = kmeans.fit(X).cluster_centers_.flatten()
    centers_sorted = np.sort(centers)
    centers_dict = {f'Tâm cụm {i+1}': float(c) for i, c in enumerate(centers_sorted)}
    
    return {
        'ten': 'Bước 2: Phân cụm K-Means (auto-select k)',
        'mo_ta': f'Tự động chọn k tối ưu bằng silhouette score, chọn được k={best_k}',
        'ket_qua': centers_dict,
        'chi_tiet': {
            'chosen_k': int(best_k),
            'best_silhouette_score': float(best_score),
            'scores': scores,
            'centers': list(map(float, centers_sorted))
        }
    }
