import numpy as np


class PSO:
    """PSO đơn giản để tối ưu hàm mục tiêu liên tục.

    - chiều bài toán tùy theo objective (n biến). Giới hạn mỗi biến qua bounds = [(min,max), ...]
    - objective: callable(x) trả về số (giá trị để minimize)
    """

    def __init__(self, objective, bounds, n_particles=20, n_iter=50, w=0.7, c1=1.5, c2=1.5, seed=42):
        self.objective = objective
        self.bounds = np.array(bounds, dtype=float)
        self.n_particles = n_particles
        self.n_iter = n_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.rng = np.random.RandomState(seed)

    def run(self):
        dim = len(self.bounds)
        lb = self.bounds[:, 0]
        ub = self.bounds[:, 1]
        X = self.rng.uniform(lb, ub, size=(self.n_particles, dim))
        V = self.rng.uniform(-np.abs(ub - lb), np.abs(ub - lb), size=(self.n_particles, dim))
        pbest = X.copy()
        pbest_val = np.array([np.inf] * self.n_particles)
        gbest = None
        gbest_val = np.inf

        for it in range(self.n_iter):
            for i in range(self.n_particles):
                x = X[i]
                val = self.objective(x)
                if val < pbest_val[i]:
                    pbest_val[i] = val
                    pbest[i] = x.copy()
                if val < gbest_val:
                    gbest_val = val
                    gbest = x.copy()
            
            r1 = self.rng.rand(self.n_particles, dim)
            r2 = self.rng.rand(self.n_particles, dim)
            V = self.w * V + self.c1 * r1 * (pbest - X) + self.c2 * r2 * (gbest - X)
            
            # Thêm nhiễu ngẫu nhiên để tránh bị mắc ở local minimum
            if it % 5 == 0 and it > 0:
                noise_scale = 0.05 * (ub - lb)
                noise = self.rng.normal(0, noise_scale, size=(self.n_particles, dim))
                V += noise
            
            X = X + V
            X = np.maximum(np.minimum(X, ub), lb)
            if (it + 1) % max(1, (self.n_iter // 5)) == 0:
                print(f"PSO tiến trình: vòng {it+1}/{self.n_iter}, best={gbest_val:.4f}")
        return gbest, gbest_val
