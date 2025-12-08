import numpy as np


def differential_evolution(func, bounds, popsize=10, iters=50, mut=0.8, crossp=0.7, seed=42):
    """A simple Differential Evolution implementation.

    - func(x) should return a scalar fitness to minimize.
    - bounds: list of (min, max) for each dimension.
    Returns best_x, best_val, history where history is list of (iter_index, best_val, best_vector)
    """
    rng = np.random.RandomState(seed)
    dim = len(bounds)
    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)

    # initialize population
    pop = rng.uniform(lb, ub, size=(popsize, dim))
    fitness = np.array([func(ind) for ind in pop])
    pbest_idx = np.argmin(fitness)
    gbest = pop[pbest_idx].copy()
    gbest_val = float(fitness[pbest_idx])
    history = [(0, gbest_val, gbest.copy())]

    for it in range(1, iters + 1):
        for i in range(popsize):
            ids = list(range(popsize))
            ids.remove(i)
            a, b, c = pop[rng.choice(ids, 3, replace=False)]
            mutant = a + mut * (b - c)
            # crossover
            cross = rng.rand(dim) < crossp
            if not np.any(cross):
                cross[rng.randint(0, dim)] = True
            trial = np.where(cross, mutant, pop[i])
            # clip
            trial = np.clip(trial, lb, ub)
            f = func(trial)
            if f < fitness[i]:
                pop[i] = trial
                fitness[i] = f
                if f < gbest_val:
                    gbest_val = float(f)
                    gbest = trial.copy()
        history.append((it, gbest_val, gbest.copy()))
    return gbest, gbest_val, history
