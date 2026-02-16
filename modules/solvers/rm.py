# Regret matching solvers

import numpy as np

class RegretMatching:
    def __init__(self, n_actions, init="uniform", plus=True, predictive=True):
        if isinstance(init, np.ndarray):
            assert init.shape == (n_actions,)
            self.x = init
        elif init == "random":
            self.x = np.random.exponential(scale=1.0, size=n_actions)
        elif init == "uniform":
            self.x = np.ones(n_actions)
        else:
            raise ValueError("`init_type` must be 'random', 'uniform', or np.ndarray")
        self.x /= self.x.sum()
        self.predictive = predictive
        self.plus = plus
        self.regrets = np.zeros_like(self.x)

    def step(self, u):
        self.last_x = self.x
        ev = u @ self.x
        u = u - ev
        self.regrets = self.regrets + u
        if self.plus: self.regrets = np.maximum(self.regrets, 0)
        if self.predictive: r_pred = np.maximum(self.regrets + u, 0)
        else: r_pred = np.maximum(self.regrets, 0)
        sum_r_pred = r_pred.sum()
        if sum_r_pred > 0: self.x = r_pred / sum_r_pred
        return ev
