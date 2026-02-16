#Gradient descent, Optimistic Gradient Descent, and AMSGrad

import numpy as np


class ProjectedGradientDescent():
    def __init__(self, n_actions, init="uniform", lr={'mode': 'constant', 'init_rate': 1e-1}):
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
        assert isinstance(lr, dict)
        if lr['mode'] == 'constant':
            self.lr = lr['init_rate']
        else:
            raise NotImplementedError("Learning rate mode not implemented")

    def step(self, grad):
        self.last_x = self.x
        ev = grad @ self.x
        self.x = self.x + self.lr * grad
        #Project back to individual simplices
        self.x = project_onto_simplex( self.x )
        return ev


class OptimisticGradientDescent():
    def __init__(self, n_actions, init="uniform", lr={'mode': 'constant', 'init_rate': 1e-1}):
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

        self.running = False

        assert isinstance(lr, dict)
        if lr['mode'] == 'constant':
            self.lr = lr['init_rate']
        else:
            raise NotImplementedError("Learning rate mode not implemented")

    def step(self, grad):
        self.last_x = self.x
        ev = grad @ self.x
        
        # At the first iteration, we should not do anything but set x_hat to x. Only after do we do normal pgd but with the gradient at x
        if self.running:
            self.x_hat = self.x_hat + self.lr * grad
            self.x_hat = project_onto_simplex( self.x_hat )
        else:
            self.x_hat = self.x.copy() 
            self.running = True
        
        # Do gradient step with the gradient at x but from point x_hat
        self.x = self.x_hat + self.lr * grad
        self.x = project_onto_simplex( self.x )

        return ev



class AMSGrad():
    def __init__(self, n_actions, init="uniform", params={'mode': 'constant', 'init_rate': 1e-1, 'beta1': 0.9, 'beta2': 0.999}, deal_with_zeros={'method': 'limit'}):
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
        assert isinstance(params, dict)
        if params['mode'] == 'constant':
            self.lr = params['init_rate']
        else:
            raise NotImplementedError("Learning rate mode not implemented")
        
        self.beta1 = params['beta1']
        self.beta2 = params['beta2']
        self.m = np.zeros(n_actions)
        self.v = np.zeros(n_actions)
        self.vhat = np.zeros(n_actions)
        if deal_with_zeros['method'] == 'limit':
            self.zeros_limit = True
        elif deal_with_zeros['method'] == 'epsilon':
            self.zeros_limit = False
            self.eps = deal_with_zeros['epsilon'] if 'epsilon' in deal_with_zeros else 1e-8
        else:
            raise ValueError("`deal_with_zeros` must be 'limit' or 'add epsilon' with an epsilon value.")
        
    def step(self, grad):
        self.last_x = self.x
        ev = grad @ self.x
        
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.square(grad)
        self.vhat = np.maximum(self.vhat, self.v)
        if self.zeros_limit:
            step = np.zeros_like(self.x)
            nonzeros = np.where(self.vhat > 1e-8)[0]
            step[nonzeros] = self.lr * np.divide(self.m[nonzeros], np.sqrt(self.vhat[nonzeros]))
            if len(nonzeros) < len(self.x):
                # If there are any zeros in vhat, we need to handle them separately
                zeros = np.setdiff1d(np.arange(len(self.vhat)), nonzeros)
                step[zeros] = self.lr * np.divide(1 - self.beta1, np.sqrt(1 - self.beta2))
        else:
            step = self.lr * np.divide(self.m, np.sqrt(self.vhat) + self.eps)
        point = self.x + step
        self.x = proj_weighted_simplex(point, np.sqrt(self.vhat))
        return ev
    

def project_onto_simplex(v):
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(u)+1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    return np.clip(v - theta, 0, None)


def proj_weighted_simplex(y, v):
    assert len(y) == len(v), "y and v must have the same length"
    nonzeros = np.where(v > 1e-8)[0]
    if len(nonzeros) == len(v):
        return proj_nonzeroweighted_simplex(y, v)
    else:
        # Split nonzeros into indices where y is positive and where it is nonpositive
        ypos = np.intersect1d(np.where(y > 0)[0], nonzeros)
        sum = np.sum(y[ypos])
        if sum <= 1.0:
            x = np.zeros_like(y)
            x[ypos] = y[ypos]
            zeros = np.setdiff1d(np.arange(len(v)), nonzeros)
            x[zeros] = (1.0 - sum) / len(zeros)
            assert np.isclose(np.sum(x), 1.0, atol=1e-8), "The sum of the projected vector x should be close to 1.0"
            assert np.all(x >= -1e-8), "All entries of the projected vector x should be nonnegative"
        else:
            x = np.zeros_like(y)
            x[ypos] = proj_nonzeroweighted_simplex(y[ypos], v[ypos])
        
        return x


def proj_nonzeroweighted_simplex(y, v, vectorized=True):
    a = v * y
    # 1) Compute breakpoints
    breakpoints = np.concatenate((a, a - v))
    sorted_indices = np.argsort(breakpoints)
    breakpoints = breakpoints[sorted_indices]

    if vectorized:
        # 2) Compute sum_xs for all breakpoints
        sum_xs = np.array([ np.sum(compute_x(y, a, v, breakpoint)) for breakpoint in breakpoints ])
        if sum_xs[0] <= 1.0 - 1e-8 or sum_xs[-1] >= 0.0 + 1e-8:
            raise ValueError("The sum_xs of the breakpoints should start above 1 and end at 0! Printing the breakpoints and sum_xs for debugging.", breakpoints, sum_xs)
        # 3) Find largest index ind_l such that sum_xs[ind_l] is bigger than 1
        ind_l = np.max( np.where( sum_xs - 1.0 >= -1e-8 )[0], initial=-1 )
        lambd = breakpoints[ind_l] + (breakpoints[ind_l+1] - breakpoints[ind_l]) * (1-sum_xs[ind_l]) / ( sum_xs[ind_l+1] - sum_xs[ind_l] )
    else:
        # 2) Initialize for Binary Search
        ind_l = 0
        ind_r = len(breakpoints) - 1
        l = len(y)
        r = 0
        pinned_down = False

        # 3) Binary Search for the correct lambda
        while not pinned_down:
            if ind_r - ind_l <= 1:
                lambd = breakpoints[ind_l] + (breakpoints[ind_r] - breakpoints[ind_l]) * (1-l) / (r-l)
                pinned_down = True
                break

            mid = (ind_l + ind_r) // 2
            inner = (a - breakpoints[mid]) / v
            C = np.sum( np.maximum( np.minimum(inner, 1.0), 0.0) )
            if C < 1.0:
                ind_r = mid
                r = C
            elif C > 1.0:
                ind_l = mid
                l = C
            else:
                lambd = breakpoints[mid]
                pinned_down = True

    # 4) Compute final projection
    x = compute_x(y, a, v, lambd)
    assert np.isclose(np.sum(x), 1.0, atol=1e-8), "The sum of the projected vector x should be close to 1.0"
    assert np.all(x >= -1e-8), "All entries of the projected vector x should be nonnegative"
    x /= np.sum(x)  # Normalize to ensure it sums to 1

    return x


def compute_x(y, a, v, lambd):
    x = np.empty_like(y)
    for i in range(len(y)):
        if lambd > a[i]:
            x[i] = 0.0
        elif lambd > a[i] - v[i]:
            x[i] = (a[i] - lambd) / v[i]
        else:
            x[i] = 1.0
    return x