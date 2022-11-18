
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern, WhiteKernel
from scipy.stats import norm
import math
import warnings 
warnings.filterwarnings('ignore')

np.random.seed(2)
domain = np.array([[0, 5]])

""" Solution """
class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration. """
        # TODO: enter your code here

        self.beta = 100
        # prior
        self.v_min = 1.2
        self.fx_kernel = 0.5 * Matern(length_scale=0.5, nu=2.5)  + WhiteKernel(0.15)
        self.vx_kernel = math.sqrt(2) * Matern(length_scale=0.5, nu=2.5) + ConstantKernel(1.5) + WhiteKernel(0.0001)
        self.fx_GP = GaussianProcessRegressor(kernel=self.fx_kernel, random_state=0)
        self.vx_GP = GaussianProcessRegressor(kernel=self.vx_kernel,  random_state=0)
        self.x = np.array([]).reshape(-1, domain.shape[0])
        self.f = np.array([]).reshape(-1, domain.shape[0])
        self.v = np.array([]).reshape(-1, domain.shape[0])

    def next_recommendation(self):
        """
        Recommend the next input to sample.
        Returns
        -------
        recommendation: np.ndarray
            1 x domain.shape[0] array containing the next point to evaluate
        """
        if len(self.x) == 0:
            next_x = 2/3 * domain[:, 0] + 1/3 * domain[:, 1]
        else:
            next_x = self.optimize_acquisition_function()
        return next_x


    def optimize_acquisition_function(self):
        """
        Optimizes the acquisition function.

        Returns
        -------
        x_opt: np.ndarray
            1 x domain.shape[0] array containing the point that maximize the acquisition function.
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * \
                 np.random.rand(domain.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=domain,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *domain[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        return np.atleast_2d(x_values[ind])

    def EI(self, x, hyper_inject):
        """ The function that returns expected improvement of current best estimate
        We implemented and modified this function from:
        https://ekamperi.github.io/machine%20learning/2021/06/11/acquisition-functions.html
        """
        mu, sigma = self.fx_GP.predict(x.reshape(-1, domain.shape[0]), return_std=True)
        mu_x_opt = np.max(self.f)
        res = mu - mu_x_opt - hyper_inject
        with np.errstate(divide='warn'):
            ei = res * norm.cdf(res/sigma) + sigma * norm.pdf(res/sigma)

        return ei

    def constraint(self, x):
        mu, sigma = self.vx_GP.predict(x.reshape(-1, domain.shape[0]), return_std=True)
        pr = norm.cdf(self.v_min, loc=mu, scale=sigma)

        return pr

    def acquisition_function(self, x):
        """
        Compute the acquisition function.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f

        Returns
        ------
        af_value: float
            Value of the acquisition function at x
        """

        # TODO: enter your code here
        ei = self.EI(x, 0.001)
        constraint = self.constraint(x)

        return float(ei - self.beta * constraint)

    def add_data_point(self, x, f, v):
        """
        Add data points to the model.

        Parameters
        ----------
        x: np.ndarray
            Hyperparameters
        f: np.ndarray
            Model accuracy
        v: np.ndarray
            Model training speed
        """

        self.x = np.vstack((self.x, x))
        self.f = np.vstack((self.f, f))
        self.v = np.vstack((self.v, v))

        self.fx_GP.fit(self.x, self.f)
        self.vx_GP.fit(self.x, self.v)



    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """
        copy_f = self.f
        copy_f[self.v <= 1.2] = -100
        max_idx = np.argmax(copy_f)
        x_opt = self.x[max_idx]
        return x_opt


""" Toy problem to check code works as expected """

def check_in_domain(x):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= domain[None, :, 0]) and np.all(x <= domain[None, :, 1])


def f(x):
    """Dummy objective"""
    mid_point = domain[:, 0] + 0.5 * (domain[:, 1] - domain[:, 0])
    return - np.linalg.norm(x - mid_point, 2)  # -(x - 2.5)^2


def v(x):
    """Dummy speed"""
    return 2


def main():
    # Init problem
    agent = BO_algo()
    #n_dim = 1
    # Add initial safe point
    x_init = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * np.random.rand(
            1)
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)

    
    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()
        # Check for valid shape
        assert x.shape == (1, domain.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {domain.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x)
        cost_val = v(x)
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = np.atleast_2d(agent.get_solution())
    assert solution.shape == (1, domain.shape[0]), \
        f"The function get solution must return a numpy array of shape (" \
        f"1, {domain.shape[0]})"
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the ' \
        f'domain, {solution} returned instead'

    # Compute regret
    if v(solution) < 1.2:
        regret = 1
    else:
        regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret{regret}')


if __name__ == "__main__":
    main()
