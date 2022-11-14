
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern
from scipy.stats import norm
import math
import GPy

np.random.seed(6)
domain = np.array([[0, 5]])

""" Solution """
class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration. """
        # TODO: enter your code here
        self.n = 0
        # prior
        self.v_min = 1.2
        self.sigma_f = 0.15
        self.sigma_v = 0.0001
        self.variance_f = 0.5
        self.variance_v = math.sqrt(2)
        self.len_scale_f = 0.5
        self.len_scale_v = 0.5
        self.constant_v = 1.5
        self.smoothness = 2.5
        self.fx_GP_kernel = GPy.kern.Matern52(
                input_dim=domain.shape[0], variance=self.variance_f ,
                lengthscale=self.len_scale_f)
        self.vx_GP_kernel = GPy.kern.Matern52(
                input_dim=domain.shape[0], variance=self.variance_v,
                lengthscale=self.len_scale_v) + GPy.kern.Bias(
                input_dim=domain.shape[0], variance=self.constant_v)

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

        if self.x.shape == 0:
           next_x = np.array([[np.random.uniform(0, 5)]])
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
        mu, sigma = self.fx_GP.predict(x.reshape(-1, domain.shape[0]))
        sigma = sigma.reshape(-1, 1)
        mu_x = self.fx_GP.predict(self.x)
        mu_x_opt = np.max(mu_x)
        res = mu - mu_x_opt - hyper_inject
        with np.errstate(divide='warn'):
            ei = res * norm.cdf(res/sigma) + sigma * norm.pdf(res/sigma)
            ei[sigma == 0.0] = 0.0
        return ei

    def constraint(self, x):
        mu, sigma = self.vx_GP.predict(x.reshape(-1, domain.shape[0]))
        if sigma != 0:
            pr = 1 - norm.cdf(self.v_min, loc=mu, scale=sigma)
        else:
            pr = 0.92 * (mu - self.v_min) if mu >= self.v_min else 0.08 * (self.v_min - mu)
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
        ei = self.EI(x, 0.00)
        constraint = self.constraint(x)
        return float(ei * constraint)

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

        if self.n == 0:
            self.fx_GP = GPy.models.gp_regression.GPRegression(
                X=self.x, Y=self.f,
                kernel=self.fx_GP_kernel, noise_var=self.sigma_f**2
            )
            self.vx_GP = GPy.models.gp_regression.GPRegression(
                X=self.x, Y=self.v,
                kernel=self.vx_GP_kernel, noise_var=self.sigma_v**2
            )

        
        self.fx_GP.set_XY(self.x, self.f)
        self.vx_GP.set_XY(self.x, self.v)


        self.fx_GP.optimize()
        self.vx_GP.optimize()

        self.n+=1

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """
        copy_f = self.f
        copy_f[self.v < 1.2] = -100
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
    return 2.0


def main():
    # Init problem
    agent = BO_algo()
    n_dim = 1
    # Add initial safe point
    x_init = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * np.random.rand(
            1, n_dim)
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

