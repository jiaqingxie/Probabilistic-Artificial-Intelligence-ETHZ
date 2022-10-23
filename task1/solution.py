import os
import re
import typing
# from sklearn.gaussian_process.kernels import *
from sklearn.cluster import KMeans
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
import gpytorch
from gpytorch.means import *
from gpytorch.kernels import *

# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = True
EVALUATION_GRID_POINTS = 300  # Number of grid points used in extended evaluation
EVALUATION_GRID_POINTS_3D = 50  # Number of points displayed in 3D during evaluation


# Cost function constants
COST_W_UNDERPREDICT = 25.0
COST_W_NORMAL = 1.0
COST_W_OVERPREDICT = 10.0


class Model(object):
    """
    Model for this task.
    You need to implement the fit_model and predict methods
    without changing their signatures, but are allowed to create additional methods.
    """

    def __init__(self):
        """
        Initialize your model here.
        We already provide a random number generator for reproducibility.
        """
        # TODO: Add custom initialization for your model here if necessary
        self.rng = np.random.default_rng(seed=0)
        self.y_mean = 33.172998 # empirical y mean
        self.y_std = 18.513832 # empirical y std
        self.x_mean = np.array([0.473093, 0.572626]) # empirical x mean
        self.x_std = np.array([0.280758, 0.277533]) # empirical x std

        self.models = []
        self.training_iter = 500 # training iteration

        self.labels = None
        self.centroids = None
        self.n_experts = 25 # number of clusters
        self.kmeans = None
        
    def make_predictions(self, test_features: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict the pollution concentration for a given set of locations.
        :param test_features: Locations as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :return:
            Tuple of three 1d NumPy float arrays, each of shape (NUM_SAMPLES,),
            containing your predictions, the GP posterior mean, and the GP posterior stddev (in that order)
        """
        # TODO: Use your GP to estimate the posterior mean and stddev for each location here
        # TODO: Use the GP posterior to form your predictions here

        x_test = (test_features - self.x_mean) / self.x_std # standardized
        experts = self.kmeans.predict(x_test)
        gp_means, gp_stds = [], []
        for model in self.models: # in each local GP model
            model.eval()
            gp_results = model(torch.tensor(x_test).float())
            gp_mean, gp_std = gp_results.mean.detach().numpy(), gp_results.variance.detach().numpy()
            gp_means.append(gp_mean) # each GP mean
            gp_stds.append(gp_std) # each GP std
        gp_means = np.array(gp_means)
        gp_stds = np.array(gp_stds)

        gp_mean, gp_std = [], []
        for _ in range(gp_means.shape[1]):
            gp_mean.append(gp_means[experts[_], _])
            gp_std.append(gp_stds[experts[_], _])
        gp_mean, gp_std = np.array(gp_mean), np.array(gp_std)
        predictions = gp_mean * self.y_std + self.y_mean # back to original
        return predictions, gp_mean * self.y_std + gp_mean, gp_std * self.y_std 

        # self.model.eval()
        # self.likelihood.eval()
        # x_test = (test_features - self.x_mean) / self.x_std
        # gp_results = self.model(torch.tensor(x_test).float())
        # gp_mean, gp_std = gp_results.mean.detach().numpy(), gp_results.variance.detach().numpy()
        # predictions = gp_mean * self.y_std + self.y_mean
        # return predictions, gp_mean * self.y_std + self.y_mean, gp_std * self.y_std

    def fitting_model(self, train_GT: np.ndarray,train_features: np.ndarray):
        """
        Fit your model on the given training data.
        :param train_features: Training features as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param train_GT: Training pollution concentrations as a 1d NumPy float array of shape (NUM_SAMPLES,)
        """
        # TODO: Fit your model here
        x_train = (train_features - self.x_mean) / self.x_std
        y_train = (train_GT - self.y_mean) / self.y_std
        self.cluster(x_train)

        for clu in range(self.n_experts): # for each cluster, train local GP
            print('training on cluster', clu+1,'/', self.n_experts)
            indices = np.where(self.labels == clu)[0]
            x_t = torch.tensor(x_train[indices]).float()
            y_t = torch.tensor(y_train[indices]).float()
            model = ExactGP(x_t, y_t)
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
            for i in range(self.training_iter):
                optimizer.zero_grad()
                output = model(x_t)
                loss = -mll(output, y_t)
                loss.backward()
                print('Iter %d/%d - Loss: %.3f' % (i + 1, self.training_iter, loss.item()))
                optimizer.step()
            self.models.append(model)

    def cluster(self, X):
        self.kmeans = KMeans(n_clusters=self.n_experts, random_state=0).fit(X) # k-means model
        self.labels = self.kmeans.labels_
        self.centroids = self.kmeans.cluster_centers_

class ExactGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y):
        super(ExactGP, self).__init__(train_x, train_y, likelihood=gpytorch.likelihoods.GaussianLikelihood())
        self.mean_module = ConstantMean()
        # self.kernel = [LinearKernel(), MaternKernel(nu = 1.5)]
        self.kernel = [MaternKernel(nu = 1.5)] # Matern kernel with nu = 1.5 is the best
        self.covar_module = ScaleKernel(AdditiveKernel(*self.kernel)) # we have try many kernels but this one is the best

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def cost_function(ground_truth: np.ndarray, predictions: np.ndarray) -> float:
    """
    Calculates the cost of a set of predictions.

    :param ground_truth: Ground truth pollution levels as a 1d NumPy float array
    :param predictions: Predicted pollution levels as a 1d NumPy float array
    :return: Total cost of all predictions as a single float
    """
    assert ground_truth.ndim == 1 and predictions.ndim == 1 and ground_truth.shape == predictions.shape

    # Unweighted cost
    cost = (ground_truth - predictions) ** 2
    weights = np.ones_like(cost) * COST_W_NORMAL

    # Case i): underprediction
    mask_1 = predictions < ground_truth
    weights[mask_1] = COST_W_UNDERPREDICT

    # Case ii): significant overprediction
    mask_2 = (predictions >= 1.2*ground_truth)
    weights[mask_2] = COST_W_OVERPREDICT

    # Weigh the cost and return the average
    return np.mean(cost * weights)



def perform_extended_evaluation(model: Model, output_dir: str = '/results'):
    """
    Visualizes the predictions of a fitted model.
    :param model: Fitted model to be visualized
    :param output_dir: Directory in which the visualizations will be stored
    """
    print('Performing extended evaluation')
    fig = plt.figure(figsize=(30, 10))
    fig.suptitle('Extended visualization of task 1')

    # Visualize on a uniform grid over the entire coordinate system
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
    )
    visualization_xs = np.stack((grid_lon.flatten(), grid_lat.flatten()), axis=1)

    # Obtain predictions, means, and stddevs over the entire map
    predictions, gp_mean, gp_stddev = model.make_predictions(visualization_xs)
    predictions = np.reshape(predictions, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_mean = np.reshape(gp_mean, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_stddev = np.reshape(gp_stddev, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))

    vmin, vmax = 0.0, 65.0
    vmax_stddev = 35.5

    # Plot the actual predictions
    ax_predictions = fig.add_subplot(1, 3, 1)
    predictions_plot = ax_predictions.imshow(predictions, vmin=vmin, vmax=vmax)
    ax_predictions.set_title('Predictions')
    fig.colorbar(predictions_plot)

    # Plot the raw GP predictions with their stddeviations
    ax_gp = fig.add_subplot(1, 3, 2, projection='3d')
    ax_gp.plot_surface(
        X=grid_lon,
        Y=grid_lat,
        Z=gp_mean,
        facecolors=cm.get_cmap()(gp_stddev / vmax_stddev),
        rcount=EVALUATION_GRID_POINTS_3D,
        ccount=EVALUATION_GRID_POINTS_3D,
        linewidth=0,
        antialiased=False
    )
    ax_gp.set_zlim(vmin, vmax)
    ax_gp.set_title('GP means, colors are GP stddev')

    # Plot the standard deviations
    ax_stddev = fig.add_subplot(1, 3, 3)
    stddev_plot = ax_stddev.imshow(gp_stddev, vmin=vmin, vmax=vmax_stddev)
    ax_stddev.set_title('GP estimated stddev')
    fig.colorbar(stddev_plot)

    # Save figure to pdf
    figure_path = os.path.join(output_dir, 'extended_evaluation.pdf')
    fig.savefig(figure_path)
    print(f'Saved extended evaluation to {figure_path}')

    plt.show()


def main():
    # Load the training dateset and test features
    train_features = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
    train_GT = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)
    #print(train_GT)
    #print(np.mean(train_features, axis=0))
    #print(np.std(train_features, axis=0))
    test_features = np.loadtxt('test_x.csv', delimiter=',', skiprows=1)

    # Fit the model
    print('Fitting model')
    model = Model()
    model.fitting_model(train_GT,train_features)

    # Predict on the test features
    print('Predicting on test features')
    predictions = model.make_predictions(test_features)
    print(predictions)

    if EXTENDED_EVALUATION:
        perform_extended_evaluation(model, output_dir='.')


if __name__ == "__main__":
    main()

