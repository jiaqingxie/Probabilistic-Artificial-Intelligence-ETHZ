import abc
import warnings

import numpy as np
import torch

import os
import matplotlib.pyplot as plt
from torch.optim import SGD

class SGLD(SGD):
    """Implementation of SGLD algorithm.
    References
    ----------
        
    """
    @torch.no_grad()
    def step(self, closure=None):
        """See `torch.optim.step’."""
        loss = super().step(closure)
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad_p = p.grad.data
                if weight_decay!=0:
                    grad_p.add_(alpha=weight_decay,other=p.data)
                langevin_noise = 0.1*torch.randn_like(p.data).mul_(group['lr']**0.5) #  use weight 0.1 to balance the noise
                p.data.add_(grad_p,alpha=-0.5*group['lr'])
                if torch.isnan(p.data).any(): 
                    exit('Exist NaN param after SGLD, Try to tune the parameter')
                if torch.isinf(p.data).any(): 
                    exit('Exist Inf param after SGLD, Try to tune the parameter')
                p.data.add_(langevin_noise)
        return loss

def ece(predicted_probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 30) -> float:
    """
    Computes the Expected Calibration Error (ECE).
    Many options are possible; in this implementation, we provide a simple version.

    Using a uniform binning scheme on the full range of probabilities, zero
    to one, we bin the probabilities of the predicted label only (ignoring
    all other probabilities). For the ith bin, we compute the avg predicted
    probability, p_i, and the bin's total accuracy, a_i.
    We then compute the ith calibration error of the bin, |p_i - a_i|.
    The final returned value is the weighted average of calibration errors of each bin.

    :param predicted_probabilities: Predicted probabilities, float array of shape (num_samples, num_classes)
    :param labels: True labels, int tensor of shape (num_samples,) with each entry in {0, ..., num_classes - 1}
    :param n_bins: Number of bins for histogram binning
    :return: ECE score as a float
    """
    num_samples, num_classes = predicted_probabilities.shape

    # Predictions are the classes with highest probability
    predictions = np.argmax(predicted_probabilities, axis=1)
    prediction_confidences = predicted_probabilities[range(num_samples), predictions]

    # Use uniform bins on the range of probabilities, i.e. closed interval [0.,1.]
    bin_upper_edges = np.histogram_bin_edges([], bins=n_bins, range=(0., 1.))
    bin_upper_edges = bin_upper_edges[1:]  # bin_upper_edges[0] = 0.

    probs_as_bin_num = np.digitize(prediction_confidences, bin_upper_edges)
    sums_per_bin = np.bincount(probs_as_bin_num, minlength=n_bins, weights=prediction_confidences)
    sums_per_bin = sums_per_bin.astype(np.float32)

    total_per_bin = np.bincount(probs_as_bin_num, minlength=n_bins) \
        + np.finfo(sums_per_bin.dtype).eps  # division by zero
    avg_prob_per_bin = sums_per_bin / total_per_bin

    onehot_labels = np.eye(num_classes)[labels]
    accuracies = onehot_labels[range(num_samples), predictions]  # accuracies[i] is 0 or 1
    accuracies_per_bin = np.bincount(probs_as_bin_num, weights=accuracies, minlength=n_bins) / total_per_bin

    prob_of_being_in_a_bin = total_per_bin / float(num_samples)

    ece_ret = np.abs(accuracies_per_bin - avg_prob_per_bin) * prob_of_being_in_a_bin
    ece_ret = np.sum(ece_ret)
    return float(ece_ret)


class ParameterDistribution(torch.nn.Module, metaclass=abc.ABCMeta):
    """
    Abstract class that models a distribution over model parameters,
    usable for Bayes by backprop.
    You can implement this class using any distribution you want
    and try out different priors and variational posteriors.
    All torch.nn.Parameter that you add in the __init__ method of this class
    will automatically be registered and know to PyTorch.
    """

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def log_likelihood(self, values: torch.Tensor) -> torch.Tensor:
        """
        Calculate the log-likelihood of the given values
        :param values: Values to calculate the log-likelihood on
        :return: Log-likelihood
        """
        pass

    @abc.abstractmethod
    def sample(self) -> torch.Tensor:
        """
        Sample from this distribution.
        Note that you only need to implement this method for variational posteriors, not priors.

        :return: Sample from this distribution. The sample shape depends on your semantics.
        """
        pass

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        # DO NOT USE THIS METHOD
        # We only implement it since torch.nn.Module requires a forward method
        warnings.warn('ParameterDistribution should not be called! Use its explicit methods!')
        return self.log_likelihood(values)


def draw_reliability_diagram(out,
                            title="Reliability Diagram", 
                            xlabel="Confidence", 
                            ylabel="Accuracy"):
    """Draws a reliability diagram into a subplot."""
    fig,ax = plt.subplots()
    plt.tight_layout()
    accuracies =  out['calib_accuracy']
    confidences = out['calib_confidence']
    counts = out['p']
    bins = out['bins']

    bin_size = 1.0 / len(counts)
    positions = bins[:-1] + bin_size/2.0

    widths = bin_size
    alphas = 0.3

    colors = np.zeros((len(counts), 4))
    colors[:, 0] = 240 / 255.
    colors[:, 1] = 60 / 255.
    colors[:, 2] = 60 / 255.
    colors[:, 3] = alphas

    gap_plt = ax.bar(positions, np.abs(accuracies - confidences), 
                     bottom=np.minimum(accuracies, confidences), width=widths,
                     edgecolor=colors, color=colors, linewidth=1, label="Gap")

    acc_plt = ax.bar(positions, 0, bottom=accuracies, width=widths,
                     edgecolor="black", color="black", alpha=1.0, linewidth=3,
                     label="Accuracy")

    ax.set_aspect("equal")
    ax.plot([0,1], [0,1], linestyle = "--", color="gray")
    

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)\
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(handles=[gap_plt, acc_plt])
    return fig


def draw_confidence_histogram(out, 
                                draw_averages=True,
                                title="Confidence Diagram",
                                xlabel="Confidence",
                                ylabel="Count"):
    """Draws a confidence histogram into a subplot."""

    fig,ax = plt.subplots()
    zs = out['p']
    bins = out['bins']
    bin_lowers = bins[:-1]
    bin_uppers = bins[1:]
    bin_middles = (bin_lowers + bin_uppers)/2

    bin_size = 1.0 / len(zs)

    ax.bar(bin_middles, zs, width=bin_size * 0.9)
   
    ax.set_xlim(0, 1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig


    


