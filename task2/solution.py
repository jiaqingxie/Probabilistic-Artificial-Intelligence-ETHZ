import os
from random import sample
import typing

import numpy as np
import torch
import torch.optim
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from torch import nn
from torch.nn import functional as F
from tqdm import trange
import tqdm
from collections import deque
from torch.distributions.normal import Normal
from torch.distributions.independent import Independent

from util import ece, ParameterDistribution, draw_reliability_diagram, draw_confidence_histogram, SGLD
from enum import Enum

# TODO: Reliability_diagram_1. Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = True

class Approach(Enum):
    Dummy_Trainer = 0
    MCDropout = 1
    Ensemble = 2
    Backprop = 3
    SGLD = 4
    SelfMade = 5 


def run_solution(dataset_train: torch.utils.data.Dataset, data_dir: str = os.curdir, output_dir: str = '/results/'):
    """
    Run your task 2 solution.
    This method should train your model, evaluate it, and return the trained model at the end.
    Make sure to preserve the method signature and to return your trained model,
    else the checker will fail!

    :param dataset_train: Training dataset
    :param data_dir: Directory containing the datasets
    :return: Your trained model
    """

    # TODO: Combined model_1: Choose if you want to combine with multiple methods or not
    combined_model = True

    if not combined_model:
        
        # TODO General_1: Choose your approach here
        #approach = Approach.MCDropout
        #approach = Approach.SGLD
        #approach == Approach.Ensemble
        #approach == Approach.SelfMade
        approach = Approach.Backprop

        if approach == Approach.Dummy_Trainer:
            trainer = DummyTrainer(dataset_train=dataset_train)
        if approach == Approach.MCDropout:
            trainer = DropoutTrainer(dataset_train=dataset_train)
        if approach == Approach.Ensemble:
            trainer = EnsembleTrainer(dataset_train=dataset_train)
        if approach == Approach.Backprop:
            trainer = BackpropTrainer(dataset_train=dataset_train)
        if approach == Approach.SGLD:
            trainer = SGLDTrainer(dataset_train=dataset_train)
        if approach == Approach.SelfMade:
            trainer = SelfTrainer(dataset_train=dataset_train)

        # Train the model
        print('Training model', approach.name)
        trainer.train()

        # Predict using the trained model
        print('Evaluating model on training data')
        eval_loader = torch.utils.data.DataLoader(
            dataset_train, batch_size=64, shuffle=False, drop_last=False
        )
        evaluate(trainer, eval_loader, data_dir, output_dir)

        # IMPORTANT: return your model here!
        return trainer
    
    elif combined_model:
        # using combined model

        # TODO: Combined model_2: If you want to use combined methods 
        # you can set the trainers you want to use like below
        trainer1 = DropoutTrainer(dataset_train=dataset_train)
        trainer2 = BackpropTrainer(dataset_train=dataset_train)
        trainer3 = EnsembleTrainer(dataset_train=dataset_train)
        trainer_list = [trainer1,trainer2,trainer3]

        # Train the combined model
        for trainer_i in trainer_list:
            trainer_i.train()

        # Evaulate each of the combined models
        for trainer_i in trainer_list:
            evaluate(trainer_i, eval_loader, data_dir, output_dir)

    # IMPORTANT: return your combined model here!
        return trainer_list


def calc_calibration_curve(predicted_probabilities: np.ndarray, labels: np.ndarray, num_bins=10)-> dict:
    """
    Calculate ece and understand what is good calibration. This task is part of the 
    extended evaluation and not required for passing. 
    """

    num_samples, num_classes = predicted_probabilities.shape
    predicted_classes = np.argmax(predicted_probabilities, axis=1) 
    confidences = predicted_probabilities[range(num_samples), predicted_classes]
    bins = np.linspace(start=0,stop=1,num=num_bins+1)

    bin_lowers = bins[:-1]
    bin_uppers = bins[1:]
    accuracies = predicted_classes == labels

    calib_confidence = np.zeros(num_bins,dtype=np.float)
    calib_accuracy = np.zeros(num_bins,dtype=np.float)
    ratios = np.zeros(num_bins,dtype=np.float)


    for bin_i, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
        # TODO: Reliability_diagram_2. Calculate calibration confidence, accuracy in every bin

        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            calib_confidence[bin_i] = np.mean(confidences[in_bin])
            calib_accuracy[bin_i] = np.mean(accuracies[in_bin])
            ratios[bin_i] = prop_in_bin 
    
    return {"calib_confidence": calib_confidence, "calib_accuracy": calib_accuracy, "p": ratios, "bins": bins}


class Framework(object):
    def __init__(self, dataset_train:torch.utils.data.Dataset, *args, **kwargs):
        """
        Basic Framework for your bayesian neural network.
        Other solutions like MC Dropout, Ensemble learning will based upon this.
        """
        self.train_set = dataset_train
        self.print_interval = 100 # number of batches until updated metrics are displayed during training

    def train(self):
        raise NotImplementedError()

    def predict(self, data_loader: torch.utils.data.DataLoader) -> np.ndarray:
        """
        Predict the class probabilities using your trained model.
        This method should return an (num_samples, 10) NumPy float array
        such that the second dimension sums up to 1 for each row.

        :param data_loader: Data loader yielding the samples to predict on
        :return: (num_samples, 10) NumPy float array where the second dimension sums up to 1 for each row
        """
        probability_batches = []
        
        for batch_x, _ in tqdm.tqdm(data_loader):
            current_probabilities = self.predict_probabilities(batch_x).detach().numpy()
            probability_batches.append(current_probabilities)

        output = np.concatenate(probability_batches, axis=0)
        assert isinstance(output, np.ndarray)
        assert output.ndim == 2 and output.shape[1] == 10
        assert np.allclose(np.sum(output, axis=1), 1.0)
        return output

    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


def combined_predict(data_loader: torch.utils.data.DataLoader, models_list: list) -> np.ndarray:
    """
    Predict the class probabilities using a combination of your trained model.
    This method should return an (num_samples, 10) NumPy float array - the same as 
    predict()- such that the second dimension sums up to 1 for each row.
    :param data_loader: Data loader yielding the samples to predict on
    :return: (num_samples, 10) NumPy float array where the second dimension sums up to 1 for each row
    """

    probability_batches = []

    for batch_x, _ in tqdm.tqdm(data_loader):
        # TODO: Combined model_3. Predict with your combined model
        current_probabilities = None
        probability_batches.append(current_probabilities)
        
    output = np.concatenate(probability_batches, axis=0)
    assert isinstance(output, np.ndarray)
    assert output.ndim == 2 and output.shape[1] == 10
    assert np.allclose(np.sum(output, axis=1), 1.0)
    return output


class DummyTrainer(Framework):
    """
    Trainer implementing a simple feedforward neural network.
    You can learn how to build your own trainer and use this model as a reference/baseline for 
    the calibration of a standard Neural Network. 
    """
    def __init__(self, dataset_train,
                 *args, **kwargs):
        super().__init__(dataset_train, *args, **kwargs)

        # Hyperparameters and general parameters
        self.batch_size = 128
        self.learning_rate = 1e-2
        self.num_epochs = 200


        self.network = MNISTNet(in_features=28*28,out_features=10)
        self.train_loader = torch.utils.data.DataLoader(
            dataset_train, batch_size=self.batch_size, shuffle=True, drop_last=True
            )
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate) 
        
    def train(self):
        self.network.train()
        progress_bar = trange(self.num_epochs)
        for _ in progress_bar:
            for batch_idx, (batch_x, batch_y) in enumerate(self.train_loader):
                # batch_x are of shape (batch_size, 784), batch_y are of shape (batch_size,)

                self.network.zero_grad()

                # Perform forward pass
                current_logits = self.network(batch_x)

                # Calculate the loss
                # We use the negative log likelihood as the loss
                # Combining nll_loss with a log_softmax is better for numeric stability
                loss = F.nll_loss(F.log_softmax(current_logits, dim=1), batch_y, reduction='sum')

                # Backpropagate to get the gradients
                loss.backward()

                self.optimizer.step()

                # Update progress bar with accuracy occasionally
                if batch_idx % self.print_interval == 0:
                    current_logits = self.network(batch_x)

                    current_accuracy = (current_logits.argmax(axis=1) == batch_y).float().mean()
                    progress_bar.set_postfix(loss=loss.item(), acc=current_accuracy.item())

    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        # using the confidence as estimated probaility, 
        self.network.eval()
        assert x.shape[1] == 28 ** 2
        estimated_probability = F.softmax(self.network(x), dim=1)
        assert estimated_probability.shape == (x.shape[0], 10)
        return estimated_probability

class MNISTNet(nn.Module):
    def __init__(self,
                in_features: int, 
                out_features: int,
                dropout_p=0,
                dropout_at_eval=False
                ):
        super().__init__()
        # TODO General_2: Play around with the network structure.
        # You could change the depth or width of the model
        # I have changed the width from 100 -> 256, 64
        self.layer1 = nn.Linear(in_features, 256) # 256 for MC-dropout
        self.layer2 = nn.Linear(256, 84) # 256 for MC-dropout
        self.layer3 = nn.Linear(84, out_features)
        self.dropout_p = dropout_p
        self.dropout_at_eval = dropout_at_eval

    def forward(self, x):
        # TODO General_2: Play around with the network structure
        # You might add different modules like Pooling 
        x = F.dropout(
                F.relu(self.layer1(x)),
                p=self.dropout_p,
                training=self.training or self.dropout_at_eval
        )
        x = F.dropout(
                F.relu(self.layer2(x)),
                p=self.dropout_p,
                training=self.training or self.dropout_at_eval
        )
        class_probs = self.layer3(x)
        return class_probs

class SelfMadeNetwork(nn.Module):
    def __init__(self,
                in_features: int, 
                out_features: int,
                dropout_p=0,
                dropout_at_eval=False
                ):
        super().__init__()
        # TODO General_2: Play around with the network structure.
        # You could change the depth or width of the model
        # I have changed the width from 100 -> 256, 64
        self.layer1 = nn.Linear(in_features, 256) # 256 for MC-dropout
        self.layer2 = nn.Linear(256, 128) # 256 for MC-dropout
        self.layer3 = nn.Linear(128, 84) 
        self.layer4 = nn.Linear(84, out_features)
        self.dropout_p = dropout_p
        self.dropout_at_eval = dropout_at_eval

    def forward(self, x):
        # TODO General_2: Play around with the network structure
        # You might add different modules like Pooling 
        x = F.dropout(
                F.relu(self.layer1(x)),
                p=self.dropout_p,
                training=self.training or self.dropout_at_eval
        )
        x = F.dropout(
                F.relu(self.layer2(x)),
                p=self.dropout_p,
                training=self.training or self.dropout_at_eval
        )

        x = F.dropout(
                F.relu(self.layer3(x)),
                p=self.dropout_p,
                training=self.training or self.dropout_at_eval
        )
        class_probs = self.layer4(x)
        return class_probs

class DropoutTrainer(Framework):
    def __init__(self, dataset_train,
                 *args, **kwargs):
        super().__init__(dataset_train, *args, **kwargs)

        # Hyperparameters and general parameters
        # TODO: MC_Dropout_4. Do experiments and tune hyperparameters
        self.batch_size = 256
        self.learning_rate = 1e-3
        self.num_epochs = 200
        torch.manual_seed(14504) # set seed for reproducibility
        
        # TODO: MC_Dropout_1. Initialize the MC_Dropout network and optimizer here
        # You can check the Dummy Trainer above for intuition about what to do
        #self.network = SelfMadeNetwork(in_features=28*28,out_features=10, dropout_p=0.18, dropout_at_eval=True)
        self.network = MNISTNet(in_features=28*28,out_features=10, dropout_p=0.15, dropout_at_eval=True)
        self.train_loader = torch.utils.data.DataLoader(
            dataset_train, batch_size=self.batch_size, shuffle=True, drop_last=True
            )
        # As pointed out in the paper, optimizer required a L2 Norm Penalty.
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate, weight_decay= 1e-3) 
        self.criterion = nn.CrossEntropyLoss()
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=4, gamma=0.9)

    def train(self):
        self.network.train()
        progress_bar = trange(self.num_epochs)
        for _ in progress_bar:
            for batch_idx, (batch_x, batch_y) in enumerate(self.train_loader):
                # batch_x are of shape (batch_size, 784), batch_y are of shape (batch_size,)

                self.network.zero_grad()
                # TODO: MC_Dropout_2. Implement MCDropout training here
                # You need to calculate the loss based on the literature
                current_logits = self.network(batch_x)
                # loss keep unchanged 
                loss = self.criterion(F.log_softmax(current_logits, dim=1), batch_y)
                #loss = F.nll_loss(F.log_softmax(current_logits, dim=1), batch_y, reduction='sum')
                # Backpropagate to get the gradients
                loss.backward()

                self.optimizer.step()
                
                # Update progress bar with accuracy occasionally
                if batch_idx % self.print_interval == 0:
                    current_logits = self.network(batch_x)
                    current_accuracy = (current_logits.argmax(axis=1) == batch_y).float().mean()
                    progress_bar.set_postfix(loss=loss.item(), acc=current_accuracy.item())
            self.lr_scheduler.step()

    def predict_probabilities(self, x: torch.Tensor, num_sample=50) -> torch.Tensor:
        # TODO: MC_Dropout_3. Implement your MC_dropout prediction here
        # You need to sample from your trained model here multiple times
        # in order to implement Monte Carlo integration
        assert x.shape[1] == 28 ** 2
        self.network.eval()
        y_preds = 0
        for i in range(num_sample):
            if i == 0:
                prob = F.softmax(self.network(x), dim=1)
                y_preds = prob.unsqueeze(0)
            else:
                prob = F.softmax(self.network(x), dim=1)
                y_preds = torch.cat((y_preds, prob.unsqueeze(0)), dim = 0) 

        estimated_probability = torch.mean(y_preds, axis = 0)
        assert estimated_probability.shape == (x.shape[0], 10)  
        return estimated_probability


class EnsembleTrainer(Framework):
    def __init__(self, dataset_train,
                 *args, **kwargs):
        super().__init__(dataset_train, *args, **kwargs)

        # Hyperparameters and general parameters
        # TODO: Ensemble_4. Do experiments and tune hyperparameters
        self.batch_size = 256
        self.learning_rate = 1e-3
        self.num_epochs = 100
        torch.manual_seed(0)
        self.train_loader = torch.utils.data.DataLoader(
            dataset_train, batch_size=self.batch_size, shuffle=True, drop_last=True
            )
        # TODO: Ensemble_1.  initialize the Ensemble network list and optimizer.
        # You can check the Dummy Trainer above for intution about what to do
        # You need to build an ensemble of initialized networks here
        self.num_ensemble = 5
        self.criterion = nn.CrossEntropyLoss()
        self.EnsembleNetworks = []
        self.optimizer = []
        self.lr_scheduler = []
        for i in range(self.num_ensemble):
            self.EnsembleNetworks.append(MNISTNet(784, 10, 0, False))
            self.optimizer.append(torch.optim.Adam(self.EnsembleNetworks[i].parameters(), lr=self.learning_rate, weight_decay=1e-3))
            self.lr_scheduler.append(torch.optim.lr_scheduler.StepLR(self.optimizer[i], step_size=5, gamma=0.9))

    def train(self):
        counter = 0
        for network in self.EnsembleNetworks:   
            network.train()
            progress_bar = trange(self.num_epochs)
            for _ in progress_bar:
                for batch_idx, (batch_x, batch_y) in enumerate(self.train_loader):
                    # batch_x are of shape (batch_size, 784), batch_y are of shape (batch_size,)
                    network.zero_grad()
                    # TODO: Ensemble_2. Implement Ensemble training here
                    # You need to calculate the loss based on the literature
                    current_logits = network(batch_x)
                    loss = self.criterion(F.log_softmax(current_logits, dim=1), batch_y)
                    # Backpropagate to get the gradients
                    loss.backward()
                    self.optimizer[counter].step()
                    # Update progress bar with accuracy occasionally
                    if batch_idx % self.print_interval == 0:
                        current_logits = network(batch_x)
                        current_accuracy = (current_logits.argmax(axis=1) == batch_y).float().mean()
                        progress_bar.set_postfix(loss=loss.item(), acc=current_accuracy.item())

                self.lr_scheduler[counter].step()
            counter+=1

    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == 28 ** 2
        # TODO: Ensemble_3. Implement Ensemble prediction here
        # You need obtain predictions from each ensemble member and think about 
        # how to combine the results from each of them

        counter = 0
        y_preds = 0
        for network in self.EnsembleNetworks:  
            network.eval() 
            if counter == 0:
                prob = F.softmax(network(x), dim=1)
                y_preds = prob.unsqueeze(0)
            else:
                prob = F.softmax(network(x), dim=1)
                y_preds = torch.cat((y_preds, prob.unsqueeze(0)), dim = 0) 
            counter+=1

        estimated_probability = torch.mean(y_preds, axis = 0)
        
        assert estimated_probability.shape == (x.shape[0], 10)  
        return estimated_probability


class SGLDTrainer(Framework):
    def __init__(self, dataset_train,
                 *args, **kwargs):
        super().__init__(dataset_train, *args, **kwargs)

        # Hyperparameters and general parameters
        # TODO: SGLD_4. Do experiments and tune hyperparameters
        self.batch_size = 1024
        self.learning_rate = 2e-4
        self.num_epochs = 100
        self.burn_in = 2
        self.sample_interval = 3
        self.max_size = 15
        torch.manual_seed(14504)
        # TODO: SGLD_1.  initialize the SGLD network.
        # You can check the Dummy Trainer above for intution about what to do
        self.network = MNISTNet(784, 10, 0.3, True)
        self.train_loader = torch.utils.data.DataLoader(
            dataset_train, batch_size=self.batch_size, shuffle=True, drop_last=True
            )
        # SGLD optimizer is provided
        self.optimizer = SGLD(self.network.parameters(),lr = self.learning_rate, weight_decay=6.5e-5, momentum=0.5, dampening = 0,
                        nesterov = True)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.68)
        # deques support bi-directional addition and deletion
        # You can add models in the right side of a deque by append()
        # You can delete models in the left side of a deque by popleft()
        self.SGLDSequence = deque() 

    def train(self):
        num_iter = 0
        print('Training model')

        self.network.train()
        progress_bar = trange(self.num_epochs)
        
        for _ in progress_bar:
            num_iter+=1

            for batch_idx, (batch_x, batch_y) in enumerate(self.train_loader):
                self.network.zero_grad()

                # Perform forward pass
                current_logits = self.network(batch_x)

                # Calculate the loss
                # TODO: SGLD_1. Implement SGLD training here
                # You need to calculate the loss based on the literature3
                loss = F.nll_loss(F.log_softmax(current_logits, dim=1), batch_y, reduction='sum')
                #loss = self.criterion(F.log_softmax(current_logits, dim=1), batch_y)
                # Backpropagate to get the gradients
                loss.backward()
                self.optimizer.step()

                if batch_idx % self.print_interval == 0:
                    current_logits = self.network(batch_x)
                    current_accuracy = (current_logits.argmax(axis=1) == batch_y).float().mean()
                    progress_bar.set_postfix(loss=loss.item(), acc=current_accuracy.item())
  
            
            # TODO: SGLD_2. save the model samples if it satisfies the following conditions:
            # We are 1) past the burn-in epochs and 2) reached one of the regular sampling intervals we save the model at
            # If the self.SGLDSequence already exceeded the maximum length then we have to delete the oldest model
            if num_iter < self.burn_in:
                continue
            else:
                if len(self.SGLDSequence) >= self.max_size:
                    if num_iter % self.sample_interval == 0:
                        self.SGLDSequence.popleft()
                        self.SGLDSequence.append(self.network)
                    else:
                        continue
                else:
                    if num_iter % self.sample_interval == 0:
                        self.SGLDSequence.append(self.network)
                    else:
                        continue
            
            self.lr_scheduler.step()
        
    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == 28 ** 2
        

        # TODO SGLD_3: Implement SGLD predictions here
        # You need to obtain the prediction from each network
        # in SGLDSequence and combine the predictions

        for i in range(len(self.SGLDSequence)):
            self.SGLDSequence[i].eval()
            if i == 0:
                prob = F.softmax(self.SGLDSequence[i](x), dim=1)
                y_preds = prob.unsqueeze(0)
            else:
                prob = F.softmax(self.SGLDSequence[i](x), dim=1)
                y_preds = torch.cat((y_preds, prob.unsqueeze(0)), dim = 0) 

        estimated_probability = torch.mean(y_preds, axis = 0)
        
        assert estimated_probability.shape == (x.shape[0], 10)  
        return estimated_probability


# 1.0 constructing priors from scratch
# 2.0 constructing BNN from pyro
class BackpropTrainer(Framework):
    def __init__(self, dataset_train,  *args, **kwargs):
        super().__init__(dataset_train, *args, **kwargs)

        # Hyperparameters and general parameters
        # TODO: Backprop_7 Tune parameters and add more if necessary
        self.hidden_features=(128, 84)
        self.batch_size = 256
        self.num_epochs = 100
        learning_rate = 1.05e-3
        torch.manual_seed(14504)
        self.network = BayesNet(in_features=28 * 28, hidden_features=self.hidden_features, out_features=10)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate, weight_decay=1e-6)
        self.train_loader = torch.utils.data.DataLoader(
            dataset_train, batch_size=self.batch_size, shuffle=True, drop_last=True
            )
        self.criterion = nn.CrossEntropyLoss() 
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.94)
        

    def train(self):

        self.network.train()

        progress_bar = trange(self.num_epochs)
        for _ in progress_bar:
            num_batches = len(self.train_loader)
            for batch_idx, (batch_x, batch_y) in enumerate(self.train_loader):
                # batch_x are of shape (batch_size, 784), batch_y are of shape (batch_size,)
                self.network.zero_grad()
                current_logits, log_prior, log_posterior = self.network(batch_x)
                # TODO: Backprop_6: Implement Bayes by backprop training here
                #loss = self.criterion(F.log_softmax(current_logits, dim=1), batch_y)
                loss = F.nll_loss(F.log_softmax(current_logits, dim=1), batch_y, reduction='sum')
                loss += (1/num_batches) * (log_posterior - log_prior)
                loss.backward()
                self.optimizer.step()
                # Update progress bar with accuracy occasionally
                if batch_idx % self.print_interval == 0:
                    current_logits, _, _ = self.network(batch_x)
                    current_accuracy = (current_logits.argmax(axis=1) == batch_y).float().mean()
                    progress_bar.set_postfix(loss=loss.item(), acc=current_accuracy.item())

            self.lr_scheduler.step()
            

    def predict_probabilities(self, x: torch.Tensor, num_mc_samples: int = 20) -> torch.Tensor:
        """
        Predict class probabilities for the given features by sampling from this BNN.

        :param x: Features to predict on, float tensor of shape (batch_size, in_features)
        :param num_mc_samples: Number of MC samples to take for prediction
        :return: Predicted class probabilities, float tensor of shape (batch_size, 10)
            such that the last dimension sums up to 1 for each row
        """
        
        probability_samples = torch.stack([F.softmax(self.network(x)[0], dim=1) for _ in range(num_mc_samples)], dim=0)
        estimated_probability = torch.mean(probability_samples, dim=0)

        assert estimated_probability.shape == (x.shape[0], 10)
        assert torch.allclose(torch.sum(estimated_probability, dim=1), torch.tensor(1.0))
        return estimated_probability

class BayesianLayer(nn.Module):
    """
    Module implementing a single Bayesian feedforward layer.
    It maintains a prior and variational posterior for the weights (and biases)
    and uses sampling to approximate the gradients via Bayes by backprop.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        Create a BayesianLayer.

        :param in_features: Number of input features
        :param out_features: Number of output features
        :param bias: If true, use a bias term (i.e., affine instead of linear transformation)
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        # Background Pytorch will backpropogate gradients to an object initialized with
        # torch.Parameter(...) and the object will be updated when computing loss.backwards()
        # during training. This will not happen for a torch.Tensor(...) object, which is by default a constant. 

        # TODO: Backprop_1. Create a suitable prior for weights and biases as an instance of ParameterDistribution.
        #  You can use the same prior for both weights and biases, but are also free to experiment with other priors.
        #  You can create constants using torch.tensor(...).
        #  Do NOT use torch.Parameter(...) here since the prior should not be optimized!
        #  Example: self.prior = MyPrior(torch.tensor(0.0), torch.tensor(1.0))

        self.prior = UnivariateGaussian(
            torch.tensor(0.0), torch.tensor(1.0)
        )
        assert isinstance(self.prior, ParameterDistribution)
        assert not any(True for _ in self.prior.parameters()), 'Prior cannot have parameters'

        # TODO: Backprop_1. Create a suitable variational posterior for weights as an instance of ParameterDistribution.
        #  You need to create separate ParameterDistribution instances for weights and biases,
        #  but you can use the same family of distributions for each if you want.
        #  IMPORTANT: You need to create a nn.Parameter(...) for each parameter
        #  and add those parameters as an attribute in the ParameterDistribution instances.
        #  If you forget to do so, PyTorch will not be able to optimize your variational posterior.
        # The variational posterior for weights is created here. For the biases it is created further down. 
        #  Example: self.weights_var_posterior = MyPosterior(
        #      torch.nn.Parameter(torch.zeros((out_features, in_features))),
        #      torch.nn.Parameter(torch.ones((out_features, in_features)))
        #  )
        self.weights_var_posterior =  MultivariateDiagonalGaussian(
            mu = nn.Parameter(
                torch.FloatTensor(self.out_features, self.in_features).normal_(mean=0.0, std=0.1)
                ),
            rho = nn.Parameter(
                torch.FloatTensor(self.out_features, self.in_features).normal_(mean=-2.9, std=0.1)
                )
        )

        assert isinstance(self.weights_var_posterior, ParameterDistribution)
        assert any(True for _ in self.weights_var_posterior.parameters()), 'Weight posterior must have parameters'

        if self.use_bias:
            # TODO: Backprop_1. Similarly as you did above for the weights, create the bias variational posterior instance here.
            #  Make sure to follow the same rules as for the weight variational posterior.
            self.bias_var_posterior = MultivariateDiagonalGaussian(
            mu = nn.Parameter(
                torch.FloatTensor(self.out_features).normal_(mean=0.0, std=0.1)
                ),
            rho = nn.Parameter(
                torch.FloatTensor(self.out_features).normal_(mean=-2.9, std=0.1)
                )
            )
            assert isinstance(self.bias_var_posterior, ParameterDistribution)
            assert any(True for _ in self.bias_var_posterior.parameters()), 'Bias posterior must have parameters'
        else:
            self.bias_var_posterior = None


    def forward(self, inputs: torch.Tensor):
        """
        Perform one forward pass through this layer.
        If you need to sample weights from the variational posterior, you can do it here during the forward pass.
        Just make sure that you use the same weights to approximate all quantities
        present in a single Bayes by backprop sampling step.

        :param inputs: Flattened input images as a (batch_size, in_features) float tensor
        :return: 3-tuple containing
            i) transformed features using stochastic weights from the variational posterior,
            ii) sample of the log-prior probability, and
            iii) sample of the log-variational-posterior probability
        """
        # TODO: Backprop_2. Perform a forward pass as described in this method's docstring.
        #  Make sure to check whether `self.use_bias` is True,
        #  and if yes, include the bias as well.
        lambd= torch.randn(self.out_features, self.in_features)
        weights = self.weights_var_posterior.mu + F.softplus(self.weights_var_posterior.rho) * lambd
        log_prior = self.prior.log_likelihood(weights).sum()
        log_variational_posterior = self.weights_var_posterior.log_likelihood(weights).sum()

        if self.use_bias:
            lambd = torch.randn(self.out_features)
            bias = self.bias_var_posterior.mu + F.softplus(self.bias_var_posterior.rho) * lambd
            log_prior += self.prior.log_likelihood(bias).sum()
            log_variational_posterior += self.bias_var_posterior.log_likelihood(bias)
        else:
            bias = None
        return F.linear(inputs, weights, bias), log_prior, log_variational_posterior


class BayesNet(nn.Module):
    """
    Module implementing a Bayesian feedforward neural network using BayesianLayer objects.
    """

    def __init__(self, in_features: int, hidden_features: typing.Tuple[int, ...], out_features: int):
        """
        Create a BNN.

        :param in_features: Number of input features
        :param hidden_features: Tuple where each entry corresponds to a (Bayesian) hidden layer with
            the corresponding number of features.
        :param out_features: Number of output features
        """

        super().__init__()

        feature_sizes = (in_features,) + hidden_features + (out_features,)
        num_affine_maps = len(feature_sizes) - 1
        self.layers = nn.ModuleList([
            BayesianLayer(feature_sizes[idx], feature_sizes[idx + 1], bias=True)
            for idx in range(num_affine_maps)
        ])
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform one forward pass through the BNN using a single set of weights
        sampled from the variational posterior.

        :param x: Input features, float tensor of shape (batch_size, in_features)
        :return: 3-tuple containing
            i) output features using stochastic weights from the variational posterior,
            ii) sample of the log-prior probability, and
            iii) sample of the log-variational-posterior probability
        """

        # TODO: Backprop_3. Perform a full pass through your BayesNet as described in this method's docstring.
        #  You can look at Dummy Trainer to get an idea how a forward pass might look like.
        #  Don't forget to apply your activation function in between BayesianLayers!
        log_prior, log_variational_posterior = 0.0, 0.0
        output_features = x
        for i, layer_i in enumerate(self.layers):
            output_features, log_prior_i, log_variational_posterior_i = layer_i(output_features)
            log_prior += log_prior_i
            log_variational_posterior += log_variational_posterior_i
            if i < len(self.layers) - 1: # last layer doesnt need activation
                output_features = self.activation(output_features)
        return output_features, log_prior, log_variational_posterior
        

    def predict_probabilities(self, x: torch.Tensor, num_mc_samples: int = 15) -> torch.Tensor:
        """
        Predict class probabilities for the given features by sampling from this BNN.

        :param x: Features to predict on, float tensor of shape (batch_size, in_features)
        :param num_mc_samples: Number of MC samples to take for prediction
        :return: Predicted class probabilities, float tensor of shape (batch_size, 10)
            such that the last dimension sums up to 1 for each row
        """
        probability_samples = torch.stack([F.softmax(self.forward(x)[0], dim=1) for _ in range(num_mc_samples)], dim=0)
        estimated_probability = torch.mean(probability_samples, dim=0)

        assert estimated_probability.shape == (x.shape[0], 10)
        assert torch.allclose(torch.sum(estimated_probability, dim=1), torch.tensor(1.0))
        return estimated_probability


class UnivariateGaussian(ParameterDistribution):
    """
    Univariate Gaussian distribution.
    For multivariate data, this assumes all elements to be i.i.d.
    """

    def __init__(self, mu: torch.Tensor, sigma: torch.Tensor):
        super(UnivariateGaussian, self).__init__()  # always make sure to include the super-class init call!
        assert mu.size() == () and sigma.size() == ()
        assert sigma > 0
        self.mu = mu
        self.sigma = sigma

    def log_likelihood(self, values: torch.Tensor) -> torch.Tensor:
        # TODO: Backprop_4. You need to complete the log likelihood function 
        # for the Univariate Gaussian distribution. 
        ans = Normal(self.mu, self.sigma).log_prob(values)
        return ans

    def sample(self) -> torch.Tensor:
        # TODO: Backprop_4. You need to complete the sample function 
        # for the Univariate Gaussian distribution. 
        ans = Normal(self.mu, self.sigma).sample()
        return ans

class MultivariateDiagonalGaussian(ParameterDistribution):
    """
    Multivariate diagonal Gaussian distribution,
    i.e., assumes all elements to be independent Gaussians
    but with different means and standard deviations.
    This parameterizes the standard deviation via a parameter rho as
    sigma = softplus(rho).
    """

    def __init__(self, mu: torch.Tensor, rho: torch.Tensor):
        super(MultivariateDiagonalGaussian, self).__init__()  # always make sure to include the super-class init call!
        assert mu.size() == rho.size()
        self.mu = mu
        self.rho = rho

    def log_likelihood(self, values: torch.Tensor) -> torch.Tensor:
        # TODO: Backprop_5. You need to complete the log likelihood function 
        # for the Multivariate DiagonalGaussian Gaussian distribution. 
        sigma = F.softplus(self.rho)
        ans = Independent(Normal(self.mu, sigma), 1).log_prob(values)
        return ans

    def sample(self) -> torch.Tensor:
        # TODO: Backprop_5. You need to complete the sample function 
        # for the Multivariate DiagonalGaussian Gaussian distribution. 
        sigma = F.softplus(self.rho)
        ans = Independent(Normal(self.mu, sigma), 1).sample(self.mu.size())
        return ans

class SelfTrainer(Framework):
    def __init__(self, dataset_train, print_interval=50,*args, **kwargs):
        """
        Basic Framework for creating your own bayesian neural network trainer.
        """
        self.train_set = dataset_train
        self.print_interval = print_interval  # number of batches until updated metrics are displayed during training

    def train(self, num_epochs):
        raise NotImplementedError()

    def predict(self, data_loader: torch.utils.data.DataLoader) -> np.ndarray:
        """
        Predict the class probabilities using your trained model.
        This method should return an (num_samples, 10) NumPy float array
        such that the second dimension sums up to 1 for each row.

        :param data_loader: Data loader yielding the samples to predict on
        :return: (num_samples, 10) NumPy float array where the second dimension sums up to 1 for each row
        """
        probability_batches = []
        
        for batch_x, batch_y in tqdm.tqdm(data_loader):
            current_probabilities = self.predict_probabilities(batch_x).detach().numpy()
            probability_batches.append(current_probabilities)

        output = np.concatenate(probability_batches, axis=0)
        assert isinstance(output, np.ndarray)
        assert output.ndim == 2 and output.shape[1] == 10
        assert np.allclose(np.sum(output, axis=1), 1.0)
        return output

    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


def evaluate(model:Framework, eval_loader: torch.utils.data.DataLoader, data_dir: str, output_dir: str):
    """
    Evaluate your model.
    :param model: Trained model to evaluate
    :param eval_loader: Data loader containing the training set for evaluation
    :param data_dir: Data directory from which additional datasets are loaded
    :param output_dir: Directory into which plots are saved
    """
    print("evaulating")
    # Predict class probabilities on test data
    predicted_probabilities = model.predict(eval_loader)

    # Calculate evaluation metrics
    predicted_classes = np.argmax(predicted_probabilities, axis=1)
    actual_classes = eval_loader.dataset.tensors[1].detach().numpy()
    accuracy = np.mean((predicted_classes == actual_classes)) 
    ece_score = ece(predicted_probabilities, actual_classes)
    print(f'Accuracy: {accuracy.item():.3f}, ECE score: {ece_score:.3f}')

    # TODO: Reliability_diagram_3. draw reliability diagram on Test Data
    # You can uncomment the below code to make it run. You can learn from
    # the graph about how to improve your model. Remember first to complete 
    # the function of calc_calibration_curve.
    
    print('Plotting reliability diagram on Test Dataset')
    out = calc_calibration_curve(predicted_probabilities, actual_classes, num_bins = 30)
    fig = draw_reliability_diagram(out)
    fig.savefig(os.path.join(output_dir, 'reliability-diagram.pdf'))


    if EXTENDED_EVALUATION:
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        ### draw reliability diagram
        print('Plotting reliability diagram')
        reliability_diagram_datax = np.load(os.path.join(data_dir, 'reliability_diagram_data_x.npy'))
        reliability_diagram_datay = np.load(os.path.join(data_dir, 'reliability_diagram_data_y.npy'))
        for i in range(3):
            demo_prob_i = reliability_diagram_datax[i]
            demo_label_i = reliability_diagram_datay[i]
            out = calc_calibration_curve(demo_prob_i, demo_label_i, num_bins = 9)
            fig = draw_reliability_diagram(out)
            fig.savefig(os.path.join(output_dir, str(i)+'_reliability-diagram.pdf'))
        
        eval_samples = eval_loader.dataset.tensors[0].detach().numpy()

        # Determine confidence per sample and sort accordingly
        confidences = np.max(predicted_probabilities, axis=1)
        sorted_confidence_indices = np.argsort(confidences)

        # Plot samples your model is most confident about
        print('Plotting most confident MNIST predictions')
        most_confident_indices = sorted_confidence_indices[-10:]
        fig, ax = plt.subplots(4, 5, figsize=(13, 11))
        for row in range(0, 4, 2):
            for col in range(5):
                sample_idx = most_confident_indices[5 * row // 2 + col]
                ax[row, col].imshow(np.reshape(eval_samples[sample_idx], (28, 28)), cmap='gray')
                ax[row, col].set_axis_off()
                ax[row + 1, col].set_title(f'predicted {predicted_classes[sample_idx]}, actual {actual_classes[sample_idx]}')
                bar_colors = ['C0'] * 10
                bar_colors[actual_classes[sample_idx]] = 'C1'
                ax[row + 1, col].bar(
                    np.arange(10), predicted_probabilities[sample_idx], tick_label=np.arange(10), color=bar_colors
                )
        fig.suptitle('Most confident predictions', size=20)
        fig.savefig(os.path.join(output_dir, 'mnist_most_confident.pdf'))

        # Plot samples your model is least confident about
        print('Plotting least confident MNIST predictions')
        least_confident_indices = sorted_confidence_indices[:10]
        fig, ax = plt.subplots(4, 5, figsize=(13, 11))
        for row in range(0, 4, 2):
            for col in range(5):
                sample_idx = least_confident_indices[5 * row // 2 + col]
                ax[row, col].imshow(np.reshape(eval_samples[sample_idx], (28, 28)), cmap='gray')
                ax[row, col].set_axis_off()
                ax[row + 1, col].set_title(f'predicted {predicted_classes[sample_idx]}, actual {actual_classes[sample_idx]}')
                bar_colors = ['C0'] * 10
                bar_colors[actual_classes[sample_idx]] = 'C1'
                ax[row + 1, col].bar(
                    np.arange(10), predicted_probabilities[sample_idx], tick_label=np.arange(10), color=bar_colors
                )
        fig.suptitle('Least confident predictions', size=20)
        fig.savefig(os.path.join(output_dir, 'mnist_least_confident.pdf'))

        print('Plotting ambiguous and rotated MNIST confidences')
        ambiguous_samples = torch.from_numpy(np.load(os.path.join(data_dir, 'test_x.npz'))['test_x']).reshape([-1, 784])[:10]
        ambiguous_dataset = torch.utils.data.TensorDataset(ambiguous_samples, torch.zeros(10))
        ambiguous_loader = torch.utils.data.DataLoader(
            ambiguous_dataset, batch_size=10, shuffle=False, drop_last=False
        )
        ambiguous_predicted_probabilities = model.predict(ambiguous_loader)
        ambiguous_predicted_classes = np.argmax(ambiguous_predicted_probabilities, axis=1)
        fig, ax = plt.subplots(4, 5, figsize=(13, 11))
        for row in range(0, 4, 2):
            for col in range(5):
                sample_idx = 5 * row // 2 + col
                ax[row, col].imshow(np.reshape(ambiguous_samples[sample_idx], (28, 28)), cmap='gray')
                ax[row, col].set_axis_off()
                ax[row + 1, col].set_title(f'predicted {ambiguous_predicted_classes[sample_idx]}')
                ax[row + 1, col].bar(
                    np.arange(10), ambiguous_predicted_probabilities[sample_idx], tick_label=np.arange(10)
                )
        fig.suptitle('Predictions on ambiguous and rotated MNIST', size=20)
        fig.savefig(os.path.join(output_dir, 'ambiguous_rotated_mnist.pdf'))


        # Do the same evaluation as on MNIST also on FashionMNIST
        print('Predicting on FashionMNIST data')
        fmnist_samples = torch.from_numpy(np.load(os.path.join(data_dir, 'fmnist.npz'))['x_test']).reshape([-1, 784])
        fmnist_dataset = torch.utils.data.TensorDataset(fmnist_samples, torch.zeros(fmnist_samples.shape[0]))
        fmnist_loader = torch.utils.data.DataLoader(
            fmnist_dataset, batch_size=64, shuffle=False, drop_last=False
        )
        fmnist_predicted_probabilities = model.predict(fmnist_loader)
        fmnist_predicted_classes = np.argmax(fmnist_predicted_probabilities, axis=1)
        fmnist_confidences = np.max(fmnist_predicted_probabilities, axis=1)
        fmnist_sorted_confidence_indices = np.argsort(fmnist_confidences)

        # Plot FashionMNIST samples your model is most confident about
        print('Plotting most confident FashionMNIST predictions')
        most_confident_indices = fmnist_sorted_confidence_indices[-10:]
        fig, ax = plt.subplots(4, 5, figsize=(13, 11))
        for row in range(0, 4, 2):
            for col in range(5):
                sample_idx = most_confident_indices[5 * row // 2 + col]
                ax[row, col].imshow(np.reshape(fmnist_samples[sample_idx], (28, 28)), cmap='gray')
                ax[row, col].set_axis_off()
                ax[row + 1, col].set_title(f'predicted {fmnist_predicted_classes[sample_idx]}')
                ax[row + 1, col].bar(
                    np.arange(10), fmnist_predicted_probabilities[sample_idx], tick_label=np.arange(10)
                )
        fig.suptitle('Most confident predictions', size=20)
        fig.savefig(os.path.join(output_dir, 'fashionmnist_most_confident.pdf'))

        # Plot FashionMNIST samples your model is least confident about
        print('Plotting least confident FashionMNIST predictions')
        least_confident_indices = fmnist_sorted_confidence_indices[:10]
        fig, ax = plt.subplots(4, 5, figsize=(13, 11))
        for row in range(0, 4, 2):
            for col in range(5):
                sample_idx = least_confident_indices[5 * row // 2 + col]
                ax[row, col].imshow(np.reshape(fmnist_samples[sample_idx], (28, 28)), cmap='gray')
                ax[row, col].set_axis_off()
                ax[row + 1, col].set_title(f'predicted {fmnist_predicted_classes[sample_idx]}')
                ax[row + 1, col].bar(
                    np.arange(10), fmnist_predicted_probabilities[sample_idx], tick_label=np.arange(10)
                )
        fig.suptitle('Least confident predictions', size=20)
        fig.savefig(os.path.join(output_dir, 'fashionmnist_least_confident.pdf'))

        print('Determining suitability of your model for OOD detection')
        all_confidences = np.concatenate([confidences, fmnist_confidences])
        dataset_labels = np.concatenate([np.ones_like(confidences), np.zeros_like(fmnist_confidences)])
        print(
            'AUROC for MNIST vs. FashionMNIST OOD detection based on confidence: '
            f'{roc_auc_score(dataset_labels, all_confidences):.3f}'
        )
        print(
            'AUPRC for MNIST vs. FashionMNIST OOD detection based on confidence: '
            f'{average_precision_score(dataset_labels, all_confidences):.3f}'
        )




def main():
    '''
    raise RuntimeError(
        'This main method is for illustrative purposes only and will NEVER be called by the checker!\n'
        'The checker always calls run_solution directly.\n'
        'Please implement your solution exclusively in the methods and classes mentioned in the task description.'
    )
    '''
    # Load training data
    data_dir = os.curdir
    output_dir = os.curdir
    raw_train_data = np.load(os.path.join(data_dir, 'train_data.npz'))
    x_train = torch.from_numpy(raw_train_data['train_x']).reshape([-1, 784])
    y_train = torch.from_numpy(raw_train_data['train_y']).long()
    dataset_train = torch.utils.data.TensorDataset(x_train, y_train)

    # Run actual solution
    run_solution(dataset_train, data_dir=data_dir, output_dir=output_dir)


if __name__ == "__main__":
    main()

