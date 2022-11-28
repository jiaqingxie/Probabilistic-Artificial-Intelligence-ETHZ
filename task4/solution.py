import random

import numpy as np
import matplotlib.pyplot as plt

import time

import scipy.signal
from gym.spaces import Box, Discrete

import torch
from torch.optim import Adam
import torch.nn as nn


def discount_cumsum(x, discount):
    """
    Compute  cumulative sums of vectors.

    Input: [x0, x1, ..., xn]
    Output: [x0 + discount * x1 + discount^2 * x2, x1 + discount * x2, ..., xn]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def combined_shape(length, shape=None):
    """Helper function that combines two array shapes."""
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    """
    The basic multilayer perceptron architecture used.

    Parameters
    ----------
    sizes: List
        List of feature sizes, i.e., 
            [indput_dim, hidden_layer_1, ..., hidden_layer_n_dim, output_dim] 

    activation: nn.Module
        Activation function for the hidden layers.

    output_activation: nn.Module
        Activation function for the output layer


    Returns
    -------
    mlp: nn.Module

    """

    # TODO: Implement this function.
    # Hint: Use nn.Sequential to stack multiple layers of the network.

    raise NotImplementedError


class Actor(nn.Module):
    """A class for the policy network."""

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)



    def _distribution(self, obs):
        """
        Takes the observation and outputs a distribution over actions.
        
        Parameters
        ----------
        obs: torch.Tensor of shape (n, obs_dim)
            State observation.

        Returns
        -------
        pi: torch.distributions.Distribution
            n action distributions for each state/obs.

        """

        # TODO: Implement this function.
        # Hint: The logits_net returns for a given observation the log 
        # probabilities. You should use them to obtain a Categorical 
        # distribution.
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        """
        Take a distribution and action, then gives the log-probability of the action
        under that distribution.

        Parameters
        ----------
        pi: torch.distributions.Distribution
            n action distributions.

        act: torch.Tensor of shape (n, act_dim)
            n action for which log likelihood is calculated.

        Returns
        -------
        log_prob: torch.Tensor of shape (n, )
            log likelihood of act.

        """

        # TODO: Implement this function.

        raise NotImplementedError

    def forward(self, obs, act=None):
        """
        Produce action distributions for given observations, and then compute the
        log-likelihood of given actions under those distributions.
        
        Parameters
        ----------
        obs: torch.Tensor of shape (n, obs_dim)
            State observation.
        act: (torch.Tensor of shape (n, act_dim), Optional). Defaults to None.
            Action for which log likelihood is calculated.

        Returns
        -------
        pi: torch.distributions.Distribution
            n action distributions.
        log_prob: torch.Tensor of shape (n, ) 
            log likelihood of act.
        """

        # TODO: Implement this function.
        # Hint: If act is None, log_prob is also None.

        raise NotImplementedError


class Critic(nn.Module):
    """The network used by the value function."""
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)
        

    def forward(self, obs):
        """
        Return the value estimate for a given observation.

        Parameters
        ----------
            obs: torch.Tensor of shape (n, obs_dim)
                State observation.

        Returns
        -------
            v: torch.Tensor of shape (n, ), i.e., where n is the number of observations.
                Value estimate for obs.
        """
        return torch.squeeze(self.v_net(obs), -1)


class VPGBuffer:
    """
    Buffer to store trajectories.
    """
    def __init__(self, obs_dim, act_dim, size, gamma, lam):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        # calculated TD residuals
        self.tdres_buf = np.zeros(size, dtype=np.float32)
        # rewards
        self.rew_buf = np.zeros(size, dtype=np.float32)
        # trajectory's remaining return
        self.ret_buf = np.zeros(size, dtype=np.float32)
        # values predicted
        self.val_buf = np.zeros(size, dtype=np.float32)
        # log probabilities of chosen actions under behavior policy
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma = gamma
        self.lam = lam
        # Pointer to the latest data point in the buffer.
        self.ptr = 0 
        # Pointer to the start of the trajectory.
        self.path_start_idx = 0
        # Maximum size of the buffer. 
        self.max_size = size

    def store(self, obs, act, rew, val, logp):
        """
        Append a single timestep to the buffer. This is called at each environment
        update to store the observed outcome in
            self.obs_buf, 
            self.act_buf,
            self.rew_buf,
            self.val_buff,
            self.logp_buff.
        
        Parameters
        ----------
        obs: torch.Tensor of shape (obs_dim, )
            State observation.

        act: torch.Tensor of shape (act_dim, )
            Applied action.

        rew: torch.Tensor of shape (1, )
            Observed rewards.

        val: torch.Tensor of shape (1, )
            Predicted values.

        logp: torch.Tensor of shape (1, )
            log probability of act under behavior policy       
        """

        # buffer has to have room so you can store
        assert self.ptr < self.max_size

        # TODO: Store new data in the respective buffers.


        # Update pointer after data is stored.
        self.ptr += 1

    def end_traj(self, last_val=0):
        """
        Calculate for a trajectory 
            1) discounted rewards-to-go, and  
            2) TD residuals.
        Store these into self.ret_buf, and self.tdres_buf respectively.

        The function is called after a trajectory ends.

        Parameters
        ----------
        last_val: np.float32
            Last value is value(state) if the rollout is cut-off at a
            certain state, or 0 if trajectory ended uninterrupted.
        """

        # Get the indexes where TD residuals and discounted 
        # rewards-to-go are stored.
        path_slice = slice(self.path_start_idx, self.ptr)
        
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        self.ret_buf[self.ptr:self.path_start_idx] = (
            np.cumsum(self.rew_buf[self.ptr:self.path_start_idx][::-1])[::-1]
        )

        # TODO: Implement TD residuals calculation.
        # Hint: use the discount_cumsum function 
        # self.tdres_buf[path_slice] = ...


        # TODO: Implement discounted rewards-to-go calculation. 
        # Hint: use the discount_cumsum function 
        # self.ret_buf[path_slice] = ...


        # Update the path_start_idx
        self.path_start_idx = self.ptr

        pass


    def get(self):
        """
        Call after an epoch ends. Resets pointers and returns the buffer contents.
        """
        # Buffer has to be full before you can get something from it.
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0

        self.tdres_buf = self.tdres_buf
        tdres_mean = np.mean(self.tdres_buf)
        tdres_std = np.std(self.tdres_buf)
        self.tdres_buf = (self.tdres_buf - tdres_mean) / tdres_std

        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    tdres=self.tdres_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}


class Agent:
    def __init__(self, env, activation=nn.Tanh):
        self.env = env
        self.hid = 64  # layer width of networks
        self.l = 2  # layer number of networks
        hidden_sizes = [self.hid]*self.l
        obs_dim = 8
        self.actor =  Actor(obs_dim, 4, hidden_sizes, activation)
        self.critic  = Critic(obs_dim, hidden_sizes, activation)

    def step(self, state):
        """
        Take an state and return action, value function, and log-likelihood
        of chosen action.

        Parameters
        ----------
        state: torch.Tensor of shape (obs_dim, )

        Returns
        -------
        act: np.ndarray of (act_dim, )
            An action sampled from the policy given a state (0, 1, 2 or 3).
        v: np.ndarray of (1, )
            The value function at the given state.
        logp: np.ndarray of (1, )
            The log-probability of the action under the policy output distribution.
        """
        
        # TODO: Implement this function.
        # Hint: This function is only called during inference. You should use
        # `torch.no_grad` to ensure that it does not interfer with the gradient computation.

        return 0, 0, 0

    def act(self, state):
        return self.step(state)[0]

    def get_action(self, obs):
        """
        Sample an action from your policy/actor.

        Parameters
        ----------
        obs: np.ndarray of shape (obs_dim, )
            State observation.

        Returns
        -------
        act: np.ndarray of shape (act_dim, )
            Action to apply.

        IMPORTANT: This function called by the checker to evaluate your agent.
        You SHOULD NOT change the arguments this function takes and what it outputs!
        """

        # TODO: Implement this function.
        # Currently, this just returns a random action.
        
        return np.random.choice([0, 1, 2, 3])


def train(env, seed=0):
    """
    Main training loop.

    IMPORTANT: This function is called by the checker to train your agent.
    You SHOULD NOT change the arguments this function takes and what it outputs!
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # TODO: In this function, you implement the actor and critic updates.

    # The observations are 8 dimensional vectors, and the actions are numbers,
    # i.e. 0-dimensional vectors (hence act_dim is an empty list).
    obs_dim = [8]
    act_dim = []

    # initialize agent
    agent = Agent(env)

    # Training parameters
    # You may wish to change the following settings for the buffer and training
    # Number of training steps per epoch
    steps_per_epoch = 3000
    # Number of epochs to train for
    epochs = 50
    # The longest an episode can go on before cutting it off
    max_ep_len = 300
    # Discount factor for weighting future rewards
    gamma = 0.99
    lam = 0.97

    # Learning rates for actor and critic function
    actor_lr = 3e-3
    critic_lr = 1e-3

    # Set up buffer
    buf = VPGBuffer(obs_dim, act_dim, steps_per_epoch, gamma, lam)

    # Initialize the ADAM optimizer using the parameters
    # of the actor and then critic networks
    # TODO: Use these optimizers later to update the actor and critic networks.
    actor_optimizer = Adam(agent.actor.parameters(), lr=actor_lr)
    critic_optimizer = Adam(agent.critic.parameters(), lr=critic_lr)

    # Initialize the environment
    state, ep_ret, ep_len = agent.env.reset(), 0, 0

    # Main training loop: collect experience in env and update / log each epoch
    for epoch in range(epochs):
        ep_returns = []
        for t in range(steps_per_epoch):
            a, v, logp = agent.step(torch.as_tensor(state, dtype=torch.float32))

            next_state, r, terminal = agent.env.transition(a)
            ep_ret += r
            ep_len += 1

            # Log transition
            buf.store(state, a, r, v, logp)

            # Update state (critical!)
            state = next_state

            timeout = ep_len == max_ep_len
            epoch_ended = (t == steps_per_epoch - 1)

            if terminal or timeout or epoch_ended:
                # if trajectory didn't reach terminal state, bootstrap value target
                if epoch_ended:
                    _, v, _ = agent.step(torch.as_tensor(state, dtype=torch.float32))
                else:
                    v = 0
                if timeout or terminal:
                    ep_returns.append(ep_ret)  # only store return when episode ended
                buf.end_traj(v)
                state, ep_ret, ep_len = agent.env.reset(), 0, 0

        mean_return = np.mean(ep_returns) if len(ep_returns) > 0 else np.nan
        print(f"Epoch: {epoch+1}/{epochs}, mean return {mean_return}")

        # This is the end of an epoch, so here is where you likely want to update
        # the actor and / or critic function.


        # TODO: Implement the policy and value function updates. Hint: some of the torch code is
        # done for you.

        data = buf.get()

        # Do 1 policy gradient update
        actor_optimizer.zero_grad() #reset the gradient in the actor optimizer

        #Hint: you need to compute a 'loss' such that its derivative with respect to the actor
        # parameters is the policy gradient. Then call loss.backwards() and actor_optimizer.step()

        # We suggest to do 100 iterations of value function updates
        for _ in range(100):
            critic_optimizer.zero_grad()
            #compute a loss for the value function, call loss.backwards() and then
            #critic_optimizer.step()


    return agent


def main():
    """
    Train and evaluate agent.

    This function basically does the same as the checker that evaluates your agent.
    You can use it for debugging your agent and visualizing what it does.
    This function is only meant for testing purposes. Any changes made here do not
    affect the submission. 
    """
    from lunar_lander import LunarLander
    from gym.wrappers.monitoring.video_recorder import VideoRecorder

    env = LunarLander()
    env.seed(0)
    
    agent = train(env)

    rec = VideoRecorder(env, "policy.mp4")
    episode_length = 300
    n_eval = 100
    returns = []
    print("Evaluating agent...")

    for i in range(n_eval):
        print(f"Testing policy: episode {i+1}/{n_eval}")
        state = env.reset()
        cumulative_return = 0
        # The environment will set terminal to True if an episode is done.
        terminal = False
        env.reset()
        for t in range(episode_length):
            if i <= 10:
                rec.capture_frame()
            # Taking an action in the environment
            action = agent.get_action(state)
            state, reward, terminal = env.transition(action)
            cumulative_return += reward
            if terminal:
                break
        returns.append(cumulative_return)
        print(f"Achieved {cumulative_return:.2f} return.")
        if i == 10:
            rec.close()
            print("Saved video of 10 episodes to 'policy.mp4'.")
    env.close()
    print(f"Average return: {np.mean(returns):.2f}")

if __name__ == "__main__":
    main()
