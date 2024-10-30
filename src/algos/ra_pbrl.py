from .experiments.algorithm import Algorithm
from .experiments.data_collector import Datum
from .experiments.environment import State, Action

import numpy as np

import torch
from torch import optim
from torch import nn

from typing import Any

import copy

class BranchedOptimization(nn.Module):
    """ Custom model used for min-max SGD"""
    def __init__(self, 
                 p: nn.Tensor, 
                 r: nn.Tensor,
                 pi: nn.Tensor) -> None:
        super().__init__()
        self.theta = nn.Linear(p.shape[1] + r.shape[1] + pi.shape[1], 1)
        # Copy tensor parameters
        with torch.no_grad():
            # theta -> [(S + A + S)*2 + (S + A)] x 1
            # Output is the risk-aware value function
            theta = torch.concat(p, r, pi)
            self.theta.weight.copy_(theta)

    def forward(self,
                x):
        """ Forward propagation """
        return self.theta(x)
        

class PbRL(Algorithm):

    def __init__(self,
                 # epochs: int,
                 # episodes: int,
                 # timesteps: int,
                 state_dim: int,
                 act_dim: int,
                 alpha: float,
                 lr_p: float=1e-2,
                 lr_r: float=1e-2,
                 lr_pi: float=1e-2):
        """"""
        ''' Learning algorithm hyperparameters '''

        self.alpha = alpha
        # Learning rates
        self.lr_p = lr_p
        self.lr_r = lr_r
        self.lr_pi = lr_pi

        # Basic state-action set dimensions
        self.state_dim = state_dim
        self.act_dim = act_dim

        ''' Reward, transition, and policy parameterizations '''
        # theta_t -> (S+A+S) x 1
        # T[S'| S, A]: prob. S' occurs given S and A. Dist over all states
        # We concatenate S, A, and S', since we may have ininfite state-actions
        # Basic shallow neural network with sigmoid activation for probability function
        self.p = nn.Sequential(
            nn.Linear(state_dim + act_dim + state_dim, 1, bias=False),
            # nn.Sigmoid(), # The remova of this allows outputs to be logits.
        )
        self.optim_p = optim.Adam(self.p.parameters(), lr=self.lr_p)

        # theta_r -> (S+A+S) x 1
        # r[S, A]: reward of taking action A @ state S
        # TODO: Gotta figure out how to do this!
        # NOTE: Possible reason for fail in sum: extremes and small can be "hidden!"
        # NOTE: The "reward function" is really a preference function taking as input both trajs' avg
        # NOTE: May need to make this a DNN... Nope because we assume linear function is used.
        self.r = nn.Sequential(
            nn.Linear((state_dim + act_dim + state_dim)*2, 1, bias=False),
            # No activation layer; only logits
        )#torch.rand([state_dim + act_dim, 1])
        self.optim_r = optim.Adam(self.r.parameters(), lr=self.lr_r)

        # theta_pi_[1 or 2] -> (S + A) x 1
        # theta_p1: Best policy
        self.theta_pi_1 = nn.Linear(state_dim + act_dim, 1)

        # theta_p2: Exploratory policy
        self.theta_pi_2 = nn.Linear(state_dim + act_dim, 1)

        ''' Additional variables used in learning '''
        # The idea is that the trajectory buffer will contain learnt
        # trajectories that will be stored in zeta_1&2 for learning.
        # Trajectories for best and exploratory policies.
        self.tau_1 = []
        self.tau_2 = []

        # Trajectory buffer for best policy
        self.zeta_1 = []
        # Trajectory buffer for exploratory policy
        self.zeta_2 = []

        # Trajectory preference
        self.reward = []

        self.k = 0

    def act(self,
            state: State,
            exploratory: bool=False) -> Action:
        """ Make decision for best or exploratory policies """
        
        # To search for an action, 1. find best action by "descent"
        if exploratory:
            # use the other policy
            action = state @ self.theta_pi_2
            # Save to trajectory
            self.tau_2.append([state, action])
        else:
            # Use policy 1
            action = state @ self.theta_pi_1
            # Save to trajectory
            self.tau_1.append([state, action])
        return action
    
    def P_bound(k):
        return np.sqrt(2 * self.state_dim * np.log( (2* 100 * 100 * self.state_dim * self.act_dim) / (k) ))

    def r_bound():
        return np.sqrt(2 * self.state_dim * np.log( (2* 100 * 100 * self.state_dim * self.act_dim) / (k) ))

    def save_trajectories(self,
                          reward: float) -> None:
        
        """ Store trajectories for learning """
        assert len(self.tau_1) == len(self.tau_2), "tau trajectories not same length!"
        # Store trajectories thus far for learning
        self.zeta_1.append(self.tau_1)
        self.zeta_2.append(self.tau_2)
        # Store trajectory preference
        self.reward.append(reward)
        # Reset trajectories for next episode
        self.tau_1 = []
        self.tau_2 = []

    def learn(self,
              reward: tuple[float, float],
              reset: bool=False) -> None:
        """ single-step Risk-Aware Preference-based RL algorithm """

        ''' Prepare data for learning '''
        # Data for transition optimization

        ''' Optimize transition & reward function '''
        for tau in (zeta := self.zeta_1 + self.zeta_2):
            for h in len(zeta) - 1:
                s, a = tau[h]
                s_, _ = tau[h + 1]
                # compute soln
                # Input tensor: (S+A+S) x 1
                
                ''' Stochastic Gradient Descent '''
                self.optim_p.zero_grad()
                out = self.p(torch.concat([s, a, s_]))
                loss_p = torch.nn.MSELoss(out, 1.) # Loss uses 1. since s_ was true transition
                loss_p.backward()
                self.optim_p.step()
        del zeta # save memory

        ''' Optimize reward (trajectory-wise) '''
        for tau_1, tau_2, reward in zip(self.zeta_1, self.zeta_2, self.reward):
            # compute the trajectory "displacement"
            t_1 = torch.sum(tau_1)
            t_2 = torch.sum(tau_2)

            self.optim_r.zero_grad()
            out = self.r(torch.concat([t_1, t_2]))
            # reward must be between 0. and 1.
            loss_r = torch.nn.MSELoss(out, reward)
            loss_r.backward()
            self.optim_r.step()

        ''' Perform min-max SGD for both policies '''
        BO_1 = BranchedOptimization(r=next(self.r.parameters()),
                                    p=next(self.p.parameters()),
                                    pi=self.theta_pi_1)
        optim_1 = nn.optim.Adam(BO_1.parameters(), lr=0.1)
        with torch.no_grad():
            BO_1_backup = next(BO_1.parameters())
        BO_2 = BranchedOptimization(r=next(self.r.parameters()),
                                    p=next(self.p.parameters()),
                                    pi=self.theta_pi_2)
        optim_2 = nn.optim.Adam(BO_2.parameters(), lr=0.1, maximize=True)
        with torch.no_grad():
            BO_2_backup = next(BO_2.parameters())
        
        self.reward.sort()
        cvar_thr = int(self.alpha * len(self.reward))
        V = self.reward[:cvar_thr] / cvar_thr
        
        optim_1.zero_grad()
        optim_2.zero_grad()
        out_1 = BO_1(torch.concat([torch.sum(self.zeta_1), self.zeta_1, self.zeta_1]) )
        out_2 = BO_2(torch.concat([torch.sum(self.zeta_2), self.zeta_2, self.zeta_2]) )
        loss_1 = nn.MSELoss(out_1, V)
        loss_2 = nn.MSELoss(out_2, V)
        optim_1.step()
        optim_2.step()

        with torch.no_grad():
            if torch.cdist(BO_1_backup, next(BO_1.parameters()[0])) < self.P_bound() or torch.cdist(BO_1_backup, next(BO_1.parameters()[1])) < self.r_bound():
                # Roll back weights
                BO_1.weight.copy_(BO_1_backup)
            if torch.cdist(BO_2_backup, next(BO_2.parameters()))[0] < self.P_bound() or torch.cdist(BO_2_backup, next(BO_2.parameters()))[1] < self.r_bound():
                # Roll back weights
                BO_2.weight.copy_(BO_2_backup)

        if reset:
            # Reset saved trajectories & timesteps
            self.reset()

    def reset(self) -> None:
        self.k += 1
        self.zeta_1 = []
        self.zeta_2 = []

    # Ignore for now the function bellow!
    def inspect(self) -> dict | Datum | Any:

        return {
            'zeta_1': self.zeta_1,
            'zeta_2': self.zeta_2,
        }
