'''

@author Jose E. Aguilar Escamilla
@date June 12th, 2024
@email aguijose(at)oregonstate.edu

'''
import numpy as np

from src.paper_experiments.experiments.algorithms.base import RLAlgorithm, LinearPolicy

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import sqrtm
import torch

from typing import Callable

State = Action = Vector = ArrayLike | torch.Tensor
# State, Action, Reward, State'
SARS = list[Vector, Vector, Vector, Vector]



class LSVIUCB(RLAlgorithm):
    def __init__(self,
                 horizon: int,
                 phi: Callable[[State, Action], Vector],
                 beta: float,
                 lambda_: float,
                 n_params: int = None,
                 bias: bool = False,):
        ''' Least Squares Value Iteration /w Upper Confidence Bound

        The LSVI-UCB algorithm [Jin et al., 2020] is a version of value
        iteration (see RL resources such as [Sutton & Barto, 2016]) that
        utilizes Least Squares to approximate a Q-function parameterized
        by a linear function. For more details in the algorithm see the
        cited paper. NOTE: We assume that the episode loop will be
        handled by outside code containing an instance of this class.
        The algorithm is implemented in the learn function.

        Pseudo-code (Algorithm 1 in paper)
        ==================================
        FOR episode k=1,...,K DO
            Receive the initial state x^k_1.

            FOR step h=H,...,1 DO
                Λ_h ← Σ^{k-1}_{τ=1}
                    ↪ φ(x^τ_h, a^τ_h)·φ(x^τ_h, a^τ_h)^T + λ·I

                w_h ← Λ^{-1}_h Σ^{k-1}_{τ=1} φ(x^τ_h, a^τ_h) *
                    ↪ [r_h(x^τ_h, a^τ_h) + max_a Q_{h+1}(x^τ_{h+1}, a)].

                Q_h(·,·) ← min{w_h^T * φ(·,·) + β[φ(·,·)^T * Λ^{-1}_h]^{1/2}, H}

            FOR step h=1,...,H DO
                Take action a_h^k ← argmax_{a ∈ A} Q_h(x^k_h, a), and
                    ↪ observe x^k_{h+2}

        parameters
        ==========
        horizon (H): int
        phi (φ): Callable[[State, Action], Vector]
        beta (β): float
        lambda_ (λ): float
        '''

        self.horizon = horizon
        self.phi = phi
        self.beta = beta
        self.lambda_ = lambda_
        self.n_params = n_params
        self.bias = bias

        self._policy = LinearPolicy(None, n_params, bias)
        # None can be a signal that buffer is done/empty

        # Latest timestep added
        self.t: int = -2 # buffer setter will increase by +1: -1 = empty
        self.buffer = [None, None, None, None]

    @property
    def policy(self):
        pass
    @policy.setter
    def set_policy(self,
                   policy: Vector):
        self._policy = policy
    @policy.getter
    def get_policy(self):
        return self._policy

    @property
    def buffer(self) -> SARS:
        return self.get_buffer()
    @buffer.setter
    def set_buffer(self,
                   transition: SARS):
        _s, _a, _r, _s_ = transition
        self._s.appenn(_s)
        self._a.appenn(_a)
        self._r.appenn(_r)
        self._s_.appenn(_s_)
        self.t += 1

    @buffer.getter
    def get_buffer(self) -> SARS:
        try:
            return self._s.pop(), self._a.pop(), self._r.pop(), self._s_.pop()
        except IndexError:
            raise IndexError('Buffer is empty.')

    def Q_(self,
           state: Vector,
           action: Vector) -> float:
        x = self.phi(state, action)
        return self.policy(x)

    def decide(self,
               state: Vector):
        '''Pseudocode
        for all actions in state:
            find greatest value.
        '''

    def _update(self, state, action, reward, next_state):
        self.t += 1
        phi = np.outer(state, action)
        self.B += phi @ phi.T
        self.f += reward * phi
        self.mu = sqrtm(self.B) @ np.random.normal(size=(self.state_dim, self.action_dim))
        self.theta = np.linalg.inv(self.B) @ (self.f + self.mu)

    def learn(self,):
        
        def _max(a: list[float]):
            ''' Implement max exactly using linear assumption '''
            # optimal value found via limit
            # Idea 0: Find gradient and approximate via binary search

        H = self.t + 1
        for h in range(H, 0, -1):
            s, a, r, s_ = self.buffer
            # Eq. 4:
            Lambda_ =(self.phi(s, a) @ self.phi(s, a).T).sum()
            I = self.lambda_ * np.identity()
            Lambda = Lambda_ + I

            # Eq. 5:
            Bellman = r +
            W = np.linalg.inv(Lambda) # Guaranteed to exist

            # Eq. 6:

    def __str__(self):
        return 'LSVI-UCB'