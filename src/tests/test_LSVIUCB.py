from ..paper_experiments.experiments.algorithms.base import LinearPolicy
from src.paper_experiments.algos import LSVIUCB
import numpy as np

def test_LinearPolicy():
    params = np.random.random([10])
    x = np.random.random([10])
    bias = 1.2

    p = LinearPolicy(params)
    p_1 = LinearPolicy(n_params=9, bias=bias)

    assert x @ p.params == p(x)
    assert p_1.bias == bias

def test_LSVIUCB():
    horizon = 10
    phi = lambda s, a: s
    beta = 0.1
    lambda_ = 0.1
    n_params = 9
    bias = True

    algo = LSVIUCB(horizon, phi, beta, lambda_, n_params, bias)

    assert True