from src.paper_experiments.experiments.environment import CautiousDeciderEnvironment

import torch
def test_init():
    difficulty = tau = 10
    complexity = kappa = 2
    def transition(state, action):
        t = state.abs().sum()
        state[t] = action
        return state

    transition_map = state_action_map = transition
    reward_param = torch.ones([tau, 1])
    env = CautiousDeciderEnvironment(
        reward_param,
        state_action_map,
        transition_map,
        difficulty=difficulty,
        complexity=complexity)

    # Assertions
    assert (env.theta == reward_param).all()
    assert env.phi == state_action_map
    assert env.T == transition_map
    assert env.tau == tau
    assert env.kappa == kappa
    assert env.t == None


def test_reset():
    difficulty = tau = 10
    complexity = kappa = 2

    def transition(state, action):
        t = state.abs().sum()
        state[t] = action
        return state

    transition_map = state_action_map = transition
    reward_param = torch.ones([tau, 1])
    env = CautiousDeciderEnvironment(
        reward_param,
        state_action_map,
        transition_map,
        difficulty=difficulty,
        complexity=complexity)

    state = env.reset()

    # Assertions
    assert env.t == 0
    assert (state == torch.zeros_like(state)).all()

def test_take_action():
    difficulty = tau = 10
    complexity = kappa = 2

    def transition(state, action, phi=False):
        t = int(state.abs().sum())
        state[t] = action
        return state

    transition_map = state_action_map = transition
    reward_param = torch.ones([tau, 1])
    env = CautiousDeciderEnvironment(
        reward_param,
        state_action_map,
        transition_map,
        difficulty=difficulty,
        complexity=complexity)

    state = env.reset().clone().detach()

    state_, info = env.take_action(action = -1)
    # Assertions
    assert (state != state_).any()
    assert state_[0] == -1
    assert state_[1] == 0
    assert info['reward'] == -1
    assert info['done'] == False

def test_done():
    difficulty = tau = 10
    complexity = kappa = 2

    def transition(state, action):
        t = int(state.abs().sum())
        state[t] = action
        return state

    transition_map = state_action_map = transition
    reward_param = torch.ones([tau, 1])
    env = CautiousDeciderEnvironment(
        reward_param,
        state_action_map,
        transition_map,
        difficulty=difficulty,
        complexity=complexity)
    state = env.reset().clone().detach()

    env.t = tau - 2

    env.state = torch.ones_like(env.theta)
    env.state[env.t + 1] = 0 # Last step untaken

    state_, info = env.take_action(action=-1)
    # Asserts
    assert info['done']
    # We do tau - 2 because 1 action is wrong.
    assert info['reward'] == (tau - 2)

# def test_overuse():