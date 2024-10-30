import gymnasium
from algos.ra_pbrl import PbRL

# Both environments used in preference-based experiments
# To render, add render_mode="human"
env_1 = gymnasium.make("HalfCheetah-v4")
env_2 = gymnasium.make("HalfCheetah-v4")

# Algorithm setup
assert env_1.observation_space.shape[0] == env_2.observation_space.shape[0]
assert env_1.action_space.shape[0] == env_2.action_space.shape[0]

algo = PbRL(env_1.observation_space.shape[0],
            env_1.action_space.shape[0])

# reset environments; ignore info
observation_1, info_1 = env_1.reset()
observation_2, info_2 = env_2.reset()

# Data collection
reward = [0., 0.]

''' Experiment Parameters '''
K = 10
H = 1_000

for k in range(K):
    for h in range(H):
        # Get action decision
        action_1 = algo.act(observation_1, exploratory=True)
        action_2 = algo.act(observation_2, exploratory=False)

        # action = env.action_space.sample()  # agent policy that uses the observation and info
        observation_1, reward_1, terminated_1, truncated_1, info_1 = env_1.step(action_1)
        observation_2, reward_2, terminated_2, truncated_2, info_2 = env_2.step(action_2)

        # Data collection
        reward[0] += reward_1
        reward[1] += reward_2

        # Note: It is assumed that when either env ends, we stop even if other is unfinished
        if terminated_1 or terminated_2 or truncated_1 or truncated_2:
            observation_1, info_1 = env_1.reset()
            observation_2, info_2 = env_2.reset()
            break

    # Apply learning algorithm
    algo.learn(reward)

env_1.close()
env_2.close()

print(reward)
