from pettingzoo.classic import texas_holdem_no_limit_v6
from time import sleep
import numpy as np


env = texas_holdem_no_limit_v6.env(
    num_players=6, render_mode="rgb_array", screen_height=500
)
env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        # mask = servation["action_mask"]
        mask = np.array([0, 1, 0, 0, 0], dtype=np.int8)
        print(mask)
        print(observation["observation"][0:13])
        print(observation["observation"][13 : 13 * 2])
        print(observation["observation"][13 * 2 : 13 * 3])
        print(observation["observation"][13 * 3 : 13 * 4])
        print(observation["observation"][13 * 4 :])
        # this is where you would insert your policy
        action = env.action_space(agent).sample(mask)
    env.step(action)
env.close()
