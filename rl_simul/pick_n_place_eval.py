import time

import gymnasium as gym
import panda_gym  # noqa: F401
from sb3_contrib import TQC

env_name = "PandaPickAndPlace-v3"
env = gym.make(env_name, render_mode="human")
observation, info = env.reset()

# load a model
model = TQC.load("data/PandaPickAndPlace-v3", env)

for _ in range(1000):
    action, _ = model.predict(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()
    time.sleep(0.1)

env.close()
