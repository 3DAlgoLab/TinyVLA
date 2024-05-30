import time

import gymnasium as gym
import panda_gym
from stable_baselines3 import PPO

print("panda-gym version:", panda_gym.__version__)

env_name = "PandaReachDense-v3"
env = gym.make(env_name, render_mode="human")
observation, info = env.reset()

# load a model
model = PPO.load("a2c_panda_reach", env)

for _ in range(1000):
    # current_position = observation["observation"][0:3]
    # desired_position = observation["desired_goal"][0:3]
    # action = 5.0 * (desired_position - current_position)
    action, _ = model.predict(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()
    time.sleep(0.1)

env.close()
