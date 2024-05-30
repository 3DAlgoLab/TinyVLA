import gymnasium as gym
import panda_gym  # noqa: F401
from stable_baselines3 import PPO

env = gym.make("PandaReachDense-v3")
model = PPO(policy="MultiInputPolicy", env=env, verbose=1)

model.learn(total_timesteps=50000, progress_bar=True)
model.save("ppo_panda_reach_dense")
