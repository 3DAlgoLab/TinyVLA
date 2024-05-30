# WARNING, This file will not be functional until stable-baselines3 is compatible
# with gymnasium. See https://github.com/DLR-RM/stable-baselines3/pull/780 for more information.
import gymnasium as gym
import panda_gym
from stable_baselines3 import DDPG, HerReplayBuffer

env = gym.make("PandaPush-v3")

model = DDPG(
    policy="MultiInputPolicy", env=env, replay_buffer_class=HerReplayBuffer, verbose=1
)

model.learn(total_timesteps=100000, progress_bar=True)
model.save("ddpg_panda_push")
