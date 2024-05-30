# %%
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3.common.env_util import make_vec_env


class ExampleMultiEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.mu = 0.8

    def step(self, action):
        pass

    def set_mu(self, new_mu):
        self.mu = new_mu


# %%

env = gym.make("CartPole-v1")
env.unwrapped.gravity

# %%
vec_env = make_vec_env(ExampleMultiEnv)
print(vec_env.env_method("get_wrapper_attr", "mu"))
# %%
