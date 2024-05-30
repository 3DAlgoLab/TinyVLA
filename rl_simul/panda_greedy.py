# %%
import gymnasium as gym
import panda_gym
from stable_baselines3 import DDPG

env = gym.make("PandaReach-v3")
model = DDPG(policy="MultiInputPolicy", env=env)
model.learn(30_000)

# %%
vec_env = model.get_env()
obs = vec_env.reset()

# %%
for i in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    print(rewards)
    vec_env.render()
    if dones:
        obs = vec_env.reset()
# %%
rewards
# %%
