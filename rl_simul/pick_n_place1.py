# %%
import gymnasium as gym
import panda_gym


print(panda_gym.__version__)
env_id = "PandaPickAndPlace-v3"


env = gym.make(env_id, render_mode="human", renderer="OpenGL")
env.reset()

# %%
dir(env)
# %%
action = env.action_space.sample()
print(action)
env.step(action)
# %%
env.action_space
# %%
