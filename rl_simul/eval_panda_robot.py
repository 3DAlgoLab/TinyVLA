# %%
import gymnasium as gym
import panda_gym
from IPython.display import Image
from moviepy.editor import ImageSequenceClip
from stable_baselines3 import A2C

# %%

env_id = "PandaReachDense-v3"
model = A2C.load("a2c_panda_reach")
env = gym.make(env_id, render_mode="rgb_array")
obs, info = env.reset()
images = [env.render()]

for i in range(100):
    action, state = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    images.append(env.render())
    if i % 5 == 0:
        print(reward)

    if done or truncated:
        observation, info = env.reset()
        images.append(env.render())


env.close()
# %%
fps = 40
clip = ImageSequenceClip(images, fps=fps)
clip.write_gif("evaluated.gif", fps=fps)
Image(filename="evaluated.gif")

# %%
