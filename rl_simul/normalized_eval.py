# %%
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import A2C
import gymnasium as gym
import panda_gym
from IPython.display import Image
from moviepy.editor import ImageSequenceClip

# %%
# Load the saved statistics
eval_env = DummyVecEnv([lambda: gym.make("PandaReachDense-v3")])
eval_env = VecNormalize.load("data/panda-reach-env.pkl", eval_env)

# We need to override the render_mode
eval_env.render_mode = "rgb_array"
eval_env.training = False
eval_env.norm_reward = False

# Load the agent
model = A2C.load("data/a2c-panda-reach")

# %%
images = []
obs = eval_env.reset()
for i in range(100):
    action, _ = model.predict(obs)
    obs, reward, done, info = eval_env.step(action)
    if i % 20 == 0:
        print(f"eval-step {i+1}:{reward}")
    if done:
        obs = eval_env.reset()
    images.append(eval_env.render(mode="rgb_array"))

# %%
fps = 40
clip = ImageSequenceClip(images, fps=fps)
clip.write_gif("data/a2c_evaluated.gif", fps=fps)
# %%
Image(filename="data/a2c_evaluated.gif")

# %%
mean_reward, std_reward = evaluate_policy(model, eval_env)
print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")
del eval_env
