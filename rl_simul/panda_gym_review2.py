# %%
import gymnasium as gym
import panda_gym  # noqa: F401
from PIL import Image


env = gym.make("PandaReach-v3", render_mode="rgb_array", renderer="OpenGL")
env.reset()

image = env.render()  # RGB rendering of shape (480, 720, 3)
env.close()
Image.fromarray(image)

# %%
env = gym.make(
    "PandaSlide-v3",
    render_mode="rgb_array",
    renderer="OpenGL",
    render_width=480,
    render_height=480,
    render_target_position=[0.2, 0, 0],
    render_distance=1.0,
    render_yaw=90,
    render_pitch=-70,
    render_roll=0,
)
env.reset()
image = env.render()  # RGB rendering of shape (480, 480, 3)
env.close()
Image.fromarray(image)

# %%
