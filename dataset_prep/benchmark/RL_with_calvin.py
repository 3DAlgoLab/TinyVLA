# %% [markdown]
# <a href="https://colab.research.google.com/github/mees/calvin/blob/main/RL_with_CALVIN.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# <h1>Reinforcement Learning with CALVIN</h1>
#
# The **CALVIN** simulated benchmark is perfectly suited for training agents with reinforcement learning, in this notebook we will demonstrate how to integrate your agents to these environments.

# %% [markdown]
# ## Installation
# The first step is to install the CALVIN github repository such that we have access to the packages

# %%
# Download repo
# %mkdir /content/calvin
# %cd /content/calvin
# !git clone https://github.com/mees/calvin_env.git
# %cd /content/calvin/calvin_env
# !git clone https://github.com/lukashermann/tacto.git
# # Install packages
# %cd /content/calvin/calvin_env/tacto/
# !pip3 install -e .
# %cd /content/calvin/calvin_env
# !pip3 install -e .
# !pip3 install -U numpy

# %%
# Run this to check if the installation was succesful
from calvin_env.envs.play_table_env import PlayTableSimEnv

# %% [markdown]
# ## Loading the environment
# After the installation has finished successfully, we can start using the environment for reinforcement Learning.
# To be able to use the environment we need to have the appropriate configuration that define the desired features, for this example, we will load the static and gripper camera.

# %%
# %cd /content/calvin
from hydra import initialize, compose

with initialize(config_path="../../benchmarks/calvin/calvin_env/conf/"):
    cfg = compose(
        config_name="config_data_collection.yaml",
        overrides=["cameras=static_and_gripper"],
    )
    cfg.env["use_egl"] = False
    cfg.env["show_gui"] = False
    cfg.env["use_vr"] = False
    cfg.env["use_scene_info"] = True
    print(cfg.env)

# %% [markdown]
# The environment has similar structure to traditional OpenAI Gym environments.
#
# *   We can restart the simulation with the *reset* function.
# *   We can perform an action in the environment with the *step* function.
# *   We can visualize images taken from the cameras in the environment by using the *render* function.
#
#
#
#

# %%

import cv2
from IPython import display
import PIL.Image


def cv2_imshow(a):
    """A replacement for cv2.imshow() for use in Jupyter notebooks.

    Args:
      a: np.ndarray. shape (N, M) or (N, M, 1) is an NxM grayscale image. For
        example, a shape of (N, M, 3) is an NxM BGR color image, and a shape of
        (N, M, 4) is an NxM BGRA color image.
    """
    a = a.clip(0, 255).astype("uint8")
    # cv2 stores colors as BGR; convert to RGB
    if a.ndim == 3:
        if a.shape[2] == 4:
            a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
        else:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    display.display(PIL.Image.fromarray(a))


# %%
import time
import hydra
import numpy as np

# from google.colab.patches import cv2_imshow

env = hydra.utils.instantiate(cfg.env)
observation = env.reset()
# The observation is given as a dictionary with different values
print(observation.keys())
for i in range(5):
    # The action consists in a pose displacement (position and orientation)
    action_displacement = np.random.uniform(low=-1, high=1, size=6)
    # And a binary gripper action, -1 for closing and 1 for opening
    action_gripper = np.random.choice([-1, 1], size=1)
    action = np.concatenate((action_displacement, action_gripper), axis=-1)
    observation, reward, done, info = env.step(action)
    rgb = env.render(mode="rgb_array")[:, :, ::-1]
    cv2_imshow(rgb)

# %% [markdown]
# ##  Custom environment for Reinforcement Learning
# There are some aspects that needs to be defined to be able to use it for reinforcement learning, including:
#
# 1.   Observation space
# 2.   Action space
# 3.   Reward function
#
# We are going to create a Custom environment that extends the **PlaytableSimEnv** to add these requirements. <br/>
# The specific task that will be solved is called "move_slider_left", here you can find a [list of possible tasks](https://github.com/mees/calvin_env/blob/main/conf/tasks/new_playtable_tasks.yaml) that can be evaluated using CALVIN.
#
#

# %%
from gym import spaces
from calvin_env.envs.play_table_env import PlayTableSimEnv


class SlideEnv(PlayTableSimEnv):
    def __init__(self, tasks: dict = {}, **kwargs):
        super(SlideEnv, self).__init__(**kwargs)
        # For this example we will modify the observation to
        # only retrieve the end effector pose
        self.action_space = spaces.Box(low=-1, high=1, shape=(7,))
        self.observation_space = spaces.Box(low=-1, high=1, shape=(7,))
        # We can use the task utility to know if the task was executed correctly
        self.tasks = hydra.utils.instantiate(tasks)

    def reset(self):
        obs = super().reset()
        self.start_info = self.get_info()
        return obs

    def get_obs(self):
        """Overwrite robot obs to only retrieve end effector position"""
        robot_obs, robot_info = self.robot.get_observation()
        return robot_obs[:7]

    def _success(self):
        """Returns a boolean indicating if the task was performed correctly"""
        current_info = self.get_info()
        task_filter = ["move_slider_left"]
        task_info = self.tasks.get_task_info_for_set(
            self.start_info, current_info, task_filter
        )
        return "move_slider_left" in task_info

    def _reward(self):
        """Returns the reward function that will be used
        for the RL algorithm"""
        reward = int(self._success()) * 10
        r_info = {"reward": reward}
        return reward, r_info

    def _termination(self):
        """Indicates if the robot has reached a terminal state"""
        success = self._success()
        done = success
        d_info = {"success": success}
        return done, d_info

    def step(self, action):
        """Performing a relative action in the environment
        input:
            action: 7 tuple containing
                    Position x, y, z.
                    Angle in rad x, y, z.
                    Gripper action
                    each value in range (-1, 1)

                    OR
                    8 tuple containing
                    Relative Joint angles j1 - j7 (in rad)
                    Gripper action
        output:
            observation, reward, done info
        """
        # Transform gripper action to discrete space
        env_action = action.copy()
        env_action[-1] = (int(action[-1] >= 0) * 2) - 1

        # for using actions in joint space
        if len(env_action) == 8:
            env_action = {"action": env_action, "type": "joint_rel"}

        self.robot.apply_action(env_action)
        for i in range(self.action_repeat):
            self.p.stepSimulation(physicsClientId=self.cid)
        obs = self.get_obs()
        info = self.get_info()
        reward, r_info = self._reward()
        done, d_info = self._termination()
        info.update(r_info)
        info.update(d_info)
        return obs, reward, done, info


# %% [markdown]
# # Training an RL agent
# After generating the wrapper training a reinforcement learning agent is straightforward, for this example we will use stable baselines 3 agents

# %%
# !pip3 install stable_baselines3

# %% [markdown]
# To train the agent we create an instance of our new environment and send it to the stable baselines agent to learn a policy.
#
#
# > Note: the example uses Soft Actor Critic (SAC) which is one of the state of the art algorithm for off-policy RL.
#
#

# %%
import gym
import numpy as np
from stable_baselines3 import SAC

new_env_cfg = {**cfg.env}
new_env_cfg["tasks"] = cfg.tasks
new_env_cfg.pop("_target_", None)
new_env_cfg.pop("_recursive_", None)
env = SlideEnv(**new_env_cfg)
model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)

# %%
model.save("sac_slider_left")
# %%
