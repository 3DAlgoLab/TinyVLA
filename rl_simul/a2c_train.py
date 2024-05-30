import gymnasium as gym
import panda_gym  # noqa: F401
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv


if __name__ == "__main__":
    env_id = "PandaReachDense-v3"
    num_cpu = 10

    env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)
    model = A2C(policy="MultiInputPolicy", env=env, verbose=1)
    model.learn(total_timesteps=1_000_000, progress_bar=True)
    model.save("data/a2c_panda_reach")
