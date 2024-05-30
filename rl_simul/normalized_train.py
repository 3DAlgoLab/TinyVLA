import gymnasium as gym
import panda_gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
import multiprocessing as mp

# Create a single environment
env_id = "PandaReachDense-v3"
print(panda_gym.__version__)

if __name__ == "__main__":
    env = make_vec_env(env_id, n_envs=mp.cpu_count(), vec_env_cls=SubprocVecEnv)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = A2C(policy="MultiInputPolicy", env=env, verbose=1)
    model.learn(200_000, progress_bar=True)
    model.save("data/a2c-panda-reach")
    env.save("data/panda-reach-env.pkl")
