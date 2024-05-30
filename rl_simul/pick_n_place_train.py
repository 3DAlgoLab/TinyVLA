import multiprocessing as mp

import panda_gym  # noqa: F401
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

num_cpu = mp.cpu_count()
env_id = "PandaPickAndPlaceDense-v3"

if __name__ == "__main__":
    env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(policy="MultiInputPolicy", env=env, verbose=1)
    model.learn(total_timesteps=10_000_000, progress_bar=True)
    model.save("data/ppo_panda_pick_and_place")
    env.save("data/ppo_panda_pick_and_place_env.pkl")
