import os
import argparse
import numpy as np
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from src.env import AppleGameEnv

parser = argparse.ArgumentParser()
parser.add_argument("--render_mode", type=str, default="agent")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--limit_step", type=int, default=2000)
parser.add_argument("--policy", type=str, default="MlpPolicy")
parser.add_argument("--n_steps", type=int, default=2048)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--gae_lambda", type=float, default=0.95)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--clip_range", type=float, default=0.2)
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--total_timesteps", type=int, default=50000)
args = parser.parse_args()

env = AppleGameEnv(render_mode=args.render_mode, limit_step=args.limit_step)
env.reset(seed=args.seed)
vec_env = DummyVecEnv([lambda: env])
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.)

model = PPO(
    policy=args.policy,
    env=vec_env,
    verbose=1,
    n_steps=args.n_steps,
    batch_size=args.batch_size,
    gae_lambda=args.gae_lambda,
    gamma=args.gamma,
    n_epochs=args.n_epochs,
    clip_range=args.clip_range,
    learning_rate=args.learning_rate,
)

model.learn(total_timesteps=args.total_timesteps)
os.makedirs("model", exist_ok=True)
model.save("model/ppo_apple_game")