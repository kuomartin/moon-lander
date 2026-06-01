import argparse

import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecFrameStack

# 註冊環境
register(
    id="PixelMoonLander-v0",
    entry_point="lander_env:PixelMoonLander",
)


def train(timesteps, resume=False, model_path="dqn_pixel_moon_lander"):
    print("Creating environment for DQN...")
    env_kwargs = {"render_mode": "rgb_array"}
    # DQN 通常使用單個環境或同步環境，對於 FrameStack 很重要
    env = make_vec_env("PixelMoonLander-v0", n_envs=1, env_kwargs=env_kwargs)
    env = VecFrameStack(env, n_stack=4)

    if resume:
        print(f"Loading existing model from {model_path}...")
        model = DQN.load(
            model_path, env=env, tensorboard_log="./dqn_lander_tensorboard/"
        )
    else:
        print("Initializing new DQN agent...")
        model = DQN(
            "CnnPolicy",
            env,
            verbose=1,
            tensorboard_log="./dqn_lander_tensorboard/",
            learning_rate=1e-4,
            buffer_size=50_000,
            learning_starts=1000,
            batch_size=32,
            gamma=0.99,
            target_update_interval=500,
            exploration_fraction=0.5,
            exploration_final_eps=0.05,
        )

    print(f"Starting DQN training for {timesteps} timesteps...")
    model.learn(
        total_timesteps=timesteps, progress_bar=True, reset_num_timesteps=not resume
    )
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")


def test(model_path):
    print(f"Loading DQN model {model_path} for testing...")
    env_kwargs = {"render_mode": "human"}
    env = make_vec_env("PixelMoonLander-v0", n_envs=1, env_kwargs=env_kwargs)
    env = VecFrameStack(env, n_stack=4)

    model = DQN.load(model_path)
    obs = env.reset()
    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            if dones[0]:
                obs = env.reset()
    except KeyboardInterrupt:
        print("Testing stopped.")
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--model-path", type=str, default="dqn_pixel_moon_lander")
    args = parser.parse_args()

    if args.test:
        test(args.model_path)
    else:
        train(args.timesteps, model_path=args.model_path)
