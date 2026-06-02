import argparse

import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# 註冊 MLP 版本的降級測試環境
register(
    id="MLPMoonLander-v0",
    entry_point="lander_env_mlp:MLPMoonLanderEnv",
)


def train(timesteps, resume=False, model_path="ppo_mlp_moon_lander"):
    print("Creating MLP environment...")

    # MLP 不再需要圖片做輸入，也不必堆疊畫面 (FrameStack)，直接傳入 1D state
    env_kwargs = {"fixed_map": True}
    env = make_vec_env("MLPMoonLander-v0", n_envs=4, env_kwargs=env_kwargs)

    if resume:
        print(f"Loading existing model from {model_path} to resume training...")
        try:
            model = PPO.load(
                model_path, env=env, tensorboard_log="./ppo_mlp_tensorboard/"
            )
        except Exception as e:
            print(f"Error loading model to resume: {e}")
            print("Starting a new model instead.")
            resume = False

    if not resume:
        print("Initializing new PPO agent for MLP...")
        # 這裡從 CnnPolicy 改成 MlpPolicy
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log="./ppo_mlp_tensorboard/",
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=64,
            ent_coef=0.01,
        )

    print(f"Starting training for {timesteps} timesteps...")
    model.learn(
        total_timesteps=timesteps, progress_bar=True, reset_num_timesteps=not resume
    )

    model.save(model_path)
    print(f"Model saved to {model_path}.zip")

    # Evaluate
    print("Evaluating model...")
    eval_env = make_vec_env("MLPMoonLander-v0", n_envs=1)
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=10, deterministic=True
    )
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")


def test(model_path):
    print(f"Loading model {model_path} for testing...")
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 測試時開啟 human 模式觀看結果
    env_kwargs = {"render_mode": "human"}
    env = make_vec_env("MLPMoonLander-v0", n_envs=1, env_kwargs=env_kwargs)

    obs = env.reset()
    try:
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)

            if dones[0]:
                print(f"Episode Finished. Reward: {rewards[0]:.2f}")
    except KeyboardInterrupt:
        print("Testing interrupted by user.")
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train or test a Stable Baselines3 agent for MLP Downgrade Test."
    )
    parser.add_argument(
        "--test", action="store_true", help="Run the trained model instead of training"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from an existing model"
    )
    parser.add_argument(
        "--timesteps", type=int, default=300_000, help="Total timesteps for training"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="ppo_mlp_moon_lander",
        help="Path to the trained model",
    )
    args = parser.parse_args()

    if args.test:
        test(args.model_path)
    else:
        train(args.timesteps, args.resume, args.model_path)
