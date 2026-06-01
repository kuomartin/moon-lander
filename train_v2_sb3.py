import argparse

import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecFrameStack

# 將我們的 V2 環境註冊到 Gymnasium 中
register(
    id="PixelMoonLander-v2",
    entry_point="lander_env_v2:PixelMoonLanderV2",
)


def train(timesteps, resume=False, model_path="ppo_pixel_moon_lander_v2"):
    print("Creating V2 environment...")
    # 建立多個平行的環境來加速訓練 (Vectorized Environments)
    # 並指定 render_mode 為 rgb_array 給 CNN 觀測
    env_kwargs = {"render_mode": "rgb_array"}
    env = make_vec_env("PixelMoonLander-v2", n_envs=4, env_kwargs=env_kwargs)

    # 由於觀察空間是單張靜態圖片，神經網路無法直接推斷速度和旋轉方向。
    # 這裡我們疊加連續 4 張畫面 (Frame Stacking)，幫助模型理解動態資訊。
    env = VecFrameStack(env, n_stack=4)

    if resume:
        print(f"Loading existing model from {model_path} to resume training...")
        try:
            # 載入現有模型，並將環境綁定上去
            model = PPO.load(
                model_path, env=env, tensorboard_log="./ppo_lander_v2_tensorboard/"
            )
        except Exception as e:
            print(f"Error loading model to resume: {e}")
            print("Starting a new model instead.")
            resume = False

    if not resume:
        print("Initializing new PPO agent for V2...")
        # 使用 CnnPolicy 來處理 84x84 的圖片輸入
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            tensorboard_log="./ppo_lander_v2_tensorboard/",
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=64,
            ent_coef=0.01,  # 鼓勵模型多做隨機探索
        )

    print(f"Starting training for {timesteps} timesteps...")
    # reset_num_timesteps=False 確保 Tensorboard 上的步數會接續下去
    model.learn(
        total_timesteps=timesteps, progress_bar=True, reset_num_timesteps=not resume
    )

    # 儲存訓練好的模型
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")

    # 評估模型
    print("Evaluating model...")
    eval_env = make_vec_env("PixelMoonLander-v2", n_envs=1, env_kwargs=env_kwargs)
    eval_env = VecFrameStack(eval_env, n_stack=4)

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

    # 測試時開啟 human 模式以便觀察
    env_kwargs = {"render_mode": "human"}
    env = make_vec_env("PixelMoonLander-v2", n_envs=1, env_kwargs=env_kwargs)
    env = VecFrameStack(env, n_stack=4)

    obs = env.reset()
    try:
        while True:
            # 模型進行預測
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
        description="Train or test a Stable Baselines3 agent for PixelMoonLander V2."
    )
    parser.add_argument(
        "--test", action="store_true", help="Run the trained model instead of training"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from an existing model"
    )
    parser.add_argument(
        "--timesteps", type=int, default=100_000, help="Total timesteps for training"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="ppo_pixel_moon_lander_v2",
        help="Path to the trained model",
    )
    args = parser.parse_args()

    if args.test:
        test(args.model_path)
    else:
        train(args.timesteps, args.resume, args.model_path)
