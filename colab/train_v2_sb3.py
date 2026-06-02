import argparse

from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecFrameStack

N_ENVS = 4
N_STEPS = (1 << 14) // N_ENVS


def train(
    timesteps,
    resume=False,
    model_path="ppo_pixel_moon_lander_v2",
    seed=42,
    map_pool_size=None,
    tolerance_mode=True,
    speed_penalty=40.0,
    angle_penalty=60.0,
):
    print(
        f"Creating V2 environment (Seed: {seed}, Pool: {map_pool_size}, Tolerance: {tolerance_mode})..."
    )
    # 建立多個平行的環境來加速訓練 (Vectorized Environments)
    # 並指定 render_mode 為 rgb_array 給 CNN 觀測
    env_kwargs = {
        "render_mode": "rgb_array",
        "fixed_map": False,
        "base_seed": seed,
        "map_pool_size": map_pool_size,
        "tolerance_mode": tolerance_mode,
        "speed_tolerance_penalty": speed_penalty,
        "angle_tolerance_penalty": angle_penalty,
    }
    env = make_vec_env(PixelMoonLanderV2, n_envs=N_ENVS, env_kwargs=env_kwargs)

    # 由於觀察空間是單張靜態圖片，神經網路無法直接推斷速度和旋轉方向。
    # 這裡我們疊加連續 4 張畫面 (Frame Stacking)，幫助模型理解動態資訊。
    env = VecFrameStack(env, n_stack=4)

    if resume:
        print(f"Loading existing model from {model_path} to resume training...")
        try:
            # 載入現有模型，並將環境綁定上去，並覆蓋超參數以提高穩定性
            model = PPO.load(
                model_path,
                env=env,
                tensorboard_log="./ppo_lander_v2_tensorboard/",
                learning_rate=3e-5,
                n_steps=N_STEPS,
                batch_size=256,
                n_epochs=10,
                ent_coef=0.05,
                target_kl=0.03,
                clip_range=0.2,
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
            learning_rate=3e-5,
            n_steps=N_STEPS,
            batch_size=256,
            n_epochs=10,
            ent_coef=0.05,
            target_kl=0.03,
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
    eval_env = make_vec_env(PixelMoonLanderV2, n_envs=1, env_kwargs=env_kwargs)
    eval_env = VecFrameStack(eval_env, n_stack=4)

    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=10, deterministic=True
    )
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")


def test(
    model_path,
    seed=42,
    map_pool_size=None,
    tolerance_mode=True,
    speed_penalty=40.0,
    angle_penalty=60.0,
):
    print(f"Loading model {model_path} for testing (Headless)...")
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 測試時使用 rgb_array 模式 (Colab 不支援 human 模式)
    env_kwargs = {
        "render_mode": "rgb_array",
        "base_seed": seed,
        "map_pool_size": map_pool_size,
        "tolerance_mode": tolerance_mode,
        "speed_tolerance_penalty": speed_penalty,
        "angle_tolerance_penalty": angle_penalty,
    }
    env = make_vec_env(PixelMoonLanderV2, n_envs=1, env_kwargs=env_kwargs)
    env = VecFrameStack(env, n_stack=4)

    obs = env.reset()
    try:
        for i in range(5):  # 測試 5 個回合
            done = False
            total_reward = 0
            obs = env.reset()
            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = env.step(action)
                total_reward += rewards[0]
                done = dones[0]
            print(f"Episode {i + 1} Finished. Reward: {total_reward:.2f}")
    except KeyboardInterrupt:
        print("Testing interrupted by user.")
    finally:
        env.close()
