import argparse

from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# Register our Radar environment
register(
    id="RadarMoonLander-v0",
    entry_point="lander_env:RadarMoonLander",
)


def train(
    timesteps,
    resume,
    model_path,
    seed,
    num_rays,
    map_pool_size,
    speed_tolerance,
    angle_tolerance,
    render_fps,
):
    print(
        f"Creating Radar environment (Seed: {seed}, Rays: {num_rays}, Pool: {map_pool_size})..."
    )

    env_kwargs = {
        "render_mode": None,
        "num_rays": num_rays,
        "fixed_map": False,
        "base_seed": seed,
        "map_pool_size": map_pool_size,
        "speed_tolerance": speed_tolerance,
        "angle_tolerance": angle_tolerance,
        "render_fps": render_fps,
    }

    # Detect CPU cores to set n_envs appropriately
    # --- Consistency Strategy ---
    # 1. Fix the total number of steps collected before each update
    # Total rollout size = n_envs * n_steps. We want this to be constant (e.g., 2048)
    target_total_rollout = 16384

    import multiprocessing

    cpu_cores = multiprocessing.cpu_count()
    n_envs = max(min(cpu_cores, 8), 2)  # Range between 2 and 8

    # Adjust n_steps so that n_envs * n_steps is always close to target_total_rollout
    n_steps = target_total_rollout // n_envs

    # 2. Fix training hyperparams (Use stable defaults that work on both)
    batch_size = 64
    n_epochs = 10
    learning_rate = 3e-4

    print(
        f"Consistency Mode: n_envs={n_envs}, n_steps={n_steps} (Total: {n_envs * n_steps}), batch_size={batch_size}"
    )

    env = make_vec_env("RadarMoonLander-v0", n_envs=n_envs, env_kwargs=env_kwargs)

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if resume:
        print(f"Loading existing model from {model_path}...")
        model = PPO.load(
            model_path,
            env=env,
            tensorboard_log="./tensorboard/",
            device=device,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            seed=seed,
        )
    else:
        print(f"Initializing new PPO on {device} (Seed: {seed})...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log="./tensorboard/",
            device=device,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            seed=seed,
        )

    print(f"Starting training for {timesteps} timesteps...")

    # 每訓練約 25,000 步 (除以 8 個並行環境) 就自動儲存一次 checkpoint
    checkpoint_callback = CheckpointCallback(
        save_freq=max(25_000 // 8, 1),
        save_path="./model_checkpoints/",
        name_prefix=model_path,
    )

    model.learn(
        total_timesteps=timesteps,
        tb_log_name=model_path,
        progress_bar=True,
        reset_num_timesteps=not resume,
        callback=checkpoint_callback,
    )

    model.save(model_path)
    print(f"Model saved to {model_path}.zip")

    # Evaluation
    print("Evaluating model...")
    eval_env = make_vec_env("RadarMoonLander-v0", n_envs=1, env_kwargs=env_kwargs)
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=10, deterministic=True
    )
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")


def test(
    model_path,
    seed,
    num_rays,
    map_pool_size,
    speed_tolerance,
    angle_tolerance,
    render_fps,
):
    import pygame

    from lander_env import WIDTH

    print(f"Loading model {model_path} for testing...")
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    env_kwargs = {
        "render_mode": "human",
        "num_rays": num_rays,
        "base_seed": seed,
        "map_pool_size": map_pool_size,
        "speed_tolerance": speed_tolerance,
        "angle_tolerance": angle_tolerance,
        "render_fps": render_fps,
    }
    env = make_vec_env("RadarMoonLander-v0", n_envs=1, env_kwargs=env_kwargs)

    obs = env.reset()
    try:
        running = True
        while running:
            # Handle pygame events for window manipulation
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key in (
                        pygame.K_PLUS,
                        pygame.K_EQUALS,
                        pygame.K_KP_PLUS,
                    ):
                        # Extract the actual environment from DummyVecEnv to set its scale
                        base_env = env.envs[0].unwrapped
                        base_env.scale = min(20, base_env.scale + 1)
                        base_env.screen = pygame.display.set_mode(
                            (int(WIDTH * base_env.scale), int(WIDTH * base_env.scale))
                        )
                    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        base_env = env.envs[0].unwrapped
                        base_env.scale = max(1, base_env.scale - 1)
                        base_env.screen = pygame.display.set_mode(
                            (int(WIDTH * base_env.scale), int(WIDTH * base_env.scale))
                        )

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
        description="Train or test Radar Moon Lander (MLP)."
    )
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--model-path", type=str, default="ppo_radar_moon_lander")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-rays", type=int, default=20)
    parser.add_argument("--map-pool-size", type=int, default=None)
    parser.add_argument("--speed-tolerance", type=float, default=0)
    parser.add_argument("--angle-tolerance", type=float, default=0)
    parser.add_argument(
        "--fps", type=int, default=30, help="Frames per second for rendering"
    )
    args = parser.parse_args()

    if args.test:
        test(
            args.model_path,
            args.seed,
            args.num_rays,
            args.map_pool_size,
            args.speed_tolerance,
            args.angle_tolerance,
            args.fps,
        )
    else:
        train(
            args.timesteps,
            args.resume,
            args.model_path,
            args.seed,
            args.num_rays,
            args.map_pool_size,
            args.speed_tolerance,
            args.angle_tolerance,
            args.fps,
        )
