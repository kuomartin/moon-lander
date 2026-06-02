import argparse

from gymnasium.envs.registration import register
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# Register our Radar environment
register(
    id="RadarMoonLander-v0",
    entry_point="lander_env:RadarMoonLander",
)


class SuccessRateCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.outcomes = []
        self.successes = []

    def _on_step(self) -> bool:
        for idx, done in enumerate(self.locals["dones"]):
            if done:
                info = self.locals["infos"][idx]
                terminal_info = info.get("terminal_info", info)

                if "outcome" in terminal_info:
                    self.outcomes.append(terminal_info["outcome"])
                if "is_success" in terminal_info:
                    self.successes.append(1.0 if terminal_info["is_success"] else 0.0)

        # Log rolling average of the last 100 episodes
        if len(self.successes) > 0 and self.n_calls % 100 == 0:
            recent_successes = self.successes[-100:]
            recent_outcomes = self.outcomes[-100:]

            success_rate = sum(recent_successes) / len(recent_successes)
            self.logger.record("custom/success_rate", success_rate)

            # 計算各個狀態的機率
            rates = {}
            for out_val in [-1, 1, 2, 3]:
                rate = sum(1 for o in recent_outcomes if o == out_val) / len(
                    recent_outcomes
                )
                rates[out_val] = rate

            # 計算 score_mean: pad_1 + 2*pad_2 + 3*pad_3 - crash
            # 這也等於 sum(recent_outcomes) / len(recent_outcomes)
            score_mean = sum(recent_outcomes) / len(recent_outcomes)
            self.logger.record("custom/score_mean", score_mean)

            # 記錄各別機率
            name_map = {
                -1: "crash_rate",
                1: "pad_1_rate",
                2: "pad_2_rate",
                3: "pad_3_rate",
            }
            for out_val, name in name_map.items():
                self.logger.record(f"custom/{name}", rates[out_val])

        return True


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
    # SAC typically benefits from a larger batch size and a replay buffer
    import multiprocessing

    cpu_cores = multiprocessing.cpu_count()
    # For SAC, it's often more stable to use fewer environments and keep the gradient steps ratio 1:1
    n_envs = max(min(cpu_cores, 4), 1)

    batch_size = 256
    learning_rate = 3e-4
    buffer_size = 1_000_000
    learning_starts = 10_000

    print(
        f"SAC Configuration: n_envs={n_envs}, batch_size={batch_size}, buffer_size={buffer_size}"
    )

    env = make_vec_env("RadarMoonLander-v0", n_envs=n_envs, env_kwargs=env_kwargs)

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if resume:
        print(f"Loading existing model from {model_path}...")
        model = SAC.load(
            model_path,
            env=env,
            tensorboard_log="./tensorboard/",
            device=device,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            learning_starts=learning_starts,
            ent_coef="auto_0.1",
            gamma=0.99,
            tau=0.005,
            train_freq=1,
            gradient_steps=n_envs,
            seed=seed,
        )
    else:
        print(f"Initializing new SAC on {device} (Seed: {seed})...")
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log="./tensorboard/",
            device=device,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            learning_starts=learning_starts,
            ent_coef="auto_0.1",
            gamma=0.99,
            tau=0.005,
            train_freq=1,
            gradient_steps=n_envs,
            seed=seed,
        )

    print(f"Starting training for {timesteps} timesteps...")

    # 每訓練約 25,000 步 (除以 8 個並行環境) 就自動儲存一次 checkpoint
    checkpoint_callback = CheckpointCallback(
        save_freq=max(25_600 // n_envs, 1),
        save_path="./model_checkpoints/",
        name_prefix=model_path,
    )

    success_callback = SuccessRateCallback()
    callbacks = CallbackList([checkpoint_callback, success_callback])

    model.learn(
        total_timesteps=timesteps,
        tb_log_name=model_path,
        progress_bar=True,
        reset_num_timesteps=not resume,
        callback=callbacks,
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
        model = SAC.load(model_path)
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
    parser.add_argument("--model-path", type=str, default="sac_radar_moon_lander")
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
