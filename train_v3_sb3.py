import argparse

from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# Register our V3 Radar environment
register(
    id="RadarMoonLander-v3",
    entry_point="lander_env_v3:RadarMoonLanderV3",
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
):
    print(
        f"Creating V3 Radar environment (Seed: {seed}, Rays: {num_rays}, Pool: {map_pool_size})..."
    )

    env_kwargs = {
        "render_mode": None,
        "num_rays": num_rays,
        "fixed_map": False,
        "base_seed": seed,
        "map_pool_size": map_pool_size,
        "speed_tolerance": speed_tolerance,
        "angle_tolerance": angle_tolerance,
    }

    # MLP usually trains well with multiple environments
    env = make_vec_env("RadarMoonLander-v3", n_envs=8, env_kwargs=env_kwargs)

    if resume:
        print(f"Loading existing model from {model_path} to resume training...")
        try:
            model = PPO.load(
                model_path,
                env=env,
                tensorboard_log="./ppo_radar_v3_tensorboard/",
                # Standard MLP hyperparams
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
            )
        except Exception as e:
            print(f"Error loading model to resume: {e}")
            resume = False

    else:
        print("Initializing new PPO agent with MlpPolicy for V3...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log="./ppo_radar_v3_tensorboard/",
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
        )

    print(f"Starting training for {timesteps} timesteps...")
    model.learn(
        total_timesteps=timesteps, progress_bar=True, reset_num_timesteps=not resume
    )

    model.save(model_path)
    print(f"Model saved to {model_path}.zip")

    # Evaluation
    print("Evaluating model...")
    eval_env = make_vec_env("RadarMoonLander-v3", n_envs=1, env_kwargs=env_kwargs)
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
):
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
    }
    env = make_vec_env("RadarMoonLander-v3", n_envs=1, env_kwargs=env_kwargs)

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
        description="Train or test Radar Moon Lander V3 (MLP)."
    )
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--model-path", type=str, default="ppo_radar_moon_lander_v3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-rays", type=int, default=20)
    parser.add_argument("--map-pool-size", type=int, default=None)
    parser.add_argument("--speed-tolerance", type=float, default=0)
    parser.add_argument("--angle-tolerance", type=float, default=0)
    args = parser.parse_args()

    if args.test:
        test(
            args.model_path,
            args.seed,
            args.num_rays,
            args.map_pool_size,
            args.speed_tolerance,
            args.angle_tolerance,
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
        )
