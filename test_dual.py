import argparse

import pygame
from gymnasium.envs.registration import register
from stable_baselines3 import PPO, SAC

from lander_env import HEIGHT, WIDTH, RadarMoonLander

# Register the environment
try:
    register(
        id="RadarMoonLander-v0",
        entry_point="lander_env:RadarMoonLander",
    )
except Exception:
    pass


def load_model(path):
    print(f"Attempting to load model from {path}...")
    # Try PPO first
    try:
        model = PPO.load(path)
        print(f"Loaded {path} as PPO model.")
        return model
    except Exception:
        pass

    # Try SAC
    try:
        model = SAC.load(path)
        print(f"Loaded {path} as SAC model.")
        return model
    except Exception as e:
        print(f"Failed to load model from {path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Run two models side-by-side.")
    parser.add_argument("model_path1", type=str, help="Path to the first model (.zip)")
    parser.add_argument("model_path2", type=str, help="Path to the second model (.zip)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-rays", type=int, default=20)
    parser.add_argument("--map-pool-size", type=int, default=None)
    parser.add_argument("--speed-tolerance", type=float, default=0)
    parser.add_argument("--angle-tolerance", type=float, default=0)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--scale", type=int, default=5)

    args = parser.parse_args()

    model1 = load_model(args.model_path1)
    model2 = load_model(args.model_path2)

    if model1 is None or model2 is None:
        return

    # Automatically detect num_rays from models
    def get_num_rays(model):
        obs_shape = model.observation_space.shape[0]
        return (obs_shape - 8) // 2

    n_rays1 = get_num_rays(model1)
    n_rays2 = get_num_rays(model2)
    print(f"Model 1: detected {n_rays1} rays. Model 2: detected {n_rays2} rays.")

    env_kwargs1 = {
        "render_mode": "rgb_array",
        "num_rays": n_rays1,
        "base_seed": args.seed,
        "map_pool_size": args.map_pool_size,
        "speed_tolerance": args.speed_tolerance,
        "angle_tolerance": args.angle_tolerance,
        "render_fps": args.fps,
    }

    env_kwargs2 = env_kwargs1.copy()
    env_kwargs2["num_rays"] = n_rays2

    env1 = RadarMoonLander(**env_kwargs1)
    env2 = RadarMoonLander(**env_kwargs2)
    env1.scale = args.scale
    env2.scale = args.scale

    pygame.init()
    pygame.display.set_caption("Dual Radar Moon Lander Comparison")

    view_width = WIDTH * args.scale
    view_height = HEIGHT * args.scale
    screen = pygame.display.set_mode(
        (view_width * 2 + 10, view_height + int(60 * (args.scale / 5)))
    )
    clock = pygame.time.Clock()

    # Scale font size based on scale
    font_size = int(24 * (args.scale / 5))
    font = pygame.font.SysFont(None, font_size)
    label_font = pygame.font.SysFont(None, int(font_size * 0.8))

    obs1, _ = env1.reset(seed=args.seed)
    obs2, _ = env2.reset(seed=args.seed)

    done1 = False
    done2 = False
    outcome1 = 0
    outcome2 = 0
    episode_reward1 = 0
    episode_reward2 = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False

        # Step 1
        if not done1:
            action1, _ = model1.predict(obs1, deterministic=True)
            obs1, reward1, terminated1, truncated1, info1 = env1.step(action1)
            episode_reward1 += reward1
            done1 = terminated1 or truncated1
            if done1:
                outcome1 = info1.get("outcome", -1)

        # Step 2
        if not done2:
            action2, _ = model2.predict(obs2, deterministic=True)
            obs2, reward2, terminated2, truncated2, info2 = env2.step(action2)
            episode_reward2 += reward2
            done2 = terminated2 or truncated2
            if done2:
                outcome2 = info2.get("outcome", -1)

        # Render
        env1.render()
        env2.render()

        screen.fill((50, 50, 50))
        if env1.screen:
            screen.blit(env1.screen, (0, 0))
        if env2.screen:
            screen.blit(env2.screen, (view_width + 10, 0))

        # Draw labels
        label1 = label_font.render(
            f"Model 1: {args.model_path1}", True, (255, 255, 255)
        )
        label2 = label_font.render(
            f"Model 2: {args.model_path2}", True, (255, 255, 255)
        )
        screen.blit(label1, (10, view_height + 5))
        screen.blit(label2, (view_width + 10, view_height + 5))

        # Draw status
        status1 = "FINISHED" if done1 else "RUNNING"
        status2 = "FINISHED" if done2 else "RUNNING"

        # Color logic for status
        color1 = (255, 255, 255)
        if done1:
            color1 = (0, 255, 0) if outcome1 > 0 else (255, 50, 50)

        color2 = (255, 255, 255)
        if done2:
            color2 = (0, 255, 0) if outcome2 > 0 else (255, 50, 50)

        txt1 = font.render(
            f"Status: {status1} Reward: {episode_reward1:.1f}", True, color1
        )
        txt2 = font.render(
            f"Status: {status2} Reward: {episode_reward2:.1f}", True, color2
        )
        offset_y = int(30 * (args.scale / 5))
        screen.blit(txt1, (10, view_height + offset_y))
        screen.blit(txt2, (view_width + 10, view_height + offset_y))

        pygame.display.flip()
        clock.tick(args.fps)

        if done1 and done2:
            print(
                f"Both finished. M1 Reward: {episode_reward1:.2f}, M2 Reward: {episode_reward2:.2f}"
            )
            pygame.time.delay(1000)  # Wait a second before reset
            obs1, _ = env1.reset()
            obs2, _ = env2.reset()
            done1 = False
            done2 = False
            episode_reward1 = 0
            episode_reward2 = 0

    env1.close()
    env2.close()
    pygame.quit()


if __name__ == "__main__":
    main()
