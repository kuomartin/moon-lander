import argparse

import gymnasium as gym
import numpy as np
import pygame

from lander_env import WIDTH
from lander_env import RadarMoonLander as MoonLander


def main(num_rays, auto_stabilize, render_fps):
    # Initialize the environment in human mode for visual feedback
    env = MoonLander(render_mode="human", num_rays=num_rays, render_fps=render_fps)
    obs, info = env.reset()

    print("========================================")
    print("      Welcome to Radar Moon Lander!      ")
    print("========================================")
    print("Controls:")
    print("  W / UP Arrow    : Fire Main Engine")
    print("  A / LEFT Arrow  : Rotate Left")
    print("  D / RIGHT Arrow : Rotate Right")
    print("  R               : Reset Environment")
    print("  S               : Toggle Auto-Stabilize (Rotation)")
    print("  P / Space       : Pause / Unpause")
    print("  + / -           : Zoom In / Zoom Out")
    print("  Q / ESC         : Quit")
    print("========================================")

    is_continuous = isinstance(env.action_space, gym.spaces.Box)

    running = True
    paused = False
    while running:
        # Pygame Event Loop for Keyboard
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                elif event.key == pygame.K_s:
                    auto_stabilize = not auto_stabilize
                    print(f"Auto-stabilize is now {'ON' if auto_stabilize else 'OFF'}")
                elif event.key == pygame.K_p or event.key == pygame.K_SPACE:
                    paused = not paused
                    print(f"Game is now {'PAUSED' if paused else 'RESUMED'}")
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    env.scale = min(20, env.scale + 1)
                    env.screen = pygame.display.set_mode(
                        (int(WIDTH * env.scale), int(WIDTH * env.scale))
                    )
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    env.scale = max(1, env.scale - 1)
                    env.screen = pygame.display.set_mode(
                        (int(WIDTH * env.scale), int(WIDTH * env.scale))
                    )

        # Check held down keys for continuous thrust/rotation
        keys = pygame.key.get_pressed()

        if is_continuous:
            # Action[0]: Main Engine (-1.0 to 1.0). > 0 is On.
            # Action[1]: Rotation (-1.0 to 1.0). < -0.5 is Left, > 0.5 is Right.
            m_power = -1.0
            r_power = 0.0

            if keys[pygame.K_w] or keys[pygame.K_UP]:
                m_power = 1.0

            if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                r_power = -1.0
            elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                r_power = 1.0
            elif auto_stabilize:
                # PD controller to keep angle at 0 when no manual rotation is applied
                target_r = -10.0 * env.angle - 40.0 * env.v_angle
                r_power = float(np.clip(target_r, -1.0, 1.0))

            action = np.array([m_power, r_power], dtype=np.float32)
        else:
            action = 0  # Default: NOP
            if keys[pygame.K_w] or keys[pygame.K_UP]:
                action = 1

            if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                action = 2  # Rotate right
            elif keys[pygame.K_a] or keys[pygame.K_LEFT]:
                action = 3  # Rotate left
            elif auto_stabilize:
                target_r = -10.0 * env.angle - 40.0 * env.v_angle
                if target_r > 0.5:
                    action = 2
                elif target_r < -0.5:
                    action = 3

        if not paused:
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)

            # If the episode ended, wait a bit and reset
            if terminated or truncated:
                if reward > 50:
                    print(">>> Landed Safely! Reward:", reward)
                else:
                    print(">>> Crashed / Out of bounds! Reward:", reward)

                pygame.time.wait(1000)  # Wait 1 second before reset
                obs, info = env.reset()
        else:
            # Still render to keep window active and constrain frame rate
            env.render()

    env.close()
    print("Game closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play Radar Moon Lander manually.")
    parser.add_argument(
        "--num-rays",
        type=int,
        default=20,
        help="Number of radar rays to use (default: 20)",
    )
    parser.add_argument(
        "--auto-stabilize",
        action="store_true",
        help="Enable auto-stabilization by default",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second (default: 30)",
    )
    args = parser.parse_args()

    main(
        num_rays=args.num_rays, auto_stabilize=args.auto_stabilize, render_fps=args.fps
    )
