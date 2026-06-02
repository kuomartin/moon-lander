import gymnasium as gym
import numpy as np
import pygame

from lander_env_v3 import RadarMoonLanderV3 as MoonLander


def main():
    # Initialize the environment in human mode for visual feedback
    env = MoonLander(render_mode="human")
    obs, info = env.reset()

    print("========================================")
    print("      Welcome to Pixel Moon Lander!      ")
    print("========================================")
    print("Controls:")
    print("  W / UP Arrow    : Fire Main Engine")
    print("  A / LEFT Arrow  : Rotate Left")
    print("  D / RIGHT Arrow : Rotate Right")
    print("  R               : Reset Environment")
    print("  Q / ESC         : Quit")
    print("========================================")

    is_continuous = isinstance(env.action_space, gym.spaces.Box)

    running = True
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

            action = np.array([m_power, r_power], dtype=np.float32)
        else:
            action = 0  # Default: NOP
            if keys[pygame.K_w] or keys[pygame.K_UP]:
                action = 1
            elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                action = 2  # Rotate right
            elif keys[pygame.K_a] or keys[pygame.K_LEFT]:
                action = 3  # Rotate left

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

    env.close()
    print("Game closed.")


if __name__ == "__main__":
    main()
