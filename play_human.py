import pygame

# from lander_env import PixelMoonLander
from lander_env_v2 import PixelMoonLanderV2 as PixelMoonLander


def main():
    # Initialize the environment in human mode for visual feedback
    env = PixelMoonLander(render_mode="human")
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

    running = True
    while running:
        action = 0  # Default: NOP (Do nothing)

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
