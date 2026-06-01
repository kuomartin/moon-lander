import numpy as np

from lander_env import PixelMoonLander


def main():
    # Setting render_mode="rgb_array" allows the CNN to get the pixel arrays
    # without opening a human-viewable window
    env = PixelMoonLander(render_mode="rgb_array")

    # 1. Reset Environment
    obs, info = env.reset()

    print(f"Observation Shape: {obs.shape}")  # Should be (84, 84, 1)
    print(f"Observation Data Type: {obs.dtype}")  # Should be uint8

    print("\nStarting simulated Random Agent for 5 steps...")

    for step in range(5):
        # 2. Pick a random action (0 to 3)
        # Normally your CNN / Deep Q-Network would predict the action here
        action = env.action_space.sample()

        # 3. Take step in the environment
        obs, reward, terminated, truncated, info = env.step(action)

        print(
            f"Step {step + 1}: Action={action}, Reward={reward:.2f}, Terminated={terminated}"
        )

        # If landed or crashed, reset
        if terminated or truncated:
            print("Episode Finished. Resetting...")
            obs, info = env.reset()

    env.close()
    print("Done!")


if __name__ == "__main__":
    main()
