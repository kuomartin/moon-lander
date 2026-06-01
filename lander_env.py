import math

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


class PixelMoonLander(gym.Env):
    """
    A simplified Moon Lander environment for CNN Reinforcement Learning.
    Observation space is an 84x84 grayscale pixel image.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        # 84x84 is a standard for CNN (e.g., Nature DQN)
        self.width = 84
        self.height = 84
        self.render_mode = render_mode

        # Actions:
        # 0: NOP (Do nothing)
        # 1: Main Engine (Thrust up)
        # 2: Rotate Right
        # 3: Rotate Left
        self.action_space = spaces.Discrete(4)

        # Observation space: 84x84 Grayscale
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8
        )

        # Physics constants
        self.gravity = 0.05
        self.thrust = 0.15
        self.rotation_speed = 0.1
        self.max_speed = 3.0

        # Setup pygame for rendering and pixel extraction
        self.screen = None
        self.clock = None
        self.scale = 5  # Scale for human mode (84 * 5 = 420x420)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initial Lander State
        self.x = self.width / 2.0
        self.y = self.height * 0.1
        self.vx = self.np_random.uniform(-0.5, 0.5)
        self.vy = self.np_random.uniform(0, 0.5)
        self.angle = self.np_random.uniform(-0.2, 0.2)

        # Landing Pad Configuration
        self.pad_w = 20
        self.pad_x1 = (self.width - self.pad_w) // 2
        self.pad_x2 = self.pad_x1 + self.pad_w
        self.pad_y = self.height - 10

        self.done = False

        # Flame visual states
        self.main_engine_on = False

        # 計算初始的 shaping 值 (Reward Shaping) - 僅依賴距離
        target_x = self.pad_x1 + self.pad_w / 2
        target_y = self.pad_y
        dx = (self.x - target_x) / (self.width / 2.0)
        dy = (self.y - target_y) / self.height
        dist = math.sqrt(dx**2 + dy**2)
        self.prev_shaping = -100.0 * dist

        if self.render_mode == "human":
            self._setup_pygame()

        return self._get_obs(), {}

    def _setup_pygame(self):
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.width * self.scale, self.height * self.scale)
                )
                pygame.display.set_caption("Pixel Moon Lander")
            else:
                self.screen = pygame.Surface((self.width, self.height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        self.main_engine_on = False

        # Apply action
        if action == 1:
            # Main engine
            self.vx += math.sin(self.angle) * self.thrust
            self.vy -= math.cos(self.angle) * self.thrust
            self.main_engine_on = True
        elif action == 2:
            # Rotate right
            self.angle += self.rotation_speed
        elif action == 3:
            # Rotate left
            self.angle -= self.rotation_speed

        # Apply gravity
        self.vy += self.gravity

        # Clip velocity
        self.vx = np.clip(self.vx, -self.max_speed, self.max_speed)
        self.vy = np.clip(self.vy, -self.max_speed, self.max_speed)

        # Update position
        self.x += self.vx
        self.y += self.vy

        # Check termination & reward

        # 1. 計算目前的 shaping 值 (僅計算距離)
        target_x = self.pad_x1 + self.pad_w / 2
        target_y = self.pad_y
        dx = (self.x - target_x) / (self.width / 2.0)
        dy = (self.y - target_y) / self.height
        dist = math.sqrt(dx**2 + dy**2)

        shaping = -100.0 * dist

        # 2. 獎勵 = 狀態進步的幅度 (離目標更近就會得到正向獎勵)
        reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        # 3. 燃料懲罰 (避免 Agent 亂噴射或瘋狂旋轉)
        if action == 1:
            reward -= 0.3  # 主引擎
        elif action in [2, 3]:
            reward -= 0.05  # 側邊旋轉

        terminated = False

        # Collision bounding box for lander roughly ~4 pixels
        lander_bottom = self.y + 4

        # Check out of bounds
        if self.x < 0 or self.x > self.width or self.y < 0:
            reward += -100
            terminated = True
        # Check landing or crash
        elif lander_bottom >= self.pad_y:
            # Check speed and angle
            if abs(self.vy) < 1.0 and abs(self.angle) < 0.3:
                # Check if landed inside the pad area
                if self.pad_x1 <= self.x <= self.pad_x2:
                    reward += 100  # Safe landing!
                else:
                    reward += +10  # Missed the pad
            else:
                reward += -100  # Crashed (too fast or tilted)
            terminated = True

        self.done = terminated

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, False, {}

    def _get_obs(self):
        """Render the 84x84 image and return as numpy array"""
        # We always draw on an 84x84 surface for the CNN observation
        surface = pygame.Surface((self.width, self.height))
        surface.fill((0, 0, 0))  # Black space

        # Draw Terrain (Gray)
        pygame.draw.rect(
            surface,
            (100, 100, 100),
            (0, self.pad_y, self.width, self.height - self.pad_y),
        )

        # Draw Landing Pad (White)
        pygame.draw.rect(
            surface, (255, 255, 255), (self.pad_x1, self.pad_y, self.pad_w, 2)
        )

        # Draw Lander
        self._draw_lander(surface, self.x, self.y, self.angle, scale=1)

        # Convert to numpy array (H, W, C)
        view = pygame.surfarray.pixels3d(surface)
        view = view.transpose([1, 0, 2])

        # Convert RGB to Grayscale using standard luminosity weights
        gray = np.dot(view[..., :3], [0.2989, 0.5870, 0.1140])
        gray = np.expand_dims(gray, axis=-1).astype(np.uint8)

        return gray

    def _draw_lander(self, surface, x, y, angle, scale=1):
        # Base points for the lander (triangle shape)
        size = 4 * scale
        points = [
            (0, -size),  # Top
            (-size, size),  # Bottom Left
            (size, size),  # Bottom Right
        ]

        # Rotate and translate
        rotated_points = []
        for px, py in points:
            rx = px * math.cos(angle) - py * math.sin(angle)
            ry = px * math.sin(angle) + py * math.cos(angle)
            rotated_points.append((x + rx, y + ry))

        pygame.draw.polygon(surface, (200, 200, 200), rotated_points)

        # Draw Flame
        if self.main_engine_on:
            flame_points = [(-size / 2.5, size), (size / 2.5, size), (0, size * 2.0)]
            rotated_flame = []
            for px, py in flame_points:
                rx = px * math.cos(angle) - py * math.sin(angle)
                ry = px * math.sin(angle) + py * math.cos(angle)
                rotated_flame.append((x + rx, y + ry))
            pygame.draw.polygon(surface, (255, 150, 0), rotated_flame)

    def render(self):
        if self.render_mode is None:
            return

        if self.screen is None:
            self._setup_pygame()

        if self.render_mode == "human":
            self.screen.fill((0, 0, 0))

            # Scaled drawing for human visibility
            scaled_pad_y = self.pad_y * self.scale

            # Terrain
            pygame.draw.rect(
                self.screen,
                (100, 100, 100),
                (
                    0,
                    scaled_pad_y,
                    self.width * self.scale,
                    (self.height - self.pad_y) * self.scale,
                ),
            )
            # Pad
            pygame.draw.rect(
                self.screen,
                (255, 255, 255),
                (
                    self.pad_x1 * self.scale,
                    scaled_pad_y,
                    self.pad_w * self.scale,
                    2 * self.scale,
                ),
            )

            # Lander
            self._draw_lander(
                self.screen,
                self.x * self.scale,
                self.y * self.scale,
                self.angle,
                scale=self.scale,
            )

            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
