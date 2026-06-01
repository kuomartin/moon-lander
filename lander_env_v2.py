import math

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


class PixelMoonLanderV2(gym.Env):
    """
    A Moon Lander environment with randomly generated terrain for CNN Reinforcement Learning.
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

    def _generate_terrain(self):
        self.terrain_x = [0]
        self.terrain_y = [self.np_random.integers(40, 80)]

        self.pads = []

        # Determine number of pads (2 to 4)
        num_pads = self.np_random.integers(2, 5)
        # Pick random x-coordinates for pad centers, keeping them away from edges
        # We need to make sure they are somewhat spaced out
        possible_centers = list(range(15, self.width - 15))
        self.np_random.shuffle(possible_centers)
        pad_centers = []
        for cx in possible_centers:
            # Check distance to existing pads
            if all(abs(cx - existing_cx) > 15 for existing_cx in pad_centers):
                pad_centers.append(cx)
            if len(pad_centers) == num_pads:
                break

        pad_centers.sort()

        current_x = 0
        for cx in pad_centers:
            # Determine pad width
            pad_w = self.np_random.integers(6, 15)
            px1 = cx - pad_w // 2
            px2 = cx + pad_w // 2

            # Generate jagged points until the pad
            while current_x < px1 - 3:
                step_x = self.np_random.integers(3, 8)
                current_x = min(current_x + step_x, px1)
                if current_x == px1:
                    break
                current_y = self.terrain_y[-1] + self.np_random.integers(-15, 16)
                current_y = np.clip(current_y, 30, 80)
                self.terrain_x.append(current_x)
                self.terrain_y.append(current_y)

            # Add the pad surface
            pad_y = self.terrain_y[-1] + self.np_random.integers(-10, 11)
            pad_y = np.clip(pad_y, 40, 72)
            self.terrain_x.append(px1)
            self.terrain_y.append(pad_y)
            self.terrain_x.append(px2)
            self.terrain_y.append(pad_y)

            # Determine score multiplier based on pad width
            if pad_w <= 7:
                mult = 3
            elif pad_w <= 10:
                mult = 2
            else:
                mult = 1

            self.pads.append(
                {"x1": px1, "x2": px2, "y": pad_y, "mult": mult, "width": pad_w}
            )
            current_x = px2

        # Generate jagged points for the rest of the terrain
        while current_x < self.width:
            step_x = self.np_random.integers(3, 8)
            current_x += step_x
            if current_x > self.width:
                current_x = self.width
            current_y = self.terrain_y[-1] + self.np_random.integers(-15, 16)
            current_y = np.clip(current_y, 30, 80)
            self.terrain_x.append(current_x)
            self.terrain_y.append(current_y)

        self.terrain_x[-1] = self.width

        # Interpolate to get heights for all integer x coordinates
        self.terrain_heights = np.interp(
            np.arange(self.width), self.terrain_x, self.terrain_y
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._generate_terrain()

        # Initial Lander State (starts near the top center)
        self.x = self.width / 2.0
        self.y = self.height * 0.1
        self.vx = self.np_random.uniform(-0.5, 0.5)
        self.vy = self.np_random.uniform(0, 0.5)
        self.angle = self.np_random.uniform(-0.2, 0.2)

        self.done = False

        # Flame visual states
        self.main_engine_on = False

        self.prev_shaping = self._calculate_shaping()

        if self.render_mode == "human":
            self._setup_pygame()

        return self._get_obs(), {}

    def _calculate_shaping(self):
        # Calculate distance to the closest pad
        distances = []
        for pad in self.pads:
            target_x = (pad["x1"] + pad["x2"]) / 2.0
            target_y = pad["y"]
            dx = (self.x - target_x) / (self.width / 2.0)
            dy = (self.y - target_y) / self.height
            dist = math.sqrt(dx**2 + dy**2)
            distances.append(dist)

        if distances:
            return -100.0 * min(distances)
        return 0.0

    def _setup_pygame(self):
        if self.screen is None:
            pygame.init()
            pygame.font.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.width * self.scale, self.height * self.scale)
                )
                pygame.display.set_caption("Pixel Moon Lander V2")
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

        # 1. Calculate current shaping value
        shaping = self._calculate_shaping()

        # 2. Reward = difference in shaping
        # 放大距離縮減的獎勵，讓它更有動力靠近
        reward = (shaping - self.prev_shaping) * 2.0
        self.prev_shaping = shaping

        # 追加引導：如果垂直速度太快(往下掉)，給予微小懲罰
        if self.vy > 1.0:
            reward -= 0.1

        # 追加引導：如果不在降落台上，且成功保持懸浮/緩降，給予微小獎勵 (鼓勵存活)
        if 0 < self.vy < 0.5:
            reward += 0.05

        # 3. Fuel penalty
        if action == 1:
            reward -= 0.1  # 再次稍微降低主引擎懲罰
        elif action in [2, 3]:
            reward -= 0.02  # 降低側邊旋轉懲罰

        terminated = False

        # Collision bounding box for lander
        lander_bottom = self.y + 2
        lander_left = self.x - 2
        lander_right = self.x + 2

        # Check out of bounds
        if self.x < 0 or self.x >= self.width or self.y < 0:
            reward += -100
            terminated = True
        else:
            # Check terrain collision at left, center, right
            tx_left = np.clip(int(lander_left), 0, self.width - 1)
            tx_center = np.clip(int(self.x), 0, self.width - 1)
            tx_right = np.clip(int(lander_right), 0, self.width - 1)

            if (
                lander_bottom >= self.terrain_heights[tx_center]
                or lander_bottom >= self.terrain_heights[tx_left]
                or lander_bottom >= self.terrain_heights[tx_right]
            ):
                # Check if landed on a pad
                landed_on_pad = None
                for pad in self.pads:
                    # Lander must be fully inside the pad's horizontal bounds
                    if pad["x1"] <= lander_left and lander_right <= pad["x2"]:
                        # Vertical check: lander bottom should be roughly at pad height
                        if abs(lander_bottom - pad["y"]) <= 5:
                            landed_on_pad = pad
                            break

                if landed_on_pad is not None:
                    # Check speed and angle
                    if abs(self.vy) < 1.0 and abs(self.angle) < 0.3:
                        reward += 200 * landed_on_pad["mult"]  # 大幅提升安全降落的獎勵
                    else:
                        reward += -100  # Crashed (too fast or tilted)
                else:
                    reward += -100  # Crashed on uneven terrain or missed pad
                terminated = True

        self.done = terminated

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, False, {}

    def _get_obs(self):
        """Render the 84x84 image and return as numpy array"""
        surface = pygame.Surface((self.width, self.height))
        surface.fill((0, 0, 0))  # Black space

        # Create polygon points for terrain
        terrain_poly = [(tx, ty) for tx, ty in zip(self.terrain_x, self.terrain_y)]
        terrain_poly.append((self.width, self.height))
        terrain_poly.append((0, self.height))

        # Draw Terrain (Gray)
        pygame.draw.polygon(surface, (100, 100, 100), terrain_poly)
        # Draw outline
        pygame.draw.lines(surface, (200, 200, 200), False, terrain_poly[:-2], 1)

        # Draw Landing Pads (White)
        for pad in self.pads:
            pygame.draw.line(
                surface,
                (255, 255, 255),
                (pad["x1"], pad["y"]),
                (pad["x2"], pad["y"]),
                1,
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
        size = 2 * scale
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
            scaled_terrain_poly = [
                (tx * self.scale, ty * self.scale)
                for tx, ty in zip(self.terrain_x, self.terrain_y)
            ]
            scaled_terrain_poly.append(
                (self.width * self.scale, self.height * self.scale)
            )
            scaled_terrain_poly.append((0, self.height * self.scale))

            # Terrain
            pygame.draw.polygon(self.screen, (100, 100, 100), scaled_terrain_poly)
            # Outline
            pygame.draw.lines(
                self.screen,
                (200, 200, 200),
                False,
                scaled_terrain_poly[:-2],
                max(1, self.scale // 2),
            )

            # Pads and multipliers
            font = pygame.font.SysFont(None, 8 * self.scale)
            for pad in self.pads:
                # Draw thick white pad
                pygame.draw.line(
                    self.screen,
                    (255, 255, 255),
                    (pad["x1"] * self.scale, pad["y"] * self.scale),
                    (pad["x2"] * self.scale, pad["y"] * self.scale),
                    max(2, self.scale // 2),
                )

                # Render multiplier text
                text = font.render(f"x{pad['mult']}", True, (200, 200, 200))
                text_rect = text.get_rect(
                    center=(
                        ((pad["x1"] + pad["x2"]) / 2) * self.scale,
                        (pad["y"] + 6) * self.scale,
                    )
                )
                self.screen.blit(text, text_rect)

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
