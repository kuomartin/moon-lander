import math

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


class MLPMoonLanderEnv(gym.Env):
    """
    A downgraded Moon Lander environment for testing logic and physics.
    Observation space is a 1D array (state variables) instead of images.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode=None,
        fixed_map=False,
        map_pool_size=None,
        base_seed=42,
        tolerance_mode=True,
        speed_tolerance_penalty=30.0,
        angle_tolerance_penalty=50.0,
    ):
        self.width = 84
        self.height = 84
        self.render_mode = render_mode
        self.fixed_map = fixed_map
        self.map_pool_size = map_pool_size
        self.base_seed = base_seed
        self.terrain_generated = False

        # 寬容模式參數
        self.tolerance_mode = tolerance_mode
        self.speed_tolerance_penalty = speed_tolerance_penalty
        self.angle_tolerance_penalty = angle_tolerance_penalty

        self.action_space = spaces.Discrete(4)

        # Observation space: 1D array [x, y, vx, vy, angle, target_x, target_y, fuel]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )

        # Gravity and thrust adjustments for easier control
        self.gravity = 0.05
        self.thrust = 0.2
        self.rotation_speed = 0.15
        self.max_speed = 4.0

        self.screen = None
        self.clock = None
        self.scale = 5  # Scale for human mode

        # Add fuel mechanics
        self.initial_fuel = 200.0

    def _generate_terrain(self):
        self.terrain_x = [0]
        self.terrain_y = [self.np_random.integers(40, 80)]
        self.pads = []

        num_pads = self.np_random.integers(2, 5)
        possible_centers = list(range(15, self.width - 15))
        self.np_random.shuffle(possible_centers)
        pad_centers = []
        for cx in possible_centers:
            if all(abs(cx - existing_cx) > 15 for existing_cx in pad_centers):
                pad_centers.append(cx)
            if len(pad_centers) == num_pads:
                break

        pad_centers.sort()

        current_x = 0
        for cx in pad_centers:
            pad_w = self.np_random.integers(6, 15)
            px1 = cx - pad_w // 2
            px2 = cx + pad_w // 2

            while current_x < px1 - 3:
                step_x = self.np_random.integers(3, 8)
                current_x = min(current_x + step_x, px1)
                if current_x == px1:
                    break
                current_y = self.terrain_y[-1] + self.np_random.integers(-15, 16)
                current_y = np.clip(current_y, 30, 80)
                self.terrain_x.append(current_x)
                self.terrain_y.append(current_y)

            pad_y = self.terrain_y[-1] + self.np_random.integers(-10, 11)
            pad_y = np.clip(pad_y, 40, 72)
            self.terrain_x.append(px1)
            self.terrain_y.append(pad_y)
            self.terrain_x.append(px2)
            self.terrain_y.append(pad_y)

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

        self.terrain_heights = np.interp(
            np.arange(self.width), self.terrain_x, self.terrain_y
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.map_pool_size is not None:
            # Deterministically pick a map from the pool
            map_seed = self.base_seed + self.np_random.integers(0, self.map_pool_size)
            # Use a temporary RNG to generate the terrain
            temp_rng = np.random.default_rng(map_seed)
            old_rng = self.np_random
            self.np_random = temp_rng
            self._generate_terrain()
            self.np_random = old_rng
        elif not self.fixed_map or not self.terrain_generated:
            self._generate_terrain()
            self.terrain_generated = True

        # Select one target pad for the entire episode to stabilize learning
        # We pick the one with the highest multiplier, or a random one if tied
        best_mult = max(pad["mult"] for pad in self.pads)
        best_pads = [p for p in self.pads if p["mult"] == best_mult]
        self.target_pad = self.np_random.choice(best_pads)

        self.x = self.width / 2.0
        self.y = self.height * 0.1
        self.vx = self.np_random.uniform(-0.5, 0.5)
        self.vy = self.np_random.uniform(0, 0.5)
        self.angle = self.np_random.uniform(-0.2, 0.2)

        self.done = False
        self.main_engine_on = False
        self.prev_shaping = self._calculate_shaping()
        self.fuel = self.initial_fuel

        if self.render_mode == "human":
            self._setup_pygame()

        return self._get_obs(), {}

    def _calculate_shaping(self):
        # Calculate distance to the FIXED target pad
        target_x = (self.target_pad["x1"] + self.target_pad["x2"]) / 2.0
        target_y = self.target_pad["y"]
        dx = (self.x - target_x) / (self.width / 2.0)
        dy = (self.y - target_y) / self.height
        dist = math.sqrt(dx**2 + dy**2)

        return -100.0 * dist

    def _setup_pygame(self):
        if self.screen is None:
            pygame.init()
            pygame.font.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.width * self.scale, self.height * self.scale)
                )
                pygame.display.set_caption("MLP Moon Lander")
            else:
                self.screen = pygame.Surface((self.width, self.height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        # Ensure we have fuel to execute actions
        if self.fuel <= 0:
            action = 0

        self.main_engine_on = False

        if action == 1:
            self.fuel -= 1.0
            self.vx += math.sin(self.angle) * self.thrust
            self.vy -= math.cos(self.angle) * self.thrust
            self.main_engine_on = True
        elif action == 2:
            self.fuel -= 0.5
            self.angle += self.rotation_speed
        elif action == 3:
            self.fuel -= 0.5
            self.angle -= self.rotation_speed

        self.fuel = max(0.0, self.fuel)

        self.vy += self.gravity

        self.vx = np.clip(self.vx, -self.max_speed, self.max_speed)
        self.vy = np.clip(self.vy, -self.max_speed, self.max_speed)

        self.x += self.vx
        self.y += self.vy

        shaping = self._calculate_shaping()
        reward = (shaping - self.prev_shaping) * 2.0
        self.prev_shaping = shaping

        # Fuel penalty (discourage hovering)
        if action == 1:
            reward -= 0.2  # Slightly reduced penalty
        elif action in [2, 3]:
            reward -= 0.02  # Reduced rotation action penalty

        # REMOVED: Penalize fast rotation (This prevented horizontal movement)
        # reward -= abs(self.angle) * 0.05

        terminated = False
        truncated = False

        lander_bottom = self.y + 2
        lander_left = self.x - 2
        lander_right = self.x + 2

        if not self.done:
            if self.x < 0 or self.x >= self.width or self.y < 0:
                reward = -100  # Override shaping
                terminated = True
            else:
                tx_left = np.clip(int(lander_left), 0, self.width - 1)
                tx_center = np.clip(int(self.x), 0, self.width - 1)
                tx_right = np.clip(int(lander_right), 0, self.width - 1)

                if (
                    lander_bottom >= self.terrain_heights[tx_center]
                    or lander_bottom >= self.terrain_heights[tx_left]
                    or lander_bottom >= self.terrain_heights[tx_right]
                ):
                    landed_on_pad = None
                    for pad in self.pads:
                        if pad["x1"] <= lander_left and lander_right <= pad["x2"]:
                            if abs(lander_bottom - pad["y"]) <= 5:
                                landed_on_pad = pad
                                break

                    if landed_on_pad is not None:
                        # Check if it's the TARGET pad for extra consistency
                        is_target = landed_on_pad == self.target_pad

                        speed_err = max(0, abs(self.vy) - 1.5)
                        angle_err = max(0, abs(self.angle) - 0.5)

                        if speed_err == 0 and angle_err == 0:
                            bonus = 200 * landed_on_pad["mult"]
                            if is_target:
                                bonus += 100
                            reward = bonus
                        elif self.tolerance_mode:
                            # 寬容模式：根據傳入的比例扣分
                            bonus = (
                                (100 * landed_on_pad["mult"])
                                - (speed_err * self.speed_tolerance_penalty)
                                - (angle_err * self.angle_tolerance_penalty)
                            )
                            if is_target:
                                bonus += 50
                            reward = max(-20, bonus)
                        else:
                            reward = -100
                    else:
                        reward = -100  # Crashed outside pad
                    terminated = True

        self.done = terminated or truncated

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        """Return a 1D array of state variables for MLP policy"""
        target_x = (self.target_pad["x1"] + self.target_pad["x2"]) / 2.0
        target_y = self.target_pad["y"]

        # Use Relative Coordinates (Very important for MLP to learn direction)
        obs = np.array(
            [
                (target_x - self.x) / self.width,  # Relative X
                (target_y - self.y) / self.height,  # Relative Y
                self.vx / self.max_speed,
                self.vy / self.max_speed,
                self.angle,
                self.fuel / self.initial_fuel,
                # Add sin/cos of angle for better orientation representation
                math.sin(self.angle),
                math.cos(self.angle),
            ],
            dtype=np.float32,
        )

        return obs

    def _draw_lander(self, surface, x, y, angle, scale=1):
        # Rocket shape: 5 points (Pentagon)
        s = scale
        # Relative points (Tip, Right Shoulder, Right Bottom, Left Bottom, Left Shoulder)
        points = [
            (0, -3 * s),  # Tip (Head)
            (1.2 * s, -0.5 * s),  # Right Shoulder
            (1.2 * s, 2.5 * s),  # Right Bottom
            (-1.2 * s, 2.5 * s),  # Left Bottom
            (-1.2 * s, -0.5 * s),  # Left Shoulder
        ]

        # Rotate and translate
        rotated_points = []
        for px, py in points:
            rx = px * math.cos(angle) - py * math.sin(angle)
            ry = px * math.sin(angle) + py * math.cos(angle)
            rotated_points.append((x + rx, y + ry))

        # Draw main body (Darker gray for contrast)
        pygame.draw.polygon(surface, (150, 150, 150), rotated_points)

        # Draw Nose Cone (Pure White for maximum brightness in grayscale)
        nose_points = [rotated_points[0], rotated_points[1], rotated_points[4]]
        pygame.draw.polygon(surface, (255, 255, 255), nose_points)

        if self.main_engine_on:
            # Flame starts from the elongated bottom
            flame_points = [(-0.8 * s, 2.5 * s), (0.8 * s, 2.5 * s), (0, 5.0 * s)]
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

            scaled_terrain_poly = [
                (tx * self.scale, ty * self.scale)
                for tx, ty in zip(self.terrain_x, self.terrain_y)
            ]
            scaled_terrain_poly.append(
                (self.width * self.scale, self.height * self.scale)
            )
            scaled_terrain_poly.append((0, self.height * self.scale))

            pygame.draw.polygon(self.screen, (100, 100, 100), scaled_terrain_poly)

            font = pygame.font.SysFont(None, 8 * self.scale)
            for pad in self.pads:
                if pad["mult"] == 3:
                    color = (0, 255, 0)  # Green
                elif pad["mult"] == 2:
                    color = (255, 255, 0)  # Yellow
                else:
                    color = (255, 255, 255)  # White

                # Draw thick pad
                pygame.draw.line(
                    self.screen,
                    color,
                    (pad["x1"] * self.scale, pad["y"] * self.scale),
                    (pad["x2"] * self.scale, pad["y"] * self.scale),
                    max(3, self.scale),  # Thicker line
                )
                text = font.render(f"x{pad['mult']}", True, color)
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

            # Render fuel bar
            fuel_pct = self.fuel / self.initial_fuel
            bar_width = 100
            pygame.draw.rect(
                self.screen, (255, 100, 100), (10, 10, int(bar_width * fuel_pct), 10)
            )
            pygame.draw.rect(self.screen, (255, 255, 255), (10, 10, bar_width, 10), 1)

            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
