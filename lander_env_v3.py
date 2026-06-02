import math

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

WIDTH = 84
HEIGHT = 84
MAX_VELOCITY = 3.0
MAX_ANGLE_VELOCITY = 1.0
GRAVITY = 0.05
THROTTLE = 0.15
SIDE_ENGINE = 0.01
MAX_FUEL = 200.0
LANDING_SPEED = 1.0
LANDING_ANGLE = 0.3

_ZERO = np.int64(0)
_WIDTH = np.int64(WIDTH)
_HEIGHT = np.int64(HEIGHT)


class RadarMoonLanderV3(gym.Env):
    """
    A Moon Lander environment with radar-based observation space for MLP Reinforcement Learning.

    Observation Space Format (1D array of size N*2 + 6):
    - [N * (distance, type)]:
        - distance (float): Normalized distance (0.0 to 1.0) of the ray hit.
        - type (float): Normalized object type (-1.0: Terrain/Boundary, 0.0: Empty, >0: Pad mult / 3.0).
    - [x, y]: Normalized position (0.0 to 1.0).
    - [vx, vy]: Normalized velocity (-1.0 to 1.0).
    - v_angle: Normalized angular velocity (-1.0 to 1.0).
    - fuel: Normalized fuel level (0.0 to 1.0).

    Actions (Continuous Box[-1.0, 1.0]):
    - Index 0: Main Engine (-1.0 to 0.0: Off, 0.0 to 1.0: On 0%-100%)
    - Index 1: Rotation (-1.0 to -0.5: Left 100%-0%, -0.5 to 0.5: NOP, 0.5 to 1.0: Right 0%-100%)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: str | None = None,
        num_rays=20,
        fixed_map=False,
        map_pool_size: int | None = None,
        base_seed=0,
        speed_tolerance=0.0,
        angle_tolerance=0.0,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.num_rays = num_rays
        self.fixed_map = fixed_map
        self.map_pool_size = map_pool_size
        self.base_seed = base_seed
        self.terrain_generated = False

        # Tolerance parameters
        self.speed_tolerance = speed_tolerance
        self.angle_tolerance = angle_tolerance

        # Actions (Continuous):
        # - Index 0: Main Engine (-1.0 to 1.0)
        # - Index 1: Rotation (-1.0 to 1.0)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation space:
        # num_rays * 2 (dist, type) + 2 (x, y) + 2 (vx, vy) + 1 (v_angle) + 1 (fuel) + 2 (sin/cos angle)
        obs_dim = self.num_rays * 2 + 8
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.screen = None
        self.clock = None
        self.scale = 5

    def _generate_terrain(self):
        self.terrain_x = [_ZERO]
        self.terrain_y = [self.np_random.integers(42, 80)]
        self.pads = []

        num_pads = self.np_random.integers(2, 5)
        possible_centers = list(range(15, WIDTH - 15))
        self.np_random.shuffle(possible_centers)
        pad_centers: list[int] = []
        for cx in possible_centers:
            if all(abs(cx - existing_cx) > 15 for existing_cx in pad_centers):
                pad_centers.append(cx)
            if len(pad_centers) == num_pads:
                break
        pad_centers.sort()

        current_x = 0
        for cx in pad_centers:
            # 調整平台的寬度範圍，使其有機會小於或等於 7 藉此產生 3 倍(綠色)的平台
            pad_w = self.np_random.integers(6, 20)
            px1 = cx - pad_w // 2
            px2 = cx + pad_w // 2

            while current_x < px1 - 3:
                step_x = self.np_random.integers(3, 8)
                current_x = min(current_x + step_x, px1)
                if current_x == px1:
                    break
                current_y = self.terrain_y[-1] + self.np_random.integers(-15, 16)
                current_y = np.clip(current_y, 42, 80)
                self.terrain_x.append(current_x)
                self.terrain_y.append(current_y)

            pad_y = self.terrain_y[-1] + self.np_random.integers(-10, 11)
            pad_y = np.clip(pad_y, 42, 72)
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

        while current_x < WIDTH:
            step_x = self.np_random.integers(3, 8)
            current_x += step_x
            if current_x > WIDTH:
                current_x = _WIDTH
            current_y = self.terrain_y[-1] + self.np_random.integers(-15, 16)
            current_y = np.clip(current_y, 42, 80)
            self.terrain_x.append(current_x)
            self.terrain_y.append(current_y)

        self.terrain_x[-1] = _WIDTH
        self.terrain_heights = np.interp(
            np.arange(_WIDTH), self.terrain_x, self.terrain_y
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.map_pool_size is not None:
            map_seed = self.base_seed + self.np_random.integers(0, self.map_pool_size)
            temp_rng = np.random.default_rng(map_seed)
            old_rng = self.np_random
            self.np_random = temp_rng
            self._generate_terrain()
            self.np_random = old_rng
        elif not self.fixed_map or not self.terrain_generated:
            self._generate_terrain()
            self.terrain_generated = True

        self.x = WIDTH * self.np_random.uniform(0.2, 0.8)
        self.y = HEIGHT * 0.1
        self.vx = self.np_random.uniform(-0.5, 0.5)
        self.vy = self.np_random.uniform(0, 0.5)
        self.angle = self.np_random.uniform(-0.2, 0.2)
        self.v_angle = 0.0
        self.fuel = MAX_FUEL

        self.done = False
        self.main_engine_on = False
        self.prev_shaping = self._calculate_shaping()

        if self.render_mode == "human":
            self._setup_pygame()

        return self._get_obs(), {}

    def _calculate_shaping(self):
        potential_rewards = []
        for pad in self.pads:
            target_x = (pad["x1"] + pad["x2"]) / 2.0
            target_y = pad["y"]
            dx = (self.x - target_x) / (WIDTH / 2.0)
            dy = (self.y - target_y) / HEIGHT
            dist = math.sqrt(dx**2 + dy**2)
            shaping_val = -50.0 * dist + (pad["mult"] - 1) * 10.0
            potential_rewards.append(shaping_val)
        return max(potential_rewards) if potential_rewards else 0.0

    def _setup_pygame(self):
        if self.screen is None:
            pygame.init()
            pygame.font.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (WIDTH * self.scale, HEIGHT * self.scale)
                )
                pygame.display.set_caption("Radar Moon Lander V3")
            else:
                self.screen = pygame.Surface((WIDTH, HEIGHT))
        if self.clock is None:
            self.clock = pygame.time.Clock()
        if not hasattr(self, "font"):
            self.font = pygame.font.SysFont(None, 24)

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        self.main_engine_on = False

        # Fuel check
        if self.fuel <= 0:
            action = np.array([-1.0, 0.0])  # Force Engine Off, Rotation 0

        # Action[0]: Main Engine (-1.0 to 1.0)
        m_power = np.clip(action[0], -1.0, 1.0)
        if m_power > 0.0:
            self.fuel -= m_power * 1.0
            self.vx += math.sin(self.angle) * THROTTLE * m_power
            self.vy -= math.cos(self.angle) * THROTTLE * m_power
            self.main_engine_on = True

        # Action[1]: Rotation (-1.0 to 1.0) with Dead Zone [-0.5, 0.5]
        raw_r_power = np.clip(action[1], -1.0, 1.0)
        if raw_r_power < -0.5:
            r_power = (raw_r_power + 0.5) * 2.0
        elif raw_r_power > 0.5:
            r_power = (raw_r_power - 0.5) * 2.0
        else:
            r_power = 0.0

        if abs(r_power) > 0.0:
            self.fuel -= abs(r_power) * 0.5
            self.v_angle += r_power * SIDE_ENGINE

        # Clip angular velocity and update angle
        self.v_angle = np.clip(self.v_angle, -MAX_ANGLE_VELOCITY, MAX_ANGLE_VELOCITY)
        self.angle += self.v_angle

        self.fuel = max(0.0, self.fuel)
        self.vy += GRAVITY
        self.vx = np.clip(self.vx, -MAX_VELOCITY, MAX_VELOCITY)
        self.vy = np.clip(self.vy, -MAX_VELOCITY, MAX_VELOCITY)
        self.x += self.vx
        self.y += self.vy

        shaping = self._calculate_shaping()
        reward = (shaping - self.prev_shaping) * 2.0
        self.prev_shaping = shaping

        # Add penalty for high angular velocity to stabilize
        reward -= abs(self.v_angle) * 0.1

        # Engine usage (fuel consumption) penalty
        if m_power > 0.0:
            reward -= m_power * 0.3
        if abs(r_power) > 0.0:
            reward -= abs(r_power) * 0.03

        # Fuel-out penalty
        if self.fuel <= 0:
            reward -= 10.0

        terminated = False
        lander_bottom = self.y + 2
        lander_left = self.x - 2
        lander_right = self.x + 2

        if self.x < 0 or self.x >= WIDTH or self.y < 0 or self.y >= HEIGHT:
            reward += -100
            terminated = True
        else:
            tx_left = np.clip(int(lander_left), 0, WIDTH - 1)
            tx_center = np.clip(int(self.x), 0, WIDTH - 1)
            tx_right = np.clip(int(lander_right), 0, WIDTH - 1)

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
                    if abs(self.vy) < 1.0 and abs(self.angle) < 0.3:
                        reward += 100 * landed_on_pad["mult"]
                    else:
                        reward += -100
                        if self.speed_tolerance > 0:
                            bound = LANDING_SPEED * self.speed_tolerance
                            tolerance = 50 * (2 - abs(self.vy) / bound)
                            reward += max(tolerance, 0)
                        if self.angle_tolerance > 0:
                            bound = LANDING_ANGLE * self.angle_tolerance
                            tolerance = 50 * (2 - abs(self.angle) / bound)
                            reward += max(tolerance, 0)
                else:
                    reward += -100
                terminated = True

        self.done = terminated
        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, False, {}

    def _get_obs(self):
        # Radar rays
        rays = []
        self.last_rays = []
        # Cast rays in a circle around the lander
        for i in range(self.num_rays):
            angle = self.angle + (i * 2 * math.pi / self.num_rays)
            dist, rtype = self._cast_ray(self.x, self.y, angle)
            self.last_rays.append((angle, dist, rtype))
            rays.append(dist / 100.0)  # Normalize distance

            if rtype == -1:
                norm_type = -1.0
            elif rtype == 0:
                norm_type = 0.0
            else:
                norm_type = rtype / 3.0
            rays.append(norm_type)

        # Other states
        obs = rays + [
            self.x / WIDTH,
            self.y / HEIGHT,
            self.vx / MAX_VELOCITY,
            self.vy / MAX_VELOCITY,
            self.v_angle / MAX_ANGLE_VELOCITY,
            self.fuel / MAX_FUEL,
            math.sin(self.angle),
            math.cos(self.angle),
        ]
        return np.array(obs, dtype=np.float32)

    def _cast_ray(self, start_x, start_y, angle, max_dist=100):
        step = 1.0
        dist = 0
        while dist < max_dist:
            dist += step
            rx = start_x + dist * math.sin(
                angle
            )  # Use sin for x, -cos for y based on how angle is defined in physics
            ry = start_y - dist * math.cos(angle)

            if rx < 0 or rx >= WIDTH or ry < 0 or ry >= HEIGHT:
                return dist, -1  # Hit boundary

            tx = int(rx)
            if ry >= self.terrain_heights[tx]:
                # Check if hit pad
                for pad in self.pads:
                    if pad["x1"] <= rx <= pad["x2"] and abs(ry - pad["y"]) < 2:
                        return dist, pad["mult"]  # Hit pad (1, 2, 3)
                return dist, -1  # Hit terrain
        return max_dist, 0  # Hit nothing

    def _draw_lander(self, surface, x, y, angle, scale=1):
        s = scale
        points = [
            (0, -3 * s),
            (1.2 * s, -0.5 * s),
            (1.2 * s, 2.5 * s),
            (-1.2 * s, 2.5 * s),
            (-1.2 * s, -0.5 * s),
        ]
        rotated_points = []
        for px, py in points:
            rx = px * math.cos(angle) - py * math.sin(angle)
            ry = px * math.sin(angle) + py * math.cos(angle)
            rotated_points.append((float(x + rx), float(y + ry)))

        # Ensure all points are cast to float to avoid pygame TypeError
        clean_points = [(float(p[0]), float(p[1])) for p in rotated_points]

        pygame.draw.polygon(
            surface,
            (0, 0, 0) if surface.get_width() == WIDTH else (150, 150, 150),
            clean_points,
        )
        nose_points = [clean_points[0], clean_points[1], clean_points[4]]
        pygame.draw.polygon(surface, (255, 255, 255), nose_points)
        if self.main_engine_on:
            flame_points = [(-0.8 * s, 2.5 * s), (0.8 * s, 2.5 * s), (0, 5.0 * s)]
            rotated_flame = []
            for px, py in flame_points:
                rx = px * math.cos(angle) - py * math.sin(angle)
                ry = px * math.sin(angle) + py * math.cos(angle)
                rotated_flame.append((float(x + rx), float(y + ry)))
            pygame.draw.polygon(surface, (255, 150, 0), rotated_flame)

    def render(self):
        if self.render_mode is None:
            return
        if self.screen is None:
            self._setup_pygame()
        if self.render_mode == "human":
            if self.screen is not None:
                self.screen.fill((0, 0, 0))
            scaled_terrain_poly = [
                (tx * self.scale, ty * self.scale)
                for tx, ty in zip(self.terrain_x, self.terrain_y)
            ]
            scaled_terrain_poly.append((_WIDTH * self.scale, _HEIGHT * self.scale))
            scaled_terrain_poly.append((_ZERO, _HEIGHT * self.scale))
            pygame.draw.polygon(self.screen, (100, 100, 100), scaled_terrain_poly)
            for pad in self.pads:
                color = (
                    (0, 255, 0)
                    if pad["mult"] == 3
                    else (255, 255, 0)
                    if pad["mult"] == 2
                    else (255, 255, 255)
                )
                pygame.draw.line(
                    self.screen,
                    color,
                    (pad["x1"] * self.scale, pad["y"] * self.scale),
                    (pad["x2"] * self.scale, pad["y"] * self.scale),
                    max(3, self.scale),
                )

            if hasattr(self, "last_rays"):
                for angle, dist, rtype in self.last_rays:
                    end_x = self.x + dist * math.sin(angle)
                    end_y = self.y - dist * math.cos(angle)
                    if rtype == 0:
                        color = (50, 50, 50)  # Empty
                    elif rtype == -1:
                        color = (150, 50, 50)  # Terrain / Boundary (darker red)
                    elif rtype == 1:
                        color = (255, 255, 255)  # Pad x1
                    elif rtype == 2:
                        color = (255, 255, 0)  # Pad x2
                    elif rtype == 3:
                        color = (0, 255, 0)  # Pad x3
                    else:
                        color = (255, 255, 255)

                    pygame.draw.line(
                        self.screen,
                        color,
                        (float(self.x * self.scale), float(self.y * self.scale)),
                        (float(end_x * self.scale), float(end_y * self.scale)),
                        1,
                    )

            self._draw_lander(
                self.screen,
                self.x * self.scale,
                self.y * self.scale,
                self.angle,
                scale=self.scale,
            )

            # Draw fuel
            fuel_text = self.font.render(
                f"Fuel: {int(self.fuel)} / {int(MAX_FUEL)}", True, (255, 255, 255)
            )
            self.screen.blit(fuel_text, (10, 10))

            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
