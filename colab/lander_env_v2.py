import math

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class PixelMoonLanderV2(gym.Env):
    """
    A Moon Lander environment with randomly generated terrain for CNN Reinforcement Learning.
    Observation space is an 84x84 grayscale pixel image.
    This version uses OpenCV instead of Pygame for compatibility with Colab.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode=None,
        fixed_map=False,
        map_pool_size=None,
        base_seed=42,
        tolerance_mode=True,
        speed_tolerance_penalty=40.0,
        angle_tolerance_penalty=60.0,
    ):
        # 84x84 is a standard for CNN (e.g., Nature DQN)
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

        self.reset()

    def _generate_terrain(self):
        self.terrain_x = [0]
        self.terrain_y = [self.np_random.integers(40, 80)]

        self.pads = []

        # Determine number of pads (2 to 4)
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
            pad_w = self.np_random.integers(10, 20)
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
            map_seed = self.base_seed + self.np_random.integers(0, self.map_pool_size)
            temp_rng = np.random.default_rng(map_seed)
            old_rng = self.np_random
            self.np_random = temp_rng
            self._generate_terrain()
            self.np_random = old_rng
        elif not self.fixed_map or not self.terrain_generated:
            self._generate_terrain()
            self.terrain_generated = True

        self.x = self.width / 2.0
        self.y = self.height * 0.1
        self.vx = self.np_random.uniform(-0.5, 0.5)
        self.vy = self.np_random.uniform(0, 0.5)
        self.angle = self.np_random.uniform(-0.2, 0.2)

        self.done = False
        self.main_engine_on = False
        self.prev_shaping = self._calculate_shaping()

        return self._get_obs(), {}

    def _calculate_shaping(self):
        potential_rewards = []
        for pad in self.pads:
            target_x = (pad["x1"] + pad["x2"]) / 2.0
            target_y = pad["y"]
            dx = (self.x - target_x) / (self.width / 2.0)
            dy = (self.y - target_y) / self.height
            dist = math.sqrt(dx**2 + dy**2)
            shaping_val = -100.0 * dist + (pad["mult"] - 1) * 10.0
            potential_rewards.append(shaping_val)

        if potential_rewards:
            return max(potential_rewards)
        return 0.0

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        self.main_engine_on = False
        if action == 1:
            self.vx += math.sin(self.angle) * self.thrust
            self.vy -= math.cos(self.angle) * self.thrust
            self.main_engine_on = True
        elif action == 2:
            self.angle += self.rotation_speed
        elif action == 3:
            self.angle -= self.rotation_speed

        self.vy += self.gravity
        self.vx = np.clip(self.vx, -self.max_speed, self.max_speed)
        self.vy = np.clip(self.vy, -self.max_speed, self.max_speed)
        self.x += self.vx
        self.y += self.vy

        shaping = self._calculate_shaping()
        reward = (shaping - self.prev_shaping) * 2.0
        self.prev_shaping = shaping

        if self.vy > 1.0:
            reward -= 0.1
        if 0 < self.vy < 0.5:
            reward += 0.05

        if action == 1:
            reward -= 0.05
        elif action in [2, 3]:
            reward -= 0.01

        terminated = False
        lander_bottom = self.y + 2
        lander_left = self.x - 2
        lander_right = self.x + 2

        if self.x < 0 or self.x >= self.width or self.y < 0:
            reward += -100
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
                    if abs(self.vy) < 1.0 and abs(self.angle) < 0.3:
                        reward += 500 * landed_on_pad["mult"]
                    elif self.tolerance_mode:
                        speed_err = max(0, abs(self.vy) - 1.0)
                        angle_err = max(0, abs(self.angle) - 0.3)
                        land_reward = (
                            (250 * landed_on_pad["mult"])
                            - (speed_err * self.speed_tolerance_penalty)
                            - (angle_err * self.angle_tolerance_penalty)
                        )
                        reward += max(-20, land_reward)
                    else:
                        reward += -100
                else:
                    reward += -100
                terminated = True

        self.done = terminated
        return self._get_obs(), reward, terminated, False, {}

    def _get_obs(self):
        """Render the 84x84 image and return as numpy array using OpenCV"""
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Terrain
        terrain_poly = [(tx, ty) for tx, ty in zip(self.terrain_x, self.terrain_y)]
        terrain_poly.append((self.width, self.height))
        terrain_poly.append((0, self.height))
        pts = np.array(terrain_poly, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(img, [pts], (60, 60, 60))

        # Pads
        for pad in self.pads:
            if pad["mult"] == 3:
                color = (255, 255, 255)  # White
            elif pad["mult"] == 2:
                color = (200, 200, 200)  # Light Gray
            else:
                color = (140, 140, 140)  # Mid Gray
            cv2.line(
                img,
                (int(pad["x1"]), int(pad["y"])),
                (int(pad["x2"]), int(pad["y"])),
                color,
                2,
            )

        # Lander
        self._draw_lander(img, self.x, self.y, self.angle, scale=2.0)

        # Grayscale conversion using OpenCV
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.expand_dims(gray, axis=-1).astype(np.uint8)
        return gray

    def _draw_lander(self, img, x, y, angle, scale=1.0):
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
            rotated_points.append((int(x + rx), int(y + ry)))

        pts = np.array(rotated_points, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(img, [pts], (0, 0, 0))

        # Nose cone (White for orientation)
        nose_points = [rotated_points[0], rotated_points[1], rotated_points[4]]
        nose_pts = np.array(nose_points, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(img, [nose_pts], (255, 255, 255))

        # Flame
        if self.main_engine_on:
            flame_points = [(-0.8 * s, 2.5 * s), (0.8 * s, 2.5 * s), (0, 5.0 * s)]
            rotated_flame = []
            for px, py in flame_points:
                rx = px * math.cos(angle) - py * math.sin(angle)
                ry = px * math.sin(angle) + py * math.cos(angle)
                rotated_flame.append((int(x + rx), int(y + ry)))
            flame_pts = np.array(rotated_flame, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(img, [flame_pts], (0, 165, 255))

    def render(self):
        if self.render_mode == "rgb_array":
            return self._get_obs()
        return None

    def close(self):
        pass
