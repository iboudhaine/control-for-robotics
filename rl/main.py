import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from stable_baselines3 import PPO
import math

class UnicycleObstacleEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60} 

    def __init__(self, render_mode=None):
        super(UnicycleObstacleEnv, self).__init__()

        # --- Physics ---
        self.TAU = 0.1
        
        # --- FIX 1: Enforce Minimum Velocity (Slide 3) ---
        # The robot MUST move at least 0.25 m/s. It cannot stop.
        self.u_min = np.array([0.25, -1.0], dtype=np.float32) 
        self.u_max = np.array([1.0, 1.0], dtype=np.float32)
        
        # --- Obstacles (x_min, x_max, y_min, y_max) ---
        self.obs_logic = [
            (3.0, 4.0, 4.0, 10.0), # Wall 1 (Left, Top)
            (6.0, 7.0, 0.0, 6.0)   # Wall 2 (Right, Bottom)
        ]
        
        self.target = np.array([9.0, 9.0]) 
        self.scale = 80 
        self.window_size = 800 

        # --- Spaces ---
        self.action_space = spaces.Box(low=self.u_min, high=self.u_max, dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.max_steps = 500 # Give it plenty of time

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Start at (1,1) facing Right (0.0 rad)
        self.state = np.array([1.0, 1.0, 0.0], dtype=np.float32)
        self.current_step = 0
        
        if self.render_mode == "human":
            self._render_frame()
            
        return self._get_obs(), {}

    def _get_obs(self):
        dx = self.target[0] - self.state[0]
        dy = self.target[1] - self.state[1]
        return np.array([self.state[0], self.state[1], self.state[2], dx, dy], dtype=np.float32)

    def _check_collision(self, x, y):
        # 1. Map Bounds (0-10)
        if not (0 <= x <= 10 and 0 <= y <= 10):
            return True
        # 2. Obstacles
        for (x1, x2, y1, y2) in self.obs_logic:
            if x1 <= x <= x2 and y1 <= y <= y2:
                return True
        return False

    def _get_proximity_penalty(self, x, y):
        penalty = 0.0
        safe_margin = 0.4
        
        # Check Wall Proximity
        for (x1, x2, y1, y2) in self.obs_logic:
            if (x1 - safe_margin <= x <= x2 + safe_margin) and \
               (y1 - safe_margin <= y <= y2 + safe_margin):
                penalty -= 0.1 # Small warning sting
        
        # Check Map Border Proximity (Don't hug the edges!)
        if x < 0.2 or x > 9.8 or y < 0.2 or y > 9.8:
            penalty -= 0.2
            
        return penalty

    def step(self, action):
        u = np.clip(action, self.u_min, self.u_max)
        x1, x2, x3 = self.state
        
        # Noise
        w = np.random.uniform(-0.05, 0.05, size=3)

        # Dynamics
        x1_new = x1 + self.TAU * (u[0] * np.cos(x3) + w[0])
        x2_new = x2 + self.TAU * (u[0] * np.sin(x3) + w[1])
        x3_new = x3 + self.TAU * (u[1] + w[2])
        x3_new = (x3_new + np.pi) % (2 * np.pi) - np.pi 

        self.state = np.array([x1_new, x2_new, x3_new], dtype=np.float32)
        self.current_step += 1

        terminated = False
        reward = 0

        # --- REWARD ENGINEERING ---
        dist = np.linalg.norm(self.state[:2] - self.target)
        
        # 1. Progress Reward (Better than pure distance)
        # We reward it for getting closer than it was last step
        prev_dist = np.linalg.norm([x1 - self.target[0], x2 - self.target[1]])
        reward += (prev_dist - dist) * 10.0 
        
        # 2. Time Penalty (Existential Crisis)
        # Even if it's safe, it loses points for wasting time. 
        # This prevents it from looping safely forever.
        reward -= 0.05

        # 3. Proximity Penalty (Soft Constraint)
        reward += self._get_proximity_penalty(x1_new, x2_new)

        # 4. Collision (Death)
        if self._check_collision(x1_new, x2_new):
            terminated = True
            reward = -200.0 # Reduced from -500 so it's not TOO afraid to try

        # 5. Success
        if dist < 0.5:
            terminated = True
            reward = 200.0

        truncated = self.current_step >= self.max_steps

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, truncated, {}

    def _render_frame(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        # Draw Obstacles
        for (x1, x2, y1, y2) in self.obs_logic:
            rect_h = (y2 - y1) * self.scale
            rect_y = (10 - y2) * self.scale
            rect_x = x1 * self.scale
            rect_w = (x2 - x1) * self.scale
            pygame.draw.rect(canvas, (200, 50, 50), pygame.Rect(rect_x, rect_y, rect_w, rect_h))

        # Draw Target
        py_tx, py_ty = int(self.target[0]*self.scale), int((10-self.target[1])*self.scale)
        pygame.draw.circle(canvas, (0, 200, 0), (py_tx, py_ty), 15)

        # Draw Robot
        rx, ry, rtheta = self.state
        py_rx, py_ry = int(rx*self.scale), int((10-ry)*self.scale)
        pygame.draw.circle(canvas, (50, 50, 255), (py_rx, py_ry), 12)
        
        # Heading
        end_x = py_rx + 25 * math.cos(rtheta)
        end_y = py_ry - 25 * math.sin(rtheta)
        pygame.draw.line(canvas, (0, 0, 0), (py_rx, py_ry), (end_x, end_y), 3)

        self.window.blit(canvas, (0, 0))
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None

if __name__ == "__main__":
    print("------------------------------------------")
    print("   TRAINING PHASE (High Velocity)         ")
    print("------------------------------------------")
    
    env = UnicycleObstacleEnv(render_mode=None)
    
    # 256x256 neurons
    policy_kwargs = dict(net_arch=[256, 256])
    
    model = PPO("MlpPolicy", env, verbose=1, 
                learning_rate=0.0003,
                ent_coef=0.01, 
                policy_kwargs=policy_kwargs)

    # 150k steps
    model.learn(total_timesteps=150000) 
    env.close()

    print("\n------------------------------------------")
    print("   VISUALIZATION PHASE                    ")
    print("------------------------------------------")
    
    env = UnicycleObstacleEnv(render_mode="human")
    obs, _ = env.reset()
    
    running = True
    while running:
        if pygame.get_init():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
        
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        
        if terminated or truncated:
            print(f"Outcome: {'Goal Reached!' if reward > 0 else 'Collision/Timeout'}")
            obs, _ = env.reset()
            pygame.time.wait(500)
    
    env.close()
    