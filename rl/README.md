# Reinforcement Learning

This directory contains reinforcement learning implementations for unicycle robot navigation with obstacle avoidance.

## What's Inside

- **main.py** - PPO-based unicycle robot training and visualization

## What It Does

Trains a unicycle robot to navigate from start position (1, 1) to target position (9, 9) while avoiding obstacles:

- Two rectangular obstacles (walls) in the 10x10 workspace
- Custom Gymnasium environment with unicycle dynamics
- PPO (Proximal Policy Optimization) algorithm from stable-baselines3
- Real-time visualization using Pygame

**Key Features:**
- Minimum velocity constraint (0.25 m/s) - robot cannot stop
- Noise injection for robustness
- Reward shaping for collision avoidance and goal reaching
- Visual feedback during testing

**Environment:**
- State: [x, y, theta, dx, dy] (position, orientation, distance to target)
- Action: [linear_velocity, angular_velocity]
- Constraints: v ∈ [0.25, 1.0], ω ∈ [-1, 1]
- Obstacles at: (3-4, 4-10) and (6-7, 0-6)

## Installation

```bash
pip install gymnasium numpy pygame stable-baselines3
```

## How to Run

```bash
python main.py
```

The script will:
1. Train the PPO agent for 150,000 timesteps (~5-10 minutes)
2. Automatically open visualization window
3. Show the trained robot navigating to the target
4. Reset and repeat on collision/success

Close the Pygame window to exit.

## Customization

Modify in `main.py`:
- `self.obs_logic`: Change obstacle positions
- `self.target`: Change goal location
- `total_timesteps`: Train for more/fewer steps
- `learning_rate`, `ent_coef`: Tune PPO hyperparameters
- `policy_kwargs`: Change neural network architecture
- Reward function in `step()`: Adjust behavior priorities
