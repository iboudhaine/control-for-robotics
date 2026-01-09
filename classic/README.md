# Classic Control

This directory contains classical path tracking implementations for discrete-time robot models.

## What's Inside

Two Jupyter notebooks implementing different control strategies:

1. **classic_control_integrator_model.ipynb** - Single integrator model path tracking
2. **classic_control_unicycle_model.ipynb** - Unicycle model path tracking

## What It Does

**Integrator Model:**
- Implements Full State Feedback control with pole placement
- Implements LQR (Linear Quadratic Regulator) optimal control
- Tracks smooth trajectories through waypoints in 2D space
- Handles disturbances and input constraints

**Unicycle Model:**
- Implements Feedback Linearization with LQR
- Uses look-ahead point transformation for non-holonomic constraints
- Implements Lyapunov indirect method with linearized feedback
- Tracks smooth trajectories with orientation control

## Installation

```bash
pip install numpy matplotlib scipy jupyter
```

## How to Run

1. Start Jupyter:
```bash
jupyter notebook
```

2. Open either notebook:
   - `classic_control_integrator_model.ipynb`
   - `classic_control_unicycle_model.ipynb`

3. Run all cells sequentially

4. Modify parameters to test different scenarios:
   - Change waypoints for different paths
   - Adjust LQR weights (Q, R matrices)
   - Modify starting positions
   - Tune feedback gains

## Key Parameters

**Integrator Model:**
- `tau`: Sampling time (0.1s)
- `K`: Feedback gain (calculated from desired pole or LQR)
- `Q`, `R`: LQR weight matrices

**Unicycle Model:**
- `L`: Look-ahead distance (0.3m)
- `u1_lim`: Linear velocity limits [0.25, 1.0] m/s
- `u2_lim`: Angular velocity limits [-1, 1] rad/s
- `Q`, `R`: LQR weight matrices for virtual point control
