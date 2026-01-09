# Data-Driven Control: Global Linear Models and Local Linearization

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Implementation and validation of **data-driven control techniques** for nonlinear systems using linear approximations learned directly from data.

## What is Data-Driven Control?

**Data-driven control** designs controllers directly from measured system data, without requiring explicit mathematical models.

**The Problem:** Traditional control requires knowing the system dynamics `ẋ = f(x, u)`, but for many real systems these models are unavailable or too complex.

**Our Solution:** 
1. Collect input-output data: `(x_k, u_k, x_{k+1})`
2. Identify linear approximations from data:
   - **Global model:** `x_{k+1} ≈ A x_k + B u_k` (single model for entire system)
   - **Local models:** `x_{k+1} ≈ A_i x_k + B_i u_k + c_i` (different models per region)
3. Design controllers using these learned models

## Experiments

We validate theoretical guarantees on three benchmark nonlinear systems:

### Systems Tested
1. **Pendulum:** `θ̈ + bθ̇ + sin(θ) = u`
2. **Van der Pol:** `ẍ - μ(1-x²)ẋ + x = u`
3. **Duffing:** `ẍ + δẋ + αx + βx³ = u`

### Experiment 1: Sample Complexity
**Tests:** Error decreases as `O(1/√N)` with more data

**Results:**
| System      | Error @ N=50 | Error @ N=5000 | Verified |
|-------------|--------------|----------------|----------|
| Pendulum    | 0.0301       | 0.0023         | ✅       |
| Van der Pol | 0.0287       | 0.0022         | ✅       |
| Duffing     | 0.0344       | 0.0029         | ✅       |

### Experiment 2: Linearization Error Bounds
**Tests:** Approximation error bounded by `(L/2)·‖x-x*‖²`

**Results:**
- **3000 test points** across all systems
- **0 violations** of theoretical bound (0.0%)
- Confirms quadratic error growth

### Experiment 3: Stability Analysis
**Tests:** Stable linearization → stable closed-loop system

**Results:**
| System      | Max Eigenvalue | Stable? | Convergence Rate |
|-------------|----------------|---------|------------------|
| Pendulum    | 0.9987         | ✅      | 70.0%            |
| Van der Pol | 1.0137         | ❌      | N/A              |
| Duffing     | 0.9987         | ✅      | 75.0%            |

### Experiment 4: Global vs Local Models
**Tests:** Local models achieve better accuracy

**Results:**
| System      | Global Error | Local Error | Improvement |
|-------------|--------------|-------------|-------------|
| Pendulum    | 0.001576     | 0.000995    | **36.9%**   |
| Van der Pol | 0.006913     | 0.003811    | **44.9%**   |
| Duffing     | 0.004689     | 0.003435    | **26.7%**   |



## Usage

Run all experiments:
```bash
python hypothesis_validation.py
```

This generates:
- Console output with detailed results
- 11 validation figures (PNG)
- Complete experimental validation (~2-3 minutes)

### Quick Example
```python
from data_driven_control_validation import *

# Create system
system = PendulumSystem(damping=0.1)

# Generate data and identify model
X, U, Y = generate_dataset(system, n_trajectories=50, trajectory_length=100)
A, B = identify_global_linear_model(X, U, Y)

print(f"Identified A matrix:\n{A}")
print(f"Identified B matrix:\n{B}")
```



## 🎓 Authors

**Students:** Mahmoud MAFTAH, Ilyas BOUDHAINE, Ilyas HAKKOU, Sami AGOURRAM

**Supervisors:** Adnane SAOUD, Sadek BELMFADDEL ALAOUI

## 📚 References

1. Ljung, L. (1999). *System Identification: Theory for the User*. Prentice Hall.
2. Brunton, S. L., & Kutz, J. N. (2019). *Data-Driven Science and Engineering*. Cambridge.
3. Dean, S., et al. (2020). "On the sample complexity of the linear quadratic regulator." *Foundations of Computational Mathematics*, 20, 633-679.


---

⭐ Star this repo if you found it helpful!