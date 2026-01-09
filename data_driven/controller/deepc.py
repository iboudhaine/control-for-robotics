"""
Data-Enabled Predictive Control (DeePC)

Implements DeePC algorithm from:
"Data-Enabled Predictive Control: In the Shallows of the DeePC"
by Coulson, Lygeros, and Dörfler (2019)

Key idea: Use Hankel matrices from data to directly predict future
outputs without explicit system identification.

For a system x_{t+1} = f(x_t, u_t), DeePC constructs:
- Hankel matrices from collected data
- Predicts future trajectory using data-driven predictor
- Solves constrained optimization for optimal control

The method uses Willems' fundamental lemma to represent all possible
trajectories as linear combinations of data columns.
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Callable
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Try to import cvxpy for optimization
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    print("Warning: cvxpy not available. DeePC will use QP fallback.")


class DeePC:
    """Data-Enabled Predictive Control.
    
    Uses Hankel matrices from pre-collected data to predict and optimize
    future trajectories without system identification.
    
    Algorithm:
    1. Build Hankel matrices from data: [U_p; Y_p; U_f; Y_f]
    2. At each step, solve for g such that:
       [U_p; Y_p; U_f; Y_f] @ g = [u_ini; y_ini; u_f; y_f]
    3. Minimize cost over predicted trajectory y_f, u_f
    """
    
    def __init__(
        self,
        T_ini: int = 5,
        N_pred: int = 10,
        lambda_g: float = 1e-3,
        lambda_y: float = 1e3,
        lambda_u: float = 1.0,
        lambda_du: float = 0.1,
        u_min: np.ndarray = None,
        u_max: np.ndarray = None
    ):
        """
        Args:
            T_ini: Initial trajectory length (past horizon)
            N_pred: Prediction horizon
            lambda_g: Regularization on g (for robustness)
            lambda_y: Weight on output tracking error
            lambda_u: Weight on control effort
            lambda_du: Weight on control rate
            u_min: Lower bound on control inputs
            u_max: Upper bound on control inputs
        """
        self.T_ini = T_ini
        self.N_pred = N_pred
        self.lambda_g = lambda_g
        self.lambda_y = lambda_y
        self.lambda_u = lambda_u
        self.lambda_du = lambda_du
        self.u_min = u_min
        self.u_max = u_max
        
        # Data matrices (set by set_data)
        self.Up = None  # Past input Hankel
        self.Yp = None  # Past output Hankel
        self.Uf = None  # Future input Hankel
        self.Yf = None  # Future output Hankel
        
        self.state_dim = None
        self.input_dim = None
        self.n_cols = None
        
    def set_data(self, dataset) -> None:
        """Set data for DeePC from dataset.
        
        Constructs Hankel matrices from the collected data.
        
        Args:
            dataset: Dataset object with states, inputs
        """
        L = self.T_ini + self.N_pred
        
        X = dataset.states
        U = dataset.inputs
        
        self.state_dim = X.shape[1]
        self.input_dim = U.shape[1]
        
        N = len(X)
        self.n_cols = N - L + 1
        
        if self.n_cols <= 0:
            raise ValueError(
                f"Not enough data. Need at least {L+1} samples, got {N}."
            )
        
        # Build Hankel matrices
        # U Hankel
        U_hankel = np.zeros((L * self.input_dim, self.n_cols))
        for i in range(self.n_cols):
            for j in range(L):
                U_hankel[j*self.input_dim:(j+1)*self.input_dim, i] = U[i + j]
                
        # Y (state) Hankel
        Y_hankel = np.zeros((L * self.state_dim, self.n_cols))
        for i in range(self.n_cols):
            for j in range(L):
                Y_hankel[j*self.state_dim:(j+1)*self.state_dim, i] = X[i + j]
        
        # Split into past and future
        self.Up = U_hankel[:self.T_ini * self.input_dim, :]
        self.Uf = U_hankel[self.T_ini * self.input_dim:, :]
        self.Yp = Y_hankel[:self.T_ini * self.state_dim, :]
        self.Yf = Y_hankel[self.T_ini * self.state_dim:, :]
        
    def solve(
        self,
        u_ini: np.ndarray,
        y_ini: np.ndarray,
        y_ref: np.ndarray,
        u_ref: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Solve DeePC optimization problem.
        
        Args:
            u_ini: Initial input trajectory (T_ini, input_dim)
            y_ini: Initial output trajectory (T_ini, state_dim)
            y_ref: Reference output trajectory (N_pred, state_dim)
            u_ref: Reference input trajectory (optional)
            
        Returns:
            (u_optimal, y_predicted, info) tuple
        """
        if not CVXPY_AVAILABLE:
            return self._solve_qp(u_ini, y_ini, y_ref, u_ref)
            
        # Flatten trajectories
        u_ini_flat = u_ini.flatten()
        y_ini_flat = y_ini.flatten()
        y_ref_flat = y_ref.flatten()
        
        if u_ref is None:
            u_ref_flat = np.zeros(self.N_pred * self.input_dim)
        else:
            u_ref_flat = u_ref.flatten()
        
        # Optimization variables
        g = cp.Variable(self.n_cols)
        u_f = cp.Variable(self.N_pred * self.input_dim)
        y_f = cp.Variable(self.N_pred * self.state_dim)
        sigma_y = cp.Variable(self.T_ini * self.state_dim)  # Slack for y_ini
        
        # Constraints: Hankel @ g = [u_ini; y_ini; u_f; y_f]
        constraints = [
            self.Up @ g == u_ini_flat,
            self.Yp @ g == y_ini_flat + sigma_y,  # Soft constraint on initial
            self.Uf @ g == u_f,
            self.Yf @ g == y_f
        ]
        
        # Input constraints
        if self.u_min is not None:
            for i in range(self.N_pred):
                idx = i * self.input_dim
                constraints.append(
                    u_f[idx:idx+self.input_dim] >= self.u_min
                )
        if self.u_max is not None:
            for i in range(self.N_pred):
                idx = i * self.input_dim
                constraints.append(
                    u_f[idx:idx+self.input_dim] <= self.u_max
                )
        
        # Cost function
        cost = 0
        
        # Output tracking
        cost += self.lambda_y * cp.sum_squares(y_f - y_ref_flat)
        
        # Control effort
        cost += self.lambda_u * cp.sum_squares(u_f - u_ref_flat)
        
        # Control rate (smoothness)
        if self.lambda_du > 0:
            for i in range(self.N_pred - 1):
                idx = i * self.input_dim
                idx_next = (i + 1) * self.input_dim
                cost += self.lambda_du * cp.sum_squares(
                    u_f[idx_next:idx_next+self.input_dim] - 
                    u_f[idx:idx+self.input_dim]
                )
            # Also penalize jump from last u_ini
            cost += self.lambda_du * cp.sum_squares(
                u_f[:self.input_dim] - u_ini_flat[-self.input_dim:]
            )
        
        # Regularization on g
        cost += self.lambda_g * cp.sum_squares(g)
        
        # Slack penalty
        cost += 1e4 * cp.sum_squares(sigma_y)
        
        # Solve
        problem = cp.Problem(cp.Minimize(cost), constraints)
        
        try:
            problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        except:
            try:
                problem.solve(solver=cp.ECOS, verbose=False)
            except:
                problem.solve(verbose=False)
        
        if problem.status not in ['optimal', 'optimal_inaccurate']:
            # Fallback to simpler solution
            return self._solve_fallback(u_ini, y_ini, y_ref, u_ref)
        
        # Extract solution
        u_opt = u_f.value.reshape(self.N_pred, self.input_dim)
        y_pred = y_f.value.reshape(self.N_pred, self.state_dim)
        
        info = {
            'status': problem.status,
            'cost': problem.value,
            'g_norm': np.linalg.norm(g.value) if g.value is not None else None
        }
        
        return u_opt, y_pred, info
    
    def _solve_qp(
        self,
        u_ini: np.ndarray,
        y_ini: np.ndarray,
        y_ref: np.ndarray,
        u_ref: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Solve using simple QP approach when cvxpy not available."""
        # Stack data matrix
        n_u_ini = self.T_ini * self.input_dim
        n_y_ini = self.T_ini * self.state_dim
        n_u_f = self.N_pred * self.input_dim
        n_y_f = self.N_pred * self.state_dim
        
        # [Up; Yp; Uf; Yf] @ g = [u_ini; y_ini; u_f; y_f]
        H = np.vstack([self.Up, self.Yp, self.Uf, self.Yf])
        
        # Initial conditions
        ini = np.concatenate([u_ini.flatten(), y_ini.flatten()])
        
        # Reference
        if u_ref is None:
            u_ref = np.zeros((self.N_pred, self.input_dim))
            
        ref = np.concatenate([u_ref.flatten(), y_ref.flatten()])
        target = np.concatenate([ini, ref])
        
        # Build cost matrices for g
        # Cost: ||Hg - target||^2 + lambda_g ||g||^2
        # Weight initial conditions heavily
        W = np.diag(np.concatenate([
            np.ones(n_u_ini) * 1e6,  # Match u_ini exactly
            np.ones(n_y_ini) * 1e6,  # Match y_ini exactly
            np.ones(n_u_f) * self.lambda_u,  # u_f tracking
            np.ones(n_y_f) * self.lambda_y   # y_f tracking
        ]))
        
        # Solve: (H^T W H + lambda_g I) g = H^T W target
        Q = H.T @ W @ H + self.lambda_g * np.eye(self.n_cols)
        p = H.T @ W @ target
        
        g = np.linalg.solve(Q, p)
        
        # Extract trajectories
        result = H @ g
        u_f = result[n_u_ini + n_y_ini:n_u_ini + n_y_ini + n_u_f]
        y_f = result[n_u_ini + n_y_ini + n_u_f:]
        
        u_opt = u_f.reshape(self.N_pred, self.input_dim)
        y_pred = y_f.reshape(self.N_pred, self.state_dim)
        
        # Apply input constraints
        if self.u_min is not None:
            u_opt = np.maximum(u_opt, self.u_min)
        if self.u_max is not None:
            u_opt = np.minimum(u_opt, self.u_max)
        
        info = {
            'status': 'qp_fallback',
            'cost': np.linalg.norm(H @ g - target),
            'g_norm': np.linalg.norm(g)
        }
        
        return u_opt, y_pred, info
    
    def _solve_fallback(
        self,
        u_ini: np.ndarray,
        y_ini: np.ndarray,
        y_ref: np.ndarray,
        u_ref: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Simple fallback when optimization fails."""
        # Return reference as open-loop solution
        if u_ref is None:
            u_opt = np.zeros((self.N_pred, self.input_dim))
        else:
            u_opt = u_ref.copy()
            
        y_pred = y_ref.copy()
        
        info = {'status': 'fallback', 'cost': np.inf, 'g_norm': None}
        
        return u_opt, y_pred, info


class DeePCController:
    """DeePC as a receding horizon controller.
    
    Wraps DeePC to provide a standard controller interface that
    maintains trajectory history and applies first control input.
    """
    
    def __init__(
        self,
        dataset,
        T_ini: int = 5,
        N_pred: int = 10,
        lambda_g: float = 1e-3,
        lambda_y: float = 1e3,
        lambda_u: float = 1.0,
        lambda_du: float = 0.1,
        u_min: np.ndarray = None,
        u_max: np.ndarray = None
    ):
        """
        Args:
            dataset: Dataset for Hankel construction
            T_ini: Past horizon
            N_pred: Prediction horizon
            lambda_g: Regularization on g
            lambda_y: Output tracking weight
            lambda_u: Control effort weight
            lambda_du: Control rate weight
            u_min: Input lower bounds
            u_max: Input upper bounds
        """
        self.deepc = DeePC(
            T_ini=T_ini,
            N_pred=N_pred,
            lambda_g=lambda_g,
            lambda_y=lambda_y,
            lambda_u=lambda_u,
            lambda_du=lambda_du,
            u_min=u_min,
            u_max=u_max
        )
        
        self.deepc.set_data(dataset)
        
        self.T_ini = T_ini
        self.N_pred = N_pred
        self.state_dim = self.deepc.state_dim
        self.input_dim = self.deepc.input_dim
        
        # Trajectory history
        self.u_history = []
        self.y_history = []
        
        # Reference trajectory generator
        self.ref_generator = None
        self.target = None
        
        # Diagnostics
        self.last_info = {}
        
    def reset(self):
        """Reset controller state."""
        self.u_history = []
        self.y_history = []
        self.last_info = {}
        
    def set_target(self, target: np.ndarray):
        """Set target state for regulation."""
        self.target = np.array(target)
        
    def set_reference_generator(self, generator):
        """Set trajectory generator for reference."""
        self.ref_generator = generator
        
    def compute_control(
        self, 
        x: np.ndarray, 
        t: float
    ) -> Tuple[np.ndarray, Dict]:
        """Compute control using DeePC.
        
        Args:
            x: Current state
            t: Current time
            
        Returns:
            (control_input, info_dict)
        """
        # Add current state to history
        self.y_history.append(x.copy())
        
        # Build reference trajectory
        if self.ref_generator is not None:
            times = t + np.arange(self.N_pred) * 0.01  # Assume dt=0.01
            y_ref = np.array([
                self.ref_generator.get_reference(ti)['state'] 
                for ti in times
            ])
        elif self.target is not None:
            y_ref = np.tile(self.target, (self.N_pred, 1))
        else:
            y_ref = np.tile(x, (self.N_pred, 1))
        
        # Check if we have enough history
        if len(self.y_history) < self.T_ini:
            # Not enough history - use simple proportional control
            if self.target is not None:
                error = self.target - x
                u = 0.5 * error[:self.input_dim]  # Simple P control
            else:
                u = np.zeros(self.input_dim)
                
            self.u_history.append(u)
            return u, {'status': 'warmup', 'history_len': len(self.y_history)}
        
        # Build initial trajectories from history
        y_ini = np.array(self.y_history[-self.T_ini:])
        
        if len(self.u_history) >= self.T_ini:
            u_ini = np.array(self.u_history[-self.T_ini:])
        else:
            # Pad with zeros
            u_ini = np.zeros((self.T_ini, self.input_dim))
            if len(self.u_history) > 0:
                u_ini[-len(self.u_history):] = np.array(self.u_history)
        
        # Solve DeePC
        u_opt, y_pred, info = self.deepc.solve(u_ini, y_ini, y_ref)
        
        # Apply first control
        u = u_opt[0]
        
        self.u_history.append(u)
        self.last_info = info
        self.last_info['y_pred'] = y_pred
        self.last_info['y_ref'] = y_ref
        
        return u, self.last_info
    
    def get_lyapunov_value(self, x: np.ndarray) -> Optional[float]:
        """Return cost as Lyapunov-like value."""
        if 'cost' in self.last_info:
            return self.last_info['cost']
        return None
