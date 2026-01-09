"""
iLQR with Learned Dynamics

Implements iterative Linear Quadratic Regulator (iLQR) using
learned dynamics models (neural networks or other).

iLQR iteratively:
1. Forward simulate using current control sequence
2. Linearize dynamics along trajectory
3. Backward pass to compute optimal feedback gains
4. Forward pass with line search to improve trajectory

This implementation uses learned dynamics models from neural_dynamics.py
or any model with predict() and get_jacobians() methods.
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Callable
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


class iLQRLearned:
    """iLQR using learned dynamics model.
    
    Solves finite-horizon optimal control:
        min sum_{t=0}^{T-1} l(x_t, u_t) + l_f(x_T)
        s.t. x_{t+1} = f(x_t, u_t)  (learned)
    
    where f is a learned dynamics model.
    """
    
    def __init__(
        self,
        dynamics_model,
        state_dim: int,
        input_dim: int,
        Q: np.ndarray = None,
        R: np.ndarray = None,
        Qf: np.ndarray = None,
        max_iterations: int = 50,
        convergence_threshold: float = 1e-4,
        regularization: float = 1e-6,
        line_search_beta: float = 0.5,
        u_min: np.ndarray = None,
        u_max: np.ndarray = None
    ):
        """
        Args:
            dynamics_model: Learned dynamics with predict() and get_jacobians()
            state_dim: State dimension
            input_dim: Input dimension
            Q: Running state cost matrix
            R: Running input cost matrix
            Qf: Terminal state cost matrix
            max_iterations: Max iLQR iterations
            convergence_threshold: Cost improvement threshold
            regularization: Regularization for backward pass
            line_search_beta: Line search decay factor
            u_min: Input lower bounds
            u_max: Input upper bounds
        """
        self.dynamics = dynamics_model
        self.state_dim = state_dim
        self.input_dim = input_dim
        
        # Cost matrices
        self.Q = Q if Q is not None else np.eye(state_dim)
        self.R = R if R is not None else 0.1 * np.eye(input_dim)
        self.Qf = Qf if Qf is not None else 10 * np.eye(state_dim)
        
        # Algorithm parameters
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.regularization = regularization
        self.line_search_beta = line_search_beta
        
        # Input constraints
        self.u_min = u_min
        self.u_max = u_max
        
    def _running_cost(
        self, 
        x: np.ndarray, 
        u: np.ndarray, 
        x_ref: np.ndarray,
        u_ref: np.ndarray = None
    ) -> float:
        """Compute running cost l(x, u)."""
        dx = x - x_ref
        if u_ref is None:
            u_ref = np.zeros(self.input_dim)
        du = u - u_ref
        
        return 0.5 * (dx @ self.Q @ dx + du @ self.R @ du)
    
    def _terminal_cost(
        self, 
        x: np.ndarray, 
        x_ref: np.ndarray
    ) -> float:
        """Compute terminal cost l_f(x)."""
        dx = x - x_ref
        return 0.5 * dx @ self.Qf @ dx
    
    def _cost_derivatives(
        self,
        x: np.ndarray,
        u: np.ndarray,
        x_ref: np.ndarray,
        u_ref: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute cost derivatives for running cost.
        
        Returns:
            (l_x, l_u, l_xx, l_uu, l_xu) derivatives
        """
        dx = x - x_ref
        if u_ref is None:
            u_ref = np.zeros(self.input_dim)
        du = u - u_ref
        
        l_x = self.Q @ dx
        l_u = self.R @ du
        l_xx = self.Q
        l_uu = self.R
        l_xu = np.zeros((self.state_dim, self.input_dim))
        
        return l_x, l_u, l_xx, l_uu, l_xu
    
    def _terminal_cost_derivatives(
        self,
        x: np.ndarray,
        x_ref: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute terminal cost derivatives.
        
        Returns:
            (l_x, l_xx) derivatives
        """
        dx = x - x_ref
        l_x = self.Qf @ dx
        l_xx = self.Qf
        
        return l_x, l_xx
    
    def _forward_pass(
        self,
        x0: np.ndarray,
        u_seq: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Forward simulate trajectory.
        
        Args:
            x0: Initial state
            u_seq: Control sequence (T, input_dim)
            
        Returns:
            (x_seq, total_cost) where x_seq is (T+1, state_dim)
        """
        T = len(u_seq)
        x_seq = np.zeros((T + 1, self.state_dim))
        x_seq[0] = x0
        
        for t in range(T):
            x_seq[t + 1] = self.dynamics.predict(x_seq[t], u_seq[t])
            
        return x_seq
    
    def _backward_pass(
        self,
        x_seq: np.ndarray,
        u_seq: np.ndarray,
        x_ref_seq: np.ndarray,
        u_ref_seq: np.ndarray = None
    ) -> Tuple[List[np.ndarray], List[np.ndarray], float]:
        """Backward pass to compute feedback gains.
        
        Returns:
            (k_seq, K_seq, expected_improvement)
        """
        T = len(u_seq)
        
        k_seq = [None] * T
        K_seq = [None] * T
        
        # Terminal cost derivatives
        V_x, V_xx = self._terminal_cost_derivatives(x_seq[T], x_ref_seq[T])
        
        expected_improvement = 0.0
        
        for t in range(T - 1, -1, -1):
            x, u = x_seq[t], u_seq[t]
            x_ref = x_ref_seq[t]
            u_ref = u_ref_seq[t] if u_ref_seq is not None else None
            
            # Get dynamics Jacobians
            try:
                A, B = self.dynamics.get_jacobians(x, u)
            except:
                # Numerical differentiation fallback
                A, B = self._numerical_jacobians(x, u)
            
            # Cost derivatives
            l_x, l_u, l_xx, l_uu, l_xu = self._cost_derivatives(x, u, x_ref, u_ref)
            
            # Q-function approximation
            Q_x = l_x + A.T @ V_x
            Q_u = l_u + B.T @ V_x
            Q_xx = l_xx + A.T @ V_xx @ A
            Q_uu = l_uu + B.T @ V_xx @ B
            Q_ux = l_xu.T + B.T @ V_xx @ A
            
            # Regularization
            Q_uu_reg = Q_uu + self.regularization * np.eye(self.input_dim)
            
            # Compute gains
            try:
                Q_uu_inv = np.linalg.inv(Q_uu_reg)
            except np.linalg.LinAlgError:
                Q_uu_inv = np.linalg.pinv(Q_uu_reg)
                
            k = -Q_uu_inv @ Q_u
            K = -Q_uu_inv @ Q_ux
            
            k_seq[t] = k
            K_seq[t] = K
            
            # Update V
            V_x = Q_x + K.T @ Q_uu @ k + K.T @ Q_u + Q_ux.T @ k
            V_xx = Q_xx + K.T @ Q_uu @ K + K.T @ Q_ux + Q_ux.T @ K
            V_xx = 0.5 * (V_xx + V_xx.T)  # Symmetrize
            
            expected_improvement += -Q_u @ k - 0.5 * k @ Q_uu @ k
            
        return k_seq, K_seq, expected_improvement
    
    def _numerical_jacobians(
        self,
        x: np.ndarray,
        u: np.ndarray,
        eps: float = 1e-5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Jacobians numerically."""
        A = np.zeros((self.state_dim, self.state_dim))
        B = np.zeros((self.state_dim, self.input_dim))
        
        f0 = self.dynamics.predict(x, u)
        
        for i in range(self.state_dim):
            x_plus = x.copy()
            x_plus[i] += eps
            A[:, i] = (self.dynamics.predict(x_plus, u) - f0) / eps
            
        for i in range(self.input_dim):
            u_plus = u.copy()
            u_plus[i] += eps
            B[:, i] = (self.dynamics.predict(x, u_plus) - f0) / eps
            
        return A, B
    
    def _line_search(
        self,
        x0: np.ndarray,
        x_seq: np.ndarray,
        u_seq: np.ndarray,
        k_seq: List[np.ndarray],
        K_seq: List[np.ndarray],
        x_ref_seq: np.ndarray,
        u_ref_seq: np.ndarray,
        current_cost: float,
        expected_improvement: float
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Line search to find best step size."""
        T = len(u_seq)
        alpha = 1.0
        
        for _ in range(10):  # Max line search iterations
            # Forward pass with feedback
            x_new = np.zeros((T + 1, self.state_dim))
            u_new = np.zeros((T, self.input_dim))
            
            x_new[0] = x0
            
            for t in range(T):
                dx = x_new[t] - x_seq[t]
                u_new[t] = u_seq[t] + alpha * k_seq[t] + K_seq[t] @ dx
                
                # Apply constraints
                if self.u_min is not None:
                    u_new[t] = np.maximum(u_new[t], self.u_min)
                if self.u_max is not None:
                    u_new[t] = np.minimum(u_new[t], self.u_max)
                    
                x_new[t + 1] = self.dynamics.predict(x_new[t], u_new[t])
            
            # Compute cost
            new_cost = 0.0
            for t in range(T):
                u_ref = u_ref_seq[t] if u_ref_seq is not None else None
                new_cost += self._running_cost(x_new[t], u_new[t], x_ref_seq[t], u_ref)
            new_cost += self._terminal_cost(x_new[T], x_ref_seq[T])
            
            # Check improvement
            improvement = current_cost - new_cost
            expected = alpha * expected_improvement
            
            if improvement > 0.1 * expected or alpha < 0.01:
                return x_new, u_new, new_cost, alpha
                
            alpha *= self.line_search_beta
            
        return x_seq, u_seq, current_cost, 0.0
    
    def solve(
        self,
        x0: np.ndarray,
        x_ref_seq: np.ndarray,
        u_ref_seq: np.ndarray = None,
        u_init: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Solve optimal control problem.
        
        Args:
            x0: Initial state
            x_ref_seq: Reference state sequence (T+1, state_dim)
            u_ref_seq: Reference input sequence (T, input_dim), optional
            u_init: Initial control guess (T, input_dim)
            
        Returns:
            (u_optimal, x_trajectory, info_dict)
        """
        T = len(x_ref_seq) - 1
        
        # Initialize control sequence
        if u_init is not None:
            u_seq = u_init.copy()
        else:
            u_seq = np.zeros((T, self.input_dim))
            
        # Forward pass
        x_seq = self._forward_pass(x0, u_seq)
        
        # Compute initial cost
        cost = 0.0
        for t in range(T):
            u_ref = u_ref_seq[t] if u_ref_seq is not None else None
            cost += self._running_cost(x_seq[t], u_seq[t], x_ref_seq[t], u_ref)
        cost += self._terminal_cost(x_seq[T], x_ref_seq[T])
        
        cost_history = [cost]
        
        for iteration in range(self.max_iterations):
            # Backward pass
            k_seq, K_seq, expected_improvement = self._backward_pass(
                x_seq, u_seq, x_ref_seq, u_ref_seq
            )
            
            # Line search
            x_seq_new, u_seq_new, new_cost, alpha = self._line_search(
                x0, x_seq, u_seq, k_seq, K_seq,
                x_ref_seq, u_ref_seq, cost, expected_improvement
            )
            
            # Check convergence
            improvement = (cost - new_cost) / (abs(cost) + 1e-8)
            
            x_seq = x_seq_new
            u_seq = u_seq_new
            cost = new_cost
            cost_history.append(cost)
            
            if improvement < self.convergence_threshold:
                break
                
        info = {
            'iterations': iteration + 1,
            'cost': cost,
            'cost_history': cost_history,
            'converged': improvement < self.convergence_threshold,
            'final_k': k_seq,
            'final_K': K_seq
        }
        
        return u_seq, x_seq, info


class TrajectoryOptimizer:
    """High-level trajectory optimizer using iLQR.
    
    Provides convenient interface for trajectory tracking tasks.
    """
    
    def __init__(
        self,
        dynamics_model,
        state_dim: int,
        input_dim: int,
        horizon: int = 20,
        Q: np.ndarray = None,
        R: np.ndarray = None,
        u_min: np.ndarray = None,
        u_max: np.ndarray = None
    ):
        """
        Args:
            dynamics_model: Learned dynamics model
            state_dim: State dimension
            input_dim: Input dimension
            horizon: Planning horizon
            Q: State cost matrix
            R: Input cost matrix
            u_min: Input lower bounds
            u_max: Input upper bounds
        """
        self.ilqr = iLQRLearned(
            dynamics_model=dynamics_model,
            state_dim=state_dim,
            input_dim=input_dim,
            Q=Q,
            R=R,
            u_min=u_min,
            u_max=u_max
        )
        
        self.horizon = horizon
        self.state_dim = state_dim
        self.input_dim = input_dim
        
        # For receding horizon
        self.u_plan = None
        self.target = None
        self.ref_generator = None
        
    def set_target(self, target: np.ndarray):
        """Set regulation target."""
        self.target = np.array(target)
        
    def set_reference_generator(self, generator):
        """Set trajectory generator."""
        self.ref_generator = generator
        
    def compute_control(
        self,
        x: np.ndarray,
        t: float
    ) -> Tuple[np.ndarray, Dict]:
        """Compute control using MPC with iLQR.
        
        Args:
            x: Current state
            t: Current time
            
        Returns:
            (control_input, info_dict)
        """
        # Build reference trajectory
        if self.ref_generator is not None:
            dt = 0.01  # Assume time step
            times = t + np.arange(self.horizon + 1) * dt
            x_ref_seq = np.array([
                self.ref_generator.get_reference(ti)['state']
                for ti in times
            ])
        elif self.target is not None:
            x_ref_seq = np.tile(self.target, (self.horizon + 1, 1))
        else:
            x_ref_seq = np.tile(x, (self.horizon + 1, 1))
        
        # Warm start from previous solution
        u_init = None
        if self.u_plan is not None and len(self.u_plan) > 1:
            u_init = np.vstack([self.u_plan[1:], self.u_plan[-1:]])
        
        # Solve iLQR
        u_opt, x_traj, info = self.ilqr.solve(
            x0=x,
            x_ref_seq=x_ref_seq,
            u_init=u_init
        )
        
        # Store plan for warm start
        self.u_plan = u_opt
        
        # Apply first control
        u = u_opt[0]
        
        info['x_predicted'] = x_traj
        info['x_reference'] = x_ref_seq
        
        return u, info
    
    def get_lyapunov_value(self, x: np.ndarray) -> Optional[float]:
        """Return optimal cost as Lyapunov value."""
        if self.target is not None:
            dx = x - self.target
            return 0.5 * dx @ self.ilqr.Q @ dx
        return None


class MPCWithLearnedModel:
    """Simple MPC using learned dynamics model.
    
    Shooting method MPC that uses learned dynamics for prediction.
    Simpler than iLQR but may be less efficient.
    """
    
    def __init__(
        self,
        dynamics_model,
        state_dim: int,
        input_dim: int,
        horizon: int = 10,
        Q: np.ndarray = None,
        R: np.ndarray = None,
        u_min: np.ndarray = None,
        u_max: np.ndarray = None,
        n_iterations: int = 20
    ):
        self.dynamics = dynamics_model
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.horizon = horizon
        
        self.Q = Q if Q is not None else np.eye(state_dim)
        self.R = R if R is not None else 0.1 * np.eye(input_dim)
        
        self.u_min = u_min
        self.u_max = u_max
        self.n_iterations = n_iterations
        
        self.target = None
        self.u_plan = None
        
    def set_target(self, target: np.ndarray):
        """Set target state."""
        self.target = np.array(target)
        
    def compute_control(
        self,
        x: np.ndarray,
        t: float = 0.0
    ) -> Tuple[np.ndarray, Dict]:
        """Compute control using gradient descent on shooting cost."""
        if self.target is None:
            self.target = x
            
        # Initialize
        if self.u_plan is not None:
            u_seq = np.vstack([self.u_plan[1:], self.u_plan[-1:]])
        else:
            u_seq = np.zeros((self.horizon, self.input_dim))
        
        lr = 0.1
        
        for _ in range(self.n_iterations):
            # Forward pass
            x_seq = [x]
            for t in range(self.horizon):
                x_next = self.dynamics.predict(x_seq[-1], u_seq[t])
                x_seq.append(x_next)
            x_seq = np.array(x_seq)
            
            # Compute cost gradient via finite differences
            grad = np.zeros_like(u_seq)
            eps = 1e-4
            
            for t in range(self.horizon):
                for i in range(self.input_dim):
                    u_plus = u_seq.copy()
                    u_plus[t, i] += eps
                    
                    u_minus = u_seq.copy()
                    u_minus[t, i] -= eps
                    
                    cost_plus = self._compute_cost(x, u_plus)
                    cost_minus = self._compute_cost(x, u_minus)
                    
                    grad[t, i] = (cost_plus - cost_minus) / (2 * eps)
            
            # Update
            u_seq = u_seq - lr * grad
            
            # Clip
            if self.u_min is not None:
                u_seq = np.maximum(u_seq, self.u_min)
            if self.u_max is not None:
                u_seq = np.minimum(u_seq, self.u_max)
        
        self.u_plan = u_seq
        
        return u_seq[0], {'cost': self._compute_cost(x, u_seq)}
    
    def _compute_cost(
        self,
        x0: np.ndarray,
        u_seq: np.ndarray
    ) -> float:
        """Compute total cost for control sequence."""
        cost = 0.0
        x = x0
        
        for t in range(len(u_seq)):
            dx = x - self.target
            cost += 0.5 * dx @ self.Q @ dx + 0.5 * u_seq[t] @ self.R @ u_seq[t]
            x = self.dynamics.predict(x, u_seq[t])
            
        # Terminal cost
        dx = x - self.target
        cost += 5 * dx @ self.Q @ dx
        
        return cost
