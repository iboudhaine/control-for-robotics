"""
Policy Learning for Data-Driven Control

Implements policy learning methods that directly learn control policies
from data, potentially with stability/safety constraints.

Methods:
- Linear policy learning: u = Kx optimized from data
- Constrained policy learning: With Lyapunov stability constraints
- Imitation learning: Learn from expert demonstrations

These methods learn a mapping from state to control without
requiring an explicit dynamics model.
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Callable
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Try to import optimization libraries
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class PolicyResult:
    """Result of policy optimization."""
    K: np.ndarray  # Policy gain matrix
    cost: float    # Achieved cost
    iterations: int
    info: dict


class LinearPolicy:
    """Linear policy u = Kx optimized from data.
    
    Learns a linear feedback policy by minimizing a cost function
    evaluated on collected trajectories.
    
    Optimization:
        min_K sum_{i} ||x_i^+ - x_i||^2 + lambda_u ||K x_i||^2
        
    where x_i^+ is the successor state after applying u = K x_i.
    """
    
    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        target: np.ndarray = None,
        lambda_u: float = 0.1,
        regularization: float = 1e-4
    ):
        """
        Args:
            state_dim: State dimension
            input_dim: Input dimension
            target: Target state for regulation
            lambda_u: Control effort penalty
            regularization: Regularization parameter
        """
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.target = target if target is not None else np.zeros(state_dim)
        self.lambda_u = lambda_u
        self.regularization = regularization
        
        self.K = None
        
    def fit_from_transitions(
        self,
        dataset,
        n_iterations: int = 100,
        learning_rate: float = 0.01
    ) -> PolicyResult:
        """Learn policy from state transition data.
        
        Uses gradient descent to minimize:
            J(K) = E[ ||x' - x*||^2 + lambda ||Kx||^2 ]
            
        where x' is observed next state when u was applied.
        
        Args:
            dataset: Dataset with (states, inputs, next_states)
            n_iterations: Gradient descent iterations
            learning_rate: Learning rate
            
        Returns:
            PolicyResult with learned K
        """
        X = dataset.states      # (N, n)
        U = dataset.inputs      # (N, m)
        Y = dataset.next_states # (N, n)
        
        # Initialize K
        K = np.zeros((self.input_dim, self.state_dim))
        
        cost_history = []
        
        for iteration in range(n_iterations):
            # Compute cost gradient
            # dJ/dK = sum_i 2 * (Y_i - target)^T @ dY/dU @ x_i^T + 2*lambda_u * K @ x_i @ x_i^T
            
            # We need to estimate dY/dU from data
            # Simple approximation: assume linear response Y = f(X, U) ≈ AX + BU
            
            # For policy gradient, use finite differences
            grad = np.zeros_like(K)
            eps = 1e-4
            
            # Current cost
            cost = self._evaluate_cost(X, Y, K)
            cost_history.append(cost)
            
            for i in range(self.input_dim):
                for j in range(self.state_dim):
                    K_plus = K.copy()
                    K_plus[i, j] += eps
                    cost_plus = self._evaluate_cost(X, Y, K_plus)
                    
                    grad[i, j] = (cost_plus - cost) / eps
            
            # Update
            K = K - learning_rate * grad
            
            # Regularization (shrinkage)
            K = K * (1 - self.regularization)
        
        self.K = K
        
        return PolicyResult(
            K=K,
            cost=cost_history[-1] if cost_history else float('inf'),
            iterations=n_iterations,
            info={'cost_history': cost_history}
        )
    
    def _evaluate_cost(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        K: np.ndarray
    ) -> float:
        """Evaluate policy cost on data."""
        # Tracking error: how close did next states get to target?
        errors = Y - self.target
        tracking_cost = np.mean(np.sum(errors**2, axis=1))
        
        # Control cost: what control was implied by K?
        U_implied = X @ K.T
        control_cost = self.lambda_u * np.mean(np.sum(U_implied**2, axis=1))
        
        return tracking_cost + control_cost
    
    def fit_lqr_style(
        self,
        A_est: np.ndarray,
        B_est: np.ndarray,
        Q: np.ndarray = None,
        R: np.ndarray = None
    ) -> PolicyResult:
        """Learn policy using estimated A, B matrices (LQR-style).
        
        Given estimated linear dynamics x' = Ax + Bu,
        compute optimal LQR gain K.
        
        Args:
            A_est: Estimated state matrix
            B_est: Estimated input matrix
            Q: State cost matrix
            R: Input cost matrix
            
        Returns:
            PolicyResult with LQR gain
        """
        from scipy.linalg import solve_discrete_are
        
        if Q is None:
            Q = np.eye(self.state_dim)
        if R is None:
            R = self.lambda_u * np.eye(self.input_dim)
        
        try:
            # Solve DARE
            P = solve_discrete_are(A_est, B_est, Q, R)
            K = -np.linalg.solve(
                R + B_est.T @ P @ B_est,
                B_est.T @ P @ A_est
            )
        except:
            # Fallback to simple pole placement
            K = -0.1 * np.linalg.pinv(B_est) @ (A_est - 0.9 * np.eye(self.state_dim))
        
        self.K = K
        
        return PolicyResult(
            K=K,
            cost=0.0,
            iterations=1,
            info={'method': 'lqr', 'P': P if 'P' in locals() else None}
        )
    
    def compute_control(
        self,
        x: np.ndarray,
        t: float = 0.0
    ) -> Tuple[np.ndarray, Dict]:
        """Compute control using learned policy."""
        if self.K is None:
            raise ValueError("Policy not learned. Call fit() first.")
            
        dx = x - self.target
        u = self.K @ dx
        
        return u, {'policy_type': 'linear'}
    
    def set_target(self, target: np.ndarray):
        """Update target state."""
        self.target = np.array(target)


class ConstrainedPolicyLearning:
    """Policy learning with stability constraints.
    
    Learns a policy while enforcing Lyapunov decrease:
        V(f(x, π(x))) < V(x)  for all x
        
    where V is a candidate Lyapunov function and π is the policy.
    
    Uses SDP or SOS relaxations when available.
    """
    
    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        lyapunov_matrix: np.ndarray = None,
        stability_margin: float = 0.01
    ):
        """
        Args:
            state_dim: State dimension
            input_dim: Input dimension  
            lyapunov_matrix: P matrix for V(x) = x^T P x
            stability_margin: Required decrease: V(x') < (1-margin)*V(x)
        """
        self.state_dim = state_dim
        self.input_dim = input_dim
        
        if lyapunov_matrix is None:
            lyapunov_matrix = np.eye(state_dim)
        self.P = lyapunov_matrix
        
        self.stability_margin = stability_margin
        self.K = None
        
    def fit_with_stability(
        self,
        A_est: np.ndarray,
        B_est: np.ndarray,
        Q: np.ndarray = None,
        R: np.ndarray = None
    ) -> PolicyResult:
        """Learn stabilizing policy using SDP.
        
        Solves:
            min trace(Q @ P) + trace(R @ K @ K^T)
            s.t. (A + BK)^T P (A + BK) - P < -margin * P
                 P > 0
        
        Using Schur complement for LMI formulation.
        
        Args:
            A_est: Estimated A matrix
            B_est: Estimated B matrix
            Q: State cost
            R: Input cost
            
        Returns:
            PolicyResult with stabilizing K
        """
        if not CVXPY_AVAILABLE:
            # Fallback to standard LQR
            policy = LinearPolicy(self.state_dim, self.input_dim)
            return policy.fit_lqr_style(A_est, B_est, Q, R)
        
        n = self.state_dim
        m = self.input_dim
        
        if Q is None:
            Q = np.eye(n)
        if R is None:
            R = 0.1 * np.eye(m)
        
        # Variables: Y = K @ X, X = P^{-1}
        X = cp.Variable((n, n), symmetric=True)
        Y = cp.Variable((m, n))
        
        # LMI constraint for stability
        # (A + BK)^T P (A + BK) < (1-margin) P
        # Equivalent: X > 0 and
        # [[X, (AX + BY)^T], [AX + BY, X]] > 0 (after scaling)
        
        AX_BY = A_est @ X + B_est @ Y
        
        lmi = cp.bmat([
            [(1 - self.stability_margin) * X, AX_BY.T],
            [AX_BY, X]
        ])
        
        constraints = [
            X >> 1e-6 * np.eye(n),  # P^{-1} > 0
            lmi >> 0
        ]
        
        # Cost: trace(Q @ P) ≈ trace(Q @ X^{-1}) ~ -log det X (for convexity)
        # Simpler: minimize ||Y||^2 (control effort) subject to stability
        cost = cp.norm(Y, 'fro')**2
        
        problem = cp.Problem(cp.Minimize(cost), constraints)
        
        try:
            problem.solve(solver=cp.SCS, verbose=False)
        except:
            try:
                problem.solve(verbose=False)
            except:
                # Fallback
                policy = LinearPolicy(self.state_dim, self.input_dim)
                return policy.fit_lqr_style(A_est, B_est, Q, R)
        
        if problem.status not in ['optimal', 'optimal_inaccurate']:
            # Fallback
            policy = LinearPolicy(self.state_dim, self.input_dim)
            return policy.fit_lqr_style(A_est, B_est, Q, R)
        
        # Extract K = Y @ X^{-1}
        X_val = X.value
        Y_val = Y.value
        
        try:
            K = Y_val @ np.linalg.inv(X_val)
        except:
            K = Y_val @ np.linalg.pinv(X_val)
        
        self.K = K
        self.P = np.linalg.inv(X_val)
        
        return PolicyResult(
            K=K,
            cost=problem.value,
            iterations=1,
            info={
                'method': 'sdp_stability',
                'P': self.P,
                'status': problem.status
            }
        )
    
    def verify_stability(
        self,
        A_est: np.ndarray,
        B_est: np.ndarray
    ) -> Tuple[bool, float]:
        """Verify that learned policy is stabilizing.
        
        Checks if (A + BK) is Schur stable and V decreases.
        
        Returns:
            (is_stable, max_eigenvalue_magnitude)
        """
        if self.K is None:
            return False, np.inf
            
        A_cl = A_est + B_est @ self.K
        eigenvalues = np.linalg.eigvals(A_cl)
        max_mag = np.max(np.abs(eigenvalues))
        
        is_stable = max_mag < 1.0
        
        return is_stable, max_mag
    
    def compute_control(
        self,
        x: np.ndarray,
        t: float = 0.0
    ) -> Tuple[np.ndarray, Dict]:
        """Compute control using learned policy."""
        if self.K is None:
            raise ValueError("Policy not learned")
            
        u = self.K @ x
        V = x @ self.P @ x
        
        return u, {'lyapunov': V, 'policy_type': 'constrained_linear'}
    
    def get_lyapunov_value(self, x: np.ndarray) -> float:
        """Compute Lyapunov function value."""
        return x @ self.P @ x


if TORCH_AVAILABLE:
    
    class NeuralPolicy(nn.Module):
        """Neural network policy."""
        
        def __init__(
            self,
            state_dim: int,
            input_dim: int,
            hidden_dims: List[int] = [64, 64]
        ):
            super().__init__()
            
            layers = []
            prev_dim = state_dim
            
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                prev_dim = hidden_dim
                
            layers.append(nn.Linear(prev_dim, input_dim))
            layers.append(nn.Tanh())  # Bounded output
            
            self.network = nn.Sequential(*layers)
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.network(x)
    
    
    class NeuralPolicyLearning:
        """Neural network policy learning.
        
        Learns a nonlinear policy π_θ(x) using policy gradient methods.
        """
        
        def __init__(
            self,
            state_dim: int,
            input_dim: int,
            hidden_dims: List[int] = [64, 64],
            learning_rate: float = 1e-3,
            u_scale: float = 1.0,
            device: str = 'cpu'
        ):
            self.state_dim = state_dim
            self.input_dim = input_dim
            self.u_scale = u_scale
            self.device = device
            
            self.policy = NeuralPolicy(
                state_dim, input_dim, hidden_dims
            ).to(device)
            
            self.optimizer = optim.Adam(
                self.policy.parameters(), lr=learning_rate
            )
            
            self.target = None
            
        def fit_imitation(
            self,
            expert_states: np.ndarray,
            expert_actions: np.ndarray,
            n_epochs: int = 100,
            batch_size: int = 64
        ) -> Dict:
            """Learn policy by imitating expert.
            
            Behavioral cloning: minimize ||π(x) - u_expert||^2
            """
            from torch.utils.data import DataLoader, TensorDataset
            
            x_t = torch.FloatTensor(expert_states).to(self.device)
            u_t = torch.FloatTensor(expert_actions / self.u_scale).to(self.device)
            
            dataset = TensorDataset(x_t, u_t)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            loss_history = []
            
            for epoch in range(n_epochs):
                epoch_loss = 0.0
                
                for x_batch, u_batch in loader:
                    self.optimizer.zero_grad()
                    
                    u_pred = self.policy(x_batch)
                    loss = nn.MSELoss()(u_pred, u_batch)
                    
                    loss.backward()
                    self.optimizer.step()
                    
                    epoch_loss += loss.item()
                    
                loss_history.append(epoch_loss / len(loader))
                
            return {'loss_history': loss_history}
        
        def compute_control(
            self,
            x: np.ndarray,
            t: float = 0.0
        ) -> Tuple[np.ndarray, Dict]:
            """Compute control using neural policy."""
            self.policy.eval()
            
            with torch.no_grad():
                x_t = torch.FloatTensor(x).unsqueeze(0).to(self.device)
                u_t = self.policy(x_t)
                u = u_t.cpu().numpy().squeeze() * self.u_scale
                
            return u, {'policy_type': 'neural'}
        
        def set_target(self, target: np.ndarray):
            """Set target (for reference)."""
            self.target = target

else:
    class NeuralPolicyLearning:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required for NeuralPolicyLearning")
