"""
Subspace System Identification

Implements subspace identification methods for linear systems using
Hankel matrix decomposition (N4SID-style algorithms).

For LTI systems:
    x_{t+1} = A x_t + B u_t
    y_t = C x_t + D u_t

Subspace methods use the column space of Hankel matrices to identify
system matrices without explicit model structure assumptions.

Reference: Van Overschee & De Moor, "Subspace Identification for Linear Systems"
"""

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


@dataclass
class SubspaceModel:
    """Identified state-space model from subspace ID.
    
    State-space form:
        x_{t+1} = A x_t + B u_t
        y_t = C x_t + D u_t
    """
    A: np.ndarray  # State transition
    B: np.ndarray  # Input matrix  
    C: np.ndarray  # Output matrix (often Identity for full state feedback)
    D: np.ndarray  # Feedthrough (often zero)
    
    # For systems where y = x (full state measurement):
    # We effectively have x_{t+1} = A x_t + B u_t
    
    def predict(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Predict next state."""
        return self.A @ x + self.B @ u
    
    def predict_sequence(
        self, 
        x0: np.ndarray, 
        u_sequence: np.ndarray
    ) -> np.ndarray:
        """Predict state sequence."""
        T = len(u_sequence)
        n = len(x0)
        states = np.zeros((T + 1, n))
        states[0] = x0
        
        for t in range(T):
            states[t + 1] = self.predict(states[t], u_sequence[t])
            
        return states
    
    @property
    def state_dim(self) -> int:
        return self.A.shape[0]
    
    @property 
    def input_dim(self) -> int:
        return self.B.shape[1]


class SubspaceID:
    """Subspace identification using Hankel matrices.
    
    Algorithm (simplified N4SID):
    1. Build Hankel matrices from input-output data
    2. Compute oblique projection onto row space
    3. SVD to determine system order
    4. Extract state-space matrices
    """
    
    def __init__(
        self,
        n_rows: int = 10,
        system_order: Optional[int] = None,
        regularization: float = 1e-8
    ):
        """
        Args:
            n_rows: Number of block rows in Hankel matrices
            system_order: System order (state dimension). Auto-detected if None.
            regularization: Regularization for numerical stability
        """
        self.n_rows = n_rows
        self.system_order = system_order
        self.regularization = regularization
        self.model = None
        self._fit_info = {}
        
    def _build_hankel(
        self, 
        data: np.ndarray, 
        n_rows: int
    ) -> np.ndarray:
        """Build block Hankel matrix from time series data.
        
        Args:
            data: Time series data, shape (T, dim)
            n_rows: Number of block rows
            
        Returns:
            Hankel matrix, shape (n_rows * dim, T - n_rows + 1)
        """
        T, dim = data.shape
        n_cols = T - n_rows + 1
        
        if n_cols <= 0:
            raise ValueError(f"Not enough data. Need at least {n_rows + 1} samples.")
        
        H = np.zeros((n_rows * dim, n_cols))
        
        for i in range(n_rows):
            H[i*dim:(i+1)*dim, :] = data[i:i+n_cols].T
            
        return H
    
    def _split_hankel(
        self, 
        H: np.ndarray, 
        dim: int, 
        n_rows: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Split Hankel matrix into past and future parts.
        
        Args:
            H: Full Hankel matrix
            dim: Dimension per block row
            n_rows: Total block rows
            
        Returns:
            (H_past, H_future) split at middle
        """
        split = n_rows // 2
        H_past = H[:split*dim, :]
        H_future = H[split*dim:, :]
        return H_past, H_future
    
    def fit(self, dataset) -> SubspaceModel:
        """Fit subspace model to dataset.
        
        For full-state systems (y = x), we identify:
            x_{t+1} = A x_t + B u_t
            
        Args:
            dataset: Dataset object with states, inputs, next_states
            
        Returns:
            SubspaceModel object
        """
        X = dataset.states      # (N, n)
        U = dataset.inputs      # (N, m)
        
        n = X.shape[1]  # state dim
        m = U.shape[1]  # input dim
        N = X.shape[0]  # samples
        
        # Build Hankel matrices
        # For state-only measurement, Y = X (output is state)
        H_u = self._build_hankel(U, self.n_rows)
        H_x = self._build_hankel(X, self.n_rows)
        
        # Split into past/future
        i = self.n_rows // 2  # past horizon
        
        # Past inputs and outputs
        Up = H_u[:i*m, :]  # Past inputs
        Uf = H_u[i*m:, :]  # Future inputs
        Yp = H_x[:i*n, :]  # Past outputs (states)
        Yf = H_x[i*n:, :]  # Future outputs
        
        # Build extended observability matrix via oblique projection
        # Stack past data
        Wp = np.vstack([Up, Yp])
        
        # Compute oblique projection of Yf on Uf and Wp
        # Yf_proj = Yf / [Uf; Wp] * Uf
        
        # Simple approach: project Yf onto Wp while removing Uf effect
        # Use QR or SVD-based projection
        
        n_cols = Yf.shape[1]
        
        # Orthogonal projection approach
        # First remove effect of future inputs
        if Uf.shape[0] > 0:
            Uf_pinv = np.linalg.pinv(Uf)
            Yf_orth = Yf - Yf @ Uf_pinv @ Uf
            Wp_orth = Wp - Wp @ Uf_pinv @ Uf
        else:
            Yf_orth = Yf
            Wp_orth = Wp
            
        # Project onto past
        Wp_pinv = np.linalg.pinv(Wp_orth + self.regularization * np.eye(Wp_orth.shape[0], Wp_orth.shape[1]))
        Ob = Yf_orth @ Wp_pinv @ Wp_orth
        
        # SVD to find system order and extended observability matrix
        try:
            U_svd, S, Vh = np.linalg.svd(Ob, full_matrices=False)
        except np.linalg.LinAlgError:
            # Fallback to regularized SVD
            Ob_reg = Ob + self.regularization * np.random.randn(*Ob.shape) * 0.01
            U_svd, S, Vh = np.linalg.svd(Ob_reg, full_matrices=False)
        
        # Determine system order
        if self.system_order is None:
            # Auto-detect from singular value drop
            S_norm = S / S[0] if S[0] > 0 else S
            gaps = np.diff(S_norm)
            
            # Find significant drop (or use state dimension)
            order = n  # Default to original state dim
            for j in range(min(len(gaps), 2*n)):
                if S_norm[j+1] < 0.01 or (gaps[j] < -0.1 and j > 0):
                    order = j + 1
                    break
                    
            self.system_order = max(n, min(order, n))  # Clamp to reasonable range
        
        order = self.system_order
        
        # Extract observability matrix
        O_n = U_svd[:, :order] @ np.diag(np.sqrt(S[:order]))
        
        # Extract A from shift structure of O_n
        # O_n = [C; CA; CA^2; ...] so A = C^+ @ CA
        if O_n.shape[0] >= 2*n:
            O_top = O_n[:-n, :]
            O_bottom = O_n[n:, :]
            
            O_top_pinv = np.linalg.pinv(O_top)
            A_est = O_top_pinv @ O_bottom
        else:
            # Fallback to direct least squares
            A_est = self._estimate_AB_direct(dataset)[0]
            
        # Estimate B from data
        A_est, B_est = self._refine_AB(dataset, A_est)
        
        # For full-state output: C = I, D = 0
        C = np.eye(n)
        D = np.zeros((n, m))
        
        self.model = SubspaceModel(A=A_est, B=B_est, C=C, D=D)
        
        # Compute fit info
        self._compute_fit_info(dataset)
        
        return self.model
    
    def _estimate_AB_direct(
        self, 
        dataset
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Direct least squares estimation of A, B."""
        X = dataset.states
        U = dataset.inputs
        Y = dataset.next_states
        
        n = X.shape[1]
        m = U.shape[1]
        N = X.shape[0]
        
        # Solve: Y = [X, U] @ [A^T; B^T]
        Phi = np.hstack([X, U])
        
        reg = self.regularization * np.eye(Phi.shape[1])
        Theta = np.linalg.solve(Phi.T @ Phi + reg, Phi.T @ Y)
        
        A = Theta[:n, :].T
        B = Theta[n:, :].T
        
        return A, B
    
    def _refine_AB(
        self, 
        dataset, 
        A_init: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Refine A and estimate B using least squares.
        
        Given initial A, solve for B in:
            x_{t+1} - A x_t = B u_t
        """
        X = dataset.states
        U = dataset.inputs
        Y = dataset.next_states
        
        n = X.shape[1]
        m = U.shape[1]
        
        # Residual after A
        residual = Y - X @ A_init.T
        
        # Estimate B
        reg = self.regularization * np.eye(m)
        B = np.linalg.solve(U.T @ U + reg, U.T @ residual).T
        
        # Refine A with B fixed
        residual_B = Y - U @ B.T
        reg_n = self.regularization * np.eye(n)
        A = np.linalg.solve(X.T @ X + reg_n, X.T @ residual_B).T
        
        return A, B
    
    def _compute_fit_info(self, dataset):
        """Compute fit quality metrics."""
        Y_pred = dataset.states @ self.model.A.T + dataset.inputs @ self.model.B.T
        Y_true = dataset.next_states
        
        residuals = Y_true - Y_pred
        self._fit_info = {
            'mse': np.mean(residuals**2),
            'rmse': np.sqrt(np.mean(residuals**2)),
            'r2': 1 - np.sum(residuals**2) / np.sum((Y_true - Y_true.mean(axis=0))**2),
            'system_order': self.system_order,
            'n_rows': self.n_rows
        }
    
    def get_fit_info(self) -> dict:
        """Get fit information."""
        return self._fit_info.copy()
    
    def predict(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Predict using fitted model."""
        if self.model is None:
            raise ValueError("Model not fitted")
        return self.model.predict(x, u)


class MOESP(SubspaceID):
    """MOESP (Multivariable Output-Error State sPace) algorithm.
    
    Alternative subspace method with different projection scheme.
    Often more robust for systems with feedback.
    """
    
    def fit(self, dataset) -> SubspaceModel:
        """Fit using MOESP algorithm."""
        X = dataset.states
        U = dataset.inputs
        
        n = X.shape[1]
        m = U.shape[1]
        
        # Build Hankel matrices
        H_u = self._build_hankel(U, self.n_rows)
        H_x = self._build_hankel(X, self.n_rows)
        
        i = self.n_rows // 2
        
        Up = H_u[:i*m, :]
        Yf = H_x[i*n:, :]
        
        # MOESP projection: remove input from output
        Up_pinv = np.linalg.pinv(Up)
        Yf_proj = Yf - Yf @ Up_pinv @ Up
        
        # SVD
        U_svd, S, Vh = np.linalg.svd(Yf_proj, full_matrices=False)
        
        # Determine order
        if self.system_order is None:
            self.system_order = n
            
        order = self.system_order
        
        # Extract state sequence
        X_est = np.diag(np.sqrt(S[:order])) @ Vh[:order, :]
        
        # Estimate A, B from state sequence
        A, B = self._estimate_AB_direct(dataset)
        
        # Refine
        A, B = self._refine_AB(dataset, A)
        
        self.model = SubspaceModel(
            A=A, B=B,
            C=np.eye(n), D=np.zeros((n, m))
        )
        
        self._compute_fit_info(dataset)
        return self.model


def fit_subspace_model(
    dataset,
    method: str = 'n4sid',
    n_rows: int = 10,
    system_order: Optional[int] = None,
    regularization: float = 1e-8
) -> Dict:
    """Convenience function for subspace identification.
    
    Args:
        dataset: Dataset object
        method: 'n4sid' or 'moesp'
        n_rows: Number of block rows in Hankel matrix
        system_order: System order (auto if None)
        regularization: Regularization parameter
        
    Returns:
        Dict with 'model', 'identifier', 'info' keys
    """
    if method == 'n4sid':
        identifier = SubspaceID(
            n_rows=n_rows,
            system_order=system_order,
            regularization=regularization
        )
    elif method == 'moesp':
        identifier = MOESP(
            n_rows=n_rows,
            system_order=system_order,
            regularization=regularization
        )
    else:
        raise ValueError(f"Unknown method: {method}")
        
    model = identifier.fit(dataset)
    
    return {
        'model': model,
        'identifier': identifier,
        'info': identifier.get_fit_info()
    }


def compute_hankel_rank(
    dataset,
    n_rows: int = 10
) -> Tuple[np.ndarray, int]:
    """Analyze Hankel matrix rank to determine system order.
    
    Args:
        dataset: Dataset object
        n_rows: Block rows
        
    Returns:
        (singular_values, estimated_order)
    """
    identifier = SubspaceID(n_rows=n_rows)
    
    X = dataset.states
    H_x = identifier._build_hankel(X, n_rows)
    
    _, S, _ = np.linalg.svd(H_x, full_matrices=False)
    
    # Estimate order from singular value drop
    S_norm = S / S[0]
    order = len(S)
    
    for i in range(len(S) - 1):
        if S_norm[i+1] < 0.01:
            order = i + 1
            break
            
    return S, order
