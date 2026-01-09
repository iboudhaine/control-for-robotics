"""
Linear System Identification

Methods for identifying linear dynamics from data:
- Global least squares: x_{t+1} = Ax_t + Bu_t
- Local linearization: Linearize around operating points

These methods assume the system can be approximated as:
    x_{t+1} ≈ A x_t + B u_t + c

where A, B are identified from data.
"""

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


@dataclass
class LinearModel:
    """Container for identified linear model.
    
    Model: x_{t+1} = A x_t + B u_t + c
    """
    A: np.ndarray  # State matrix
    B: np.ndarray  # Input matrix
    c: np.ndarray  # Offset/bias term
    
    def predict(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Predict next state.
        
        Args:
            x: Current state
            u: Control input
            
        Returns:
            Predicted next state
        """
        return self.A @ x + self.B @ u + self.c
    
    def predict_sequence(
        self, 
        x0: np.ndarray, 
        u_sequence: np.ndarray
    ) -> np.ndarray:
        """Predict state sequence given initial state and input sequence.
        
        Args:
            x0: Initial state
            u_sequence: Input sequence, shape (T, input_dim)
            
        Returns:
            State sequence, shape (T+1, state_dim)
        """
        T = len(u_sequence)
        states = np.zeros((T + 1, len(x0)))
        states[0] = x0
        
        for t in range(T):
            states[t + 1] = self.predict(states[t], u_sequence[t])
            
        return states


class LinearSystemID:
    """Linear system identification via least squares.
    
    Solves:
        min_{A,B,c} sum_i ||x_{i+1} - (A x_i + B u_i + c)||^2
        
    Using standard least squares regression.
    """
    
    def __init__(self, regularization: float = 1e-6):
        """
        Args:
            regularization: Ridge regularization parameter
        """
        self.regularization = regularization
        self.model = None
        self._fit_info = {}
        
    def fit(self, dataset) -> LinearModel:
        """Fit linear model to dataset.
        
        Args:
            dataset: Dataset object with states, inputs, next_states
            
        Returns:
            Fitted LinearModel
        """
        X = dataset.states      # (N, n)
        U = dataset.inputs      # (N, m)
        Y = dataset.next_states # (N, n)
        
        n = X.shape[1]  # state dimension
        m = U.shape[1]  # input dimension
        N = X.shape[0]  # number of samples
        
        # Build regression matrix: [x, u, 1]
        Phi = np.hstack([X, U, np.ones((N, 1))])  # (N, n+m+1)
        
        # Solve least squares with regularization: Y = Phi @ Theta^T
        # Theta = [A | B | c]^T
        
        # Ridge regression solution
        lambda_I = self.regularization * np.eye(Phi.shape[1])
        Theta = np.linalg.solve(
            Phi.T @ Phi + lambda_I,
            Phi.T @ Y
        )  # (n+m+1, n)
        
        # Extract A, B, c
        A = Theta[:n, :].T       # (n, n)
        B = Theta[n:n+m, :].T    # (n, m)
        c = Theta[n+m, :]        # (n,)
        
        self.model = LinearModel(A=A, B=B, c=c)
        
        # Compute fit statistics
        Y_pred = Phi @ Theta
        residuals = Y - Y_pred
        self._fit_info = {
            'mse': np.mean(residuals**2),
            'rmse': np.sqrt(np.mean(residuals**2)),
            'r2': 1 - np.sum(residuals**2) / np.sum((Y - Y.mean(axis=0))**2),
            'n_samples': N
        }
        
        return self.model
    
    def get_fit_info(self) -> dict:
        """Get information about the fit quality."""
        return self._fit_info.copy()
    
    def predict(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Predict next state using fitted model."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(x, u)


class LocalLinearization:
    """Local linearization around operating points.
    
    Fits separate linear models around different operating regions,
    then interpolates or selects based on current state.
    
    For unicycle around (x*, u*):
        x_{t+1} - x* ≈ A (x_t - x*) + B (u_t - u*)
    """
    
    def __init__(
        self, 
        n_clusters: int = 5,
        regularization: float = 1e-6
    ):
        """
        Args:
            n_clusters: Number of local linearization regions
            regularization: Ridge regularization
        """
        self.n_clusters = n_clusters
        self.regularization = regularization
        self.local_models = []
        self.cluster_centers = None
        
    def fit(self, dataset) -> None:
        """Fit local linear models to dataset.
        
        Args:
            dataset: Dataset object
        """
        from scipy.cluster.vq import kmeans2
        
        X = dataset.states
        U = dataset.inputs
        Y = dataset.next_states
        
        # Cluster based on state-input pairs
        features = np.hstack([X, U])
        
        try:
            centers, labels = kmeans2(features, self.n_clusters, minit='points')
        except:
            # Fallback if clustering fails
            self.n_clusters = 1
            centers = np.mean(features, axis=0, keepdims=True)
            labels = np.zeros(len(X), dtype=int)
            
        self.cluster_centers = centers
        
        # Fit linear model for each cluster
        self.local_models = []
        state_dim = X.shape[1]
        input_dim = U.shape[1]
        
        for k in range(self.n_clusters):
            mask = labels == k
            
            if np.sum(mask) < state_dim + input_dim + 2:
                # Not enough data in cluster, use global model
                mask = np.ones(len(X), dtype=bool)
                
            X_k = X[mask]
            U_k = U[mask]
            Y_k = Y[mask]
            
            # Center point
            x_center = np.mean(X_k, axis=0)
            u_center = np.mean(U_k, axis=0)
            
            # Fit local linear model
            dX = X_k - x_center
            dU = U_k - u_center
            dY = Y_k  # Next states
            
            # Regression: dY = [dX, dU, 1] @ Theta^T
            N = len(X_k)
            Phi = np.hstack([dX, dU, np.ones((N, 1))])
            
            lambda_I = self.regularization * np.eye(Phi.shape[1])
            Theta = np.linalg.solve(
                Phi.T @ Phi + lambda_I,
                Phi.T @ dY
            )
            
            A = Theta[:state_dim, :].T
            B = Theta[state_dim:state_dim+input_dim, :].T
            c = Theta[-1, :]
            
            self.local_models.append({
                'A': A,
                'B': B,
                'c': c,
                'x_center': x_center,
                'u_center': u_center,
                'center': centers[k]
            })
            
    def _find_nearest_cluster(self, x: np.ndarray, u: np.ndarray) -> int:
        """Find nearest cluster center."""
        feature = np.concatenate([x, u])
        distances = np.linalg.norm(self.cluster_centers - feature, axis=1)
        return np.argmin(distances)
    
    def predict(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Predict using nearest local model."""
        k = self._find_nearest_cluster(x, u)
        model = self.local_models[k]
        
        dx = x - model['x_center']
        du = u - model['u_center']
        
        return model['A'] @ dx + model['B'] @ du + model['c']
    
    def get_local_model(
        self, 
        x: np.ndarray, 
        u: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get A, B matrices for local linearization at (x, u).
        
        Returns:
            (A, B) matrices for the nearest local model
        """
        k = self._find_nearest_cluster(x, u)
        model = self.local_models[k]
        return model['A'], model['B']


def fit_linear_model(
    dataset,
    method: str = 'global',
    regularization: float = 1e-6,
    n_clusters: int = 5
) -> Dict:
    """Convenience function to fit a linear model.
    
    Args:
        dataset: Dataset object
        method: 'global' for single linear model, 'local' for local linearization
        regularization: Ridge regularization parameter
        n_clusters: Number of clusters for local method
        
    Returns:
        Dictionary with 'model' and 'info' keys
    """
    if method == 'global':
        identifier = LinearSystemID(regularization=regularization)
        model = identifier.fit(dataset)
        return {
            'model': model,
            'identifier': identifier,
            'info': identifier.get_fit_info()
        }
    elif method == 'local':
        identifier = LocalLinearization(
            n_clusters=n_clusters,
            regularization=regularization
        )
        identifier.fit(dataset)
        return {
            'model': identifier,
            'identifier': identifier,
            'info': {'n_clusters': len(identifier.local_models)}
        }
    else:
        raise ValueError(f"Unknown method: {method}")


def estimate_linearization(
    dataset,
    x_eq: np.ndarray,
    u_eq: np.ndarray,
    radius: float = 0.5,
    regularization: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate A, B matrices by linearization around an equilibrium.
    
    Selects data near (x_eq, u_eq) and fits a local linear model.
    
    Args:
        dataset: Dataset object
        x_eq: Equilibrium state
        u_eq: Equilibrium input
        radius: Data selection radius
        regularization: Ridge regularization
        
    Returns:
        (A, B) matrices of local linearization
    """
    X = dataset.states
    U = dataset.inputs
    Y = dataset.next_states
    
    # Select nearby data
    state_dist = np.linalg.norm(X - x_eq, axis=1)
    input_dist = np.linalg.norm(U - u_eq, axis=1)
    total_dist = state_dist + input_dist
    
    mask = total_dist < radius
    
    if np.sum(mask) < X.shape[1] + U.shape[1] + 2:
        # Not enough local data, expand radius
        idx = np.argsort(total_dist)[:max(20, X.shape[1] + U.shape[1] + 5)]
        mask = np.zeros(len(X), dtype=bool)
        mask[idx] = True
        
    X_local = X[mask]
    U_local = U[mask]
    Y_local = Y[mask]
    
    # Fit local model: Y = A(X - x_eq) + B(U - u_eq) + c
    dX = X_local - x_eq
    dU = U_local - u_eq
    
    n = X.shape[1]
    m = U.shape[1]
    N = len(X_local)
    
    Phi = np.hstack([dX, dU, np.ones((N, 1))])
    
    lambda_I = regularization * np.eye(Phi.shape[1])
    Theta = np.linalg.solve(
        Phi.T @ Phi + lambda_I,
        Phi.T @ Y_local
    )
    
    A = Theta[:n, :].T
    B = Theta[n:n+m, :].T
    
    return A, B
