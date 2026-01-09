"""
Neural Network Dynamics Model

Implements neural network models for learning nonlinear dynamics:
    x_{t+1} = f_θ(x_t, u_t)

Uses PyTorch for neural network training with support for:
- MLP (Multi-Layer Perceptron) architecture
- Residual connections
- Uncertainty estimation (optional)
- Model for use with MPC/iLQR

Key features:
- Training with early stopping and validation
- Batch normalization for stable training
- Jacobian computation for linearization (LQR/iLQR)
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Callable
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Neural dynamics will be limited.")


class NeuralDynamics:
    """Base class for neural network dynamics models."""
    
    def __init__(self, state_dim: int, input_dim: int):
        self.state_dim = state_dim
        self.input_dim = input_dim
        
    def predict(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Predict next state."""
        raise NotImplementedError
        
    def predict_batch(
        self, 
        x: np.ndarray, 
        u: np.ndarray
    ) -> np.ndarray:
        """Predict next states for batch of inputs."""
        raise NotImplementedError
        
    def get_jacobians(
        self, 
        x: np.ndarray, 
        u: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get Jacobians df/dx and df/du at (x, u)."""
        raise NotImplementedError


if TORCH_AVAILABLE:
    
    class MLPNetwork(nn.Module):
        """MLP network for dynamics prediction."""
        
        def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dims: List[int] = [64, 64],
            activation: str = 'relu',
            use_batch_norm: bool = False,
            dropout: float = 0.0
        ):
            super().__init__()
            
            self.input_dim = input_dim
            self.output_dim = output_dim
            
            # Build layers
            layers = []
            prev_dim = input_dim
            
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                    
                if activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'tanh':
                    layers.append(nn.Tanh())
                elif activation == 'leaky_relu':
                    layers.append(nn.LeakyReLU(0.1))
                elif activation == 'elu':
                    layers.append(nn.ELU())
                    
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                    
                prev_dim = hidden_dim
                
            # Output layer
            layers.append(nn.Linear(prev_dim, output_dim))
            
            self.network = nn.Sequential(*layers)
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.network(x)
    
    
    class ResidualMLPNetwork(nn.Module):
        """MLP that predicts residual: x_{t+1} = x_t + f_θ(x_t, u_t).
        
        Residual formulation often trains faster and generalizes better
        for dynamics learning since the identity mapping is easy to learn.
        """
        
        def __init__(
            self,
            state_dim: int,
            input_dim: int,
            hidden_dims: List[int] = [64, 64],
            activation: str = 'relu',
            use_batch_norm: bool = False
        ):
            super().__init__()
            
            self.state_dim = state_dim
            self.input_dim = input_dim
            
            self.mlp = MLPNetwork(
                input_dim=state_dim + input_dim,
                output_dim=state_dim,
                hidden_dims=hidden_dims,
                activation=activation,
                use_batch_norm=use_batch_norm
            )
            
        def forward(
            self, 
            x: torch.Tensor, 
            u: torch.Tensor
        ) -> torch.Tensor:
            """Forward pass with residual connection.
            
            Args:
                x: Current state (batch_size, state_dim)
                u: Control input (batch_size, input_dim)
                
            Returns:
                Next state (batch_size, state_dim)
            """
            xu = torch.cat([x, u], dim=-1)
            residual = self.mlp(xu)
            return x + residual
        
        def forward_combined(self, xu: torch.Tensor) -> torch.Tensor:
            """Forward with combined input for Jacobian computation."""
            x = xu[..., :self.state_dim]
            u = xu[..., self.state_dim:]
            return self.forward(x, u)
    
    
    class MLPDynamics(NeuralDynamics):
        """Neural network dynamics model using MLP.
        
        Learns f_θ such that x_{t+1} = f_θ(x_t, u_t) from data.
        """
        
        def __init__(
            self,
            state_dim: int,
            input_dim: int,
            hidden_dims: List[int] = [64, 64],
            activation: str = 'relu',
            use_residual: bool = True,
            use_batch_norm: bool = False,
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-5,
            device: str = 'cpu'
        ):
            """
            Args:
                state_dim: State dimension
                input_dim: Input dimension
                hidden_dims: List of hidden layer dimensions
                activation: Activation function ('relu', 'tanh', 'elu')
                use_residual: Use residual connection (x + f(x,u))
                use_batch_norm: Use batch normalization
                learning_rate: Learning rate for optimizer
                weight_decay: L2 regularization
                device: 'cpu' or 'cuda'
            """
            super().__init__(state_dim, input_dim)
            
            self.device = device
            self.use_residual = use_residual
            
            # Build network
            if use_residual:
                self.network = ResidualMLPNetwork(
                    state_dim=state_dim,
                    input_dim=input_dim,
                    hidden_dims=hidden_dims,
                    activation=activation,
                    use_batch_norm=use_batch_norm
                ).to(device)
            else:
                self.network = MLPNetwork(
                    input_dim=state_dim + input_dim,
                    output_dim=state_dim,
                    hidden_dims=hidden_dims,
                    activation=activation,
                    use_batch_norm=use_batch_norm
                ).to(device)
                
            self.optimizer = optim.Adam(
                self.network.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
            
            self.loss_fn = nn.MSELoss()
            self.training_history = []
            
            # Input/output normalization
            self.x_mean = None
            self.x_std = None
            self.u_mean = None
            self.u_std = None
            
        def _normalize_inputs(
            self, 
            x: np.ndarray, 
            u: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:
            """Normalize inputs using stored statistics."""
            if self.x_mean is not None:
                x = (x - self.x_mean) / (self.x_std + 1e-8)
            if self.u_mean is not None:
                u = (u - self.u_mean) / (self.u_std + 1e-8)
            return x, u
        
        def _denormalize_output(self, x: np.ndarray) -> np.ndarray:
            """Denormalize output."""
            if self.x_mean is not None:
                x = x * (self.x_std + 1e-8) + self.x_mean
            return x
        
        def fit(
            self,
            dataset,
            n_epochs: int = 100,
            batch_size: int = 64,
            validation_split: float = 0.2,
            early_stopping_patience: int = 10,
            normalize: bool = True,
            verbose: bool = True
        ) -> Dict:
            """Train the neural network on dataset.
            
            Args:
                dataset: Dataset object
                n_epochs: Maximum number of epochs
                batch_size: Batch size for training
                validation_split: Fraction for validation
                early_stopping_patience: Epochs without improvement before stopping
                normalize: Normalize inputs/outputs
                verbose: Print training progress
                
            Returns:
                Training history dictionary
            """
            # Split data
            train_data, val_data = dataset.split(
                train_ratio=1-validation_split,
                shuffle=True
            )
            
            # Compute normalization statistics
            if normalize:
                self.x_mean = train_data.states.mean(axis=0)
                self.x_std = train_data.states.std(axis=0)
                self.u_mean = train_data.inputs.mean(axis=0)
                self.u_std = train_data.inputs.std(axis=0)
            
            # Prepare tensors
            def prepare_tensors(data):
                x, u = self._normalize_inputs(data.states, data.inputs)
                y = data.next_states
                if normalize:
                    y = (y - self.x_mean) / (self.x_std + 1e-8)
                    
                x_t = torch.FloatTensor(x).to(self.device)
                u_t = torch.FloatTensor(u).to(self.device)
                y_t = torch.FloatTensor(y).to(self.device)
                return x_t, u_t, y_t
            
            x_train, u_train, y_train = prepare_tensors(train_data)
            x_val, u_val, y_val = prepare_tensors(val_data)
            
            # DataLoader
            train_dataset = TensorDataset(x_train, u_train, y_train)
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            
            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            best_state = None
            
            for epoch in range(n_epochs):
                # Training
                self.network.train()
                train_loss = 0.0
                
                for x_batch, u_batch, y_batch in train_loader:
                    self.optimizer.zero_grad()
                    
                    if self.use_residual:
                        y_pred = self.network(x_batch, u_batch)
                    else:
                        xu = torch.cat([x_batch, u_batch], dim=1)
                        y_pred = self.network(xu)
                        
                    loss = self.loss_fn(y_pred, y_batch)
                    loss.backward()
                    self.optimizer.step()
                    
                    train_loss += loss.item() * len(x_batch)
                    
                train_loss /= len(train_dataset)
                
                # Validation
                self.network.eval()
                with torch.no_grad():
                    if self.use_residual:
                        y_val_pred = self.network(x_val, u_val)
                    else:
                        xu_val = torch.cat([x_val, u_val], dim=1)
                        y_val_pred = self.network(xu_val)
                        
                    val_loss = self.loss_fn(y_val_pred, y_val).item()
                    
                self.training_history.append({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss
                })
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in 
                                  self.network.state_dict().items()}
                else:
                    patience_counter += 1
                    
                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch}: train_loss={train_loss:.6f}, "
                          f"val_loss={val_loss:.6f}")
                    
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break
                    
            # Restore best model
            if best_state is not None:
                self.network.load_state_dict(best_state)
                
            return {
                'history': self.training_history,
                'best_val_loss': best_val_loss
            }
        
        def predict(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
            """Predict next state.
            
            Args:
                x: Current state (state_dim,) or (batch, state_dim)
                u: Control input (input_dim,) or (batch, input_dim)
                
            Returns:
                Next state prediction
            """
            self.network.eval()
            
            # Handle single sample
            single = x.ndim == 1
            if single:
                x = x.reshape(1, -1)
                u = u.reshape(1, -1)
                
            # Normalize
            x_norm, u_norm = self._normalize_inputs(x, u)
            
            with torch.no_grad():
                x_t = torch.FloatTensor(x_norm).to(self.device)
                u_t = torch.FloatTensor(u_norm).to(self.device)
                
                if self.use_residual:
                    y_t = self.network(x_t, u_t)
                else:
                    xu_t = torch.cat([x_t, u_t], dim=1)
                    y_t = self.network(xu_t)
                    
                y = y_t.cpu().numpy()
                
            # Denormalize
            y = self._denormalize_output(y)
            
            if single:
                y = y.squeeze(0)
                
            return y
        
        def predict_batch(
            self, 
            x: np.ndarray, 
            u: np.ndarray
        ) -> np.ndarray:
            """Batch prediction."""
            return self.predict(x, u)
        
        def predict_sequence(
            self,
            x0: np.ndarray,
            u_sequence: np.ndarray
        ) -> np.ndarray:
            """Predict state sequence.
            
            Args:
                x0: Initial state
                u_sequence: Control sequence (T, input_dim)
                
            Returns:
                State sequence (T+1, state_dim)
            """
            T = len(u_sequence)
            states = np.zeros((T + 1, self.state_dim))
            states[0] = x0
            
            for t in range(T):
                states[t + 1] = self.predict(states[t], u_sequence[t])
                
            return states
        
        def get_jacobians(
            self, 
            x: np.ndarray, 
            u: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:
            """Compute Jacobians df/dx and df/du using autograd.
            
            Returns:
                (A, B) where A = df/dx, B = df/du at (x, u)
            """
            x_norm, u_norm = self._normalize_inputs(
                x.reshape(1, -1), u.reshape(1, -1)
            )
            
            x_t = torch.FloatTensor(x_norm).to(self.device).requires_grad_(True)
            u_t = torch.FloatTensor(u_norm).to(self.device).requires_grad_(True)
            
            if self.use_residual:
                y = self.network(x_t, u_t)
            else:
                xu = torch.cat([x_t, u_t], dim=1)
                y = self.network(xu)
            
            # Compute Jacobians
            A = torch.zeros(self.state_dim, self.state_dim).to(self.device)
            B = torch.zeros(self.state_dim, self.input_dim).to(self.device)
            
            for i in range(self.state_dim):
                # Gradient of y[i] w.r.t. x and u
                grad_outputs = torch.zeros_like(y)
                grad_outputs[0, i] = 1.0
                
                grads = torch.autograd.grad(
                    y, [x_t, u_t], 
                    grad_outputs=grad_outputs,
                    retain_graph=True
                )
                
                A[i] = grads[0].squeeze()
                B[i] = grads[1].squeeze()
            
            # Account for normalization
            if self.x_std is not None:
                # dy_norm/dx_norm * dx_norm/dx * dy/dy_norm
                scale_x = self.x_std / (self.x_std + 1e-8)  # x_std_out / x_std_in
                A = A.cpu().numpy() * scale_x.reshape(-1, 1)
                
                scale_u = self.x_std / (self.u_std + 1e-8)
                B = B.cpu().numpy() * scale_u.reshape(-1, 1)
            else:
                A = A.cpu().numpy()
                B = B.cpu().numpy()
                
            return A, B
        
        def save(self, filepath: str):
            """Save model to file."""
            torch.save({
                'state_dict': self.network.state_dict(),
                'state_dim': self.state_dim,
                'input_dim': self.input_dim,
                'x_mean': self.x_mean,
                'x_std': self.x_std,
                'u_mean': self.u_mean,
                'u_std': self.u_std,
                'use_residual': self.use_residual,
                'training_history': self.training_history
            }, filepath)
            
        @classmethod
        def load(cls, filepath: str, device: str = 'cpu') -> 'MLPDynamics':
            """Load model from file."""
            checkpoint = torch.load(filepath, map_location=device)
            
            model = cls(
                state_dim=checkpoint['state_dim'],
                input_dim=checkpoint['input_dim'],
                use_residual=checkpoint['use_residual'],
                device=device
            )
            
            model.network.load_state_dict(checkpoint['state_dict'])
            model.x_mean = checkpoint['x_mean']
            model.x_std = checkpoint['x_std']
            model.u_mean = checkpoint['u_mean']
            model.u_std = checkpoint['u_std']
            model.training_history = checkpoint['training_history']
            
            return model

else:
    # Fallback when PyTorch not available
    class MLPDynamics(NeuralDynamics):
        """Placeholder when PyTorch is not available."""
        
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required for MLPDynamics. "
                "Install with: pip install torch"
            )


def train_neural_dynamics(
    dataset,
    state_dim: int = None,
    input_dim: int = None,
    hidden_dims: List[int] = [64, 64],
    n_epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    use_residual: bool = True,
    device: str = 'cpu',
    verbose: bool = True
) -> Dict:
    """Convenience function to train neural dynamics model.
    
    Args:
        dataset: Dataset object
        state_dim: State dimension (inferred if None)
        input_dim: Input dimension (inferred if None)
        hidden_dims: Hidden layer dimensions
        n_epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        use_residual: Use residual connection
        device: 'cpu' or 'cuda'
        verbose: Print progress
        
    Returns:
        Dict with 'model', 'history', 'info' keys
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required. Install with: pip install torch")
        
    if state_dim is None:
        state_dim = dataset.state_dim
    if input_dim is None:
        input_dim = dataset.input_dim
        
    model = MLPDynamics(
        state_dim=state_dim,
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        use_residual=use_residual,
        learning_rate=learning_rate,
        device=device
    )
    
    result = model.fit(
        dataset,
        n_epochs=n_epochs,
        batch_size=batch_size,
        verbose=verbose
    )
    
    return {
        'model': model,
        'history': result['history'],
        'info': {
            'best_val_loss': result['best_val_loss'],
            'state_dim': state_dim,
            'input_dim': input_dim,
            'hidden_dims': hidden_dims
        }
    }
