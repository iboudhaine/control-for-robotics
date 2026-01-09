"""
System Identification Module

Provides methods for learning system dynamics from data:
- Linear system identification (least squares)
- Subspace identification (Hankel-based)
- Neural network dynamics models
"""

from .linear_id import (
    LinearSystemID,
    LocalLinearization,
    fit_linear_model
)

from .subspace_id import (
    SubspaceID,
    fit_subspace_model
)

from .neural_dynamics import (
    NeuralDynamics,
    MLPDynamics,
    train_neural_dynamics
)

__all__ = [
    'LinearSystemID',
    'LocalLinearization',
    'fit_linear_model',
    'SubspaceID',
    'fit_subspace_model',
    'NeuralDynamics',
    'MLPDynamics',
    'train_neural_dynamics'
]
