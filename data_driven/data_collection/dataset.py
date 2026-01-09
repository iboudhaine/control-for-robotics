"""
Dataset Management for Data-Driven Control

Provides utilities for collecting, storing, and managing datasets
from robotic systems. Supports noise injection and data splitting.

Key classes:
- Dataset: Container for (x, u, x_next) tuples
- Functions for data collection from simulations
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Callable
import sys
import os

# Add parent path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


@dataclass
class Dataset:
    """Dataset container for system identification.
    
    Stores input-output data in the form (x_t, u_t, x_{t+1}).
    
    Attributes:
        states: Current states, shape (N, state_dim)
        inputs: Control inputs, shape (N, input_dim)
        next_states: Next states, shape (N, state_dim)
        times: Time stamps, shape (N,)
    """
    states: np.ndarray
    inputs: np.ndarray
    next_states: np.ndarray
    times: np.ndarray = field(default_factory=lambda: np.array([]))
    
    def __post_init__(self):
        """Validate data shapes."""
        assert len(self.states) == len(self.inputs) == len(self.next_states), \
            "All arrays must have the same length"
        if len(self.times) == 0:
            self.times = np.arange(len(self.states))
            
    @property
    def n_samples(self) -> int:
        """Number of data samples."""
        return len(self.states)
    
    @property
    def state_dim(self) -> int:
        """Dimension of state space."""
        return self.states.shape[1] if len(self.states.shape) > 1 else 1
    
    @property
    def input_dim(self) -> int:
        """Dimension of input space."""
        return self.inputs.shape[1] if len(self.inputs.shape) > 1 else 1
    
    def add_noise(
        self, 
        state_noise_std: float = 0.0,
        input_noise_std: float = 0.0,
        measurement_noise_std: float = 0.0,
        seed: Optional[int] = None
    ) -> 'Dataset':
        """Add noise to the dataset.
        
        Args:
            state_noise_std: Std dev of noise added to current states
            input_noise_std: Std dev of noise added to inputs
            measurement_noise_std: Std dev of noise added to next states (measurements)
            seed: Random seed
            
        Returns:
            New Dataset with added noise
        """
        rng = np.random.default_rng(seed)
        
        noisy_states = self.states + rng.normal(
            0, state_noise_std, self.states.shape
        ) if state_noise_std > 0 else self.states.copy()
        
        noisy_inputs = self.inputs + rng.normal(
            0, input_noise_std, self.inputs.shape
        ) if input_noise_std > 0 else self.inputs.copy()
        
        noisy_next_states = self.next_states + rng.normal(
            0, measurement_noise_std, self.next_states.shape
        ) if measurement_noise_std > 0 else self.next_states.copy()
        
        return Dataset(
            states=noisy_states,
            inputs=noisy_inputs,
            next_states=noisy_next_states,
            times=self.times.copy()
        )
    
    def split(
        self,
        train_ratio: float = 0.8,
        shuffle: bool = True,
        seed: Optional[int] = None
    ) -> Tuple['Dataset', 'Dataset']:
        """Split dataset into training and validation sets.
        
        Args:
            train_ratio: Fraction of data for training
            shuffle: Whether to shuffle before splitting
            seed: Random seed
            
        Returns:
            (train_dataset, val_dataset)
        """
        n = self.n_samples
        indices = np.arange(n)
        
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)
            
        n_train = int(n * train_ratio)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]
        
        train_data = Dataset(
            states=self.states[train_idx],
            inputs=self.inputs[train_idx],
            next_states=self.next_states[train_idx],
            times=self.times[train_idx]
        )
        
        val_data = Dataset(
            states=self.states[val_idx],
            inputs=self.inputs[val_idx],
            next_states=self.next_states[val_idx],
            times=self.times[val_idx]
        )
        
        return train_data, val_data
    
    def get_batch(
        self,
        batch_size: int,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get a random batch of data.
        
        Args:
            batch_size: Number of samples in batch
            seed: Random seed
            
        Returns:
            (states, inputs, next_states) batch
        """
        rng = np.random.default_rng(seed)
        indices = rng.choice(self.n_samples, size=min(batch_size, self.n_samples), replace=False)
        
        return self.states[indices], self.inputs[indices], self.next_states[indices]
    
    def subsample(self, n_samples: int, seed: Optional[int] = None) -> 'Dataset':
        """Create a subsampled dataset.
        
        Args:
            n_samples: Number of samples to keep
            seed: Random seed
            
        Returns:
            Subsampled Dataset
        """
        if n_samples >= self.n_samples:
            return Dataset(
                states=self.states.copy(),
                inputs=self.inputs.copy(),
                next_states=self.next_states.copy(),
                times=self.times.copy()
            )
            
        rng = np.random.default_rng(seed)
        indices = rng.choice(self.n_samples, size=n_samples, replace=False)
        indices = np.sort(indices)  # Keep temporal order
        
        return Dataset(
            states=self.states[indices],
            inputs=self.inputs[indices],
            next_states=self.next_states[indices],
            times=self.times[indices]
        )
    
    def concatenate(self, other: 'Dataset') -> 'Dataset':
        """Concatenate with another dataset.
        
        Args:
            other: Another Dataset
            
        Returns:
            Combined Dataset
        """
        return Dataset(
            states=np.vstack([self.states, other.states]),
            inputs=np.vstack([self.inputs, other.inputs]),
            next_states=np.vstack([self.next_states, other.next_states]),
            times=np.concatenate([self.times, other.times + self.times[-1] + 0.01])
        )
    
    def to_hankel(self, L: int) -> Tuple[np.ndarray, np.ndarray]:
        """Convert to Hankel matrices for DeePC.
        
        Args:
            L: Number of rows (sequence length)
            
        Returns:
            (U_hankel, X_hankel) Hankel matrices
        """
        N = self.n_samples - L + 1
        if N <= 0:
            raise ValueError(f"Not enough data for L={L}. Need at least {L} samples.")
            
        # State Hankel matrix
        X_hankel = np.zeros((L * self.state_dim, N))
        for i in range(N):
            for j in range(L):
                X_hankel[j*self.state_dim:(j+1)*self.state_dim, i] = self.states[i + j]
                
        # Input Hankel matrix
        U_hankel = np.zeros((L * self.input_dim, N))
        for i in range(N):
            for j in range(L):
                U_hankel[j*self.input_dim:(j+1)*self.input_dim, i] = self.inputs[i + j]
                
        return U_hankel, X_hankel
    
    def save(self, filepath: str):
        """Save dataset to file.
        
        Args:
            filepath: Path to save file (.npz)
        """
        np.savez(
            filepath,
            states=self.states,
            inputs=self.inputs,
            next_states=self.next_states,
            times=self.times
        )
        
    @classmethod
    def load(cls, filepath: str) -> 'Dataset':
        """Load dataset from file.
        
        Args:
            filepath: Path to .npz file
            
        Returns:
            Loaded Dataset
        """
        data = np.load(filepath)
        return cls(
            states=data['states'],
            inputs=data['inputs'],
            next_states=data['next_states'],
            times=data['times']
        )


def collect_data(
    model,
    excitation_signal,
    x0: np.ndarray,
    duration: float,
    dt: float = 0.01,
    process_noise_std: float = 0.0,
    seed: Optional[int] = None
) -> Dataset:
    """Collect data from a model using excitation signal.
    
    This simulates the TRUE system to collect data. The collected data
    can then be used for system identification without knowing the model.
    
    Args:
        model: Robot model with dynamics(x, u) method
        excitation_signal: ExcitationSignal object
        x0: Initial state
        duration: Total simulation duration
        dt: Time step
        process_noise_std: Standard deviation of process noise
        seed: Random seed for noise
        
    Returns:
        Dataset with collected (x, u, x_next) tuples
    """
    rng = np.random.default_rng(seed)
    
    times = np.arange(0, duration, dt)
    n_steps = len(times)
    
    state_dim = len(x0)
    input_dim = excitation_signal.input_dim
    
    states = np.zeros((n_steps, state_dim))
    inputs = np.zeros((n_steps, input_dim))
    next_states = np.zeros((n_steps, state_dim))
    
    x = x0.copy()
    
    for i, t in enumerate(times):
        # Get excitation input
        u = excitation_signal.generate(t)
        
        # Apply input saturation if model has it
        if hasattr(model, 'saturate_input'):
            u = model.saturate_input(u)
            
        # Process noise
        w = rng.normal(0, process_noise_std, state_dim) if process_noise_std > 0 else None
        
        # Simulate one step
        x_next = model.dynamics(x, u, w)
        
        # Store data
        states[i] = x
        inputs[i] = u
        next_states[i] = x_next
        
        # Update state
        x = x_next
        
    return Dataset(
        states=states,
        inputs=inputs,
        next_states=next_states,
        times=times
    )


def collect_trajectory_data(
    model,
    controller,
    trajectory_generator,
    x0: np.ndarray,
    duration: float,
    dt: float = 0.01,
    exploration_noise_std: float = 0.0,
    process_noise_std: float = 0.0,
    seed: Optional[int] = None
) -> Tuple[Dataset, np.ndarray]:
    """Collect data while following a trajectory with a controller.
    
    Uses a controller to track a reference trajectory while optionally
    adding exploration noise to collect richer data.
    
    Args:
        model: Robot model
        controller: Controller object with compute_control(x, t) method
        trajectory_generator: TrajectoryGenerator object
        x0: Initial state
        duration: Total duration
        dt: Time step
        exploration_noise_std: Std of added exploration noise to control
        process_noise_std: Std of process noise
        seed: Random seed
        
    Returns:
        (dataset, reference_states) tuple
    """
    rng = np.random.default_rng(seed)
    
    times = np.arange(0, duration, dt)
    n_steps = len(times)
    
    state_dim = len(x0)
    
    # Get reference trajectory
    ref_data = trajectory_generator.get_reference_sequence(times)
    reference_states = ref_data['states']
    
    # Determine input dimension from first control output
    x = x0.copy()
    
    # Set controller reference
    if hasattr(controller, 'set_target'):
        controller.set_target(reference_states[0])
        
    # Get input dimension
    try:
        test_u, _ = controller.compute_control(x, 0)
        input_dim = len(test_u)
    except:
        input_dim = 2  # Default for unicycle
    
    states = np.zeros((n_steps, state_dim))
    inputs = np.zeros((n_steps, input_dim))
    next_states = np.zeros((n_steps, state_dim))
    
    x = x0.copy()
    
    for i, t in enumerate(times):
        # Update reference
        ref = trajectory_generator.get_reference(t)
        if hasattr(controller, 'set_target'):
            controller.set_target(ref['state'])
            
        # Compute control
        u, _ = controller.compute_control(x, t)
        u = np.atleast_1d(u)
        
        # Add exploration noise
        if exploration_noise_std > 0:
            u = u + rng.normal(0, exploration_noise_std, len(u))
            
        # Apply saturation
        if hasattr(model, 'saturate_input'):
            u = model.saturate_input(u)
            
        # Process noise
        w = rng.normal(0, process_noise_std, state_dim) if process_noise_std > 0 else None
        
        # Simulate
        x_next = model.dynamics(x, u, w)
        
        # Store
        states[i] = x
        inputs[i] = u
        next_states[i] = x_next
        
        x = x_next
        
    dataset = Dataset(
        states=states,
        inputs=inputs,
        next_states=next_states,
        times=times
    )
    
    return dataset, reference_states


def create_datasets(
    model,
    excitation_signal,
    x0_list: List[np.ndarray],
    duration: float,
    dt: float = 0.01,
    sizes: dict = None,
    seed: Optional[int] = None
) -> dict:
    """Create datasets of different sizes for comparison.
    
    Args:
        model: Robot model
        excitation_signal: Excitation signal generator
        x0_list: List of initial states to collect from
        duration: Duration per trajectory
        dt: Time step
        sizes: Dict mapping size name to number of samples, e.g.,
               {'small': 100, 'medium': 500, 'large': 2000}
        seed: Random seed
        
    Returns:
        Dict mapping size name to Dataset
    """
    if sizes is None:
        sizes = {'small': 100, 'medium': 500, 'large': 2000}
        
    rng = np.random.default_rng(seed)
    
    # Collect all data
    all_data = []
    for x0 in x0_list:
        # Reset excitation signal
        excitation_signal._last_switch_time = -np.inf  # For random signals
        
        data = collect_data(
            model, excitation_signal, x0, duration, dt,
            seed=rng.integers(10000)
        )
        all_data.append(data)
        
    # Concatenate
    full_data = all_data[0]
    for data in all_data[1:]:
        full_data = full_data.concatenate(data)
        
    # Create different sized datasets
    datasets = {}
    for name, n_samples in sizes.items():
        datasets[name] = full_data.subsample(n_samples, seed=rng.integers(10000))
        
    datasets['full'] = full_data
    
    return datasets
