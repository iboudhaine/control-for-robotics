"""
Abstraction Module - Grid discretization and transition computation.

This module is MODEL-AGNOSTIC. It works with any Dynamics subclass.
The grid and transitions are computed using the dynamics.post() method.
"""

import numpy as np
from tqdm.auto import tqdm
from itertools import product
from .dynamics import Dynamics


class Abstraction:
    """
    Finite-state abstraction of a continuous dynamical system.
    
    Discretizes the state space into a uniform grid and computes
    transitions between cells using over-approximation.
    """
    
    def __init__(self, dynamics: Dynamics, state_bounds: np.ndarray, eta):
        """
        Args:
            dynamics: A Dynamics object (e.g., IntegratorDynamics)
            state_bounds: Array of shape (state_dim, 2) with [min, max] per dimension
                          Example: [[-10, 10], [-10, 10]] for 2D
            eta: Grid cell size. Can be:
                 - A single float (same for all dimensions)
                 - A list/array of floats (one per dimension)
        """
        self.dynamics = dynamics
        self.state_bounds = np.array(state_bounds)
        
        # Handle per-dimension eta
        if isinstance(eta, (int, float)):
            self.eta = np.array([float(eta)] * dynamics.state_dim)
        else:
            self.eta = np.array(eta, dtype=float)
            if len(self.eta) != dynamics.state_dim:
                raise ValueError(f"eta must have {dynamics.state_dim} elements, got {len(self.eta)}")
        
        # Compute grid dimensions (different for each dimension now)
        self.grid_shape = tuple(
            int(np.ceil((self.state_bounds[d, 1] - self.state_bounds[d, 0]) / self.eta[d]))
            for d in range(dynamics.state_dim)
        )
        self.num_cells = int(np.prod(self.grid_shape))
        
        # Transitions: {(cell_idx, u_idx): set of successor cell indices}
        self.transitions = {}
    
    def cell_to_bounds(self, cell_idx: int) -> tuple:
        """
        Convert a flat cell index to its spatial bounds.
        
        Returns:
            (lo, hi): Lower-left and upper-right corners of the cell
        """
        # Convert flat index to multi-dimensional index
        multi_idx = np.unravel_index(cell_idx, self.grid_shape)
        
        lo = np.array([
            self.state_bounds[d, 0] + multi_idx[d] * self.eta[d]
            for d in range(self.dynamics.state_dim)
        ])
        hi = np.array([
            lo[d] + self.eta[d]
            for d in range(self.dynamics.state_dim)
        ])
        
        return lo, hi
    
    def bounds_to_cells(self, lo: np.ndarray, hi: np.ndarray) -> set:
        """
        Find all cell indices that overlap with the rectangle [lo, hi].
        
        Args:
            lo: Lower bound of the rectangle
            hi: Upper bound of the rectangle
        
        Returns:
            Set of cell indices
        """
        cells = set()
        
        for d in range(self.dynamics.state_dim):
            if lo[d] < self.state_bounds[d, 0] or hi[d] > self.state_bounds[d, 1]:
                # Successor extends outside grid - not safe!
                return set()
        
        # Compute index ranges for each dimension
        idx_ranges = []
        for d in range(self.dynamics.state_dim):
            # Clamp to grid bounds
            lo_clamped = max(lo[d], self.state_bounds[d, 0])
            hi_clamped = min(hi[d], self.state_bounds[d, 1])
            
            if lo_clamped >= hi_clamped:
                # No overlap with grid in this dimension
                return set()
            
            idx_lo = int((lo_clamped - self.state_bounds[d, 0]) / self.eta[d])
            idx_hi = int(np.ceil((hi_clamped - self.state_bounds[d, 0]) / self.eta[d]))
            
            # Clamp to valid indices
            idx_lo = max(0, idx_lo)
            idx_hi = min(self.grid_shape[d], idx_hi)
            
            idx_ranges.append(range(idx_lo, idx_hi))
        
        # Generate all combinations
        for multi_idx in product(*idx_ranges):
            flat_idx = np.ravel_multi_index(multi_idx, self.grid_shape)
            cells.add(flat_idx)
        
        return cells
    
    def build_transitions(self, progress_callback=None):
        """
        Compute all transitions for every (cell, control) pair.
        
        For each cell ξ and control u:
            1. Get cell bounds [x_lo, x_hi]
            2. Compute over-approximated successor via dynamics.post()
            3. Find all cells overlapping with successor bounds
            
        Args:
            progress_callback: Optional callable(current, total, message) for progress updates
        """
        total = self.num_cells
        num_controls = len(self.dynamics.control_set)
        print(f"Building transitions for {total} cells × {num_controls} controls...")
        
        for cell_idx in tqdm(range(self.num_cells)):
            x_lo, x_hi = self.cell_to_bounds(cell_idx)
            
            for u_idx, u in enumerate(self.dynamics.control_set):
                # Over-approximated successor bounds
                succ_lo, succ_hi = self.dynamics.post(x_lo, x_hi, u)
                
                # Find overlapping cells
                succ_cells = self.bounds_to_cells(succ_lo, succ_hi)
                
                self.transitions[(cell_idx, u_idx)] = succ_cells
            
            # Call progress callback if provided
            if progress_callback is not None and cell_idx % max(1, total // 100) == 0:
                progress_callback(cell_idx + 1, total, f"Building transitions: {cell_idx + 1}/{total} cells")
        
        print(f"Done. Total transitions: {len(self.transitions)}")
    
    def get_cell_center(self, cell_idx: int) -> np.ndarray:
        """Get the center point of a cell (useful for simulation)."""
        lo, hi = self.cell_to_bounds(cell_idx)
        return (lo + hi) / 2
    
    def point_to_cell(self, x: np.ndarray) -> int:
        """Convert a continuous state to its cell index. Returns -1 if out of bounds."""
        multi_idx = []
        for d in range(self.dynamics.state_dim):
            idx = int((x[d] - self.state_bounds[d, 0]) / self.eta[d])
            # Check if out of bounds
            if idx < 0 or idx >= self.grid_shape[d]:
                return -1  # Out of bounds
            multi_idx.append(idx)
        return np.ravel_multi_index(tuple(multi_idx), self.grid_shape)
    
    def is_in_bounds(self, x: np.ndarray) -> bool:
        """Check if a state is within the grid bounds."""
        for d in range(self.dynamics.state_dim):
            if x[d] < self.state_bounds[d, 0] or x[d] >= self.state_bounds[d, 1]:
                return False
        return True
