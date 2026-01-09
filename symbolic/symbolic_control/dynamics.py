"""
Dynamics Module - Abstract base class and concrete implementations.

This module isolates all model-specific math. To add a new robot model,
simply create a new class inheriting from Dynamics.
"""

from abc import ABC, abstractmethod
import numpy as np


class Dynamics(ABC):
    @property
    def disturbance_dim(self) -> int:
        """Dimension of the disturbance vector. Override if different from state_dim."""
        return self.state_dim

    @property
    def w_min(self) -> np.ndarray:
        """Minimum disturbance (vector)."""
        return -self.w_bound * np.ones(self.disturbance_dim)

    @property
    def w_max(self) -> np.ndarray:
        """Maximum disturbance (vector)."""
        return self.w_bound * np.ones(self.disturbance_dim)
    """
    Abstract base class for dynamical systems.
    
    Any concrete model must implement:
    - post(): Over-approximated successor set (for abstraction)
    - step(): Exact next state (for simulation)
    - state_dim, control_set: System properties
    """
    
    @abstractmethod
    def post(self, x_lo: np.ndarray, x_hi: np.ndarray, u: np.ndarray) -> tuple:
        """
        Compute over-approximated reachable set from cell [x_lo, x_hi] under input u.
        
        Args:
            x_lo: Lower-left corner of the cell
            x_hi: Upper-right corner of the cell
            u: Control input
            
        Returns:
            (succ_lo, succ_hi): Bounding box of all possible successors
        """
        pass
    
    @abstractmethod
    def step(self, x: np.ndarray, u: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        Compute exact next state (for simulation).
        
        Args:
            x: Current state
            u: Control input
            w: Disturbance
            
        Returns:
            Next state
        """
        pass
    
    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Dimension of the state space."""
        pass
    
    @property
    @abstractmethod
    def control_set(self) -> list:
        """List of all discrete control inputs."""
        pass


class IntegratorDynamics(Dynamics):
    """
    2D Integrator (velocity-controlled point robot).
    
    Dynamics:
        x1(t+1) = x1(t) + τ * (u1 + w1)
        x2(t+1) = x2(t) + τ * (u2 + w2)
    
    This system is MONOTONE: increasing x or u always increases the successor.
    This allows exact interval arithmetic (no sampling needed).
    """
    
    def __init__(self, tau=1.0, w_bound=0.05, u_values=None):
        """
        Args:
            tau: Sampling period
            w_bound: Disturbance bound (symmetric: w ∈ [-w_bound, w_bound])
            u_values: List of discrete control values per axis (default: [-1, -0.5, 0, 0.5, 1])
        """
        self.tau = tau
        self.w_bound = w_bound
        
        # Default control discretization
        if u_values is None:
            u_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
        
        # Build control set: all combinations of (u1, u2)
        self._control_set = [
            np.array([u1, u2]) 
            for u1 in u_values 
            for u2 in u_values
        ]
    
    @property
    def state_dim(self) -> int:
        return 2
    
    @property
    def control_set(self) -> list:
        return self._control_set
    
    def post(self, x_lo: np.ndarray, x_hi: np.ndarray, u: np.ndarray) -> tuple:
        """
        Over-approximation using MONOTONICITY.
        
        Since f(x, u, w) = x + τ(u + w) is monotone increasing in x and w:
        - Minimum successor: f(x_lo, u, w_min)
        - Maximum successor: f(x_hi, u, w_max)
        """
        w_lo = self.w_min
        w_hi = self.w_max
        
        succ_lo = self.step(x_lo, u, w_lo)
        succ_hi = self.step(x_hi, u, w_hi)
        
        return succ_lo, succ_hi
    
    def step(self, x: np.ndarray, u: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Exact state update for simulation."""
        return x + self.tau * (u + w)


class UnicycleDynamics(Dynamics):
    @property
    def disturbance_dim(self) -> int:
        return 3
    """
    3D Unicycle model (mobile robot with heading).
    
    Dynamics:
        x1(t+1) = x1(t) + τ * (u1 * cos(x3(t)) + w1)
        x2(t+1) = x2(t) + τ * (u1 * sin(x3(t)) + w2)
        x3(t+1) = x3(t) + τ * (u2 + w3)  (mod 2π, wrapped to [-π, π])
    
    where:
        - (x1, x2) is position, x3 is heading angle
        - u1 is linear velocity, u2 is angular velocity
        - w = (w1, w2, w3) is disturbance
    
    This system is NON-MONOTONE due to sin/cos terms.
    We use Jacobian-based growth bounds for over-approximation.
    """
    
    def __init__(self, tau=1.0, w_bound=0.05, v_values=None, omega_values=None):
        """
        Args:
            tau: Sampling period
            w_bound: Disturbance bound (symmetric: w ∈ [-w_bound, w_bound]^3)
            v_values: List of linear velocity values (u1 ∈ [0.25, 1])
            omega_values: List of angular velocity values (u2 ∈ [-1, 1])
        """
        self.tau = tau
        self.w_bound = w_bound
        
        # Default control discretization
        if v_values is None:
            v_values = [0.25, 0.5, 0.75, 1.0]
        if omega_values is None:
            omega_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
        
        self.v_values = np.array(v_values)
        self.omega_values = np.array(omega_values)
        self.v_max = np.max(np.abs(v_values))
        
        # Build control set: all combinations of (v, omega)
        self._control_set = [
            np.array([v, omega]) 
            for v in v_values 
            for omega in omega_values
        ]
    
    @property
    def state_dim(self) -> int:
        return 3
    
    @property
    def control_set(self) -> list:
        return self._control_set
    
    def _wrap_angle(self, theta: float) -> float:
        """Wrap angle to [-π, π]."""
        return ((theta + np.pi) % (2 * np.pi)) - np.pi
    
    def step(self, x: np.ndarray, u: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        Exact state update for simulation.
        
        Args:
            x: State [x1, x2, x3] (position and heading)
            u: Control [u1, u2] (linear and angular velocity)
            w: Disturbance [w1, w2, w3]
            
        Returns:
            Next state with angle wrapped to [-π, π]
        """
        x1, x2, x3 = x
        u1, u2 = u
        w1, w2, w3 = w
        
        x1_next = x1 + self.tau * (u1 * np.cos(x3) + w1)
        x2_next = x2 + self.tau * (u1 * np.sin(x3) + w2)
        x3_next = self._wrap_angle(x3 + self.tau * (u2 + w3))
        
        return np.array([x1_next, x2_next, x3_next])
    
    def post(self, x_lo: np.ndarray, x_hi: np.ndarray, u: np.ndarray) -> tuple:
        """
        Over-approximation using Jacobian-based growth bounds.
        
        For non-monotone systems, we use:
            successor ∈ [f(x*, u, w*) - Δ, f(x*, u, w*) + Δ]
        where:
            x* = center of cell
            w* = center of disturbance (= 0 for symmetric bounds)
            Δ = D_x * δ_x + D_w * δ_w
            D_x, D_w = bounds on Jacobians
            δ_x, δ_w = half-widths of state cell and disturbance
        
        Jacobian analysis for unicycle:
            ∂f/∂x = [[1, 0, -τ*u1*sin(x3)],
                     [0, 1,  τ*u1*cos(x3)],
                     [0, 0,  1]]
            
            |∂f/∂x| ≤ [[1, 0, τ*|u1|],
                       [0, 1, τ*|u1|],
                       [0, 0, 1]]
            
            ∂f/∂w = τ * I_3
        """
        u1, _u2 = u
        
        # Cell center and half-width
        x_center = (x_lo + x_hi) / 2
        delta_x = (x_hi - x_lo) / 2
        
        # Disturbance center (0 for symmetric) and half-width
        delta_w = np.array([self.w_bound, self.w_bound, self.w_bound])
        w_center = np.zeros(3)
        
        # Compute center transition
        f_center = self.step(x_center, u, w_center)
        
        # Jacobian bounds (element-wise absolute values)
        # d_x: bound on |∂f_i/∂x_j|
        # For x3 derivative: |∂f1/∂x3| = τ*|u1|*|sin(x3)| ≤ τ*|u1|
        #                    |∂f2/∂x3| = τ*|u1|*|cos(x3)| ≤ τ*|u1|
        d_x = np.array([
            [1.0, 0.0, self.tau * np.abs(u1)],
            [0.0, 1.0, self.tau * np.abs(u1)],
            [0.0, 0.0, 1.0]
        ])
        
        # d_w: bound on |∂f_i/∂w_j| = τ * I_3
        d_w = self.tau * np.eye(3)
        
        # Growth bound: Δ = d_x @ δ_x + d_w @ δ_w
        growth = d_x @ delta_x + d_w @ delta_w
        
        # Over-approximation bounds
        succ_lo = f_center - growth
        succ_hi = f_center + growth
        
        # Wrap angle bounds (conservative: if interval crosses ±π, expand to full range)
        theta_lo, theta_hi = succ_lo[2], succ_hi[2]
        if theta_hi - theta_lo >= 2 * np.pi:
            # Interval covers full circle
            succ_lo[2] = -np.pi
            succ_hi[2] = np.pi
        else:
            # Wrap both bounds
            theta_lo_wrapped = self._wrap_angle(theta_lo)
            theta_hi_wrapped = self._wrap_angle(theta_hi)
            
            # Check if wrapping caused inversion (crossed ±π boundary)
            if theta_lo_wrapped > theta_hi_wrapped:
                # Conservative: expand to full range
                succ_lo[2] = -np.pi
                succ_hi[2] = np.pi
            else:
                succ_lo[2] = theta_lo_wrapped
                succ_hi[2] = theta_hi_wrapped
        
        return succ_lo, succ_hi


class ManipulatorDynamics(Dynamics):
    """
    4D Two-Link Planar Robotic Manipulator.
    
    State: x = [θ₁, θ₂, θ̇₁, θ̇₂] (joint angles and velocities)
    Control: u = [τ₁, τ₂] (joint torques)
    
    Dynamics:
        x(t+1) = x(t) + τ * f(x(t), u(t))
        
        f(x, u) = [θ̇, M(θ)⁻¹(u - c(θ, θ̇) - g(θ))]
    
    where:
        M(θ) = mass matrix (2x2)
        c(θ, θ̇) = Coriolis/centrifugal forces (2x1)
        g(θ) = gravity forces (2x1)
    
    This system is NON-MONOTONE due to trigonometric nonlinearities.
    We use sampling-based over-approximation for the post() method.
    """
    
    def __init__(self, tau=0.1, w_bound=0.01, 
                 m1=1.0, m2=1.0, l1=0.5, l2=0.5, gravity=9.81,
                 torque_values=None):
        """
        Args:
            tau: Sampling period
            w_bound: Disturbance bound (symmetric)
            m1, m2: Link masses (kg)
            l1, l2: Link lengths (m)
            gravity: Gravitational acceleration (m/s²)
            torque_values: List of torque values for each joint
        """
        self.tau = tau
        self.w_bound = w_bound
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.gravity = gravity
        
        # Default torque discretization
        if torque_values is None:
            torque_values = [-10.0, -5.0, 0.0, 5.0, 10.0]
        
        self.torque_values = np.array(torque_values)
        
        # Build control set: all combinations of (τ₁, τ₂)
        self._control_set = [
            np.array([t1, t2])
            for t1 in torque_values
            for t2 in torque_values
        ]
    
    @property
    def state_dim(self) -> int:
        return 4
    
    @property
    def disturbance_dim(self) -> int:
        return 4
    
    @property
    def control_set(self) -> list:
        return self._control_set
    
    def _mass_matrix(self, theta: np.ndarray) -> np.ndarray:
        """
        Compute the mass matrix M(θ).
        
        M = [M11, M12]
            [M21, M22]
        """
        theta1, theta2 = theta
        c2 = np.cos(theta2)
        
        m1, m2 = self.m1, self.m2
        l1, l2 = self.l1, self.l2
        
        M11 = m1 * l1**2 + m2 * (l1**2 + 2*l1*l2*c2 + l2**2)
        M12 = m2 * (l1*l2*c2 + l2**2)
        M21 = M12
        M22 = m2 * l2**2
        
        return np.array([[M11, M12], [M21, M22]])
    
    def _coriolis(self, theta: np.ndarray, theta_dot: np.ndarray) -> np.ndarray:
        """
        Compute Coriolis/centrifugal forces c(θ, θ̇).
        """
        theta1, theta2 = theta
        dtheta1, dtheta2 = theta_dot
        s2 = np.sin(theta2)
        
        m2 = self.m2
        l1, l2 = self.l1, self.l2
        
        h = m2 * l1 * l2 * s2
        
        c1 = -h * (2*dtheta1*dtheta2 + dtheta2**2)
        c2 = h * dtheta1**2
        
        return np.array([c1, c2])
    
    def _gravity_forces(self, theta: np.ndarray) -> np.ndarray:
        """
        Compute gravity forces g(θ).
        """
        theta1, theta2 = theta
        c1 = np.cos(theta1)
        c12 = np.cos(theta1 + theta2)
        
        m1, m2 = self.m1, self.m2
        l1, l2 = self.l1, self.l2
        g = self.gravity
        
        g1 = (m1 + m2) * g * l1 * c1 + m2 * g * l2 * c12
        g2 = m2 * g * l2 * c12
        
        return np.array([g1, g2])
    
    def step(self, x: np.ndarray, u: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        Exact state update for simulation.
        
        Args:
            x: State [θ₁, θ₂, θ̇₁, θ̇₂]
            u: Control [τ₁, τ₂]
            w: Disturbance [w₁, w₂, w₃, w₄]
            
        Returns:
            Next state
        """
        theta = x[:2]
        theta_dot = x[2:]
        
        # Compute dynamics
        M = self._mass_matrix(theta)
        c = self._coriolis(theta, theta_dot)
        g = self._gravity_forces(theta)
        
        # Joint accelerations: θ̈ = M⁻¹(u - c - g)
        theta_ddot = np.linalg.solve(M, u - c - g)
        
        # Euler integration with disturbance
        theta_new = theta + self.tau * (theta_dot + w[:2])
        theta_dot_new = theta_dot + self.tau * (theta_ddot + w[2:])
        
        return np.array([theta_new[0], theta_new[1], theta_dot_new[0], theta_dot_new[1]])
    
    def post(self, x_lo: np.ndarray, x_hi: np.ndarray, u: np.ndarray) -> tuple:
        """
        Over-approximation using sampling at corners and center.
        
        For this highly nonlinear system, we sample the cell corners
        and center, then expand by the Lipschitz constant estimate.
        """
        # Sample points: corners + center
        samples = []
        
        # Center
        x_center = (x_lo + x_hi) / 2
        samples.append(x_center)
        
        # Corners (2^4 = 16 corners for 4D)
        for i in range(16):
            corner = np.array([
                x_lo[0] if (i & 1) == 0 else x_hi[0],
                x_lo[1] if (i & 2) == 0 else x_hi[1],
                x_lo[2] if (i & 4) == 0 else x_hi[2],
                x_lo[3] if (i & 8) == 0 else x_hi[3],
            ])
            samples.append(corner)
        
        # Compute transitions for all samples with extreme disturbances
        w_lo = self.w_min
        w_hi = self.w_max
        
        all_successors = []
        for x_sample in samples:
            # Try with min and max disturbance
            for w in [w_lo, w_hi, np.zeros(4)]:
                try:
                    succ = self.step(x_sample, u, w)
                    all_successors.append(succ)
                except np.linalg.LinAlgError:
                    # Singular matrix - skip this sample
                    continue
        
        if not all_successors:
            # Fallback: return empty (will be pruned)
            return x_lo.copy(), x_lo.copy()
        
        all_successors = np.array(all_successors)
        
        # Bounding box of all samples
        succ_lo = np.min(all_successors, axis=0)
        succ_hi = np.max(all_successors, axis=0)
        
        # Add margin for Lipschitz continuity
        # The dynamics are Lipschitz, so we add a safety margin
        delta_x = (x_hi - x_lo) / 2
        delta_w = np.array([self.w_bound] * 4)
        
        # Conservative Lipschitz estimate
        L_x = 1.0 + self.tau * 10.0  # Conservative bound
        L_w = self.tau
        
        margin = L_x * np.max(delta_x) * 0.5 + L_w * np.max(delta_w)
        
        succ_lo -= margin
        succ_hi += margin
        
        return succ_lo, succ_hi

