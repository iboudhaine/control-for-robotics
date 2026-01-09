"""
Trajectory Generation for Reference Tracking

Provides reference trajectory generators for trajectory tracking tasks:
- Circle trajectory
- Figure-8 (lemniscate) trajectory
- Line/straight trajectory
- Custom waypoint trajectories

Each generator provides position, velocity, and optionally acceleration
references for the unicycle robot.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Callable


class TrajectoryGenerator(ABC):
    """Base class for reference trajectory generators."""
    
    def __init__(self, center: np.ndarray = None):
        """
        Args:
            center: Center point of the trajectory (x, y)
        """
        self.center = np.array([0.0, 0.0]) if center is None else np.array(center)
        
    @abstractmethod
    def get_reference(self, t: float) -> dict:
        """Get reference state at time t.
        
        Args:
            t: Time
            
        Returns:
            Dictionary with keys:
            - 'position': (x, y) position
            - 'theta': heading angle
            - 'velocity': linear velocity
            - 'omega': angular velocity
            - 'state': full state [x, y, theta]
        """
        pass
    
    def get_reference_sequence(self, times: np.ndarray) -> dict:
        """Get reference sequence over time array.
        
        Args:
            times: Array of time points
            
        Returns:
            Dictionary with arrays for each reference quantity
        """
        refs = [self.get_reference(t) for t in times]
        return {
            'positions': np.array([r['position'] for r in refs]),
            'thetas': np.array([r['theta'] for r in refs]),
            'velocities': np.array([r['velocity'] for r in refs]),
            'omegas': np.array([r['omega'] for r in refs]),
            'states': np.array([r['state'] for r in refs])
        }


class CircleTrajectory(TrajectoryGenerator):
    """Circular reference trajectory.
    
    The unicycle follows a circle of given radius at constant speed.
    
    Parametrization:
        x(t) = cx + r * cos(ω*t + φ)
        y(t) = cy + r * sin(ω*t + φ)
        θ(t) = ω*t + φ + π/2  (tangent direction)
    
    where ω = v/r is the angular rate around the circle.
    """
    
    def __init__(
        self,
        radius: float = 1.0,
        speed: float = 0.5,
        center: np.ndarray = None,
        clockwise: bool = False,
        start_angle: float = 0.0
    ):
        """
        Args:
            radius: Circle radius
            speed: Linear speed along the circle
            center: Center point (x, y)
            clockwise: If True, traverse clockwise
            start_angle: Starting angle on the circle (radians)
        """
        super().__init__(center)
        self.radius = radius
        self.speed = speed
        self.direction = -1 if clockwise else 1
        self.start_angle = start_angle
        
        # Angular rate around the circle
        self.omega_circle = self.direction * speed / radius
        
    def get_reference(self, t: float) -> dict:
        """Get circular reference at time t."""
        # Angle on the circle
        angle = self.start_angle + self.omega_circle * t
        
        # Position
        x = self.center[0] + self.radius * np.cos(angle)
        y = self.center[1] + self.radius * np.sin(angle)
        
        # Heading (tangent to circle)
        theta = angle + self.direction * np.pi / 2
        theta = np.arctan2(np.sin(theta), np.cos(theta))  # Normalize to [-π, π]
        
        # Velocities
        v = self.speed
        omega = self.omega_circle
        
        return {
            'position': np.array([x, y]),
            'theta': theta,
            'velocity': v,
            'omega': omega,
            'state': np.array([x, y, theta])
        }


class Figure8Trajectory(TrajectoryGenerator):
    """Figure-8 (lemniscate) reference trajectory.
    
    Creates a figure-8 pattern using Lemniscate of Bernoulli parametrization:
        x(t) = a * sin(ωt)
        y(t) = a * sin(ωt) * cos(ωt) = (a/2) * sin(2ωt)
    
    Or using two circles (smoother, more practical for robots):
        x(t) = a * sin(ωt)  
        y(t) = (a/2) * sin(2ωt)
    """
    
    def __init__(
        self,
        scale: float = 1.0,
        period: float = 10.0,
        center: np.ndarray = None
    ):
        """
        Args:
            scale: Size scale factor
            period: Time to complete one figure-8
            center: Center point (x, y)
        """
        super().__init__(center)
        self.scale = scale
        self.period = period
        self.omega = 2 * np.pi / period
        
    def get_reference(self, t: float) -> dict:
        """Get figure-8 reference at time t."""
        omega_t = self.omega * t
        
        # Position (figure-8 parametrization)
        x = self.center[0] + self.scale * np.sin(omega_t)
        y = self.center[1] + self.scale * 0.5 * np.sin(2 * omega_t)
        
        # Velocity components
        dx_dt = self.scale * self.omega * np.cos(omega_t)
        dy_dt = self.scale * self.omega * np.cos(2 * omega_t)
        
        # Heading (direction of motion)
        theta = np.arctan2(dy_dt, dx_dt)
        
        # Linear velocity magnitude
        v = np.sqrt(dx_dt**2 + dy_dt**2)
        
        # Angular velocity (derivative of heading)
        # θ = atan2(dy/dt, dx/dt)
        # dθ/dt = (d²y/dt² * dx/dt - d²x/dt² * dy/dt) / (dx/dt² + dy/dt²)
        d2x_dt2 = -self.scale * self.omega**2 * np.sin(omega_t)
        d2y_dt2 = -2 * self.scale * self.omega**2 * np.sin(2 * omega_t)
        
        denom = dx_dt**2 + dy_dt**2
        if denom > 1e-6:
            omega_heading = (d2y_dt2 * dx_dt - d2x_dt2 * dy_dt) / denom
        else:
            omega_heading = 0.0
            
        return {
            'position': np.array([x, y]),
            'theta': theta,
            'velocity': v,
            'omega': omega_heading,
            'state': np.array([x, y, theta])
        }


class LineTrajectory(TrajectoryGenerator):
    """Straight line trajectory between waypoints.
    
    Moves at constant speed along a straight line from start to end point.
    """
    
    def __init__(
        self,
        start: np.ndarray,
        end: np.ndarray,
        speed: float = 0.5
    ):
        """
        Args:
            start: Starting point (x, y)
            end: Ending point (x, y)
            speed: Linear speed
        """
        super().__init__(center=(np.array(start) + np.array(end)) / 2)
        self.start = np.array(start)
        self.end = np.array(end)
        self.speed = speed
        
        # Direction and length
        self.direction = self.end - self.start
        self.length = np.linalg.norm(self.direction)
        if self.length > 1e-6:
            self.direction = self.direction / self.length
        else:
            self.direction = np.array([1.0, 0.0])
            
        self.theta = np.arctan2(self.direction[1], self.direction[0])
        self.duration = self.length / speed if speed > 0 else np.inf
        
    def get_reference(self, t: float) -> dict:
        """Get line reference at time t."""
        # Distance traveled
        distance = min(self.speed * t, self.length)
        
        # Position
        position = self.start + distance * self.direction
        
        # Check if finished
        if t >= self.duration:
            v = 0.0
        else:
            v = self.speed
            
        return {
            'position': position,
            'theta': self.theta,
            'velocity': v,
            'omega': 0.0,
            'state': np.array([position[0], position[1], self.theta])
        }


class WaypointTrajectory(TrajectoryGenerator):
    """Trajectory through a sequence of waypoints.
    
    Connects waypoints with straight line segments at constant speed.
    """
    
    def __init__(
        self,
        waypoints: np.ndarray,
        speed: float = 0.5,
        loop: bool = False
    ):
        """
        Args:
            waypoints: Array of waypoints, shape (n_points, 2)
            speed: Linear speed between waypoints
            loop: Whether to loop back to start
        """
        self.waypoints = np.array(waypoints)
        self.speed = speed
        self.loop = loop
        
        if loop:
            self.waypoints = np.vstack([self.waypoints, self.waypoints[0]])
            
        # Compute segment lengths and cumulative distances
        self.segments = []
        self.cum_distances = [0.0]
        
        for i in range(len(self.waypoints) - 1):
            segment = LineTrajectory(
                self.waypoints[i], self.waypoints[i+1], speed
            )
            self.segments.append(segment)
            self.cum_distances.append(
                self.cum_distances[-1] + segment.length
            )
            
        self.total_length = self.cum_distances[-1]
        self.total_duration = self.total_length / speed if speed > 0 else np.inf
        
        super().__init__(center=np.mean(waypoints, axis=0))
        
    def get_reference(self, t: float) -> dict:
        """Get waypoint trajectory reference at time t."""
        if self.loop:
            t = t % self.total_duration
            
        # Find current segment
        distance = self.speed * t
        
        for i, segment in enumerate(self.segments):
            if distance <= self.cum_distances[i+1]:
                segment_t = (distance - self.cum_distances[i]) / self.speed
                return segment.get_reference(segment_t)
                
        # Past all waypoints - return last position
        return self.segments[-1].get_reference(self.segments[-1].duration)


class SmoothTrajectory(TrajectoryGenerator):
    """Smooth trajectory from arbitrary position/velocity functions.
    
    Allows defining custom trajectories with smooth (differentiable) motion.
    """
    
    def __init__(
        self,
        x_func: Callable[[float], float],
        y_func: Callable[[float], float],
        dx_func: Callable[[float], float] = None,
        dy_func: Callable[[float], float] = None
    ):
        """
        Args:
            x_func: Function x(t)
            y_func: Function y(t)
            dx_func: Function dx/dt (optional, computed numerically if None)
            dy_func: Function dy/dt (optional, computed numerically if None)
        """
        super().__init__()
        self.x_func = x_func
        self.y_func = y_func
        self.dx_func = dx_func
        self.dy_func = dy_func
        self._dt = 1e-5  # For numerical differentiation
        
    def get_reference(self, t: float) -> dict:
        """Get smooth trajectory reference at time t."""
        x = self.x_func(t)
        y = self.y_func(t)
        
        # Compute velocities
        if self.dx_func is not None:
            dx = self.dx_func(t)
        else:
            dx = (self.x_func(t + self._dt) - self.x_func(t - self._dt)) / (2 * self._dt)
            
        if self.dy_func is not None:
            dy = self.dy_func(t)
        else:
            dy = (self.y_func(t + self._dt) - self.y_func(t - self._dt)) / (2 * self._dt)
            
        theta = np.arctan2(dy, dx)
        v = np.sqrt(dx**2 + dy**2)
        
        # Angular velocity (numerical)
        theta_next = np.arctan2(
            (self.y_func(t + self._dt) - self.y_func(t - self._dt)) / (2 * self._dt),
            (self.x_func(t + self._dt) - self.x_func(t - self._dt)) / (2 * self._dt)
        )
        theta_prev = np.arctan2(
            (self.y_func(t) - self.y_func(t - 2*self._dt)) / (2 * self._dt),
            (self.x_func(t) - self.x_func(t - 2*self._dt)) / (2 * self._dt)
        )
        
        # Handle angle wrapping
        omega = np.arctan2(np.sin(theta_next - theta_prev), np.cos(theta_next - theta_prev)) / (2 * self._dt)
        
        return {
            'position': np.array([x, y]),
            'theta': theta,
            'velocity': v,
            'omega': omega,
            'state': np.array([x, y, theta])
        }
