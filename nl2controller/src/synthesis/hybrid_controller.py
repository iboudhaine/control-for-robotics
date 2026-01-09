"""
Hybrid Controller for Objective 2 Integration

Bridges symbolic control (FSA from GR(1) synthesis) with continuous dynamics
(robot models from colleagues' Robotics-project).

Two-layer architecture:
1. High-level: FSA provides discrete waypoints/goals
2. Low-level: Continuous controller tracks waypoints with real dynamics
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

# Add colleagues' project to path
ROBOTICS_PROJECT_PATH = Path(__file__).parent.parent.parent.parent.parent / "Robotics-project"
if ROBOTICS_PROJECT_PATH.exists():
    sys.path.insert(0, str(ROBOTICS_PROJECT_PATH))

try:
    import numpy as np
    from .controller import Controller

    # Import colleagues' components
    from models.unicycle import UnicycleModel
    from models.manipulator import TwoLinkManipulator
    from models.integrator import IntegratorModel
    from symbolic.grid_abstraction import GridAbstraction
    from controllers.unicycle.polar_controller import PolarCoordinateController as PolarController
    from sim.simulator import Simulator

    COLLEAGUES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Colleagues' components not available: {e}")
    COLLEAGUES_AVAILABLE = False
    np = None


@dataclass
class HybridControllerConfig:
    """Configuration for hybrid controller."""
    model_type: str = "unicycle"  # "unicycle", "manipulator", "integrator"
    grid_resolution: Tuple[int, int] = (20, 20)
    state_bounds: np.ndarray = None
    disturbance_mode: str = "random"  # "random", "worst_case", "none"
    low_level_controller: str = "polar"  # "polar", "lqr", "mpc"

    def __post_init__(self):
        if self.state_bounds is None:
            # Default bounds for unicycle model
            self.state_bounds = np.array([[0, 10], [0, 10]])


class HybridController:
    """
    Two-layer hybrid controller for continuous robot control with symbolic planning.

    Architecture:
    - High-level: Symbolic FSA (from GR(1)) provides discrete plan
    - Low-level: Continuous controller executes motion with real dynamics
    - Bridge: Grid abstraction maps between discrete and continuous

    Fills gaps in NL2Controller:
    ✅ Continuous dynamics (Models 1-3 from slides)
    ✅ Disturbance handling (W sets)
    ✅ Low-level control (tracking controllers)
    """

    def __init__(self,
                 fsa_controller: Controller,
                 config: Optional[HybridControllerConfig] = None):
        """
        Initialize hybrid controller.

        Args:
            fsa_controller: Symbolic controller from GR(1) synthesis
            config: Configuration for continuous control
        """
        if not COLLEAGUES_AVAILABLE:
            raise ImportError(
                "Colleagues' Robotics-project not found. "
                f"Expected at: {ROBOTICS_PROJECT_PATH}"
            )

        self.fsa = fsa_controller
        self.config = config or HybridControllerConfig()

        # Initialize robot model
        self.model = self._create_model()

        # Initialize grid abstraction (bridge)
        self.grid = GridAbstraction(
            bounds=self.config.state_bounds,
            resolution=self.config.grid_resolution,
            model=self.model
        )

        # Initialize low-level controller
        self.low_level_controller = self._create_low_level_controller()

        # Extract waypoints from FSA
        self.waypoints = self._extract_waypoints_from_fsa()
        self.current_waypoint_idx = 0
        self.current_fsa_state = self.fsa.initial_states[0] if self.fsa.initial_states else 0

    def _create_model(self):
        """Create robot model based on configuration."""
        if self.config.model_type == "unicycle":
            return UnicycleModel()
        elif self.config.model_type == "manipulator":
            return TwoLinkManipulator()
        elif self.config.model_type == "integrator":
            return IntegratorModel()
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")

    def _create_low_level_controller(self):
        """Create low-level tracking controller."""
        if self.config.low_level_controller == "polar":
            return PolarController(k_rho=0.5, k_alpha=1.0, k_beta=-0.3)
        else:
            # For now, only polar controller is implemented
            # Can extend to LQR, MPC later
            return PolarController(k_rho=0.5, k_alpha=1.0, k_beta=-0.3)

    def _extract_waypoints_from_fsa(self) -> List[np.ndarray]:
        """
        Extract waypoint sequence from FSA states.

        The FSA provides discrete states. We map these to grid cells,
        then to continuous waypoints.
        """
        waypoints = []

        # Start from initial state
        if not self.fsa.initial_states:
            return [self.grid.get_cell_center(0)]

        current_state = self.fsa.initial_states[0]
        visited = set()

        # Follow transitions to extract path
        while current_state not in visited:
            visited.add(current_state)

            # Get state info
            state = self.fsa.get_state(current_state)

            # Map state to waypoint
            # Strategy: use state ID as cell ID (simple mapping)
            # For more complex FSAs, would need better mapping
            cell_id = min(current_state, self.grid.n_cells - 1)
            waypoint = self.grid.get_cell_center(cell_id)
            waypoints.append(waypoint)

            # Move to next state
            successors = self.fsa.get_successors(current_state)
            if not successors:
                break

            # Take first valid transition
            current_state = successors[0].target

            # Prevent infinite loops
            if len(waypoints) > 100:
                break

        return waypoints if waypoints else [self.grid.get_cell_center(0)]

    def set_obstacles(self, obstacle_polygons: List[np.ndarray]):
        """Set obstacles in grid abstraction."""
        self.grid.set_obstacles(obstacle_polygons)

    def set_goal_region(self, goal_polygon: np.ndarray):
        """Set goal region in grid abstraction."""
        self.grid.set_goal_region(goal_polygon)

    def compute_control(self, x: np.ndarray, t: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute control input for current state.

        Args:
            x: Current continuous state
            t: Current time

        Returns:
            u: Control input
            diagnostics: Dictionary with debug info
        """
        # Get current target from waypoint sequence
        if self.current_waypoint_idx >= len(self.waypoints):
            self.current_waypoint_idx = len(self.waypoints) - 1

        target = self.waypoints[self.current_waypoint_idx]

        # Check if reached current waypoint
        distance = np.linalg.norm(x[:2] - target)
        if distance < 0.3:  # Threshold for waypoint reached
            self.current_waypoint_idx = min(
                self.current_waypoint_idx + 1,
                len(self.waypoints) - 1
            )
            target = self.waypoints[self.current_waypoint_idx]

        # Compute low-level control
        u, diag = self.low_level_controller.compute_control(x, t, target=target)

        # Add diagnostics
        diag.update({
            'waypoint_idx': self.current_waypoint_idx,
            'target': target,
            'distance_to_target': distance,
            'fsa_state': self.current_fsa_state,
        })

        return u, diag

    def simulate(self,
                 x0: np.ndarray,
                 duration: float,
                 dt: float = 0.1,
                 obstacles: Optional[List[np.ndarray]] = None) -> Dict[str, Any]:
        """
        Simulate closed-loop system with continuous dynamics.

        Args:
            x0: Initial state
            duration: Simulation duration
            dt: Time step
            obstacles: List of obstacle polygons

        Returns:
            Dictionary with simulation results
        """
        # Set obstacles if provided
        if obstacles:
            self.set_obstacles(obstacles)

        # Create simulator with colleagues' code
        simulator = Simulator(
            model=self.model,
            controller=self,
            disturbance_mode=self.config.disturbance_mode
        )

        # Run simulation
        result = simulator.run(x0, duration, dt=dt)

        # Add metadata
        result['model_type'] = self.config.model_type
        result['disturbance_mode'] = self.config.disturbance_mode
        result['num_waypoints'] = len(self.waypoints)

        return result

    def visualize(self, simulation_result: Dict[str, Any], save_path: Optional[str] = None):
        """
        Visualize simulation results.

        Args:
            simulation_result: Output from simulate()
            save_path: Path to save figure
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Extract data
        t = simulation_result['time']
        x = simulation_result['state']
        u = simulation_result.get('control', [])

        # Plot 1: Trajectory
        ax = axes[0, 0]
        self.grid.visualize(ax=ax, show_labels=False)
        ax.plot(x[:, 0], x[:, 1], 'b-', linewidth=2, label='Trajectory')
        ax.plot(x[0, 0], x[0, 1], 'go', markersize=10, label='Start')
        ax.plot(x[-1, 0], x[-1, 1], 'ro', markersize=10, label='End')

        # Plot waypoints
        waypoints_array = np.array(self.waypoints)
        ax.plot(waypoints_array[:, 0], waypoints_array[:, 1],
                'r*', markersize=12, label='Waypoints')

        ax.set_title('Trajectory (Continuous Dynamics)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: State evolution
        ax = axes[0, 1]
        ax.plot(t, x[:, 0], 'b-', label='x₁ (position)')
        ax.plot(t, x[:, 1], 'r-', label='x₂ (position)')
        if x.shape[1] > 2:
            ax.plot(t, x[:, 2], 'g-', label='x₃ (angle/velocity)')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('State')
        ax.set_title('State Evolution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Control inputs
        ax = axes[1, 0]
        if len(u) > 0:
            u_array = np.array(u)
            ax.plot(t[:-1], u_array[:, 0], 'b-', label='u₁')
            if u_array.shape[1] > 1:
                ax.plot(t[:-1], u_array[:, 1], 'r-', label='u₂')
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Control Input')
            ax.set_title('Control Inputs', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add control bounds from model
            if hasattr(self.model, 'u_min') and hasattr(self.model, 'u_max'):
                ax.axhline(self.model.u_max[0], color='b', linestyle='--', alpha=0.5)
                ax.axhline(self.model.u_min[0], color='b', linestyle='--', alpha=0.5)

        # Plot 4: Distance to target
        ax = axes[1, 1]
        distances = [np.linalg.norm(x[i, :2] - self.waypoints[min(i // 10, len(self.waypoints) - 1)])
                    for i in range(len(x))]
        ax.plot(t, distances, 'b-', linewidth=2)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Distance to Target [m]')
        ax.set_title('Tracking Performance', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Visualization saved: {save_path}")

        return fig


def create_hybrid_controller_from_pipeline_result(
    pipeline_result: Dict[str, Any],
    model_type: str = "unicycle",
    disturbance_mode: str = "random"
) -> HybridController:
    """
    Create hybrid controller from NL2Controller pipeline result.

    Args:
        pipeline_result: Output from extended_pipeline.py
        model_type: Which robot model to use
        disturbance_mode: How to handle disturbances

    Returns:
        Configured HybridController
    """
    # Extract FSA controller
    controller_data = pipeline_result['synthesis']['controller']
    fsa_controller = Controller.from_dict(controller_data)

    # Create configuration
    config = HybridControllerConfig(
        model_type=model_type,
        disturbance_mode=disturbance_mode
    )

    # Create hybrid controller
    return HybridController(fsa_controller, config)
