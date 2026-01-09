"""
Continuous Dynamics Simulator

Wrapper for colleagues' continuous simulation that integrates with NL2Controller.
Provides the same interface as GridWorld but uses real robot models.

OBJECTIVE 2 GAP MITIGATION:
✅ Continuous dynamics (Models 1-3 from slides)
✅ Disturbance handling (bounded sets W)
✅ Real robot models (unicycle, manipulator, integrator)
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

# Add colleagues' project to path
ROBOTICS_PROJECT_PATH = Path(__file__).parent.parent.parent.parent.parent / "Robotics-project"
if ROBOTICS_PROJECT_PATH.exists():
    sys.path.insert(0, str(ROBOTICS_PROJECT_PATH))

try:
    from models.unicycle import UnicycleModel
    from models.manipulator import TwoLinkManipulator
    from models.integrator import IntegratorModel
    from sim.simulator import Simulator
    COLLEAGUES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Colleagues' components not available: {e}")
    COLLEAGUES_AVAILABLE = False


class ContinuousSimulator:
    """
    Continuous dynamics simulator using colleagues' robot models.

    Replaces grid-world simulation with real continuous dynamics:
    - Model 1: Double integrator (IntegratorModel)
    - Model 2: Unicycle with orientation (UnicycleModel)
    - Model 3: Two-link manipulator (TwoLinkManipulator)

    All models include:
    - Continuous state space (not discrete grid)
    - Differential equations with real dynamics
    - Bounded disturbances W (configurable)
    - Input constraints from slides
    """

    def __init__(self,
                 model_type: str = "unicycle",
                 disturbance_mode: str = "random",
                 enable_visualization: bool = True):
        """
        Initialize continuous simulator.

        Args:
            model_type: "unicycle", "manipulator", or "integrator"
            disturbance_mode: "random", "worst_case", or "none"
            enable_visualization: Whether to enable plotting
        """
        if not COLLEAGUES_AVAILABLE:
            raise ImportError(
                f"Colleagues' Robotics-project not found at {ROBOTICS_PROJECT_PATH}. "
                "Cannot use continuous dynamics."
            )

        self.model_type = model_type
        self.disturbance_mode = disturbance_mode
        self.enable_visualization = enable_visualization

        # Create robot model
        self.model = self._create_model()

        # Store simulation results
        self.last_result = None

    def _create_model(self):
        """Create robot model based on type."""
        if self.model_type == "unicycle":
            return UnicycleModel()
        elif self.model_type == "manipulator":
            return TwoLinkManipulator()
        elif self.model_type == "integrator":
            return IntegratorModel()
        else:
            raise ValueError(
                f"Unknown model type: {self.model_type}. "
                "Valid options: 'unicycle', 'manipulator', 'integrator'"
            )

    def run_controller(self,
                      controller_dict: Dict[str, Any],
                      initial_state: Optional[np.ndarray] = None,
                      duration: float = 10.0,
                      dt: float = 0.1) -> Dict[str, Any]:
        """
        Run controller in continuous simulation.

        This is the main interface called by extended_pipeline.py.

        Args:
            controller_dict: Controller from synthesis stage
            initial_state: Initial continuous state (if None, uses default)
            duration: Simulation duration [s]
            dt: Time step [s]

        Returns:
            Dictionary with simulation results
        """
        from ..synthesis.hybrid_controller import (
            HybridController,
            HybridControllerConfig,
            Controller
        )

        # Convert controller dict to Controller object
        fsa_controller = Controller.from_dict(controller_dict)

        # Create hybrid controller configuration
        config = HybridControllerConfig(
            model_type=self.model_type,
            disturbance_mode=self.disturbance_mode
        )

        # Create hybrid controller
        hybrid = HybridController(fsa_controller, config)

        # Set initial state
        if initial_state is None:
            if self.model_type == "unicycle":
                initial_state = np.array([1.0, 1.0, 0.0])  # x, y, theta
            elif self.model_type == "manipulator":
                initial_state = np.zeros(4)  # theta1, theta2, theta1_dot, theta2_dot
            elif self.model_type == "integrator":
                initial_state = np.array([1.0, 1.0, 0.0, 0.0])  # x, y, x_dot, y_dot
            else:
                initial_state = np.zeros(3)

        # Run simulation
        result = hybrid.simulate(initial_state, duration, dt)

        # Store result for visualization
        self.last_result = result
        self.last_hybrid_controller = hybrid

        return self._format_result(result)

    def _format_result(self, sim_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format simulation result to match expected interface.

        Converts colleagues' simulator output to format expected by
        existing NL2Controller code.
        """
        return {
            'success': True,
            'trajectory': sim_result['state'].tolist(),
            'time': sim_result['time'].tolist(),
            'control_inputs': sim_result.get('control', []),
            'model_type': sim_result.get('model_type', self.model_type),
            'disturbance_mode': sim_result.get('disturbance_mode', self.disturbance_mode),
            'continuous_dynamics': True,  # Flag to indicate this is NOT grid-world
            'num_waypoints': sim_result.get('num_waypoints', 0),
            'metadata': {
                'simulator': 'ContinuousSimulator',
                'model': self.model.__class__.__name__,
                'state_dim': self.model.state_dim,
                'input_dim': self.model.input_dim,
            }
        }

    def visualize(self, save_path: Optional[str] = None):
        """
        Visualize last simulation result.

        Args:
            save_path: Path to save figure (if None, displays)
        """
        if self.last_result is None:
            print("No simulation results to visualize. Run simulation first.")
            return

        if not self.enable_visualization:
            print("Visualization disabled.")
            return

        # Use hybrid controller's visualization
        fig = self.last_hybrid_controller.visualize(
            self.last_result,
            save_path=save_path
        )

        if save_path is None:
            import matplotlib.pyplot as plt
            plt.show()

        return fig

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about current robot model."""
        info = {
            'model_type': self.model_type,
            'model_class': self.model.__class__.__name__,
            'state_dim': self.model.state_dim,
            'input_dim': self.model.input_dim,
            'tau': self.model.tau,
        }

        # Add model-specific parameters
        if hasattr(self.model, 'u_min'):
            info['u_min'] = self.model.u_min.tolist()
        if hasattr(self.model, 'u_max'):
            info['u_max'] = self.model.u_max.tolist()
        if hasattr(self.model, 'W'):
            info['disturbance_bounds'] = self.model.W.tolist()

        return info


class SimulatorFactory:
    """
    Factory for creating simulators.

    Automatically selects between grid-world (legacy) and continuous
    based on availability and configuration.
    """

    @staticmethod
    def create_simulator(use_continuous: bool = True,
                        model_type: str = "unicycle",
                        disturbance_mode: str = "random") -> Any:
        """
        Create appropriate simulator.

        Args:
            use_continuous: If True, use continuous dynamics (recommended)
            model_type: Robot model type
            disturbance_mode: Disturbance handling mode

        Returns:
            Simulator instance (Continuous or GridWorld)
        """
        if use_continuous and COLLEAGUES_AVAILABLE:
            return ContinuousSimulator(
                model_type=model_type,
                disturbance_mode=disturbance_mode
            )
        else:
            # Fallback to grid-world
            from .robot_sim import RobotSimulator
            print("⚠️  Using grid-world simulator (continuous dynamics not available)")
            return RobotSimulator()

    @staticmethod
    def is_continuous_available() -> bool:
        """Check if continuous dynamics are available."""
        return COLLEAGUES_AVAILABLE
