"""
Data Collection Module

Provides tools for collecting data from robotic systems:
- Excitation signals for persistent excitation
- Trajectory references (circle, figure-8, etc.)
- Dataset management with noise injection
"""

from .excitation_signals import (
    ExcitationSignal,
    RandomExcitation,
    SinusoidalExcitation,
    ChirpExcitation,
    PRBSExcitation,
    create_rich_excitation
)

from .trajectory_generation import (
    TrajectoryGenerator,
    CircleTrajectory,
    Figure8Trajectory,
    LineTrajectory
)

from .dataset import (
    Dataset,
    collect_data,
    collect_trajectory_data
)

__all__ = [
    'ExcitationSignal',
    'RandomExcitation', 
    'SinusoidalExcitation',
    'ChirpExcitation',
    'PRBSExcitation',
    'TrajectoryGenerator',
    'CircleTrajectory',
    'Figure8Trajectory',
    'LineTrajectory',
    'Dataset',
    'collect_data',
    'collect_trajectory_data',
    'create_rich_excitation'
]
