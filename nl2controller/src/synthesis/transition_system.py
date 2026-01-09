"""
Transition System - Robot Environment Model

Defines the discrete model of the robot's capabilities and environment.
This is combined with the LTL specification for synthesis.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional
from enum import Enum


class VariableType(Enum):
    """Type of variable in the transition system."""
    ENVIRONMENT = "env"     # Uncontrollable (environment chooses)
    SYSTEM = "sys"          # Controllable (robot chooses)


@dataclass
class Variable:
    """
    A variable in the transition system.
    
    Attributes:
        name: Variable identifier
        domain: Set of possible values
        var_type: Environment or system variable
        initial: Initial value (optional)
    """
    name: str
    domain: Set[str] = field(default_factory=lambda: {"true", "false"})
    var_type: VariableType = VariableType.SYSTEM
    initial: Optional[str] = None
    
    @property
    def is_boolean(self) -> bool:
        """Check if this is a boolean variable."""
        return self.domain == {"true", "false"} or self.domain == {True, False}


@dataclass
class TransitionSystem:
    """
    A discrete transition system modeling the robot and environment.
    
    This defines:
    - System variables (controllable by robot)
    - Environment variables (uncontrollable)
    - Transition constraints (dynamics)
    - Initial conditions
    """
    
    # Variables
    env_vars: List[Variable] = field(default_factory=list)
    sys_vars: List[Variable] = field(default_factory=list)
    
    # Constraints
    env_init: List[str] = field(default_factory=list)    # Initial environment
    sys_init: List[str] = field(default_factory=list)    # Initial system state
    env_safety: List[str] = field(default_factory=list)  # Environment constraints
    sys_safety: List[str] = field(default_factory=list)  # System constraints
    
    # GR(1) components
    env_prog: List[str] = field(default_factory=list)    # Environment liveness
    sys_prog: List[str] = field(default_factory=list)    # System goals (from spec)
    
    def add_env_var(
        self, 
        name: str, 
        domain: Set[str] = None, 
        initial: str = None
    ) -> None:
        """Add an environment variable."""
        var = Variable(
            name=name,
            domain=domain or {"true", "false"},
            var_type=VariableType.ENVIRONMENT,
            initial=initial
        )
        self.env_vars.append(var)
    
    def add_sys_var(
        self, 
        name: str, 
        domain: Set[str] = None, 
        initial: str = None
    ) -> None:
        """Add a system (controllable) variable."""
        var = Variable(
            name=name,
            domain=domain or {"true", "false"},
            var_type=VariableType.SYSTEM,
            initial=initial
        )
        self.sys_vars.append(var)
    
    def add_env_init(self, constraint: str) -> None:
        """Add initial environment constraint."""
        self.env_init.append(constraint)
    
    def add_sys_init(self, constraint: str) -> None:
        """Add initial system constraint."""
        self.sys_init.append(constraint)
    
    def add_env_safety(self, constraint: str) -> None:
        """Add environment safety constraint (always holds)."""
        self.env_safety.append(constraint)
    
    def add_sys_safety(self, constraint: str) -> None:
        """Add system safety constraint."""
        self.sys_safety.append(constraint)
    
    def add_sys_goal(self, goal: str) -> None:
        """Add a system liveness goal (GF goal)."""
        self.sys_prog.append(goal)
    
    def get_all_vars(self) -> List[Variable]:
        """Get all variables."""
        return self.env_vars + self.sys_vars
    
    def get_var_names(self) -> List[str]:
        """Get names of all variables."""
        return [v.name for v in self.get_all_vars()]
    
    def to_dict(self) -> Dict:
        """Export to dictionary format."""
        return {
            "env_vars": [
                {"name": v.name, "domain": list(v.domain), "initial": v.initial}
                for v in self.env_vars
            ],
            "sys_vars": [
                {"name": v.name, "domain": list(v.domain), "initial": v.initial}
                for v in self.sys_vars
            ],
            "env_init": self.env_init,
            "sys_init": self.sys_init,
            "env_safety": self.env_safety,
            "sys_safety": self.sys_safety,
            "env_prog": self.env_prog,
            "sys_prog": self.sys_prog,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "TransitionSystem":
        """Create from dictionary."""
        ts = cls()
        
        for v in data.get("env_vars", []):
            ts.add_env_var(v["name"], set(v.get("domain", [])), v.get("initial"))
        
        for v in data.get("sys_vars", []):
            ts.add_sys_var(v["name"], set(v.get("domain", [])), v.get("initial"))
        
        ts.env_init = data.get("env_init", [])
        ts.sys_init = data.get("sys_init", [])
        ts.env_safety = data.get("env_safety", [])
        ts.sys_safety = data.get("sys_safety", [])
        ts.env_prog = data.get("env_prog", [])
        ts.sys_prog = data.get("sys_prog", [])
        
        return ts


def create_gridworld_ts(width: int, height: int) -> TransitionSystem:
    """
    Create a simple gridworld transition system.
    
    This is useful for testing and demos.
    """
    ts = TransitionSystem()
    
    # Location variable
    locations = {f"loc_{x}_{y}" for x in range(width) for y in range(height)}
    ts.add_sys_var("location", locations, initial=f"loc_0_0")
    
    # Add adjacency constraints (can only move to adjacent cells)
    for x in range(width):
        for y in range(height):
            loc = f"loc_{x}_{y}"
            neighbors = []
            
            if x > 0:
                neighbors.append(f"loc_{x-1}_{y}")
            if x < width - 1:
                neighbors.append(f"loc_{x+1}_{y}")
            if y > 0:
                neighbors.append(f"loc_{x}_{y-1}")
            if y < height - 1:
                neighbors.append(f"loc_{x}_{y+1}")
            neighbors.append(loc)  # Can stay in place
            
            if neighbors:
                # If in loc, next location must be neighbor or self
                neighbor_str = " | ".join(f"(location' = {n})" for n in neighbors)
                ts.add_sys_safety(f"(location = {loc}) -> ({neighbor_str})")
    
    return ts
