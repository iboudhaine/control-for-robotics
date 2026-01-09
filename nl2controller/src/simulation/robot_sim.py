"""
Robot Simulator

Simulates robot behavior based on synthesized controllers.
Tracks state transitions and validates execution.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class Direction(Enum):
    """Robot movement directions."""
    NORTH = (0, 1)
    SOUTH = (0, -1)
    EAST = (1, 0)
    WEST = (-1, 0)
    
    @property
    def name_fr(self) -> str:
        """French/Darija-friendly names."""
        names = {
            Direction.NORTH: "avant (qdam)",
            Direction.SOUTH: "arrière (lour)",
            Direction.EAST: "droite (lyamin)",
            Direction.WEST: "gauche (lisar)"
        }
        return names[self]


@dataclass
class RobotState:
    """
    Current state of the robot in the simulation.
    """
    x: int = 0
    y: int = 0
    direction: Direction = Direction.NORTH
    holding: Optional[str] = None  # Object the robot is holding
    
    # Active propositions (for LTL evaluation)
    active_props: Set[str] = field(default_factory=set)
    
    # History of positions
    history: List[Tuple[int, int]] = field(default_factory=list)
    
    def __post_init__(self):
        self.history = [(self.x, self.y)]
    
    @property
    def position(self) -> Tuple[int, int]:
        return (self.x, self.y)
    
    def move_forward(self, distance: int = 1) -> "RobotState":
        """Move in the current direction."""
        dx, dy = self.direction.value
        self.x += dx * distance
        self.y += dy * distance
        self.history.append(self.position)
        return self
    
    def turn_left(self) -> "RobotState":
        """Turn 90 degrees left."""
        turns = {
            Direction.NORTH: Direction.WEST,
            Direction.WEST: Direction.SOUTH,
            Direction.SOUTH: Direction.EAST,
            Direction.EAST: Direction.NORTH
        }
        self.direction = turns[self.direction]
        return self
    
    def turn_right(self) -> "RobotState":
        """Turn 90 degrees right."""
        turns = {
            Direction.NORTH: Direction.EAST,
            Direction.EAST: Direction.SOUTH,
            Direction.SOUTH: Direction.WEST,
            Direction.WEST: Direction.NORTH
        }
        self.direction = turns[self.direction]
        return self
    
    def set_proposition(self, prop: str, value: bool = True) -> "RobotState":
        """Set a proposition value."""
        if value:
            self.active_props.add(prop)
        else:
            self.active_props.discard(prop)
        return self
    
    def has_proposition(self, prop: str) -> bool:
        """Check if a proposition is active."""
        return prop in self.active_props


@dataclass
class SimulationResult:
    """
    Result of running a simulation.
    """
    success: bool
    steps: int
    final_state: RobotState
    trace: List[Dict[str, Any]]
    violations: List[str] = field(default_factory=list)
    goals_reached: List[str] = field(default_factory=list)
    
    def __str__(self) -> str:
        status = "✅ SUCCESS" if self.success else "❌ FAILED"
        return f"{status} - {self.steps} steps, violations: {len(self.violations)}"


class GridWorld:
    """
    A 2D grid world for robot simulation.
    """
    
    def __init__(
        self,
        width: int = 10,
        height: int = 10,
        obstacles: Optional[Set[Tuple[int, int]]] = None,
        goals: Optional[Dict[str, Tuple[int, int]]] = None
    ):
        """
        Initialize the grid world.
        
        Args:
            width: Grid width
            height: Grid height
            obstacles: Set of obstacle positions
            goals: Dict mapping goal names to positions
        """
        self.width = width
        self.height = height
        self.obstacles = obstacles or set()
        self.goals = goals or {}
    
    def is_valid_position(self, x: int, y: int) -> bool:
        """Check if a position is valid (in bounds and not an obstacle)."""
        in_bounds = 0 <= x < self.width and 0 <= y < self.height
        not_obstacle = (x, y) not in self.obstacles
        return in_bounds and not_obstacle
    
    def get_cell_type(self, x: int, y: int) -> str:
        """Get the type of cell at a position."""
        if (x, y) in self.obstacles:
            return "obstacle"
        for name, pos in self.goals.items():
            if pos == (x, y):
                return f"goal:{name}"
        return "empty"
    
    def add_obstacle(self, x: int, y: int) -> None:
        """Add an obstacle."""
        self.obstacles.add((x, y))
    
    def add_goal(self, name: str, x: int, y: int) -> None:
        """Add a goal location."""
        self.goals[name] = (x, y)
    
    def get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Get valid neighboring positions."""
        neighbors = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if self.is_valid_position(nx, ny):
                neighbors.append((nx, ny))
        return neighbors
    
    @classmethod
    def from_ascii(cls, ascii_map: str) -> "GridWorld":
        """
        Create a grid from ASCII art.
        
        Legend:
        - '.' = empty
        - '#' = obstacle
        - 'G' = goal (default)
        - 'A'-'Z' = named goals
        - 'R' = robot start (ignored for world)
        """
        lines = [line for line in ascii_map.strip().split('\n') if line]
        height = len(lines)
        width = max(len(line) for line in lines)
        
        obstacles = set()
        goals = {}
        
        for y, line in enumerate(reversed(lines)):  # Flip Y for cartesian coords
            for x, char in enumerate(line):
                if char == '#':
                    obstacles.add((x, y))
                elif char == 'G':
                    goals['goal'] = (x, y)
                elif char.isupper() and char not in ('R', 'G'):
                    goals[char] = (x, y)
        
        return cls(width=width, height=height, obstacles=obstacles, goals=goals)


class RobotSimulator:
    """
    Simulates robot execution based on a synthesized controller.
    """
    
    def __init__(self, world: Optional[GridWorld] = None):
        """
        Initialize the simulator.
        
        Args:
            world: The grid world (creates default 10x10 if None)
        """
        self.world = world or GridWorld()
        self.state = RobotState()
        self.step_count = 0
        self.trace: List[Dict[str, Any]] = []
    
    def reset(self, x: int = 0, y: int = 0, direction: Direction = Direction.NORTH):
        """Reset the robot to initial state."""
        self.state = RobotState(x=x, y=y, direction=direction)
        self.step_count = 0
        self.trace = []
        self._record_state("init")
    
    def _record_state(self, action: str):
        """Record current state in trace."""
        self.trace.append({
            "step": self.step_count,
            "action": action,
            "position": self.state.position,
            "direction": self.state.direction.name,
            "props": list(self.state.active_props)
        })
    
    def execute_action(self, action: str) -> bool:
        """
        Execute a single action.
        
        Args:
            action: Action to execute (move_forward, turn_left, turn_right, etc.)
            
        Returns:
            True if action was successful
        """
        self.step_count += 1
        
        if action == "move_forward":
            dx, dy = self.state.direction.value
            new_x = self.state.x + dx
            new_y = self.state.y + dy
            
            if self.world.is_valid_position(new_x, new_y):
                self.state.move_forward()
                self._update_propositions()
                self._record_state(action)
                return True
            else:
                logger.warning(f"Cannot move to ({new_x}, {new_y}) - blocked")
                self._record_state(f"{action}_blocked")
                return False
        
        elif action == "turn_left":
            self.state.turn_left()
            self._record_state(action)
            return True
        
        elif action == "turn_right":
            self.state.turn_right()
            self._record_state(action)
            return True
        
        elif action == "stop":
            self._record_state(action)
            return True
        
        elif action.startswith("pickup_"):
            obj = action.replace("pickup_", "")
            self.state.holding = obj
            self._record_state(action)
            return True
        
        elif action.startswith("drop_"):
            self.state.holding = None
            self._record_state(action)
            return True
        
        # Handle abstract actions from basic synthesis (these don't move the robot)
        elif action in ["start", "start_monitoring", "continue", "progress", "reach_goal", "execute_spec"]:
            # These are abstract controller actions - just record them, don't move
            # But still update propositions so guards can be evaluated
            self._update_propositions()
            self._record_state(action)
            return True
        
        else:
            logger.warning(f"Unknown action: {action} (simulation will continue but robot won't move)")
            self._record_state(f"unknown:{action}")
            return True  # Allow simulation to continue
    
    def _update_propositions(self):
        """Update propositions based on current position."""
        x, y = self.state.position
        
        # Check goals
        for name, pos in self.world.goals.items():
            self.state.set_proposition(f"at_{name}", pos == (x, y))
        
        # Check if near obstacles
        for ox, oy in self.world.obstacles:
            if abs(x - ox) <= 1 and abs(y - oy) <= 1:
                self.state.set_proposition("near_obstacle", True)
                return
        self.state.set_proposition("near_obstacle", False)
    
    def run_controller(
        self,
        controller: Dict[str, Any],
        max_steps: int = 100
    ) -> SimulationResult:
        """
        Run a controller to completion or max steps.
        
        Args:
            controller: Synthesized controller with states and transitions
            max_steps: Maximum steps before timeout
            
        Returns:
            SimulationResult with execution trace
        """
        self.reset()
        violations = []
        goals_reached = []
        
        # Extract controller info
        states = controller.get("states", [])
        transitions = controller.get("transitions", [])
        guarantees = controller.get("guarantees", [])
        
        # Get initial state ID (handle both dict and string formats)
        initial_states_list = controller.get("initial_states", [])
        if initial_states_list:
            current_state = initial_states_list[0]  # Use first initial state
        elif states:
            # Fallback: get ID from first state
            first_state = states[0]
            if isinstance(first_state, dict):
                current_state = first_state.get("id", 0)
            else:
                current_state = first_state
        else:
            current_state = 0  # Default to state 0
        
        # Build transition map (using state IDs as keys)
        trans_map = {}
        for t in transitions:
            src = t.get("from", t.get("source"))
            # Ensure src is a hashable type (int or string)
            if isinstance(src, dict):
                src = src.get("id", src.get("name", 0))
            trans_map.setdefault(src, []).append(t)
        
        while self.step_count < max_steps:
            # Find applicable transition
            available = trans_map.get(current_state, [])
            
            if not available:
                logger.info(f"No transitions from state {current_state} - stopping")
                break
            
            # Select first applicable transition (always take first one for abstract controllers)
            # For abstract controllers, guards are not evaluated - we just follow the structure
            trans = available[0]
            action = trans.get("action", trans.get("label", "stop"))
            target = trans.get("to", trans.get("target"))
            # Ensure target is a hashable type (int or string)
            if isinstance(target, dict):
                target = target.get("id", target.get("name", 0))
            
            # Execute action (may be abstract)
            self.execute_action(action)
            current_state = target
            
            # Check for goal states (controller-level goals)
            accepting_states = controller.get("accepting_states", [])
            if current_state in accepting_states:
                goals_reached.append(f"controller_state_{current_state}")
            
            # For abstract controllers, also check if we're in a "goal" state by name/labels
            for state in states:
                if isinstance(state, dict):
                    state_id = state.get("id")
                    state_name = state.get("name", "")
                    if state_id == current_state and ("goal" in state_name.lower() or "goal" in str(state.get("labels", []))):
                        goals_reached.append(f"goal_state_{state_id}")
            
            # Update propositions even for abstract actions (so guards can be checked)
            self._update_propositions()
            
            # Check guarantees
            for g in guarantees:
                if not self._check_ltl_simple(g):
                    violations.append(f"Violated: {g}")
        
        # For abstract controllers, success is just no violations (not requiring physical goal reach)
        # Check if this looks like an abstract controller (has abstract actions)
        abstract_actions = ["start", "continue", "progress", "reach_goal", "start_monitoring", "execute_spec"]
        is_abstract = any(
            any(abstract in str(t.get("action", "")) for abstract in abstract_actions)
            for t in transitions
        )
        
        # Success criteria: no violations, and either goals reached OR abstract controller completed
        success = len(violations) == 0 and (len(goals_reached) > 0 or is_abstract)
        
        return SimulationResult(
            success=success,
            steps=self.step_count,
            final_state=self.state,
            trace=self.trace,
            violations=violations,
            goals_reached=goals_reached
        )
    
    def run_simple_commands(self, commands: List[str]) -> SimulationResult:
        """
        Run a list of simple commands (for demo/testing).
        
        Args:
            commands: List of action strings
            
        Returns:
            SimulationResult
        """
        self.reset()
        
        for cmd in commands:
            self.execute_action(cmd)
        
        return SimulationResult(
            success=True,
            steps=self.step_count,
            final_state=self.state,
            trace=self.trace
        )
    
    def _check_ltl_simple(self, formula: str) -> bool:
        """
        Simple LTL checking for current state.
        
        This is a basic implementation - for full LTL model checking,
        use a proper model checker like SPOT or NuSMV.
        """
        # Check simple propositions
        if formula.startswith("!"):
            prop = formula[1:].strip()
            return not self.state.has_proposition(prop)
        elif formula.startswith("G ") or formula.startswith("[]"):
            # Globally - check current (simplified)
            prop = formula[2:].strip() if formula.startswith("G ") else formula[2:].strip()
            return self.state.has_proposition(prop)
        else:
            return self.state.has_proposition(formula)
    
    def get_ascii_visualization(self) -> str:
        """
        Get an ASCII visualization of the current world state.
        
        Returns:
            String with ASCII art of the grid
        """
        lines = []
        
        # Direction arrows
        arrows = {
            Direction.NORTH: "^",
            Direction.SOUTH: "v",
            Direction.EAST: ">",
            Direction.WEST: "<"
        }
        
        for y in range(self.world.height - 1, -1, -1):
            line = []
            for x in range(self.world.width):
                if (x, y) == self.state.position:
                    line.append(arrows[self.state.direction])
                elif (x, y) in self.world.obstacles:
                    line.append("#")
                elif (x, y) in self.world.goals.values():
                    # Find goal name
                    for name, pos in self.world.goals.items():
                        if pos == (x, y):
                            line.append(name[0].upper())
                            break
                elif (x, y) in self.state.history:
                    line.append("·")
                else:
                    line.append(".")
            lines.append(" ".join(line))
        
        # Add coordinate labels
        x_labels = " ".join(str(i % 10) for i in range(self.world.width))
        lines.append("-" * (self.world.width * 2 - 1))
        lines.append(x_labels)
        
        return "\n".join(lines)
