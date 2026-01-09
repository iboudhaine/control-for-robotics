"""
Controller Data Structures

Defines the output of the synthesis stage: a finite-state controller
that can be executed by the robot.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any
from enum import Enum


@dataclass(frozen=True)
class State:
    """
    A state in the controller automaton.
    
    Attributes:
        id: Unique identifier
        name: Human-readable name
        is_initial: Whether this is an initial state
        labels: Set of atomic propositions true in this state
    """
    id: int
    name: str = ""
    is_initial: bool = False
    labels: frozenset = field(default_factory=frozenset)
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if isinstance(other, State):
            return self.id == other.id
        return False


@dataclass
class Transition:
    """
    A transition in the controller automaton.
    
    Attributes:
        source: Source state ID
        target: Target state ID
        guard: Condition (environment input) triggering this transition
        action: System output/action when taking this transition
    """
    source: int
    target: int
    guard: str = "true"     # Condition on environment inputs
    action: str = ""        # System output
    
    def __str__(self):
        return f"{self.source} --[{self.guard}]--> {self.target}: {self.action}"


class ControllerType(Enum):
    """Type of synthesized controller."""
    MEALY = "mealy"         # Output depends on state AND input
    MOORE = "moore"         # Output depends only on state
    STRATEGY = "strategy"   # Generic winning strategy


@dataclass
class Controller:
    """
    A finite-state reactive controller.
    
    This is the output of the synthesis stage - an automaton that
    can be executed by the robot to satisfy the LTL specification.
    """
    
    states: List[State] = field(default_factory=list)
    transitions: List[Transition] = field(default_factory=list)
    initial_states: List[int] = field(default_factory=list)
    controller_type: ControllerType = ControllerType.MEALY
    
    # Metadata
    specification: str = ""         # The LTL spec this controller realizes
    synthesis_time: float = 0.0     # Time taken to synthesize
    is_realizable: bool = True      # Whether the spec was realizable
    
    def __post_init__(self):
        """Set initial states from state list if not provided."""
        if not self.initial_states:
            self.initial_states = [s.id for s in self.states if s.is_initial]
    
    @property
    def num_states(self) -> int:
        """Number of states in the controller."""
        return len(self.states)
    
    @property
    def num_transitions(self) -> int:
        """Number of transitions in the controller."""
        return len(self.transitions)
    
    def get_state(self, state_id: int) -> Optional[State]:
        """Get state by ID."""
        for state in self.states:
            if state.id == state_id:
                return state
        return None
    
    def get_successors(self, state_id: int) -> List[Transition]:
        """Get all outgoing transitions from a state."""
        return [t for t in self.transitions if t.source == state_id]
    
    def get_predecessors(self, state_id: int) -> List[Transition]:
        """Get all incoming transitions to a state."""
        return [t for t in self.transitions if t.target == state_id]
    
    def step(self, current_state: int, environment_input: Dict[str, Any]) -> Optional[Transition]:
        """
        Execute one step of the controller.
        
        Args:
            current_state: Current state ID
            environment_input: Dictionary of environment variable values
            
        Returns:
            The transition to take, or None if no valid transition
        """
        successors = self.get_successors(current_state)
        
        for trans in successors:
            if self._evaluate_guard(trans.guard, environment_input):
                return trans
        
        return None
    
    def _evaluate_guard(self, guard: str, env: Dict[str, Any]) -> bool:
        """
        Evaluate a guard condition against environment input.
        
        This is a simple implementation - for production, use a proper
        expression evaluator.
        """
        if guard == "true" or guard == "True" or not guard:
            return True
        if guard == "false" or guard == "False":
            return False
        
        # Simple evaluation for basic guards
        try:
            # WARNING: eval is used here for simplicity
            # In production, use a safe expression parser
            return eval(guard, {}, env)
        except:
            return True  # Default to true if can't evaluate
    
    def to_dict(self) -> Dict:
        """Export controller to dictionary format."""
        # Extract accepting states (states with goal labels or special markers)
        accepting_states = []
        for s in self.states:
            if "goal" in s.labels or "accepting" in s.labels:
                accepting_states.append(s.id)
        
        return {
            "states": [
                {
                    "id": s.id,
                    "name": s.name,
                    "is_initial": s.is_initial,
                    "labels": list(s.labels)
                }
                for s in self.states
            ],
            "transitions": [
                {
                    "source": t.source,
                    "target": t.target,
                    "guard": t.guard,
                    "action": t.action
                }
                for t in self.transitions
            ],
            "initial_states": self.initial_states,
            "accepting_states": accepting_states,
            "controller_type": self.controller_type.value,
            "specification": self.specification,
            "is_realizable": self.is_realizable,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Controller":
        """Create controller from dictionary."""
        states = [
            State(
                id=s["id"],
                name=s.get("name", ""),
                is_initial=s.get("is_initial", False),
                labels=frozenset(s.get("labels", []))
            )
            for s in data.get("states", [])
        ]
        
        transitions = [
            Transition(
                source=t["source"],
                target=t["target"],
                guard=t.get("guard", "true"),
                action=t.get("action", "")
            )
            for t in data.get("transitions", [])
        ]
        
        return cls(
            states=states,
            transitions=transitions,
            initial_states=data.get("initial_states", []),
            controller_type=ControllerType(data.get("controller_type", "mealy")),
            specification=data.get("specification", ""),
            is_realizable=data.get("is_realizable", True),
        )
    
    def __repr__(self):
        return (
            f"Controller(states={self.num_states}, "
            f"transitions={self.num_transitions}, "
            f"realizable={self.is_realizable})"
        )
    
    def to_dot(self) -> str:
        """Export controller to DOT format for visualization."""
        lines = ["digraph Controller {"]
        lines.append("  rankdir=LR;")
        
        # States
        for state in self.states:
            shape = "doublecircle" if state.is_initial else "circle"
            label = state.name or f"s{state.id}"
            lines.append(f'  s{state.id} [label="{label}", shape={shape}];')
        
        # Transitions
        for trans in self.transitions:
            label = f"{trans.guard} / {trans.action}" if trans.action else trans.guard
            lines.append(f'  s{trans.source} -> s{trans.target} [label="{label}"];')
        
        lines.append("}")
        return "\n".join(lines)
