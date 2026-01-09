"""
Synthesizer - Core synthesis logic.

This module defines the interface for controller synthesis from
LTL specifications. The actual synthesis uses GR(1) game solving
with TuLiP when available, or a basic implementation otherwise.
"""

import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .controller import Controller, State, Transition
from .transition_system import TransitionSystem
from ..grounding.grounding_filter import GroundingResult, MultiGroundingResult

logger = logging.getLogger(__name__)


class SynthesisError(Exception):
    """
    Raised when synthesis fails.
    
    This typically means the specification is UNREALIZABLE -
    there is no controller that can satisfy the spec against
    all possible environment behaviors.
    """
    
    def __init__(
        self, 
        message: str, 
        is_unrealizable: bool = False,
        counter_strategy: Optional[str] = None
    ):
        super().__init__(message)
        self.is_unrealizable = is_unrealizable
        self.counter_strategy = counter_strategy


@dataclass
class SynthesisResult:
    """
    Result of the synthesis stage.
    """
    controller: Controller
    grounded_ltl: str
    synthesis_time: float = 0.0
    is_realizable: bool = True
    warnings: List[str] = field(default_factory=list)
    
    @property
    def success(self) -> bool:
        """Whether synthesis succeeded."""
        return self.is_realizable and self.controller is not None


class BaseSynthesizer(ABC):
    """Abstract base class for synthesizers."""
    
    @abstractmethod
    def synthesize(
        self, 
        grounding_result: GroundingResult,
        transition_system: Optional[TransitionSystem] = None
    ) -> SynthesisResult:
        """
        Synthesize a controller from a grounded specification.
        
        Args:
            grounding_result: The grounded LTL specification
            transition_system: Optional robot transition system
            
        Returns:
            SynthesisResult with the controller
            
        Raises:
            SynthesisError: If synthesis fails (e.g., unrealizable spec)
        """
        pass


class Synthesizer(BaseSynthesizer):
    """
    GR(1) controller synthesizer.
    
    Uses TuLiP library for formal GR(1) synthesis when available.
    Falls back to a basic direct controller construction when TuLiP
    is not installed.
    
    For production robotic systems, TuLiP installation is recommended
    as it provides formal correctness guarantees.
    """
    
    def __init__(
        self,
        timeout: float = 60.0,
        verbose: bool = False,
        solver: str = "omega"
    ):
        """
        Initialize the synthesizer.
        
        Args:
            timeout: Maximum synthesis time in seconds
            verbose: Enable verbose output
            solver: GR(1) solver backend ("omega", "gr1c", or "slugs")
        """
        self.timeout = timeout
        self.verbose = verbose
        self.solver = solver
        self._tulip_available = self._check_tulip()
    
    def _check_tulip(self) -> bool:
        """Check if TuLiP is available."""
        try:
            import tulip
            return True
        except ImportError:
            return False
    
    @property
    def tulip_available(self) -> bool:
        """Whether TuLiP is available for synthesis."""
        return self._tulip_available
    
    def synthesize(
        self, 
        grounding_result: GroundingResult,
        transition_system: Optional[TransitionSystem] = None
    ) -> SynthesisResult:
        """
        Synthesize a controller from the grounded specification.
        
        Uses TuLiP for formal GR(1) synthesis if available,
        otherwise falls back to basic controller construction.
        """
        start_time = time.time()
        
        if self._tulip_available:
            return self._synthesize_with_tulip(
                grounding_result, transition_system, start_time
            )
        else:
            logger.info("TuLiP not available, using basic controller construction")
            return self._synthesize_basic(
                grounding_result, transition_system, start_time
            )
    
    def _synthesize_basic(
        self,
        grounding_result: GroundingResult,
        transition_system: Optional[TransitionSystem],
        start_time: float
    ) -> SynthesisResult:
        """
        Basic controller construction without formal synthesis.
        
        This creates a simple controller structure based on the LTL specification.
        NOTE: This does NOT provide formal correctness guarantees.
        For safety-critical systems, use TuLiP-based synthesis.
        """
        warnings = [
            "TuLiP not installed - using basic controller construction.",
            "For formal correctness guarantees, install TuLiP: pip install tulip"
        ]
        
        # Parse the grounded LTL to extract structure
        ltl = grounding_result.grounded_ltl
        
        # Extract variables from the grounding
        variables = set()
        for entity in grounding_result.grounded_entities:
            variables.add(entity.system_var)
        
        # Create a basic controller structure
        # This is a simplified version - real synthesis would solve a game
        states = []
        transitions = []
        
        # Create states based on the LTL pattern
        if "G(" in ltl:  # Safety property
            # Create a monitoring state that tracks the safety condition
            states.append(State(
                id=0,
                name="init",
                is_initial=True,
                labels=frozenset(["safe"])
            ))
            states.append(State(
                id=1,
                name="monitor",
                is_initial=False,
                labels=frozenset(["monitoring"])
            ))
            
            # Transition to monitoring after init
            transitions.append(Transition(
                source=0,
                target=1,
                guard="true",
                action="start_monitoring"
            ))
            
            # Self-loop on monitor while safe
            transitions.append(Transition(
                source=1,
                target=1,
                guard="safe",
                action="continue"
            ))
            
        elif "F(" in ltl:  # Liveness property
            # Create states to eventually reach the goal
            states.append(State(
                id=0,
                name="init",
                is_initial=True,
                labels=frozenset()
            ))
            states.append(State(
                id=1,
                name="progress",
                is_initial=False,
                labels=frozenset(["progressing"])
            ))
            states.append(State(
                id=2,
                name="goal",
                is_initial=False,
                labels=frozenset(["goal_reached"])
            ))
            
            transitions.append(Transition(
                source=0,
                target=1,
                guard="true",
                action="start"
            ))
            transitions.append(Transition(
                source=1,
                target=1,
                guard="!goal",
                action="progress"
            ))
            transitions.append(Transition(
                source=1,
                target=2,
                guard="goal",
                action="reach_goal"
            ))
            
        else:
            # Generic controller with single state
            states.append(State(
                id=0,
                name="execute",
                is_initial=True,
                labels=frozenset(["active"])
            ))
            transitions.append(Transition(
                source=0,
                target=0,
                guard="true",
                action="execute_spec"
            ))
        
        controller = Controller(
            states=states,
            transitions=transitions,
            specification=ltl
        )
        
        synthesis_time = time.time() - start_time
        
        return SynthesisResult(
            controller=controller,
            grounded_ltl=ltl,
            synthesis_time=synthesis_time,
            is_realizable=True,
            warnings=warnings
        )
    
    def _synthesize_with_tulip(
        self,
        grounding_result: GroundingResult,
        transition_system: Optional[TransitionSystem],
        start_time: float
    ) -> SynthesisResult:
        """
        Perform synthesis using TuLiP.
        
        This is the actual GR(1) synthesis implementation.
        """
        import tulip
        from tulip import spec, synth
        import re
        
        # Create GR(1) specification
        sys_spec = spec.GRSpec()
        
        # Extract variables from grounded entities and add them as system variables
        for entity in grounding_result.grounded_entities:
            var_name = entity.system_var
            # Add as boolean system variable
            sys_spec.sys_vars[var_name] = 'boolean'
        
        # Also extract any variables from the LTL formula that might not be in entities
        # This handles cases where variables are directly in the formula
        ltl_formula = grounding_result.grounded_ltl
        # Extract identifiers (variable names) from the formula
        # Match word characters that are not LTL operators
        ltl_operators = {'G', 'F', 'X', 'U', 'W', 'R', 'M', 'true', 'false', 'True', 'False'}
        # Find all potential variable names (sequences of word chars with underscores)
        potential_vars = re.findall(r'\b([a-z][a-z0-9_]*)\b', ltl_formula, re.IGNORECASE)
        for var in potential_vars:
            if var not in ltl_operators and var not in sys_spec.sys_vars:
                sys_spec.sys_vars[var] = 'boolean'
        
        # Add variables from transition system if provided
        if transition_system:
            for var in transition_system.env_vars:
                if var.is_boolean:
                    sys_spec.env_vars[var.name] = 'boolean'
            
            for var in transition_system.sys_vars:
                if var.is_boolean:
                    sys_spec.sys_vars[var.name] = 'boolean'
            
            # Add constraints from transition system (using list.extend)
            if transition_system.env_init:
                sys_spec.env_init.extend(transition_system.env_init)
            if transition_system.sys_init:
                sys_spec.sys_init.extend(transition_system.sys_init)
            if transition_system.env_safety:
                sys_spec.env_safety.extend(transition_system.env_safety)
            if transition_system.sys_safety:
                sys_spec.sys_safety.extend(transition_system.sys_safety)
            if transition_system.sys_prog:
                sys_spec.sys_prog.extend(transition_system.sys_prog)
        
        # Add the main specification as system progress (liveness) or safety based on pattern
        if 'G(F(' in ltl_formula or 'GF' in ltl_formula:
            # Liveness property - extract the inner formula and add to sys_prog
            # G(F(x)) means we should eventually always reach x
            match = re.search(r'G\s*\(\s*F\s*\(\s*(.+?)\s*\)\s*\)', ltl_formula)
            if match:
                sys_spec.sys_prog.append(match.group(1))
            else:
                sys_spec.sys_prog.append(ltl_formula)
        elif 'G(' in ltl_formula:
            # Safety property - add to sys_safety
            # For G(!x), we add !x as safety (always not x)
            match = re.search(r'G\s*\(\s*(.+?)\s*\)', ltl_formula)
            if match:
                inner = match.group(1)
                # Handle response patterns G(a -> F(b))
                if '->' in inner and 'F(' in inner:
                    # This is a response pattern, handle specially
                    sys_spec.sys_safety.append(inner)
                else:
                    sys_spec.sys_safety.append(inner)
            else:
                sys_spec.sys_safety.append(ltl_formula)
        else:
            # Generic - add to safety
            sys_spec.sys_safety.append(ltl_formula)
        
        # Synthesize
        try:
            ctrl = synth.synthesize(sys_spec, solver=self.solver)
            
            if ctrl is None:
                raise SynthesisError(
                    "Specification is UNREALIZABLE - no winning strategy exists.",
                    is_unrealizable=True
                )
            
            # Convert TuLiP controller to our format
            controller = self._convert_tulip_controller(ctrl)
            
        except Exception as e:
            if "unrealizable" in str(e).lower():
                raise SynthesisError(
                    f"Specification is unrealizable: {e}",
                    is_unrealizable=True
                )
            raise SynthesisError(f"Synthesis failed: {e}")
        
        synthesis_time = time.time() - start_time
        
        return SynthesisResult(
            controller=controller,
            grounded_ltl=grounding_result.grounded_ltl,
            synthesis_time=synthesis_time,
            is_realizable=True
        )
    
    def _convert_tulip_controller(self, tulip_ctrl) -> Controller:
        """Convert TuLiP controller to our Controller format."""
        from .controller import Controller, State, Transition
        
        states = []
        transitions = []
        
        # Extract states
        for i, node in enumerate(tulip_ctrl.nodes()):
            state_data = tulip_ctrl.nodes[node]
            states.append(State(
                id=i,
                name=f"s{i}",
                is_initial=(i == 0),  # Simplified
                labels=frozenset()
            ))
        
        # Extract transitions
        for source, target in tulip_ctrl.edges():
            transitions.append(Transition(
                source=source,
                target=target,
                guard="true",
                action=""
            ))
        
        return Controller(states=states, transitions=transitions)


def synthesize_from_grounding(
    grounding_result: GroundingResult,
    transition_system: Optional[TransitionSystem] = None,
    timeout: int = 60
) -> SynthesisResult:
    """
    Convenience function for synthesis.
    
    Args:
        grounding_result: The grounded LTL specification
        transition_system: Optional robot model
        timeout: Maximum synthesis time
        
    Returns:
        SynthesisResult with the controller
    """
    synthesizer = Synthesizer(timeout=timeout)
    return synthesizer.synthesize(grounding_result, transition_system)
