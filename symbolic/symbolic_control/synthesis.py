"""
Synthesis Module - Combined Safety and Reachability Synthesis.

Pipeline:
1. SafetyPruning: Prunes the automaton to get safe states and transitions
2. ReachabilitySynthesis: Computes winning set on the safe automaton

The output is a controller that guarantees:
- Safety (never leaves safe set)
- Reachability (reaches target from winning set)
"""

import random
from typing import Dict, Set, List, Tuple, Optional
from dataclasses import dataclass

from .abstraction import Abstraction


@dataclass
class SafeAutomaton:
    """
    A pruned automaton containing only safe states and transitions.
    
    This is the output of safety synthesis and input to reachability synthesis.
    """
    abstraction: Abstraction          # Original abstraction (for dynamics, grid info)
    safe_states: Set[int]             # S* - maximal controlled invariant set
    safe_transitions: Dict[Tuple[int, int], Set[int]]  # {(cell, u_idx): successors} restricted to safe
    
    def get_successors(self, cell_idx: int, u_idx: int) -> Set[int]:
        """Get successors for a (cell, control) pair in the safe automaton."""
        return self.safe_transitions.get((cell_idx, u_idx), set())
    
    def get_valid_controls(self, cell_idx: int) -> List[int]:
        """Get all controls that keep the system safe from this cell."""
        valid = []
        num_controls = len(self.abstraction.dynamics.control_set)
        for u_idx in range(num_controls):
            succ = self.get_successors(cell_idx, u_idx)
            if succ:  # Non-empty means this control is safe
                valid.append(u_idx)
        return valid


def compute_safe_automaton(
    abstraction: Abstraction,
    safe_spec: Optional[Set[int]] = None,
    verbose: bool = True
) -> SafeAutomaton:
    """
    Compute the maximal controlled invariant set and prune the automaton.
    
    Algorithm (Safety Fixed-Point):
        R₀ = Qₛ
        Rₖ₊₁ = Qₛ ∩ Pre(Rₖ)
        Stop when Rₖ₊₁ = Rₖ
    
    Args:
        abstraction: Grid abstraction with precomputed transitions
        safe_spec: Qₛ - safe cells (default: all cells)
        verbose: Print progress
    
    Returns:
        SafeAutomaton with pruned states and transitions
    """
    q_s = safe_spec if safe_spec is not None else set(range(abstraction.num_cells))
    num_controls = len(abstraction.dynamics.control_set)
    
    if verbose:
        print("=" * 50)
        print("SAFETY SYNTHESIS (Automaton Pruning)")
        print("=" * 50)
    
    # Fixed-point iteration
    safe_set = q_s.copy()
    
    if verbose:
        print(f"Qₛ: {len(q_s)} cells | R₀: {len(safe_set)} cells")
    
    iteration = 0
    while True:
        iteration += 1
        
        # Compute Pre(safe_set) ∩ Qₛ
        new_safe_set = set()
        for cell_idx in q_s:
            # Cell is safe if ∃u such that Post(cell, u) ⊆ safe_set and Post ≠ ∅
            for u_idx in range(num_controls):
                successors = abstraction.transitions.get((cell_idx, u_idx), set())
                if successors and successors.issubset(safe_set):
                    new_safe_set.add(cell_idx)
                    break
        
        removed = len(safe_set) - len(new_safe_set)
        safe_set = new_safe_set
        
        if verbose:
            print(f"Iteration {iteration}: {len(safe_set)} cells (removed {removed})")
        
        if removed == 0:
            break
    
    if verbose:
        pct = 100 * len(safe_set) / abstraction.num_cells
        print(f"Converged! S* = {len(safe_set)} cells ({pct:.1f}%)")
    
    # Build safe transitions: only transitions that stay within S*
    safe_transitions = {}
    for cell_idx in safe_set:
        for u_idx in range(num_controls):
            successors = abstraction.transitions.get((cell_idx, u_idx), set())
            if successors and successors.issubset(safe_set):
                safe_transitions[(cell_idx, u_idx)] = successors
    
    if verbose:
        print(f"Safe transitions: {len(safe_transitions)}")
        print("=" * 50)
    
    return SafeAutomaton(
        abstraction=abstraction,
        safe_states=safe_set,
        safe_transitions=safe_transitions
    )


def compute_reachability(
    safe_automaton: SafeAutomaton,
    target_set: Set[int],
    verbose: bool = True
) -> Tuple[Set[int], List[Set[int]], Dict[int, List[int]]]:
    """
    Compute reachability to target on the safe automaton.
    
    Algorithm (Reachability Fixed-Point):
        R₀ = Qₐ ∩ S*
        Rₖ₊₁ = Qₐ ∪ Pre(Rₖ)  (Pre computed on safe automaton)
        Stop when Rₖ₊₁ = Rₖ
    
    Args:
        safe_automaton: Output from compute_safe_automaton
        target_set: Qₐ - target cells to reach
        verbose: Print progress
    
    Returns:
        (winning_set, reach_sets, controller)
        - winning_set: cells from which target is reachable
        - reach_sets: [R₀, R₁, ...] for controller extraction
        - controller: {cell_idx: [valid_control_indices]}
    """
    if verbose:
        print("=" * 50)
        print("REACHABILITY SYNTHESIS")
        print("=" * 50)
    
    # Target must be within safe set
    safe_target = target_set & safe_automaton.safe_states
    
    if verbose:
        print(f"Target Qₐ: {len(target_set)} cells")
        print(f"Safe target (Qₐ ∩ S*): {len(safe_target)} cells")
    
    if not safe_target:
        if verbose:
            print("WARNING: No target cells in safe set!")
        return set(), [], {}
    
    num_controls = len(safe_automaton.abstraction.dynamics.control_set)
    
    # Fixed-point iteration on safe automaton
    r_k = safe_target.copy()
    reach_sets = [r_k.copy()]
    
    if verbose:
        print(f"R₀: {len(r_k)} cells")
    
    iteration = 0
    while True:
        iteration += 1
        
        # Pre(Rₖ) on safe automaton
        pre_r_k = set()
        for cell_idx in safe_automaton.safe_states:
            for u_idx in range(num_controls):
                successors = safe_automaton.get_successors(cell_idx, u_idx)
                if successors and successors.issubset(r_k):
                    pre_r_k.add(cell_idx)
                    break
        
        # Rₖ₊₁ = Qₐ ∪ Pre(Rₖ)
        r_k_plus_1 = safe_target | pre_r_k
        added = len(r_k_plus_1) - len(r_k)
        
        if verbose:
            print(f"Iteration {iteration}: {len(r_k_plus_1)} cells (+{added})")
        
        reach_sets.append(r_k_plus_1.copy())
        
        if r_k_plus_1 == r_k:
            break
        r_k = r_k_plus_1
    
    winning_set = r_k
    
    if verbose:
        pct = 100 * len(winning_set) / safe_automaton.abstraction.num_cells
        print(f"Converged! Winning set = {len(winning_set)} cells ({pct:.1f}%)")
    
    # Extract controller
    controller = _extract_controller(safe_automaton, safe_target, reach_sets)
    
    if verbose:
        print(f"Controller covers {len(controller)} cells")
        print("=" * 50)
    
    return winning_set, reach_sets, controller


def _extract_controller(
    safe_automaton: SafeAutomaton,
    safe_target: Set[int],
    reach_sets: List[Set[int]]
) -> Dict[int, List[int]]:
    """
    Extract controller from reachability layers.
    
    For cell in layer k:
    - k=0 (target): any safe control
    - k>0: controls whose successors ⊆ Rₖ₋₁
    """
    controller = {}
    num_controls = len(safe_automaton.abstraction.dynamics.control_set)
    winning_set = reach_sets[-1] if reach_sets else set()
    
    for cell_idx in winning_set:
        # Find layer
        layer = -1
        for k, r_k in enumerate(reach_sets):
            if cell_idx in r_k:
                layer = k
                break
        
        if layer == 0:
            # At target: any safe control
            valid = safe_automaton.get_valid_controls(cell_idx)
        else:
            # Not at target: controls that reach layer k-1
            target_layer = reach_sets[layer - 1]
            valid = []
            for u_idx in range(num_controls):
                succ = safe_automaton.get_successors(cell_idx, u_idx)
                if succ and succ.issubset(target_layer):
                    valid.append(u_idx)
        
        if valid:
            controller[cell_idx] = valid
    
    return controller


class Synthesis:
    """
    Combined Safety + Reachability Synthesis.
    
    Usage:
        synth = Synthesis(abstraction, obstacles, target)
        synth.run()
        u_idx = synth.get_control(cell_idx)
    """
    
    def __init__(
        self,
        abstraction: Abstraction,
        obstacles: Set[int],
        target: Set[int]
    ):
        """
        Args:
            abstraction: Grid abstraction with precomputed transitions
            obstacles: Cells to avoid (unsafe)
            target: Cells to reach
        """
        self.abstraction = abstraction
        self.obstacles = obstacles
        self.target = target
        
        # Results (populated by run())
        self.safe_automaton: Optional[SafeAutomaton] = None
        self.winning_set: Set[int] = set()
        self.reach_sets: List[Set[int]] = []
        self.controller: Dict[int, List[int]] = {}
    
    def run(self, verbose: bool = True) -> Set[int]:
        """
        Run the full synthesis pipeline.
        
        Returns:
            Winning set
        """
        # Step 1: Safety - prune automaton
        safe_spec = set(range(self.abstraction.num_cells)) - self.obstacles
        self.safe_automaton = compute_safe_automaton(
            self.abstraction, safe_spec, verbose=verbose
        )
        
        if not self.safe_automaton.safe_states:
            if verbose:
                print("No safe states!")
            return set()
        
        # Step 2: Reachability on safe automaton
        self.winning_set, self.reach_sets, self.controller = compute_reachability(
            self.safe_automaton, self.target, verbose=verbose
        )
        
        return self.winning_set
    
    def get_control(self, cell_idx: int, random_choice: bool = True) -> Optional[int]:
        """Get a control for a cell, or None if not in winning set."""
        if cell_idx not in self.controller:
            return None
        controls = self.controller[cell_idx]
        return random.choice(controls) if random_choice else controls[0]
    
    def get_distance_to_target(self, cell_idx: int) -> int:
        """Get minimum steps to reach target, or -1 if unreachable."""
        for k, r_k in enumerate(self.reach_sets):
            if cell_idx in r_k:
                return k
        return -1
    
    @property
    def safe_set(self) -> Set[int]:
        """Get the safe set (maximal controlled invariant set)."""
        if self.safe_automaton is None:
            return set()
        return self.safe_automaton.safe_states
    
    def is_safe(self, cell_idx: int) -> bool:
        """Check if a cell is in the safe set."""
        if self.safe_automaton is None:
            return False
        return cell_idx in self.safe_automaton.safe_states
    
    def is_winning(self, cell_idx: int) -> bool:
        """Check if a cell is in the winning set."""
        return cell_idx in self.winning_set
    
    def is_at_target(self, cell_idx: int) -> bool:
        """Check if a cell is in the target."""
        safe_target = self.target & (self.safe_automaton.safe_states if self.safe_automaton else set())
        return cell_idx in safe_target
