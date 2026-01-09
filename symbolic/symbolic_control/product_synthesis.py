"""
Product Synthesis Module - Automata-Based Specifications.

This module implements controller synthesis for complex temporal specifications
expressed as regular expressions. It works by constructing a Product Automaton
that combines:
    1. The physical system abstraction (grid cells + transitions)
    2. A specification NFA (from regex)

The problem of satisfying the specification on the original system becomes
a simple reachability problem on the Product Automaton.

Pipeline:
    1. Transition pruning (remove boundary cells with no valid controls)
    2. Build NFA from regex specification
    3. Construct Product Automaton (NFA states × grid cells)
    4. Run reachability synthesis on product to find winning states
    5. Extract controller that satisfies the specification

Note: Obstacle/region avoidance is specified directly in the regex using [^...]:
    - "A[^O]*B"  : Go from A to B, avoiding region O
    - "A[^]*B"   : Go from A to B through any cells (wildcard)

Example specs:
    - "AB"           : Visit region A, then region B
    - "A|B"          : Visit region A or region B
    - "(A|B)C"       : Visit A or B, then visit C
    - "A[^O]*B"      : Visit A, avoid O, then visit B
"""

import random
from typing import Dict, Set, List, Tuple, Optional, FrozenSet
from dataclasses import dataclass
from collections import deque

import numpy as np

from .abstraction import Abstraction
from .nfa import NFA, RegionLabeler, regex_to_nfa
from .synthesis import SafeAutomaton, compute_safe_automaton


# =============================================================================
# Product State
# =============================================================================

@dataclass(frozen=True)
class ProductState:
    """
    State in the Product Automaton = (NFA states, grid cell, last label).
    
    Attributes:
        nfa_states: Frozenset of NFA states (handles nondeterminism)
        cell_idx: Grid cell index in the physical abstraction
        last_label: Last emitted label (prevents re-emitting when staying in region)
    
    The last_label field is crucial: when the robot stays in a labeled region,
    we only emit the label once (on entry), not repeatedly. This prevents
    specs like "AB" from being satisfied by just entering A once.
    """
    nfa_states: FrozenSet[int]
    cell_idx: int
    last_label: Optional[str] = None
    
    def __hash__(self):
        return hash((self.nfa_states, self.cell_idx, self.last_label))
    
    def __eq__(self, other):
        if not isinstance(other, ProductState):
            return False
        return (self.nfa_states == other.nfa_states and 
                self.cell_idx == other.cell_idx and
                self.last_label == other.last_label)


# =============================================================================
# Product Automaton
# =============================================================================

class ProductAutomaton:
    """
    Product Automaton = Physical Abstraction × Specification NFA.
    
    States are ProductState instances (nfa_states, cell_idx, last_label).
    Transitions couple physical movement with NFA evolution.
    
    Construction uses BFS from initial states, exploring only safe transitions.
    """
    
    def __init__(
        self,
        safe_automaton: SafeAutomaton,
        nfa: NFA,
        labeler: RegionLabeler
    ):
        """
        Args:
            safe_automaton: Pruned automaton from safety synthesis
            nfa: Specification NFA (from regex)
            labeler: Maps grid cells to symbolic labels
        """
        self.safe_automaton = safe_automaton
        self.nfa = nfa
        self.labeler = labeler
        self.abstraction = safe_automaton.abstraction
        
        # Product automaton data
        self.states: Set[ProductState] = set()
        self.initial_states: Set[ProductState] = set()
        self.accepting_states: Set[ProductState] = set()
        
        # Transitions: {(ProductState, u_idx): Set[ProductState]}
        self.transitions: Dict[Tuple[ProductState, int], Set[ProductState]] = {}
        
        # Reverse map for controller lookup: cell_idx -> set of ProductStates
        self.cell_to_product_states: Dict[int, Set[ProductState]] = {}
    
    def _get_cell_label(self, cell_idx: int) -> Optional[str]:
        """Get the label of a grid cell."""
        cell_lo, cell_hi = self.abstraction.cell_to_bounds(cell_idx)
        return self.labeler.get_cell_label(cell_lo, cell_hi)
    
    def _make_product_state(
        self,
        nfa_states: FrozenSet[int],
        cell_idx: int,
        prev_label: Optional[str]
    ) -> ProductState:
        """
        Create a ProductState, handling label emission logic.
        
        If the cell has a label different from prev_label, we emit it
        (advancing the NFA). Otherwise, we keep the same NFA states.
        """
        current_label = self._get_cell_label(cell_idx)
        
        if current_label is not None and current_label != prev_label:
            # New region: emit label and advance NFA
            new_nfa_states = frozenset(self.nfa.get_next_states(set(nfa_states), current_label))
            return ProductState(new_nfa_states, cell_idx, current_label)
        else:
            # Same region or no region: NFA stays (implicit self-loop)
            return ProductState(nfa_states, cell_idx, current_label)
    
    def build(self, verbose: bool = True) -> None:
        """
        Build the Product Automaton via BFS from initial states.
        """
        if verbose:
            print("=" * 50)
            print("PRODUCT AUTOMATON CONSTRUCTION")
            print("=" * 50)
        
        num_controls = len(self.abstraction.dynamics.control_set)
        
        # Initialize: (initial NFA states, each safe cell, no previous label)
        initial_nfa = frozenset(self.nfa.get_initial_states())
        
        queue = deque()
        
        for cell_idx in self.safe_automaton.safe_states:
            ps = self._make_product_state(initial_nfa, cell_idx, None)
            if ps not in self.states:
                self.states.add(ps)
                self.initial_states.add(ps)
                queue.append(ps)
                
                # Track accepting states
                if self.nfa.has_accepting(set(ps.nfa_states)):
                    self.accepting_states.add(ps)
                
                # Reverse lookup
                if cell_idx not in self.cell_to_product_states:
                    self.cell_to_product_states[cell_idx] = set()
                self.cell_to_product_states[cell_idx].add(ps)
        
        if verbose:
            print(f"Initial product states: {len(self.initial_states)}")
            print(f"Accepting in initial: {len(self.accepting_states)}")
        
        # BFS exploration
        explored = 0
        while queue:
            ps = queue.popleft()
            explored += 1
            
            if explored % 10000 == 0 and verbose:
                print(f"  Explored {explored} states, queue size: {len(queue)}")
            
            # Try all controls
            for u_idx in range(num_controls):
                # Get physical successors (from safe automaton)
                phys_successors = self.safe_automaton.get_successors(ps.cell_idx, u_idx)
                
                if not phys_successors:
                    continue
                
                # Compute product successors
                product_successors = set()
                for next_cell in phys_successors:
                    next_ps = self._make_product_state(
                        ps.nfa_states, next_cell, ps.last_label
                    )
                    product_successors.add(next_ps)
                    
                    # Add to states if new
                    if next_ps not in self.states:
                        self.states.add(next_ps)
                        queue.append(next_ps)
                        
                        # Track accepting
                        if self.nfa.has_accepting(set(next_ps.nfa_states)):
                            self.accepting_states.add(next_ps)
                        
                        # Reverse lookup
                        if next_cell not in self.cell_to_product_states:
                            self.cell_to_product_states[next_cell] = set()
                        self.cell_to_product_states[next_cell].add(next_ps)
                
                # Store transition
                self.transitions[(ps, u_idx)] = product_successors
        
        if verbose:
            print(f"Product states: {len(self.states)}")
            print(f"Accepting states: {len(self.accepting_states)}")
            print(f"Product transitions: {len(self.transitions)}")
            print("=" * 50)
    
    def get_successors(self, ps: ProductState, u_idx: int) -> Set[ProductState]:
        """Get successor product states for a (state, control) pair."""
        return self.transitions.get((ps, u_idx), set())


# =============================================================================
# Product Synthesis
# =============================================================================

class ProductSynthesis:
    """
    Controller synthesis for automata-based specifications.
    
    Combines:
        1. Product automaton construction (physical abstraction × spec NFA)
        2. Reachability synthesis on product (reach accepting states)
    
    Obstacle avoidance is specified directly in the regex using [^...] syntax:
        - "A[^O]*B" : Go to A, then to B while avoiding region O
        - "A[^CD]*B" : Go to A, then to B while avoiding regions C and D
    
    Usage:
        # Define regions
        regions = {
            'A': [-4, -3, 3, 4],      # [x_min, x_max, y_min, y_max]
            'B': [3, 4, 3, 4],
            'O': [-1, 1, -1, 1]       # Obstacle region
        }
        
        # Create synthesis (obstacles in regex via [^O])
        synth = ProductSynthesis(
            abstraction=abstraction,
            regions=regions,
            spec="A[^O]*B"           # Visit A, avoid O, then B
        )
        synth.run()
        
        # Simulate
        trajectory, nfa_trace = synth.simulate(start_pos)
    """
    
    def __init__(
        self,
        abstraction: Abstraction,
        regions: Dict[str, List[float]],
        spec: str
    ):
        """
        Args:
            abstraction: Grid abstraction with precomputed transitions
            regions: Dict mapping region names to bounds [x_min, x_max, y_min, y_max]
            spec: Regular expression specification with optional [^...] for avoidance
                  (e.g., "AB", "A[^O]*B", "(A|B)[^CD]*E")
        """
        self.abstraction = abstraction
        self.regions = regions
        self.spec = spec
        
        # Build labeler for all regions
        self.labeler = RegionLabeler()
        state_dim = abstraction.dynamics.state_dim
        
        for name, bounds_list in regions.items():
            # Parse bounds list [min, max, min, max...] into [[min, max], [min, max], ...]
            region_bounds = []
            for i in range(0, len(bounds_list), 2):
                region_bounds.append([bounds_list[i], bounds_list[i+1]])
            
            # If region has fewer dimensions than state space, pad with full state bounds
            # This allows defining 2D regions for 3D systems (e.g., unicycle)
            if len(region_bounds) < state_dim:
                for d in range(len(region_bounds), state_dim):
                    region_bounds.append([
                        abstraction.state_bounds[d, 0],
                        abstraction.state_bounds[d, 1]
                    ])
            
            self.labeler.add_region(name, region_bounds)
        
        # Build NFA from spec
        self.nfa = regex_to_nfa(spec)
        
        # Results (populated by run())
        self.safe_automaton: Optional[SafeAutomaton] = None
        self.product: Optional[ProductAutomaton] = None
        self.winning_set: Set[ProductState] = set()
        self.reach_sets: List[Set[ProductState]] = []
        self.controller: Dict[ProductState, List[int]] = {}
    
    def run(self, verbose: bool = True) -> Set[ProductState]:
        """
        Run the full synthesis pipeline.
        
        Returns:
            Winning set of ProductStates
        """
        # Step 1: Prune cells with no valid controls (boundary cells that force robot out)
        # Note: This is NOT obstacle avoidance - obstacles are handled by [^...] in the regex.
        # This just removes cells where the robot physically cannot stay in bounds.
        if verbose:
            print(f"Specification: {self.spec}")
            print(f"NFA states: {len(self.nfa.states)}, alphabet: {self.nfa.alphabet}")
            print()
        
        safe_spec = set(range(self.abstraction.num_cells))
        self.safe_automaton = compute_safe_automaton(
            self.abstraction, safe_spec, verbose=verbose
        )
        
        if not self.safe_automaton.safe_states:
            if verbose:
                print("No safe states!")
            return set()
        
        # Step 2: Build Product Automaton
        self.product = ProductAutomaton(self.safe_automaton, self.nfa, self.labeler)
        self.product.build(verbose=verbose)
        
        if not self.product.accepting_states:
            if verbose:
                print("No accepting states in product!")
            return set()
        
        # Step 3: Reachability on Product
        self.winning_set, self.reach_sets, self.controller = self._compute_reachability(
            verbose=verbose
        )
        
        return self.winning_set
    
    def _compute_reachability(
        self,
        verbose: bool = True
    ) -> Tuple[Set[ProductState], List[Set[ProductState]], Dict[ProductState, List[int]]]:
        """
        Compute reachability to accepting states on the Product Automaton.
        """
        if verbose:
            print("=" * 50)
            print("PRODUCT REACHABILITY SYNTHESIS")
            print("=" * 50)
        
        num_controls = len(self.abstraction.dynamics.control_set)
        target = self.product.accepting_states
        
        if verbose:
            print(f"Target (accepting): {len(target)} product states")
        
        # Fixed-point: backward reachability
        r_k = target.copy()
        reach_sets = [r_k.copy()]
        
        if verbose:
            print(f"R₀: {len(r_k)} states")
        
        iteration = 0
        while True:
            iteration += 1
            
            # Pre(Rₖ): states that can reach Rₖ in one step
            pre_r_k = set()
            for ps in self.product.states:
                for u_idx in range(num_controls):
                    successors = self.product.get_successors(ps, u_idx)
                    # Robust: ALL successors must be in r_k
                    if successors and successors.issubset(r_k):
                        pre_r_k.add(ps)
                        break
            
            # Rₖ₊₁ = target ∪ Pre(Rₖ)
            r_k_plus_1 = target | pre_r_k
            added = len(r_k_plus_1) - len(r_k)
            
            if verbose:
                print(f"Iteration {iteration}: {len(r_k_plus_1)} states (+{added})")
            
            reach_sets.append(r_k_plus_1.copy())
            
            if r_k_plus_1 == r_k:
                break
            r_k = r_k_plus_1
        
        winning_set = r_k
        
        if verbose:
            # Count unique cells in winning set
            winning_cells = {ps.cell_idx for ps in winning_set}
            pct = 100 * len(winning_cells) / self.abstraction.num_cells
            print(f"Converged! Winning set: {len(winning_set)} product states")
            print(f"Winning cells: {len(winning_cells)} ({pct:.1f}%)")
        
        # Extract controller
        controller = self._extract_controller(reach_sets)
        
        if verbose:
            print(f"Controller covers {len(controller)} product states")
            print("=" * 50)
        
        return winning_set, reach_sets, controller
    
    def _extract_controller(
        self,
        reach_sets: List[Set[ProductState]]
    ) -> Dict[ProductState, List[int]]:
        """Extract controller from reachability layers."""
        controller = {}
        num_controls = len(self.abstraction.dynamics.control_set)
        winning_set = reach_sets[-1] if reach_sets else set()
        
        for ps in winning_set:
            # Find layer
            layer = -1
            for k, r_k in enumerate(reach_sets):
                if ps in r_k:
                    layer = k
                    break
            
            if layer == 0:
                # At accepting: any control that stays safe
                valid = []
                for u_idx in range(num_controls):
                    succ = self.product.get_successors(ps, u_idx)
                    if succ:  # Non-empty = safe
                        valid.append(u_idx)
            else:
                # Not at accepting: controls that reach layer k-1
                target_layer = reach_sets[layer - 1]
                valid = []
                for u_idx in range(num_controls):
                    succ = self.product.get_successors(ps, u_idx)
                    if succ and succ.issubset(target_layer):
                        valid.append(u_idx)
            
            if valid:
                controller[ps] = valid
        
        return controller
    
    def get_control(
        self,
        nfa_states: FrozenSet[int],
        cell_idx: int,
        last_label: Optional[str] = None,
        random_choice: bool = True
    ) -> Optional[int]:
        """
        Get a control for a given (NFA states, cell) combination.
        
        Args:
            nfa_states: Current NFA states
            cell_idx: Current grid cell
            last_label: Last emitted label (for state reconstruction)
            random_choice: Randomly select from valid controls
        
        Returns:
            Control index, or None if not in winning set
        """
        ps = ProductState(nfa_states, cell_idx, last_label)
        
        if ps not in self.controller:
            return None
        
        controls = self.controller[ps]
        return random.choice(controls) if random_choice else controls[0]
    
    def get_control_for_position(
        self,
        pos: np.ndarray,
        nfa_states: FrozenSet[int],
        last_label: Optional[str] = None,
        random_choice: bool = True
    ) -> Optional[int]:
        """Get control for a continuous position."""
        cell_idx = self.abstraction.point_to_cell(pos)
        if cell_idx == -1:
            return None
        return self.get_control(nfa_states, cell_idx, last_label, random_choice)
    
    def is_winning(self, nfa_states: FrozenSet[int], cell_idx: int, last_label: Optional[str] = None) -> bool:
        """Check if a product state is in the winning set."""
        ps = ProductState(nfa_states, cell_idx, last_label)
        return ps in self.winning_set
    
    def is_accepting(self, nfa_states: FrozenSet[int]) -> bool:
        """Check if NFA states include an accepting state."""
        return self.nfa.has_accepting(set(nfa_states))
    
    def get_distance_to_accepting(
        self,
        nfa_states: FrozenSet[int],
        cell_idx: int,
        last_label: Optional[str] = None
    ) -> int:
        """Get minimum steps to reach accepting state, or -1 if unreachable."""
        ps = ProductState(nfa_states, cell_idx, last_label)
        for k, r_k in enumerate(self.reach_sets):
            if ps in r_k:
                return k
        return -1
    
    def simulate(
        self,
        start_pos: np.ndarray,
        max_steps: int = 100,
        verbose: bool = True
    ) -> Tuple[np.ndarray, List[FrozenSet[int]]]:
        """
        Simulate the controller from a starting position.
        
        Returns:
            (trajectory, nfa_trace)
            - trajectory: Nx2 array of positions
            - nfa_trace: List of NFA state sets at each step
        """
        trajectory = [np.array(start_pos)]
        nfa_trace = []
        
        pos = np.array(start_pos)
        nfa_states = frozenset(self.nfa.get_initial_states())
        last_label = None
        
        # Get initial cell and update NFA
        cell_idx = self.abstraction.point_to_cell(pos)
        if cell_idx != -1:
            current_label = self._get_cell_label(cell_idx)
            if current_label is not None:
                nfa_states = frozenset(self.nfa.get_next_states(set(nfa_states), current_label))
                last_label = current_label
        
        nfa_trace.append(nfa_states)
        
        for step in range(max_steps):
            cell_idx = self.abstraction.point_to_cell(pos)
            if cell_idx == -1:
                if verbose:
                    print(f"Step {step}: Out of bounds at {pos}")
                break
            
            # Check if accepting
            if self.is_accepting(nfa_states):
                if verbose:
                    print(f"Step {step}: Reached accepting state at {pos}")
                break
            
            # Get control
            u_idx = self.get_control(nfa_states, cell_idx, last_label)
            if u_idx is None:
                if verbose:
                    print(f"Step {step}: No control at cell {cell_idx}, NFA states {nfa_states}")
                break
            
            u = self.abstraction.dynamics.control_set[u_idx]
            
            # Apply dynamics with random disturbance
            w_min = self.abstraction.dynamics.w_min
            w_max = self.abstraction.dynamics.w_max
            disturbance_dim = self.abstraction.dynamics.disturbance_dim
            w = np.random.uniform(w_min, w_max, size=disturbance_dim)
            next_pos = self.abstraction.dynamics.step(pos, u, w)
            
            # Update NFA state
            next_cell = self.abstraction.point_to_cell(next_pos)
            if next_cell != -1:
                current_label = self._get_cell_label(next_cell)
                if current_label is not None and current_label != last_label:
                    nfa_states = frozenset(self.nfa.get_next_states(set(nfa_states), current_label))
                    last_label = current_label
                elif current_label is None:
                    last_label = None
            
            trajectory.append(next_pos.copy())
            nfa_trace.append(nfa_states)
            pos = next_pos
        
        return np.array(trajectory), nfa_trace
    
    def _get_cell_label(self, cell_idx: int) -> Optional[str]:
        """Get label for a cell."""
        cell_lo, cell_hi = self.abstraction.cell_to_bounds(cell_idx)
        return self.labeler.get_cell_label(cell_lo, cell_hi)
    
    def get_winning_cells(self) -> Set[int]:
        """Get set of grid cells that are in the winning set (for any NFA state)."""
        return {ps.cell_idx for ps in self.winning_set}
    
    def get_winning_cells_for_nfa_state(self, nfa_states: FrozenSet[int]) -> Set[int]:
        """Get set of grid cells that are winning for a specific NFA state."""
        return {
            ps.cell_idx for ps in self.winning_set 
            if ps.nfa_states == nfa_states
        }
    
    def get_initial_nfa_states(self) -> FrozenSet[int]:
        """Get the initial NFA states (epsilon closure of start state)."""
        return frozenset(self.nfa.get_initial_states())
    
    def update_nfa_states(
        self, 
        nfa_states: FrozenSet[int], 
        cell_idx: int
    ) -> FrozenSet[int]:
        """
        Update NFA states based on the label of a cell.
        
        Args:
            nfa_states: Current NFA states
            cell_idx: Grid cell index
            
        Returns:
            Updated NFA states after processing the cell's label
        """
        label = self._get_cell_label(cell_idx)
        if label is not None:
            return frozenset(self.nfa.get_next_states(set(nfa_states), label))
        return nfa_states
