"""
NFA Module - Specification automaton from Regular Expressions.

Regular expressions are a subset of LTL for finite-trace properties.
We convert RegEx → NFA using Thompson's construction.

Supported syntax:
    - Single-char symbols: A, B, C (concatenation: AB means A then B)
    - Multi-char symbols: <region1>, <goal_A> (use angle brackets)
    - Concatenation: AB or <region1><region2>
    - Union: A|B (visit A or B)
    - Kleene star: A* (visit A zero or more times)
    - Parentheses: (A|B)C
    - Negative char class: [^XY] (matches any symbol EXCEPT X and Y)
    
Obstacle avoidance example:
    - A[^O]*B : Go to A, then navigate to B while avoiding region O
    - A[^CD]*B : Go to A, then navigate to B while avoiding C and D
"""

import numpy as np
from dataclasses import dataclass


# =============================================================================
# Negative Symbol Set (for [^...] syntax)
# =============================================================================

class NegativeSymbolSet(frozenset):
    """
    Represents a negative character class [^...] in regex.
    
    A transition labeled with NegativeSymbolSet({A, B}) fires when
    the input symbol is anything EXCEPT A or B.
    
    Example: [^CD] matches any symbol except C and D.
    """
    
    def __repr__(self):
        return f"[^{''.join(sorted(self))}]"
    
    def matches(self, symbol) -> bool:
        """Check if symbol matches this negative set (i.e., symbol NOT in set)."""
        return symbol is not None and symbol not in self


# =============================================================================
# NFA Class
# =============================================================================

class NFA:
    """
    Non-deterministic Finite Automaton.
    
    States are integers. Supports epsilon transitions (symbol = None).
    """
    
    def __init__(self):
        self.states = set()
        self.alphabet = set()
        self.transitions = {}  # {(state, symbol): set of next states}
        self.initial = None
        self.accepting = set()
    
    def add_transition(self, from_state: int, symbol, to_state: int):
        """Add a transition. symbol=None means epsilon transition."""
        self.states.add(from_state)
        self.states.add(to_state)
        if symbol is not None:
            self.alphabet.add(symbol)
        
        key = (from_state, symbol)
        if key not in self.transitions:
            self.transitions[key] = set()
        self.transitions[key].add(to_state)
    
    def get_next_states(self, current_states: set, symbol) -> set:
        """
        Get reachable states from current states via symbol (with epsilon closure).
        
        Self-loop semantics: If symbol is not in alphabet (e.g., None for free space),
        the NFA stays in place (implicit self-loops on unrecognized symbols).
        
        Negative transitions: A transition labeled with NegativeSymbolSet fires
        when the symbol is NOT in the exclusion set.
        """
        # First: epsilon closure of current states
        current_closure = self._epsilon_closure(current_states)
        
        # Take symbol transitions (including negative symbol sets)
        next_states = set()
        for s in current_closure:
            # Check exact symbol match
            next_states |= self.transitions.get((s, symbol), set())
            
            # Check negative symbol set transitions
            for (from_state, label), to_states in self.transitions.items():
                if from_state == s and isinstance(label, NegativeSymbolSet):
                    if label.matches(symbol):
                        next_states |= to_states
        
        # Check if symbol is "recognized" by the alphabet
        # A symbol is recognized if it's explicitly in the alphabet OR
        # if it's covered by a NegativeSymbolSet (either matches or is excluded)
        symbol_recognized = symbol in self.alphabet
        if not symbol_recognized:
            for alpha_sym in self.alphabet:
                if isinstance(alpha_sym, NegativeSymbolSet):
                    # This negative set "knows about" the symbol domain
                    # If symbol is in the excluded set, it's explicitly blocked
                    # If symbol matches, it was already handled above
                    symbol_recognized = True
                    break
        
        # If no transitions taken and symbol not recognized, stay in place
        if not next_states and not symbol_recognized:
            return current_closure
        
        # Epsilon closure of result
        return self._epsilon_closure(next_states)
    
    def _epsilon_closure(self, states: set) -> set:
        """Compute epsilon closure of a set of states."""
        closure = set(states)
        stack = list(states)
        
        while stack:
            s = stack.pop()
            for next_s in self.transitions.get((s, None), set()):
                if next_s not in closure:
                    closure.add(next_s)
                    stack.append(next_s)
        
        return closure
    
    def get_initial_states(self) -> set:
        """Get epsilon closure of initial state."""
        return self._epsilon_closure({self.initial})
    
    def has_accepting(self, states: set) -> bool:
        """Check if any state in the set is accepting."""
        return bool(states & self.accepting)


# =============================================================================
# RegEx Parser → NFA (Thompson's Construction)
# =============================================================================

@dataclass
class _Fragment:
    """NFA fragment with single entry and exit."""
    start: int
    end: int


class RegexParser:
    """
    Parse regex and build NFA via Thompson's construction.
    
    Grammar:
        expr   → term ('|' term)*
        term   → factor factor*
        factor → base ('*')?
        base   → symbol | '(' expr ')'
    """
    
    def __init__(self, regex: str):
        self.regex = regex
        self.pos = 0
        self.state_counter = 0
        self.nfa = NFA()
    
    def parse(self) -> NFA:
        """Parse the regex and return the NFA."""
        fragment = self._parse_expr()
        self.nfa.initial = fragment.start
        self.nfa.accepting.add(fragment.end)
        return self.nfa
    
    def _new_state(self) -> int:
        s = self.state_counter
        self.state_counter += 1
        return s
    
    def _current(self) -> str:
        return self.regex[self.pos] if self.pos < len(self.regex) else None
    
    def _consume(self, expected=None):
        c = self._current()
        if expected and c != expected:
            raise ValueError(f"Expected '{expected}' but got '{c}'")
        self.pos += 1
        return c
    
    def _parse_expr(self) -> _Fragment:
        """expr → term ('|' term)*"""
        left = self._parse_term()
        while self._current() == '|':
            self._consume('|')
            right = self._parse_term()
            left = self._union(left, right)
        return left
    
    def _parse_term(self) -> _Fragment:
        """term → factor factor*"""
        if self._current() in (None, ')', '|'):
            s = self._new_state()
            return _Fragment(s, s)
        
        left = self._parse_factor()
        while self._current() not in (None, ')', '|'):
            right = self._parse_factor()
            left = self._concat(left, right)
        return left
    
    def _parse_factor(self) -> _Fragment:
        """factor → base ('*')?"""
        base = self._parse_base()
        if self._current() == '*':
            self._consume('*')
            return self._star(base)
        return base
    
    def _parse_base(self) -> _Fragment:
        """base → symbol | '(' expr ')' | '<' multi_char_symbol '>' | '[^' chars ']'"""
        c = self._current()
        if c == '(':
            self._consume('(')
            frag = self._parse_expr()
            self._consume(')')
            return frag
        if c == '[':
            # Negative character class: [^ABC]
            self._consume('[')
            if self._current() != '^':
                raise ValueError("Expected '^' after '[' (only [^...] syntax is supported)")
            self._consume('^')
            # Parse single-char symbols until ']'
            excluded = set()
            while self._current() and self._current() != ']':
                ch = self._consume()
                if ch.isalnum():
                    excluded.add(ch)
                # Skip commas/spaces if user writes [^A,B,C] or [^A B C]
                elif ch not in (',', ' '):
                    raise ValueError(f"Unexpected character in [^...]: {ch}")
            self._consume(']')
            if not excluded:
                # [^] with no exclusions = match ANY symbol (wildcard)
                return self._wildcard()
            return self._negative_symbol(excluded)
        if c == '<':
            # Multi-character symbol: <region1>
            self._consume('<')
            sym = self._parse_until('>')
            self._consume('>')
            return self._symbol(sym)
        if c and c.isalnum():
            # Single character symbol
            sym = self._consume()
            return self._symbol(sym)
        raise ValueError(f"Unexpected: {c}")
    
    def _parse_until(self, end_char: str) -> str:
        """Parse characters until end_char is found."""
        start = self.pos
        while self._current() and self._current() != end_char:
            self.pos += 1
        return self.regex[start:self.pos]
    
    def _parse_symbol(self) -> str:
        """Parse alphanumeric symbol."""
        start = self.pos
        while self._current() and (self._current().isalnum() or self._current() == '_'):
            self.pos += 1
        return self.regex[start:self.pos]
    
    def _symbol(self, sym: str) -> _Fragment:
        s1, s2 = self._new_state(), self._new_state()
        self.nfa.add_transition(s1, sym, s2)
        return _Fragment(s1, s2)
    
    def _negative_symbol(self, excluded: set) -> _Fragment:
        """Create a fragment that matches any symbol EXCEPT those in excluded."""
        s1, s2 = self._new_state(), self._new_state()
        neg_set = NegativeSymbolSet(excluded)
        self.nfa.add_transition(s1, neg_set, s2)
        return _Fragment(s1, s2)
    
    def _wildcard(self) -> _Fragment:
        """Create a fragment that matches ANY symbol (empty negative set)."""
        s1, s2 = self._new_state(), self._new_state()
        # NegativeSymbolSet with empty exclusions matches everything
        neg_set = NegativeSymbolSet(set())
        self.nfa.add_transition(s1, neg_set, s2)
        return _Fragment(s1, s2)
    
    def _concat(self, left: _Fragment, right: _Fragment) -> _Fragment:
        self.nfa.add_transition(left.end, None, right.start)
        return _Fragment(left.start, right.end)
    
    def _union(self, left: _Fragment, right: _Fragment) -> _Fragment:
        s1, s2 = self._new_state(), self._new_state()
        self.nfa.add_transition(s1, None, left.start)
        self.nfa.add_transition(s1, None, right.start)
        self.nfa.add_transition(left.end, None, s2)
        self.nfa.add_transition(right.end, None, s2)
        return _Fragment(s1, s2)
    
    def _star(self, inner: _Fragment) -> _Fragment:
        s1, s2 = self._new_state(), self._new_state()
        self.nfa.add_transition(s1, None, s2)          # skip (zero times)
        self.nfa.add_transition(s1, None, inner.start) # enter
        self.nfa.add_transition(inner.end, None, s2)   # exit
        self.nfa.add_transition(inner.end, None, inner.start)  # repeat
        return _Fragment(s1, s2)


def regex_to_nfa(regex: str) -> NFA:
    """Convert a regular expression string to an NFA."""
    return RegexParser(regex).parse()


# =============================================================================
# Region Labeler
# =============================================================================

class RegionLabeler:
    """
    Maps grid cells to symbolic labels (region names).
    Priority: first added = highest priority.
    """
    
    def __init__(self):
        self.regions = []
    
    def add_region(self, name: str, bounds: list):
        """Add region with bounds [[x_min, x_max], [y_min, y_max], ...]."""
        self.regions.append((name, np.array(bounds)))
    
    def get_label(self, x: np.ndarray):
        """Get label of a point. Returns None if no region matches."""
        for name, bounds in self.regions:
            # Only check dimensions defined in the region (e.g., ignore theta if region is 2D)
            dims = min(len(x), len(bounds))
            if all(bounds[d, 0] <= x[d] <= bounds[d, 1] for d in range(dims)):
                return name
        return None
    
    def get_cell_label(self, cell_lo: np.ndarray, cell_hi: np.ndarray):
        """Get label if cell overlaps with any region. Returns None if no match."""
        for name, bounds in self.regions:
            # Only check dimensions defined in the region (e.g., ignore theta if region is 2D)
            dims = min(len(cell_lo), len(bounds))
            if all(cell_lo[d] < bounds[d, 1] and bounds[d, 0] < cell_hi[d] for d in range(dims)):
                return name
        return None
