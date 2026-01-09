"""
Declare Patterns - The constrained set of temporal logic templates.

These patterns are derived from the Declare process modeling language and
represent the most common temporal behaviors in robotics/control systems.
By constraining LLM output to these patterns, we prevent LTL hallucination.

Reference: van der Aalst, W.M.P. et al. "Declarative workflows: Balancing 
between flexibility and support" (2009)
"""

from dataclasses import dataclass
from typing import List, Callable
from enum import Enum


class PatternType(Enum):
    """Categories of Declare patterns."""
    EXISTENCE = "existence"      # Patterns about occurrence
    RELATION = "relation"        # Patterns about relationships between events
    TEMPORAL = "temporal"        # Patterns with explicit temporal operators


@dataclass(frozen=True)
class DeclarePattern:
    """
    A Declare pattern template.
    
    Attributes:
        name: Unique identifier for the pattern
        description: Human-readable description
        arity: Number of atomic propositions required (1 or 2)
        ltl_template: LTL formula template with placeholders {0}, {1}
        nl_indicators: Keywords/phrases that suggest this pattern
        pattern_type: Category of the pattern
    """
    name: str
    description: str
    arity: int
    ltl_template: str
    nl_indicators: List[str]
    pattern_type: PatternType
    
    def instantiate(self, *props: str) -> str:
        """
        Create a concrete LTL formula by substituting propositions.
        
        Args:
            props: Atomic propositions to substitute
            
        Returns:
            Concrete LTL formula string
            
        Raises:
            ValueError: If wrong number of propositions provided
        """
        if len(props) != self.arity:
            raise ValueError(
                f"Pattern '{self.name}' requires {self.arity} propositions, "
                f"got {len(props)}"
            )
        return self.ltl_template.format(*props)


# =============================================================================
# DECLARE PATTERN LIBRARY
# =============================================================================

DECLARE_PATTERNS = {
    # -------------------------------------------------------------------------
    # EXISTENCE PATTERNS (Unary)
    # -------------------------------------------------------------------------
    "existence": DeclarePattern(
        name="existence",
        description="A must occur at least once",
        arity=1,
        ltl_template="F({0})",
        nl_indicators=["eventually", "at some point", "must happen", "will occur"],
        pattern_type=PatternType.EXISTENCE,
    ),
    
    "absence": DeclarePattern(
        name="absence",
        description="A must never occur",
        arity=1,
        ltl_template="G(!{0})",
        nl_indicators=["never", "avoid", "don't", "must not", "forbidden", "prohibited"],
        pattern_type=PatternType.EXISTENCE,
    ),
    
    "exactly_once": DeclarePattern(
        name="exactly_once",
        description="A must occur exactly once",
        arity=1,
        ltl_template="F({0}) & G({0} -> X(G(!{0})))",
        nl_indicators=["exactly once", "only once", "single time"],
        pattern_type=PatternType.EXISTENCE,
    ),
    
    "init": DeclarePattern(
        name="init",
        description="A must be true initially",
        arity=1,
        ltl_template="{0}",
        nl_indicators=["start with", "initially", "begin with", "at start"],
        pattern_type=PatternType.EXISTENCE,
    ),
    
    "always": DeclarePattern(
        name="always",
        description="A must always be true (invariant)",
        arity=1,
        ltl_template="G({0})",
        nl_indicators=["always", "continuously", "at all times", "invariably", "maintain"],
        pattern_type=PatternType.TEMPORAL,
    ),
    
    "eventually": DeclarePattern(
        name="eventually",
        description="A must eventually become true",
        arity=1,
        ltl_template="F({0})",
        nl_indicators=["eventually", "finally", "at some point", "ultimately"],
        pattern_type=PatternType.TEMPORAL,
    ),
    
    "always_eventually": DeclarePattern(
        name="always_eventually",
        description="A must occur infinitely often (liveness)",
        arity=1,
        ltl_template="G(F({0}))",
        nl_indicators=["repeatedly", "infinitely often", "keep doing", "patrol", "cycle"],
        pattern_type=PatternType.TEMPORAL,
    ),
    
    # -------------------------------------------------------------------------
    # RELATION PATTERNS (Binary)
    # -------------------------------------------------------------------------
    "response": DeclarePattern(
        name="response",
        description="If A occurs, then B must eventually occur afterwards",
        arity=2,
        ltl_template="G({0} -> F({1}))",
        nl_indicators=["if then", "when then", "triggers", "leads to", "causes", "implies"],
        pattern_type=PatternType.RELATION,
    ),
    
    "precedence": DeclarePattern(
        name="precedence",
        description="B can only occur if A has occurred before",
        arity=2,
        ltl_template="(!{1}) U ({0})",
        nl_indicators=["before", "precedes", "first", "prior to", "only after"],
        pattern_type=PatternType.RELATION,
    ),
    
    "succession": DeclarePattern(
        name="succession",
        description="A must be followed by B, and B requires A before",
        arity=2,
        ltl_template="G({0} -> F({1})) & ((!{1}) U ({0}))",
        nl_indicators=["followed by", "then", "and then", "sequence"],
        pattern_type=PatternType.RELATION,
    ),
    
    "alternate_response": DeclarePattern(
        name="alternate_response",
        description="Each A must be followed by B before the next A",
        arity=2,
        ltl_template="G({0} -> X((!{0}) U ({1})))",
        nl_indicators=["alternating", "each time", "every time"],
        pattern_type=PatternType.RELATION,
    ),
    
    "chain_response": DeclarePattern(
        name="chain_response",
        description="If A occurs, B must occur immediately after",
        arity=2,
        ltl_template="G({0} -> X({1}))",
        nl_indicators=["immediately", "right after", "directly", "next"],
        pattern_type=PatternType.RELATION,
    ),
    
    "not_coexistence": DeclarePattern(
        name="not_coexistence",
        description="A and B cannot both occur",
        arity=2,
        ltl_template="!(F({0}) & F({1}))",
        nl_indicators=["exclusive", "either or", "not both", "mutually exclusive"],
        pattern_type=PatternType.RELATION,
    ),
    
    "mutual_exclusion": DeclarePattern(
        name="mutual_exclusion",
        description="A and B cannot be true at the same time",
        arity=2,
        ltl_template="G(!({0} & {1}))",
        nl_indicators=["not at same time", "exclusive", "mutex", "conflict"],
        pattern_type=PatternType.RELATION,
    ),
    
    "conditional_always": DeclarePattern(
        name="conditional_always",
        description="If condition A, then B must always hold",
        arity=2,
        ltl_template="G({0} -> {1})",
        nl_indicators=["whenever", "if always", "when", "during"],
        pattern_type=PatternType.RELATION,
    ),
    
    "conditional_eventually": DeclarePattern(
        name="conditional_eventually",
        description="If condition A, then B must eventually happen",
        arity=2,
        ltl_template="G({0} -> F({1}))",
        nl_indicators=["if eventually", "when finally", "condition leads to"],
        pattern_type=PatternType.RELATION,
    ),
    
    "until": DeclarePattern(
        name="until",
        description="A must hold until B becomes true",
        arity=2,
        ltl_template="({0}) U ({1})",
        nl_indicators=["until", "till", "up to", "before"],
        pattern_type=PatternType.TEMPORAL,
    ),
    
    "weak_until": DeclarePattern(
        name="weak_until",
        description="A must hold until B (or forever if B never occurs)",
        arity=2,
        ltl_template="({0}) W ({1})",
        nl_indicators=["until or always", "unless", "as long as"],
        pattern_type=PatternType.TEMPORAL,
    ),
}


def get_pattern(name: str) -> DeclarePattern:
    """
    Retrieve a Declare pattern by name.
    
    Args:
        name: Pattern identifier
        
    Returns:
        The DeclarePattern object
        
    Raises:
        KeyError: If pattern not found
    """
    if name not in DECLARE_PATTERNS:
        raise KeyError(
            f"Unknown pattern '{name}'. Available: {list(DECLARE_PATTERNS.keys())}"
        )
    return DECLARE_PATTERNS[name]


def get_patterns_by_type(pattern_type: PatternType) -> List[DeclarePattern]:
    """Get all patterns of a specific type."""
    return [p for p in DECLARE_PATTERNS.values() if p.pattern_type == pattern_type]


def get_unary_patterns() -> List[DeclarePattern]:
    """Get all patterns that require only one proposition."""
    return [p for p in DECLARE_PATTERNS.values() if p.arity == 1]


def get_binary_patterns() -> List[DeclarePattern]:
    """Get all patterns that require two propositions."""
    return [p for p in DECLARE_PATTERNS.values() if p.arity == 2]
