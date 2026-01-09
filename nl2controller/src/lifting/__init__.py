"""
Lifting Module - Stage 1 of the NL2Controller Pipeline

This module handles the extraction of temporal logic patterns from natural language.
It uses the "Constrained Lifting" approach where the LLM is constrained to output
only from a set of known Declare patterns, preventing LTL hallucination.

Declare Patterns (based on process mining literature):
- Response: If A occurs, then B must eventually occur
- Precedence: B can only occur if A has occurred before
- Existence: A must occur at least once
- Absence: A must never occur
- ExactlyOnce: A must occur exactly once
- Init: A must be true at the initial state
- End: A must be true at the final state
- Always: A must always be true (Globally)
- Eventually: A must eventually be true (Finally)
- Until: A must hold until B occurs
- AlwaysEventually: A must occur infinitely often (GF)
"""

from .patterns import DeclarePattern, DECLARE_PATTERNS, get_pattern
from .engine import (
    LiftingEngine,
    LiftingResult,
    MultiLiftingResult,
    ExtractedEntity,
    BaseLiftingEngine
)

__all__ = [
    "DeclarePattern",
    "DECLARE_PATTERNS",
    "get_pattern",
    "LiftingEngine",
    "LiftingResult",
    "MultiLiftingResult",
    "ExtractedEntity",
    "BaseLiftingEngine",
]
