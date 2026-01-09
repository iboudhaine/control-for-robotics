"""
Synthesis Module - Stage 3 of the NL2Controller Pipeline

This module handles the synthesis of reactive controllers from
grounded LTL specifications. It integrates with TuLiP for formal
GR(1) synthesis when available, or provides a basic implementation.

The synthesis stage:
1. Takes grounded LTL formulas
2. Combines with robot transition system (environment model)
3. Solves a GR(1) game to produce a winning strategy
4. Outputs a finite-state controller (Mealy machine)

For formal correctness guarantees, install TuLiP:
    pip install tulip
"""

from .controller import Controller, State, Transition
from .synthesizer import (
    Synthesizer,
    BaseSynthesizer,
    SynthesisResult,
    SynthesisError,
    synthesize_from_grounding
)
from .transition_system import TransitionSystem, Variable

__all__ = [
    "Controller",
    "State", 
    "Transition",
    "Synthesizer",
    "BaseSynthesizer",
    "SynthesisResult",
    "SynthesisError",
    "synthesize_from_grounding",
    "TransitionSystem",
    "Variable",
]
