"""
NL2Controller - Natural Language to Controller Pipeline

A robust software pipeline that accepts Natural Language (NLP) commands
and automatically synthesizes correct-by-construction feedback controllers
for robotic systems.
"""

from .src.pipeline import (
    NL2ControllerPipeline,
    PipelineResult,
    create_pipeline,
)
from .src.lifting import (
    DeclarePattern,
    DECLARE_PATTERNS,
    LiftingResult,
    BaseLiftingEngine,
)
from .src.grounding import (
    RobotVocabulary,
    DictionaryGroundingFilter,
    GroundingResult,
    GroundingError,
)
from .src.synthesis import (
    Controller,
    State,
    Transition,
    Synthesizer,
    BaseSynthesizer,
    SynthesisResult,
    SynthesisError,
    TransitionSystem,
)

__version__ = "0.1.0"
__author__ = "NL2Controller Team"

__all__ = [
    # Pipeline
    "NL2ControllerPipeline",
    "PipelineResult",
    "create_pipeline",
    # Lifting
    "DeclarePattern",
    "DECLARE_PATTERNS",
    "LiftingResult",
    "BaseLiftingEngine",
    # Grounding
    "RobotVocabulary",
    "DictionaryGroundingFilter",
    "GroundingResult",
    "GroundingError",
    # Synthesis
    "Controller",
    "State",
    "Transition",
    "Synthesizer",
    "BaseSynthesizer",
    "SynthesisResult",
    "SynthesisError",
    "TransitionSystem",
]
