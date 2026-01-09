"""
Grounding Module - Stage 2 of the NL2Controller Pipeline

This module handles the mapping of extracted entities (from Stage 1) to
actual robot variables. It uses a CLOSED DICTIONARY approach for safety:
- Only variables in the dictionary can be used
- Unknown entities cause explicit failures (not silent mismatches)

This is the "FIX" for the Grounding Gap problem identified in VLTL-Bench.
"""

from .vocabulary import RobotVocabulary
from .grounding_filter import (
    DictionaryGroundingFilter,
    GroundingResult,
    GroundingError,
)

__all__ = [
    "RobotVocabulary",
    "DictionaryGroundingFilter",
    "GroundingResult",
    "GroundingError",
]
