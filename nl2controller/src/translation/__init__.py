"""
Translation Module - Darija to English Translation

This module handles translation from Moroccan Arabic (Darija) to English,
which is the first step in the NL2Controller pipeline for Darija input.
"""

from .translator import DarijaTranslator, TranslationResult

__all__ = [
    "DarijaTranslator",
    "TranslationResult",
]
