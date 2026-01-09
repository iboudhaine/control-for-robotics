"""
Dictionary Grounding Filter - Maps abstract LTL to grounded LTL.

This is the core component that solves the "Grounding Gap" problem.
It takes the output of Stage 1 (abstract LTL with placeholders and entities)
and produces grounded LTL with actual robot variables.

Key Safety Feature: FAIL LOUD
If an entity cannot be grounded (not in dictionary), the filter raises
an explicit error rather than silently producing incorrect output.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .vocabulary import RobotVocabulary
from ..lifting.engine import LiftingResult, MultiLiftingResult, ExtractedEntity


class GroundingError(Exception):
    """
    Raised when grounding fails.
    
    This is a SAFETY FEATURE - we fail explicitly rather than
    producing incorrect specifications.
    """
    
    def __init__(
        self, 
        message: str, 
        ungrounded_entities: List[str] = None,
        suggestions: Dict[str, List[str]] = None
    ):
        super().__init__(message)
        self.ungrounded_entities = ungrounded_entities or []
        self.suggestions = suggestions or {}


@dataclass
class GroundedEntity:
    """
    An entity that has been successfully grounded.
    """
    original_text: str      # Original from NL command
    placeholder: str        # Placeholder used in abstract LTL
    system_var: str         # The grounded system variable
    confidence: float = 1.0


@dataclass
class GroundingResult:
    """
    Result of the grounding stage.
    
    Contains the grounded LTL formula with actual robot variables.
    """
    grounded_ltl: str                       # Final LTL with system variables
    grounded_entities: List[GroundedEntity] # Mapping details
    original_abstract_ltl: str              # For reference
    original_command: str = ""
    confidence: float = 1.0
    
    def get_variable_map(self) -> Dict[str, str]:
        """Return mapping from original text to system variable."""
        return {e.original_text: e.system_var for e in self.grounded_entities}


@dataclass
class MultiGroundingResult:
    """
    Result when grounding multiple patterns.
    """
    results: List[GroundingResult]
    combined_grounded_ltl: str = ""
    
    def __post_init__(self):
        if not self.combined_grounded_ltl and self.results:
            formulas = [r.grounded_ltl for r in self.results]
            self.combined_grounded_ltl = " & ".join(f"({f})" for f in formulas)


class DictionaryGroundingFilter:
    """
    Grounding filter using closed dictionary lookup.
    
    This filter maps entities to robot variables using ONLY
    entries in the provided vocabulary. Unknown entities cause
    explicit failures (the key safety feature).
    
    Attributes:
        vocabulary: The RobotVocabulary containing valid mappings
        strict_mode: If True, fail on any ungrounded entity
        fuzzy_threshold: If not strict, try fuzzy matching above this threshold
    """
    
    def __init__(
        self,
        vocabulary: RobotVocabulary,
        strict_mode: bool = True,
        fuzzy_threshold: float = 0.6
    ):
        """
        Initialize the grounding filter.
        
        Args:
            vocabulary: RobotVocabulary with valid variable mappings
            strict_mode: If True, fail on unknown entities (recommended)
            fuzzy_threshold: Threshold for fuzzy matching (if not strict)
        """
        self.vocabulary = vocabulary
        self.strict_mode = strict_mode
        self.fuzzy_threshold = fuzzy_threshold
    
    def ground_entity(self, entity: ExtractedEntity) -> GroundedEntity:
        """
        Ground a single entity to a system variable.
        
        Args:
            entity: The extracted entity from lifting stage
            
        Returns:
            GroundedEntity with the system variable
            
        Raises:
            GroundingError: If entity cannot be grounded
        """
        text = entity.text.lower().strip()
        
        # Try exact lookup first
        system_var = self.vocabulary.lookup(text)
        
        if system_var is not None:
            return GroundedEntity(
                original_text=entity.text,
                placeholder=entity.placeholder,
                system_var=system_var,
                confidence=entity.confidence
            )
        
        # Try fuzzy lookup if not strict
        if not self.strict_mode:
            system_var = self.vocabulary.lookup_fuzzy(text, self.fuzzy_threshold)
            if system_var is not None:
                return GroundedEntity(
                    original_text=entity.text,
                    placeholder=entity.placeholder,
                    system_var=system_var,
                    confidence=entity.confidence * 0.7  # Lower confidence for fuzzy
                )
        
        # Cannot ground - FAIL LOUD
        suggestions = self._find_suggestions(text)
        raise GroundingError(
            f"Cannot ground entity '{entity.text}' - not found in robot vocabulary. "
            f"This is a safety feature to prevent incorrect specifications.",
            ungrounded_entities=[entity.text],
            suggestions={entity.text: suggestions}
        )
    
    def _find_suggestions(self, text: str, max_suggestions: int = 5) -> List[str]:
        """
        Find similar phrases in the vocabulary as suggestions.
        """
        text_lower = text.lower()
        suggestions = []
        
        for phrase in self.vocabulary.get_all_phrases():
            # Simple heuristic: share any words
            text_words = set(text_lower.split())
            phrase_words = set(phrase.split())
            
            if text_words & phrase_words:
                suggestions.append(phrase)
                if len(suggestions) >= max_suggestions:
                    break
        
        return suggestions
    
    def ground(self, lifting_result: LiftingResult) -> GroundingResult:
        """
        Ground a lifting result to produce final LTL.
        
        Args:
            lifting_result: Output from the lifting stage
            
        Returns:
            GroundingResult with grounded LTL
            
        Raises:
            GroundingError: If any entity cannot be grounded
        """
        grounded_entities = []
        ungrounded = []
        all_suggestions = {}
        
        # Ground each entity
        for entity in lifting_result.entities:
            try:
                grounded = self.ground_entity(entity)
                grounded_entities.append(grounded)
            except GroundingError as e:
                ungrounded.extend(e.ungrounded_entities)
                all_suggestions.update(e.suggestions)
        
        # If any entities failed, raise combined error
        if ungrounded:
            raise GroundingError(
                f"Cannot ground {len(ungrounded)} entities: {ungrounded}. "
                f"Please add these to the robot vocabulary or rephrase the command.",
                ungrounded_entities=ungrounded,
                suggestions=all_suggestions
            )
        
        # Substitute placeholders with system variables
        grounded_ltl = lifting_result.abstract_ltl
        for ge in grounded_entities:
            grounded_ltl = grounded_ltl.replace(ge.placeholder, ge.system_var)
        
        # Calculate overall confidence
        confidences = [ge.confidence for ge in grounded_entities]
        overall_confidence = min(confidences) if confidences else 0.0
        
        return GroundingResult(
            grounded_ltl=grounded_ltl,
            grounded_entities=grounded_entities,
            original_abstract_ltl=lifting_result.abstract_ltl,
            original_command=lifting_result.original_command,
            confidence=overall_confidence
        )
    
    def ground_multi(
        self, 
        multi_result: MultiLiftingResult
    ) -> MultiGroundingResult:
        """
        Ground multiple lifting results.
        
        Args:
            multi_result: Output from lift_multi
            
        Returns:
            MultiGroundingResult with all grounded formulas
            
        Raises:
            GroundingError: If any entity in any formula cannot be grounded
        """
        grounding_results = []
        all_ungrounded = []
        all_suggestions = {}
        
        for lr in multi_result.results:
            try:
                gr = self.ground(lr)
                grounding_results.append(gr)
            except GroundingError as e:
                all_ungrounded.extend(e.ungrounded_entities)
                all_suggestions.update(e.suggestions)
        
        if all_ungrounded:
            raise GroundingError(
                f"Cannot ground {len(all_ungrounded)} entities across formulas: "
                f"{all_ungrounded}",
                ungrounded_entities=all_ungrounded,
                suggestions=all_suggestions
            )
        
        return MultiGroundingResult(results=grounding_results)
    
    def validate_ltl_syntax(self, ltl_formula: str) -> Tuple[bool, Optional[str]]:
        """
        Basic LTL syntax validation.
        
        This is a simple check - for production, integrate with
        a proper LTL parser (e.g., Spot).
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check balanced parentheses
        paren_count = 0
        for char in ltl_formula:
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
            if paren_count < 0:
                return False, "Unbalanced parentheses (extra closing)"
        
        if paren_count != 0:
            return False, "Unbalanced parentheses (unclosed)"
        
        # Check for valid LTL operators
        valid_operators = {'G', 'F', 'X', 'U', 'W', 'R', '!', '&', '|', '->', '<->'}
        
        # Simple token validation
        # This is a basic check; a real parser would be more thorough
        
        return True, None
