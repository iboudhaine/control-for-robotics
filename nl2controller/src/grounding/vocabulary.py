"""
Robot Vocabulary - The closed dictionary of valid robot variables.

This module defines the structure and loading of the robot's variable
dictionary. The dictionary maps natural language phrases to system
variables, ensuring only valid variables can be used in the final
LTL specification.
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class VocabularyEntry:
    """
    A single entry in the robot vocabulary.
    
    Attributes:
        phrase: The natural language phrase (normalized to lowercase)
        system_var: The corresponding system variable expression
        category: Category of the variable (location, sensor, action, etc.)
        aliases: Alternative phrases that map to the same variable
    """
    phrase: str
    system_var: str
    category: str
    aliases: List[str] = field(default_factory=list)


class RobotVocabulary:
    """
    The closed dictionary of valid robot variables.
    
    This class loads and manages the mapping from natural language
    phrases to system variables. It ensures:
    1. Only known phrases can be grounded
    2. Aliases are properly resolved
    3. Case-insensitive matching
    """
    
    def __init__(self, dictionary_path: Optional[str] = None):
        """
        Initialize the vocabulary.
        
        Args:
            dictionary_path: Path to JSON vocabulary file, or None for empty vocab
        """
        self._entries: Dict[str, VocabularyEntry] = {}
        self._phrase_to_var: Dict[str, str] = {}
        self._categories: Dict[str, List[str]] = {}
        self._metadata: Dict = {}
        
        if dictionary_path:
            self.load_from_file(dictionary_path)
    
    def load_from_file(self, path: str) -> None:
        """
        Load vocabulary from a JSON file.
        
        Expected format:
        {
            "metadata": {...},
            "variables": {
                "category": {
                    "phrase": "system_var",
                    ...
                },
                ...
            },
            "aliases": {
                "alias": "target_phrase",
                ...
            }
        }
        """
        with open(path, 'r') as f:
            data = json.load(f)
        
        self._metadata = data.get("metadata", {})
        
        # Load variables by category
        variables = data.get("variables", {})
        for category, entries in variables.items():
            self._categories[category] = []
            for phrase, system_var in entries.items():
                normalized_phrase = phrase.lower().strip()
                entry = VocabularyEntry(
                    phrase=normalized_phrase,
                    system_var=system_var,
                    category=category
                )
                self._entries[normalized_phrase] = entry
                self._phrase_to_var[normalized_phrase] = system_var
                self._categories[category].append(normalized_phrase)
        
        # Load aliases
        aliases = data.get("aliases", {})
        for alias, target in aliases.items():
            normalized_alias = alias.lower().strip()
            normalized_target = target.lower().strip()
            
            if normalized_target in self._entries:
                # Map alias to the same system variable as target
                target_var = self._phrase_to_var[normalized_target]
                self._phrase_to_var[normalized_alias] = target_var
                
                # Add alias to the entry
                self._entries[normalized_target].aliases.append(normalized_alias)
    
    def load_from_dict(self, data: Dict) -> None:
        """Load vocabulary from a Python dictionary (same format as JSON)."""
        self._metadata = data.get("metadata", {})
        
        variables = data.get("variables", {})
        for category, entries in variables.items():
            self._categories[category] = []
            for phrase, system_var in entries.items():
                normalized_phrase = phrase.lower().strip()
                entry = VocabularyEntry(
                    phrase=normalized_phrase,
                    system_var=system_var,
                    category=category
                )
                self._entries[normalized_phrase] = entry
                self._phrase_to_var[normalized_phrase] = system_var
                self._categories[category].append(normalized_phrase)
        
        aliases = data.get("aliases", {})
        for alias, target in aliases.items():
            normalized_alias = alias.lower().strip()
            normalized_target = target.lower().strip()
            
            if normalized_target in self._entries:
                target_var = self._phrase_to_var[normalized_target]
                self._phrase_to_var[normalized_alias] = target_var
                self._entries[normalized_target].aliases.append(normalized_alias)
    
    def lookup(self, phrase: str) -> Optional[str]:
        """
        Look up a phrase in the vocabulary.
        
        Args:
            phrase: Natural language phrase to look up
            
        Returns:
            System variable if found, None otherwise
        """
        normalized = phrase.lower().strip()
        return self._phrase_to_var.get(normalized)
    
    def lookup_strict(self, phrase: str) -> str:
        """
        Look up a phrase, raising an error if not found.
        
        Args:
            phrase: Natural language phrase to look up
            
        Returns:
            System variable
            
        Raises:
            KeyError: If phrase not in vocabulary
        """
        result = self.lookup(phrase)
        if result is None:
            raise KeyError(
                f"Phrase '{phrase}' not found in robot vocabulary. "
                f"Available phrases: {self.get_all_phrases()[:10]}..."
            )
        return result
    
    def lookup_fuzzy(self, phrase: str, threshold: float = 0.6) -> Optional[str]:
        """
        Fuzzy lookup with simple substring matching.
        
        This is a simple implementation - for production, consider
        using proper fuzzy matching libraries.
        
        Args:
            phrase: Phrase to look up
            threshold: Minimum similarity threshold (0-1)
            
        Returns:
            Best matching system variable, or None if no good match
        """
        normalized = phrase.lower().strip()
        
        # First try exact match
        if normalized in self._phrase_to_var:
            return self._phrase_to_var[normalized]
        
        # Try substring matching
        best_match = None
        best_score = 0.0
        
        for known_phrase, system_var in self._phrase_to_var.items():
            # Simple similarity: longest common substring ratio
            score = self._simple_similarity(normalized, known_phrase)
            if score > best_score and score >= threshold:
                best_score = score
                best_match = system_var
        
        return best_match
    
    def _simple_similarity(self, s1: str, s2: str) -> float:
        """
        Simple similarity score between two strings.
        
        Uses word overlap for simplicity.
        """
        words1 = set(s1.split())
        words2 = set(s2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def contains(self, phrase: str) -> bool:
        """Check if a phrase exists in the vocabulary."""
        return phrase.lower().strip() in self._phrase_to_var
    
    def get_all_phrases(self) -> List[str]:
        """Get all known phrases (including aliases)."""
        return list(self._phrase_to_var.keys())
    
    def get_all_variables(self) -> Set[str]:
        """Get all unique system variables."""
        return set(self._phrase_to_var.values())
    
    def get_phrases_by_category(self, category: str) -> List[str]:
        """Get all phrases in a specific category."""
        return self._categories.get(category, [])
    
    def get_categories(self) -> List[str]:
        """Get all category names."""
        return list(self._categories.keys())
    
    def get_entry(self, phrase: str) -> Optional[VocabularyEntry]:
        """Get the full entry for a phrase."""
        normalized = phrase.lower().strip()
        
        # Direct entry
        if normalized in self._entries:
            return self._entries[normalized]
        
        # Check if it's an alias
        if normalized in self._phrase_to_var:
            # Find the canonical phrase
            target_var = self._phrase_to_var[normalized]
            for entry in self._entries.values():
                if entry.system_var == target_var:
                    return entry
        
        return None
    
    @property
    def metadata(self) -> Dict:
        """Get vocabulary metadata."""
        return self._metadata
    
    def __len__(self) -> int:
        """Number of entries (not counting aliases)."""
        return len(self._entries)
    
    def __contains__(self, phrase: str) -> bool:
        """Support 'in' operator."""
        return self.contains(phrase)
    
    def __repr__(self) -> str:
        return (
            f"RobotVocabulary("
            f"entries={len(self._entries)}, "
            f"categories={list(self._categories.keys())})"
        )
