"""
Lifting Engine - Core logic for NL to Pattern extraction.

This module implements the "Constrained Lifting" approach where an LLM
extracts temporal patterns from natural language, but is constrained to
output only valid Declare patterns (not raw LTL formulas).
"""

import json
import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from abc import ABC, abstractmethod

from .patterns import DeclarePattern, DECLARE_PATTERNS, get_pattern

logger = logging.getLogger(__name__)


@dataclass
class ExtractedEntity:
    """
    An entity extracted from natural language.
    
    This represents a noun phrase that should later be grounded
    to a robot variable in Stage 2.
    """
    text: str               # Original text from NL command
    placeholder: str        # Placeholder name (e.g., "p1", "p2")
    confidence: float = 1.0 # Extraction confidence


@dataclass
class LiftingResult:
    """
    Result of the lifting stage.
    
    Contains the extracted pattern, entities (not yet grounded),
    and the abstract LTL formula with placeholders.
    """
    pattern: DeclarePattern
    entities: List[ExtractedEntity]
    abstract_ltl: str           # LTL with placeholders, e.g., "G(p1 -> F(p2))"
    confidence: float = 1.0
    original_command: str = ""
    
    def get_entity_map(self) -> Dict[str, str]:
        """Return mapping from placeholder to original text."""
        return {e.placeholder: e.text for e in self.entities}


@dataclass
class MultiLiftingResult:
    """
    Result when multiple patterns are extracted from a complex command.
    """
    results: List[LiftingResult]
    combined_ltl: str = ""      # Combined formula (conjunction of all)
    
    def __post_init__(self):
        if not self.combined_ltl and self.results:
            # Combine with conjunction
            formulas = [r.abstract_ltl for r in self.results]
            self.combined_ltl = " & ".join(f"({f})" for f in formulas)


class BaseLiftingEngine(ABC):
    """Abstract base class for lifting engines."""
    
    @abstractmethod
    def lift(self, nl_command: str) -> LiftingResult:
        """
        Extract a single temporal pattern from natural language.
        
        Args:
            nl_command: Natural language specification
            
        Returns:
            LiftingResult with pattern, entities, and abstract LTL
        """
        pass
    
    @abstractmethod
    def lift_multi(self, nl_command: str) -> MultiLiftingResult:
        """
        Extract multiple patterns from a complex command.
        
        Args:
            nl_command: Natural language specification (may contain multiple clauses)
            
        Returns:
            MultiLiftingResult with list of patterns
        """
        pass


class LiftingEngine(BaseLiftingEngine):
    """
    LLM-based lifting engine using OpenAI-compatible API.
    
    This engine sends the NL command to an LLM with a carefully crafted
    prompt that constrains output to Declare patterns only.
    
    Supports any OpenAI-compatible API (OpenAI, Azure, Ollama, vLLM, etc.)
    by configuring the base_url parameter.
    """
    
    SYSTEM_PROMPT = """You are a temporal logic expert assistant. Your task is to analyze natural language robot commands and extract temporal patterns.

IMPORTANT RULES:
1. You must ONLY output patterns from the allowed list
2. You must NOT invent variable names - use generic placeholders (p1, p2, etc.)
3. Extract entities as they appear in the user's text

ALLOWED PATTERNS:
- existence: "A must occur at least once" -> F(p1)
- absence: "A must never occur" -> G(!p1)
- always: "A must always be true" -> G(p1)
- eventually: "A must eventually happen" -> F(p1)
- always_eventually: "A must occur infinitely often" -> G(F(p1))
- response: "If A then eventually B" -> G(p1 -> F(p2))
- precedence: "B only after A" -> (!p2) U (p1)
- succession: "A followed by B" -> G(p1 -> F(p2)) & (!p2) U (p1)
- chain_response: "A immediately followed by B" -> G(p1 -> X(p2))
- mutual_exclusion: "A and B never together" -> G(!(p1 & p2))
- conditional_always: "Whenever A, B" -> G(p1 -> p2)
- until: "A until B" -> (p1) U (p2)

OUTPUT FORMAT (JSON):
{
    "patterns": [
        {
            "pattern_name": "<name from allowed list>",
            "entities": ["<entity 1 text>", "<entity 2 text if binary pattern>"],
            "confidence": <0.0-1.0>
        }
    ]
}

EXAMPLES:
Input: "Always patrol Zone A"
Output: {"patterns": [{"pattern_name": "always_eventually", "entities": ["Zone A"], "confidence": 0.95}]}

Input: "If battery is low, go to charging station"
Output: {"patterns": [{"pattern_name": "response", "entities": ["battery is low", "charging station"], "confidence": 0.9}]}
NOTE: Extract only the noun phrase, not action verbs. "go to charging station" -> extract "charging station" only.

Input: "Avoid Zone C"
Output: {"patterns": [{"pattern_name": "absence", "entities": ["Zone C"], "confidence": 0.95}]}"""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        timeout: float = 30.0,
        organization: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the lifting engine with OpenAI-compatible API.
        
        Args:
            api_key: API key for authentication
            model: Model to use (default: gpt-4o-mini)
            base_url: API endpoint URL (default: OpenAI, can be Ollama, Azure, vLLM, OpenRouter, etc.)
            temperature: Sampling temperature (0 for deterministic)
            max_tokens: Maximum tokens in response (None for model default)
            timeout: Request timeout in seconds
            organization: Optional organization ID (OpenAI only)
            default_headers: Optional default headers (useful for OpenRouter HTTP-Referer, X-Title)
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")
        
        # Build client kwargs
        client_kwargs = {
            "api_key": api_key,
            "base_url": base_url,
            "timeout": timeout,
        }
        if organization:
            client_kwargs["organization"] = organization
        if default_headers:
            client_kwargs["default_headers"] = default_headers
            
        self.client = OpenAI(**client_kwargs)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    @classmethod
    def from_config(cls, config: "OpenAIConfig", default_headers: Optional[Dict[str, str]] = None) -> "LiftingEngine":
        """
        Create a LiftingEngine from an OpenAIConfig object.
        
        Args:
            config: OpenAIConfig with API settings
            default_headers: Optional default headers (useful for OpenRouter)
            
        Returns:
            Configured LiftingEngine instance
        """
        from ..config import OpenAIConfig
        return cls(
            api_key=config.api_key,
            model=config.model,
            base_url=config.base_url,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout,
            organization=config.organization,
            default_headers=default_headers
        )
    
    def _call_llm(self, nl_command: str) -> Dict:
        """Call the LLM and parse JSON response."""
        # Build request kwargs
        request_kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"Extract temporal patterns from: \"{nl_command}\""}
            ],
        }
        
        # Add max_tokens if specified
        if self.max_tokens:
            request_kwargs["max_tokens"] = self.max_tokens
        
        # Try to use JSON mode if supported (OpenAI, some others)
        try:
            request_kwargs["response_format"] = {"type": "json_object"}
            response = self.client.chat.completions.create(**request_kwargs)
        except Exception as e:
            # Fallback without JSON mode for models that don't support it
            logger.warning(f"JSON mode not supported, falling back to text mode: {e}")
            if "response_format" in request_kwargs:
                del request_kwargs["response_format"]
            response = self.client.chat.completions.create(**request_kwargs)
        
        if not response or not response.choices or len(response.choices) == 0:
            raise ValueError(f"LLM returned no response choices for command: '{nl_command}'")
        
        content = response.choices[0].message.content
        
        if not content or not content.strip():
            # Check for function calls or other response formats
            message = response.choices[0].message
            if hasattr(message, 'function_call') and message.function_call:
                raise ValueError(
                    f"LLM returned function call instead of content for command: '{nl_command}'. "
                    f"This model may not support standard chat completions. "
                    f"Try a different model or check model capabilities."
                )
            if hasattr(message, 'tool_calls') and message.tool_calls:
                raise ValueError(
                    f"LLM returned tool calls instead of content for command: '{nl_command}'. "
                    f"This model may require tool/function calling setup. "
                    f"Try a different model that supports direct JSON output."
                )
            raise ValueError(
                f"LLM returned empty response for command: '{nl_command}'. "
                f"This may indicate:\n"
                f"  1. The model doesn't support JSON mode (try removing response_format)\n"
                f"  2. The model encountered an error\n"
                f"  3. The free model has rate limits or usage restrictions\n"
                f"Try:\n"
                f"  - A simpler command\n"
                f"  - A different model (e.g., gpt-4o-mini)\n"
                f"  - Check OpenRouter logs/status"
            )
        
        # Extract JSON from response (handle markdown code blocks)
        if "```json" in content:
            match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if match:
                content = match.group(1)
        elif "```" in content:
            match = re.search(r'```\s*(.*?)\s*```', content, re.DOTALL)
            if match:
                content = match.group(1)
        
        content = content.strip()
        if not content:
            raise ValueError("LLM response contained no valid JSON after extraction")
        
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            # Better error message with more context
            error_msg = (
                f"LLM returned invalid JSON for command: '{nl_command}'. "
                f"Content preview: {content[:300]}... "
                f"Error: {e}"
            )
            raise ValueError(error_msg) from e
    
    def _parse_pattern_output(
        self, 
        pattern_data: Dict,
        original_command: str
    ) -> LiftingResult:
        """Convert LLM output to LiftingResult."""
        pattern_name = pattern_data["pattern_name"]
        entity_texts = pattern_data.get("entities", [])
        confidence = pattern_data.get("confidence", 0.8)
        
        # Get the pattern template
        try:
            pattern = get_pattern(pattern_name)
        except KeyError:
            # Fallback to 'existence' if unknown pattern
            pattern = get_pattern("existence")
            confidence *= 0.5
        
        # Create entities with placeholders
        entities = []
        placeholders = []
        for i, text in enumerate(entity_texts[:pattern.arity]):
            placeholder = f"p{i+1}"
            entities.append(ExtractedEntity(
                text=text,
                placeholder=placeholder,
                confidence=confidence
            ))
            placeholders.append(placeholder)
        
        # Ensure we have enough placeholders
        while len(placeholders) < pattern.arity:
            placeholder = f"p{len(placeholders)+1}"
            placeholders.append(placeholder)
            entities.append(ExtractedEntity(
                text=f"<missing_{placeholder}>",
                placeholder=placeholder,
                confidence=0.0
            ))
        
        # Create abstract LTL
        abstract_ltl = pattern.instantiate(*placeholders)
        
        return LiftingResult(
            pattern=pattern,
            entities=entities,
            abstract_ltl=abstract_ltl,
            confidence=confidence,
            original_command=original_command
        )
    
    def lift(self, nl_command: str) -> LiftingResult:
        """Extract a single temporal pattern from natural language."""
        result = self._call_llm(nl_command)
        patterns = result.get("patterns", [])
        
        if not patterns:
            # Default to existence if nothing extracted
            return LiftingResult(
                pattern=get_pattern("existence"),
                entities=[ExtractedEntity(text=nl_command, placeholder="p1")],
                abstract_ltl="F(p1)",
                confidence=0.3,
                original_command=nl_command
            )
        
        return self._parse_pattern_output(patterns[0], nl_command)
    
    def lift_multi(self, nl_command: str) -> MultiLiftingResult:
        """Extract multiple patterns from a complex command."""
        result = self._call_llm(nl_command)
        patterns = result.get("patterns", [])
        
        if not patterns:
            single = self.lift(nl_command)
            return MultiLiftingResult(results=[single])
        
        results = [
            self._parse_pattern_output(p, nl_command) 
            for p in patterns
        ]
        
        return MultiLiftingResult(results=results)
