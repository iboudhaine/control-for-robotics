"""
Gemini-specific Lifting Engine
Adapts Google's Gemini API to work with NL2Controller
"""

import json
import logging
import requests
from typing import Optional, Dict
from .engine import BaseLiftingEngine, LiftingResult, MultiLiftingResult, ExtractedEntity
from .patterns import get_pattern, DECLARE_PATTERNS

logger = logging.getLogger(__name__)


class GeminiLiftingEngine(BaseLiftingEngine):
    """
    Lifting engine using Google's Gemini API.

    Gemini API is different from OpenAI's, so this provides a custom implementation.
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

Input: "Avoid Zone C"
Output: {"patterns": [{"pattern_name": "absence", "entities": ["Zone C"], "confidence": 0.95}]}"""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.0,
        max_tokens: Optional[int] = 1000,
        timeout: float = 30.0
    ):
        """
        Initialize Gemini lifting engine.

        Args:
            api_key: Google API key
            model: Gemini model name (default: gemini-2.0-flash)
            temperature: Sampling temperature (0 for deterministic)
            max_tokens: Maximum tokens in response
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"

    def lift(self, nl_command: str) -> LiftingResult:
        """
        Extract temporal pattern from natural language using Gemini.

        Args:
            nl_command: Natural language command

        Returns:
            LiftingResult with extracted pattern
        """
        # Prepare the request
        url = f"{self.base_url}/models/{self.model}:generateContent"

        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": self.api_key
        }

        # Construct prompt
        user_prompt = f"{self.SYSTEM_PROMPT}\n\nInput: \"{nl_command}\"\nOutput:"

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": user_prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
            }
        }

        try:
            # Make API request
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            # Parse response
            result = response.json()

            # Extract text from Gemini response format
            if "candidates" in result and len(result["candidates"]) > 0:
                candidate = result["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    text = candidate["content"]["parts"][0]["text"]

                    # Parse the JSON response
                    return self._parse_llm_response(text, nl_command)

            logger.error(f"Unexpected Gemini response format: {result}")
            return self._fallback_extraction(nl_command)

        except requests.exceptions.RequestException as e:
            logger.error(f"Gemini API request failed: {e}")
            return self._fallback_extraction(nl_command)
        except Exception as e:
            logger.error(f"Error in Gemini lifting: {e}")
            return self._fallback_extraction(nl_command)

    def _parse_llm_response(self, text: str, nl_command: str) -> LiftingResult:
        """Parse LLM JSON response into LiftingResult."""
        try:
            # Extract JSON from response (might have markdown code blocks)
            text = text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            data = json.loads(text)

            # Extract first pattern
            if "patterns" in data and len(data["patterns"]) > 0:
                pattern_data = data["patterns"][0]
                pattern_name = pattern_data["pattern_name"]
                entities_text = pattern_data["entities"]
                confidence = pattern_data.get("confidence", 0.8)

                # Get pattern template
                pattern = get_pattern(pattern_name)

                # Create entity objects
                entities = [
                    ExtractedEntity(text=ent, placeholder=f"p{i+1}")
                    for i, ent in enumerate(entities_text)
                ]

                # Generate LTL
                placeholders = [e.placeholder for e in entities]
                ltl = pattern.instantiate(*placeholders)

                return LiftingResult(
                    pattern=pattern,
                    entities=entities,
                    abstract_ltl=ltl,
                    confidence=confidence,
                    original_command=nl_command
                )
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            logger.debug(f"Response text: {text}")

        return self._fallback_extraction(nl_command)

    def _fallback_extraction(self, nl_command: str) -> LiftingResult:
        """Fallback to simple rule-based extraction."""
        logger.warning("Using fallback extraction")

        # Simple keyword matching
        cmd_lower = nl_command.lower()

        # Default pattern (always use unary for fallback to avoid arity issues)
        pattern = get_pattern("always_eventually")

        if any(word in cmd_lower for word in ["never", "avoid", "don't"]):
            pattern = get_pattern("absence")
        elif any(word in cmd_lower for word in ["always", "constantly", "continuously"]):
            pattern = get_pattern("always_eventually")

        # Extract simple entity (always single entity for safety)
        entity = ExtractedEntity(text=nl_command, placeholder="p1")
        ltl = pattern.instantiate("p1")

        return LiftingResult(
            pattern=pattern,
            entities=[entity],
            abstract_ltl=ltl,
            confidence=0.5,
            original_command=nl_command
        )

    def lift_multi(self, nl_command: str) -> MultiLiftingResult:
        """Extract multiple patterns from complex command."""
        # For now, just wrap single pattern
        result = self.lift(nl_command)
        return MultiLiftingResult(results=[result])

    @classmethod
    def from_api_key(cls, api_key: str, model: str = "gemini-2.0-flash") -> "GeminiLiftingEngine":
        """
        Create engine from API key.

        Args:
            api_key: Google API key
            model: Gemini model name

        Returns:
            GeminiLiftingEngine instance
        """
        return cls(api_key=api_key, model=model)
