"""
Gemini-based Darija Translator
Uses Google's Gemini API for Darija to English translation
"""

import json
import logging
import requests
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GeminiTranslationResult:
    """Result of Darija to English translation."""
    darija: str
    english: str
    confidence: float = 1.0


class GeminiDarijaTranslator:
    """
    Translates Darija (Moroccan Arabic) to English using Gemini.
    """

    SYSTEM_PROMPT = """You are a Moroccan Arabic (Darija) to English translator specializing in robot navigation commands.

Your task: Translate Darija commands to clear, simple English.

Important:
- Translate naturally, don't transliterate
- Focus on the command's intent
- Keep it simple and direct
- Common Darija navigation terms:
  * sir/sˆır = go
  * wqef = stop
  * dor = turn
  * lyamin/limin/yamin = right
  * lisar/ysar = left
  * qdam/gdam = forward
  * lour = backward
  * zone/منطقة = zone/area
  * tjneb = avoid
  * lhayt = wall/obstacle

Output format: Just the English translation, no explanations.

Examples:
Darija: "sir l lyamin"
English: go to the right

Darija: "sir l zone a u tjneb lhayt"
English: go to zone a and avoid the wall

Darija: "wqef"
English: stop"""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.0,
        timeout: float = 30.0
    ):
        """
        Initialize Gemini translator.

        Args:
            api_key: Google API key
            model: Gemini model name
            temperature: Sampling temperature
            timeout: Request timeout
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"

    def translate(self, darija_text: str) -> GeminiTranslationResult:
        """
        Translate Darija to English.

        Args:
            darija_text: Darija command

        Returns:
            GeminiTranslationResult with translation
        """
        url = f"{self.base_url}/models/{self.model}:generateContent"

        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": self.api_key
        }

        # Construct prompt
        user_prompt = f"{self.SYSTEM_PROMPT}\n\nDarija: \"{darija_text}\"\nEnglish:"

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
                "maxOutputTokens": 100,
            }
        }

        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()

            # Extract translation
            if "candidates" in result and len(result["candidates"]) > 0:
                candidate = result["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    english = candidate["content"]["parts"][0]["text"].strip()

                    return GeminiTranslationResult(
                        darija=darija_text,
                        english=english,
                        confidence=0.95
                    )

            logger.error(f"Unexpected Gemini response: {result}")
            return self._fallback_translation(darija_text)

        except Exception as e:
            logger.error(f"Gemini translation failed: {e}")
            return self._fallback_translation(darija_text)

    def _fallback_translation(self, darija_text: str) -> GeminiTranslationResult:
        """Fallback to simple dictionary-based translation."""
        logger.warning("Using fallback translation")

        # Simple dictionary
        translations = {
            "sir": "go",
            "wqef": "stop",
            "dor": "turn",
            "lyamin": "right",
            "limin": "right",
            "yamin": "right",
            "lisar": "left",
            "ysar": "left",
            "qdam": "forward",
            "gdam": "forward",
            "lour": "backward",
            "tjneb": "avoid",
            "lhayt": "wall",
            "zone": "zone",
            "l": "to",
            "u": "and",
        }

        words = darija_text.lower().split()
        english_words = [translations.get(word, word) for word in words]
        english = " ".join(english_words)

        return GeminiTranslationResult(
            darija=darija_text,
            english=english,
            confidence=0.5
        )

    @classmethod
    def from_api_key(cls, api_key: str, model: str = "gemini-2.0-flash") -> "GeminiDarijaTranslator":
        """Create translator from API key."""
        return cls(api_key=api_key, model=model)
