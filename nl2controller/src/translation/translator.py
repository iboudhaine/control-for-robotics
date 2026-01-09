"""
Darija to English Translator

Uses an LLM to translate Moroccan Arabic (Darija) commands to English
before processing through the NL2Controller pipeline.
"""

import json
import re
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class TranslationResult:
    """
    Result of Darija to English translation.
    """
    original_darija: str
    english_translation: str
    transliteration: Optional[str] = None  # Latin script version of Darija
    confidence: float = 1.0
    detected_intent: Optional[str] = None  # e.g., "navigation", "avoidance"
    
    def __str__(self) -> str:
        return f"'{self.original_darija}' → '{self.english_translation}'"


class DarijaTranslator:
    """
    Translator for Moroccan Arabic (Darija) to English.
    
    Uses an OpenAI-compatible LLM API to perform translation with
    context-aware understanding of robot commands.
    
    Supports both Arabic script and Latin transliteration of Darija.
    """
    
    SYSTEM_PROMPT = """You are a specialized translator for Moroccan Arabic (Darija) to English.
Your task is to translate robot commands from Darija to clear, simple English.

IMPORTANT RULES:
1. Translate to simple, clear English suitable for robot commands
2. Preserve the imperative/command nature of the input
3. Handle both Arabic script and Latin transliteration (Franco-Arabic)
4. Recognize common Darija robot/navigation vocabulary

COMMON DARIJA TERMS:
- سير / sˆır / sir = go, move
- وقف / wqef = stop
- دور / dor = turn
- يمين / limin / yamin = right
- ليسر / lisar / ysar = left
- قدام / qdam / gdam = forward
- لور / lour / mor = backward, behind
- تجنب / tjneb = avoid
- حيط / heit / hayt = wall
- باب / bab = door
- كرسي / korsi = chair
- طاولة / tabla = table
- شي حاجة / chi haja = something, obstacle
- بزربة / bzzerba = quickly, fast
- بشوية / bchwiya = slowly
- حتى / hta = until
- و / w / u = and
- أولا / wla = or
- ل / l = to, towards
- من / mn / men = from

OUTPUT FORMAT (JSON):
{
    "english": "<clear English translation>",
    "transliteration": "<Latin script version if Arabic input>",
    "intent": "<navigation|avoidance|conditional|compound>",
    "confidence": <0.0-1.0>
}

EXAMPLES:
Input: "سير ل اليمين"
Output: {"english": "go to the right", "transliteration": "sir l lyamin", "intent": "navigation", "confidence": 0.95}

Input: "sir l lyamin u tjneb lhayt"
Output: {"english": "go to the right and avoid the wall", "transliteration": "sir l lyamin u tjneb lhayt", "intent": "compound", "confidence": 0.9}

Input: "وقف ملي تشوف شي حاجة"
Output: {"english": "stop when you see an obstacle", "transliteration": "wqef mli tchof chi haja", "intent": "conditional", "confidence": 0.85}

Input: "dor l lisar u sir l qdam"
Output: {"english": "turn left and go forward", "transliteration": "dor l lisar u sir l qdam", "intent": "compound", "confidence": 0.9}"""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.1,
        timeout: float = 30.0
    ):
        """
        Initialize the Darija translator.
        
        Args:
            api_key: API key for the LLM service
            model: Model to use for translation
            base_url: API endpoint URL
            temperature: Sampling temperature
            timeout: Request timeout in seconds
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout
        )
        self.model = model
        self.temperature = temperature
    
    @classmethod
    def from_config(cls, config: "OpenAIConfig") -> "DarijaTranslator":
        """
        Create a translator from OpenAIConfig.
        
        Args:
            config: OpenAIConfig with API settings
            
        Returns:
            Configured DarijaTranslator instance
        """
        return cls(
            api_key=config.api_key,
            model=config.model,
            base_url=config.base_url,
            temperature=config.temperature,
            timeout=config.timeout
        )
    
    def translate(self, darija_text: str) -> TranslationResult:
        """
        Translate Darija text to English.
        
        Args:
            darija_text: Text in Darija (Arabic script or transliteration)
            
        Returns:
            TranslationResult with English translation
        """
        logger.info(f"Translating: '{darija_text}'")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": f"Translate this Darija robot command to English: \"{darija_text}\""}
                ],
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            result = self._parse_response(content, darija_text)
            
            logger.info(f"Translation: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            # Fallback: return original text
            return TranslationResult(
                original_darija=darija_text,
                english_translation=darija_text,
                confidence=0.0
            )
    
    def _parse_response(self, content: str, original: str) -> TranslationResult:
        """Parse LLM response into TranslationResult."""
        # Handle markdown code blocks
        if "```json" in content:
            match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if match:
                content = match.group(1)
        elif "```" in content:
            match = re.search(r'```\s*(.*?)\s*```', content, re.DOTALL)
            if match:
                content = match.group(1)
        
        try:
            data = json.loads(content.strip())
            
            return TranslationResult(
                original_darija=original,
                english_translation=data.get("english", original),
                transliteration=data.get("transliteration"),
                confidence=data.get("confidence", 0.8),
                detected_intent=data.get("intent")
            )
        except json.JSONDecodeError:
            # If parsing fails, try to extract English from response
            return TranslationResult(
                original_darija=original,
                english_translation=content.strip(),
                confidence=0.5
            )
    
    def translate_batch(self, texts: list) -> list:
        """
        Translate multiple Darija texts.
        
        Args:
            texts: List of Darija texts
            
        Returns:
            List of TranslationResults
        """
        return [self.translate(text) for text in texts]


class DemoTranslator:
    """
    Demo translator that works without API access.
    
    Uses a simple dictionary-based approach for common Darija phrases.
    Useful for testing and demonstration.
    """
    
    # Common Darija to English mappings
    TRANSLATIONS = {
        # Basic navigation
        "sir": "go",
        "sˆır": "go",
        "سير": "go",
        "wqef": "stop",
        "وقف": "stop",
        "dor": "turn",
        "دور": "turn",
        
        # Directions
        "lyamin": "right",
        "limin": "right",
        "yamin": "right",
        "اليمين": "right",
        "lisar": "left",
        "ysar": "left",
        "ليسار": "left",
        "qdam": "forward",
        "gdam": "forward",
        "قدام": "forward",
        "lour": "backward",
        "mor": "backward",
        "لور": "backward",
        
        # Obstacles/objects
        "lhayt": "the wall",
        "hayt": "wall",
        "heit": "wall",
        "حيط": "wall",
        "bab": "door",
        "باب": "door",
        "korsi": "chair",
        "كرسي": "chair",
        "tabla": "table",
        "طاولة": "table",
        "chi haja": "obstacle",
        "شي حاجة": "obstacle",
        
        # Verbs
        "tjneb": "avoid",
        "تجنب": "avoid",
        "chof": "see",
        "شوف": "see",
        
        # Connectors
        "u": "and",
        "w": "and",
        "و": "and",
        "wla": "or",
        "أولا": "or",
        "l": "to",
        "ل": "to",
        "hta": "until",
        "حتى": "until",
        "mli": "when",
        "ملي": "when",
        "ila": "if",
        "إلا": "if",
    }
    
    def translate(self, darija_text: str) -> TranslationResult:
        """
        Translate Darija using dictionary lookup.
        
        Args:
            darija_text: Text in Darija
            
        Returns:
            TranslationResult with English translation
        """
        # Normalize and tokenize
        text_lower = darija_text.lower().strip()
        
        # Replace known words
        english_words = []
        words = re.split(r'\s+', text_lower)
        
        for word in words:
            # Try direct lookup
            if word in self.TRANSLATIONS:
                english_words.append(self.TRANSLATIONS[word])
            else:
                # Try partial matching for compound phrases
                found = False
                for darija, english in self.TRANSLATIONS.items():
                    if darija in word:
                        english_words.append(english)
                        found = True
                        break
                
                if not found:
                    english_words.append(word)  # Keep original if not found
        
        english = " ".join(english_words)
        
        # Clean up common patterns
        english = english.replace(" to right", " to the right")
        english = english.replace(" to left", " to the left")
        english = english.replace(" to forward", " forward")
        english = english.replace("go to forward", "go forward")
        
        # Detect intent
        intent = self._detect_intent(english)
        
        return TranslationResult(
            original_darija=darija_text,
            english_translation=english,
            transliteration=text_lower if self._is_arabic(darija_text) else None,
            confidence=0.7,
            detected_intent=intent
        )
    
    def _is_arabic(self, text: str) -> bool:
        """Check if text contains Arabic characters."""
        return any('\u0600' <= char <= '\u06FF' for char in text)
    
    def _detect_intent(self, english: str) -> str:
        """Detect the intent of the translated command."""
        english_lower = english.lower()
        
        if "and" in english_lower or "," in english_lower:
            return "compound"
        elif "if" in english_lower or "when" in english_lower:
            return "conditional"
        elif "avoid" in english_lower or "not" in english_lower:
            return "avoidance"
        else:
            return "navigation"
    
    def translate_batch(self, texts: list) -> list:
        """Translate multiple texts."""
        return [self.translate(text) for text in texts]
