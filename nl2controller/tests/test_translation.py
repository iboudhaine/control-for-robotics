"""
Tests for the Darija to English Translation Module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.translation.translator import (
    DarijaTranslator,
    DemoTranslator,
    TranslationResult
)


class TestTranslationResult:
    """Tests for TranslationResult dataclass."""
    
    def test_basic_result(self):
        result = TranslationResult(
            original_darija="sir l lyamin",
            english_translation="go to the right"
        )
        assert result.original_darija == "sir l lyamin"
        assert result.english_translation == "go to the right"
        assert result.confidence == 1.0
    
    def test_result_with_metadata(self):
        result = TranslationResult(
            original_darija="سير",
            english_translation="go",
            transliteration="sir",
            confidence=0.95,
            detected_intent="navigation"
        )
        assert result.transliteration == "sir"
        assert result.confidence == 0.95
        assert result.detected_intent == "navigation"
    
    def test_str_representation(self):
        result = TranslationResult(
            original_darija="wqef",
            english_translation="stop"
        )
        assert "'wqef'" in str(result)
        assert "'stop'" in str(result)


class TestDemoTranslator:
    """Tests for the demo (dictionary-based) translator."""
    
    @pytest.fixture
    def translator(self):
        return DemoTranslator()
    
    def test_simple_words(self, translator):
        """Test translation of individual words."""
        # Navigation
        result = translator.translate("sir")
        assert "go" in result.english_translation.lower()
        
        result = translator.translate("wqef")
        assert "stop" in result.english_translation.lower()
    
    def test_directions(self, translator):
        """Test direction words."""
        result = translator.translate("lyamin")
        assert "right" in result.english_translation.lower()
        
        result = translator.translate("lisar")
        assert "left" in result.english_translation.lower()
        
        result = translator.translate("qdam")
        assert "forward" in result.english_translation.lower()
    
    def test_compound_commands(self, translator):
        """Test compound commands with 'u' (and)."""
        result = translator.translate("sir u wqef")
        assert "go" in result.english_translation.lower()
        assert "and" in result.english_translation.lower()
        assert "stop" in result.english_translation.lower()
    
    def test_navigation_command(self, translator):
        """Test full navigation command."""
        result = translator.translate("sir l lyamin")
        eng = result.english_translation.lower()
        assert "go" in eng
        assert "right" in eng
    
    def test_avoidance_command(self, translator):
        """Test avoidance command."""
        result = translator.translate("tjneb lhayt")
        eng = result.english_translation.lower()
        assert "avoid" in eng or "tjneb" in eng
        assert "wall" in eng
    
    def test_arabic_script(self, translator):
        """Test Arabic script translation."""
        result = translator.translate("سير")
        assert "go" in result.english_translation.lower()
        
        result = translator.translate("وقف")
        assert "stop" in result.english_translation.lower()
    
    def test_arabic_detection(self, translator):
        """Test Arabic script detection."""
        # Arabic script should have transliteration
        result = translator.translate("سير")
        assert result.transliteration is not None or "go" in result.english_translation
    
    def test_intent_detection(self, translator):
        """Test intent detection."""
        # Simple navigation
        result = translator.translate("sir")
        assert result.detected_intent == "navigation"
        
        # Compound
        result = translator.translate("sir u wqef")
        assert result.detected_intent == "compound"
        
        # Avoidance
        result = translator.translate("tjneb")
        assert result.detected_intent == "avoidance"
    
    def test_batch_translation(self, translator):
        """Test batch translation."""
        texts = ["sir", "wqef", "dor"]
        results = translator.translate_batch(texts)
        
        assert len(results) == 3
        assert all(isinstance(r, TranslationResult) for r in results)
    
    def test_unknown_words_kept(self, translator):
        """Test that unknown words are preserved."""
        result = translator.translate("xyz123")
        assert "xyz123" in result.english_translation


class TestDarijaTranslator:
    """Tests for the LLM-based Darija translator."""
    
    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI client."""
        with patch('openai.OpenAI') as mock:
            client = MagicMock()
            mock.return_value = client
            yield client
    
    def test_init_requires_openai(self, mock_openai_client):
        """Test that translator initializes with API key."""
        translator = DarijaTranslator(
            api_key="test-key",
            model="gpt-4o-mini"
        )
        assert translator.model == "gpt-4o-mini"
    
    def test_translate_success(self, mock_openai_client):
        """Test successful translation."""
        # Mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '''
        {
            "english": "go to the right",
            "transliteration": "sir l lyamin",
            "intent": "navigation",
            "confidence": 0.95
        }
        '''
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        translator = DarijaTranslator(api_key="test-key")
        result = translator.translate("sir l lyamin")
        
        assert result.english_translation == "go to the right"
        assert result.confidence == 0.95
    
    def test_translate_handles_markdown(self, mock_openai_client):
        """Test handling of markdown-wrapped JSON."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '''
        ```json
        {"english": "stop", "confidence": 0.9}
        ```
        '''
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        translator = DarijaTranslator(api_key="test-key")
        result = translator.translate("wqef")
        
        assert result.english_translation == "stop"
    
    def test_translate_api_error_fallback(self, mock_openai_client):
        """Test fallback on API error."""
        mock_openai_client.chat.completions.create.side_effect = Exception("API Error")
        
        translator = DarijaTranslator(api_key="test-key")
        result = translator.translate("sir")
        
        # Should return original text with low confidence
        assert result.original_darija == "sir"
        assert result.confidence == 0.0
    
    def test_from_config(self, mock_openai_client):
        """Test creating translator from config."""
        from src.config import OpenAIConfig
        
        config = OpenAIConfig(
            api_key="test-key",
            model="gpt-4",
            base_url="https://custom.api/v1"
        )
        
        translator = DarijaTranslator.from_config(config)
        assert translator.model == "gpt-4"
    
    def test_batch_translation(self, mock_openai_client):
        """Test batch translation."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"english": "go", "confidence": 0.9}'
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        translator = DarijaTranslator(api_key="test-key")
        results = translator.translate_batch(["sir", "wqef"])
        
        assert len(results) == 2


class TestTranslatorEdgeCases:
    """Edge case tests for translators."""
    
    @pytest.fixture
    def demo_translator(self):
        return DemoTranslator()
    
    def test_empty_input(self, demo_translator):
        result = demo_translator.translate("")
        assert result.english_translation == ""
    
    def test_whitespace_handling(self, demo_translator):
        result = demo_translator.translate("  sir   l   lyamin  ")
        assert "go" in result.english_translation.lower()
    
    def test_mixed_script(self, demo_translator):
        """Test mixed Arabic and Latin script."""
        result = demo_translator.translate("سير l lyamin")
        # Should handle gracefully
        assert result.english_translation is not None
    
    def test_numbers_preserved(self, demo_translator):
        result = demo_translator.translate("sir 5")
        assert "5" in result.english_translation
