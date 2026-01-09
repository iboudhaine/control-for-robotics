"""
Tests for the Lifting module (Stage 1).

Uses pytest-mock to mock LLM API calls without requiring actual API access.
"""

import pytest
import json
from unittest.mock import MagicMock, patch

from src.lifting.patterns import (
    DeclarePattern,
    DECLARE_PATTERNS,
    get_pattern,
    get_unary_patterns,
    get_binary_patterns,
    PatternType,
)
from src.lifting.engine import (
    LiftingEngine,
    LiftingResult,
    ExtractedEntity,
    MultiLiftingResult,
    BaseLiftingEngine
)


class TestDeclarePatterns:
    """Tests for the Declare pattern library."""
    
    def test_pattern_library_not_empty(self):
        """Ensure pattern library has entries."""
        assert len(DECLARE_PATTERNS) > 0
    
    def test_get_valid_pattern(self):
        """Test retrieving a valid pattern."""
        pattern = get_pattern("response")
        assert pattern.name == "response"
        assert pattern.arity == 2
    
    def test_get_invalid_pattern_raises(self):
        """Test that invalid pattern raises KeyError."""
        with pytest.raises(KeyError):
            get_pattern("nonexistent_pattern")
    
    def test_unary_patterns_have_arity_1(self):
        """Verify all unary patterns have arity 1."""
        for pattern in get_unary_patterns():
            assert pattern.arity == 1
    
    def test_binary_patterns_have_arity_2(self):
        """Verify all binary patterns have arity 2."""
        for pattern in get_binary_patterns():
            assert pattern.arity == 2
    
    def test_pattern_instantiation_unary(self):
        """Test instantiating a unary pattern."""
        pattern = get_pattern("always")
        ltl = pattern.instantiate("safe_mode")
        assert ltl == "G(safe_mode)"
    
    def test_pattern_instantiation_binary(self):
        """Test instantiating a binary pattern."""
        pattern = get_pattern("response")
        ltl = pattern.instantiate("trigger", "action")
        assert ltl == "G(trigger -> F(action))"
    
    def test_pattern_instantiation_wrong_arity_raises(self):
        """Test that wrong number of props raises ValueError."""
        pattern = get_pattern("response")  # arity 2
        with pytest.raises(ValueError):
            pattern.instantiate("only_one")
    
    def test_all_patterns_have_required_fields(self):
        """Verify all patterns have necessary attributes."""
        for name, pattern in DECLARE_PATTERNS.items():
            assert pattern.name == name
            assert pattern.description
            assert pattern.arity in (1, 2)
            assert pattern.ltl_template
            assert pattern.nl_indicators
            assert isinstance(pattern.pattern_type, PatternType)
    
    def test_pattern_templates_are_valid(self):
        """Test that pattern templates can be instantiated."""
        for name, pattern in DECLARE_PATTERNS.items():
            props = [f"p{i}" for i in range(pattern.arity)]
            ltl = pattern.instantiate(*props)
            assert ltl  # Should produce non-empty string
            # Check placeholders are substituted
            for p in props:
                assert p in ltl


class TestLiftingResult:
    """Tests for LiftingResult dataclass."""
    
    def test_lifting_result_creation(self):
        """Test creating a LiftingResult."""
        pattern = get_pattern("response")
        entities = [
            ExtractedEntity(text="battery low", placeholder="p1"),
            ExtractedEntity(text="charging", placeholder="p2"),
        ]
        result = LiftingResult(
            pattern=pattern,
            entities=entities,
            abstract_ltl="G(p1 -> F(p2))",
            confidence=0.9,
            original_command="If battery low, go charge"
        )
        assert result.pattern.name == "response"
        assert len(result.entities) == 2
        assert result.confidence == 0.9
    
    def test_entity_map(self):
        """Test get_entity_map returns correct mapping."""
        pattern = get_pattern("response")
        entities = [
            ExtractedEntity(text="battery low", placeholder="p1"),
            ExtractedEntity(text="charging", placeholder="p2"),
        ]
        result = LiftingResult(
            pattern=pattern,
            entities=entities,
            abstract_ltl="G(p1 -> F(p2))"
        )
        entity_map = result.get_entity_map()
        assert entity_map == {"p1": "battery low", "p2": "charging"}


class TestMultiLiftingResult:
    """Tests for MultiLiftingResult."""
    
    def test_multi_result_combines_ltl(self):
        """Test that multiple results are combined with conjunction."""
        pattern1 = get_pattern("always_eventually")
        pattern2 = get_pattern("absence")
        
        r1 = LiftingResult(
            pattern=pattern1,
            entities=[ExtractedEntity("Zone A", "p1")],
            abstract_ltl="G(F(p1))"
        )
        r2 = LiftingResult(
            pattern=pattern2,
            entities=[ExtractedEntity("Zone C", "p1")],
            abstract_ltl="G(!p1)"
        )
        
        multi = MultiLiftingResult(results=[r1, r2])
        
        assert "G(F(p1))" in multi.combined_ltl
        assert "G(!p1)" in multi.combined_ltl
        assert "&" in multi.combined_ltl


class TestLiftingEngine:
    """Tests for the LiftingEngine using mocked LLM calls."""
    
    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI client."""
        mock_client = MagicMock()
        return mock_client
    
    @pytest.fixture
    def engine(self, mock_openai_client):
        """Create an engine with mocked OpenAI client."""
        with patch('openai.OpenAI') as MockOpenAI:
            MockOpenAI.return_value = mock_openai_client
            engine = LiftingEngine(
                api_key="test-key",
                model="gpt-4o-mini",
                base_url="https://api.openai.com/v1"
            )
            engine.client = mock_openai_client
            return engine
    
    def _mock_llm_response(self, client, response_data):
        """Helper to set up mock LLM response."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(response_data)
        client.chat.completions.create.return_value = mock_response
    
    def test_engine_creation(self, engine):
        """Test engine can be created with mock client."""
        assert engine is not None
        assert engine.model == "gpt-4o-mini"
    
    def test_lift_patrol_command(self, engine, mock_openai_client):
        """Test lifting a patrol command."""
        self._mock_llm_response(mock_openai_client, {
            "patterns": [{
                "pattern_name": "always_eventually",
                "entities": ["Zone A"],
                "confidence": 0.95
            }]
        })
        
        result = engine.lift("Patrol Zone A")
        assert result.pattern.name == "always_eventually"
        assert "Zone A" in result.entities[0].text
    
    def test_lift_avoid_command(self, engine, mock_openai_client):
        """Test lifting an avoidance command."""
        self._mock_llm_response(mock_openai_client, {
            "patterns": [{
                "pattern_name": "absence",
                "entities": ["Zone C"],
                "confidence": 0.95
            }]
        })
        
        result = engine.lift("Avoid Zone C")
        assert result.pattern.name == "absence"
        assert "Zone C" in result.entities[0].text
    
    def test_lift_if_then_command(self, engine, mock_openai_client):
        """Test lifting an if-then command."""
        self._mock_llm_response(mock_openai_client, {
            "patterns": [{
                "pattern_name": "response",
                "entities": ["battery is low", "charging station"],
                "confidence": 0.9
            }]
        })
        
        result = engine.lift("If battery is low, go to charging station")
        assert result.pattern.name == "response"
        assert len(result.entities) == 2
    
    def test_lift_always_command(self, engine, mock_openai_client):
        """Test lifting an 'always' command."""
        self._mock_llm_response(mock_openai_client, {
            "patterns": [{
                "pattern_name": "always",
                "entities": ["safe distance"],
                "confidence": 0.9
            }]
        })
        
        result = engine.lift("Always maintain safe distance")
        assert result.pattern.name == "always"
    
    def test_lift_multi_complex(self, engine, mock_openai_client):
        """Test multi-lift with complex command."""
        self._mock_llm_response(mock_openai_client, {
            "patterns": [
                {
                    "pattern_name": "always_eventually",
                    "entities": ["Zone A"],
                    "confidence": 0.9
                },
                {
                    "pattern_name": "absence",
                    "entities": ["Zone C"],
                    "confidence": 0.9
                }
            ]
        })
        
        result = engine.lift_multi("Patrol Zone A and avoid Zone C")
        assert len(result.results) == 2
        assert result.combined_ltl
        assert "&" in result.combined_ltl
    
    def test_abstract_ltl_has_placeholders(self, engine, mock_openai_client):
        """Test that abstract LTL uses placeholders, not raw variables."""
        self._mock_llm_response(mock_openai_client, {
            "patterns": [{
                "pattern_name": "response",
                "entities": ["battery is low", "charging station"],
                "confidence": 0.9
            }]
        })
        
        result = engine.lift("If battery is low, go to charging station")
        # Should have p1, p2 not actual variable names
        assert "p1" in result.abstract_ltl
        assert "p2" in result.abstract_ltl
        assert "battery" not in result.abstract_ltl
        assert "charging" not in result.abstract_ltl
    
    def test_confidence_is_set(self, engine, mock_openai_client):
        """Test that confidence is between 0 and 1."""
        self._mock_llm_response(mock_openai_client, {
            "patterns": [{
                "pattern_name": "always_eventually",
                "entities": ["Zone A"],
                "confidence": 0.85
            }]
        })
        
        result = engine.lift("Patrol Zone A")
        assert 0 <= result.confidence <= 1
    
    def test_empty_response_fallback(self, engine, mock_openai_client):
        """Test handling when LLM returns no patterns."""
        self._mock_llm_response(mock_openai_client, {"patterns": []})
        
        result = engine.lift("Do something")
        # Should fall back to existence pattern with low confidence
        assert result is not None
        assert result.confidence < 0.5
    
    def test_unknown_pattern_fallback(self, engine, mock_openai_client):
        """Test handling when LLM returns unknown pattern name."""
        self._mock_llm_response(mock_openai_client, {
            "patterns": [{
                "pattern_name": "unknown_pattern",
                "entities": ["something"],
                "confidence": 0.9
            }]
        })
        
        result = engine.lift("Do something")
        # Should fall back to existence with reduced confidence
        assert result is not None
        assert result.pattern.name == "existence"


class TestLiftingEngineFromConfig:
    """Test creating LiftingEngine from config."""
    
    def test_from_config(self):
        """Test creating engine from OpenAIConfig."""
        from src.config import OpenAIConfig
        
        config = OpenAIConfig(
            api_key="test-key",
            base_url="http://localhost:11434/v1",
            model="llama3",
            temperature=0.1,
            max_tokens=500
        )
        
        with patch('openai.OpenAI') as MockOpenAI:
            MockOpenAI.return_value = MagicMock()
            engine = LiftingEngine.from_config(config)
            
            assert engine.model == "llama3"
            assert engine.temperature == 0.1
            assert engine.max_tokens == 500


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def engine(self):
        """Create engine with mocked client."""
        with patch('openai.OpenAI') as MockOpenAI:
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client
            engine = LiftingEngine(api_key="test")
            engine.client = mock_client
            return engine
    
    def test_json_in_markdown_code_block(self, engine):
        """Test handling JSON wrapped in markdown code blocks."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '''```json
{
    "patterns": [{
        "pattern_name": "always",
        "entities": ["safe"],
        "confidence": 0.9
    }]
}
```'''
        engine.client.chat.completions.create.return_value = mock_response
        
        result = engine.lift("Always be safe")
        assert result.pattern.name == "always"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
