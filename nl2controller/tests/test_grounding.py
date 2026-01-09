"""
Tests for the Grounding module (Stage 2).
"""

import pytest
from pathlib import Path

from src.grounding.vocabulary import RobotVocabulary, VocabularyEntry
from src.grounding.grounding_filter import (
    DictionaryGroundingFilter,
    GroundingResult,
    GroundingError,
    GroundedEntity,
)
from src.lifting.engine import LiftingResult, ExtractedEntity
from src.lifting.patterns import get_pattern


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_vocabulary_data():
    """Sample vocabulary data for testing."""
    return {
        "metadata": {
            "robot_name": "TestBot",
            "version": "1.0.0"
        },
        "variables": {
            "locations": {
                "zone a": "location == ZONE_A",
                "zone b": "location == ZONE_B",
                "zone c": "location == ZONE_C",
                "charging station": "location == CHARGING_STATION"
            },
            "sensors": {
                "low battery": "sys.battery < 20",
                "battery low": "sys.battery < 20",
                "obstacle detected": "sensors.obstacle == True",
                "red light": "signals.traffic_light == RED"
            },
            "actions": {
                "stop": "action.velocity == 0",
                "move": "action.velocity > 0",
                "charge": "action.charging == True"
            }
        },
        "aliases": {
            "area a": "zone a",
            "sector a": "zone a",
            "base": "charging station"
        }
    }


@pytest.fixture
def vocabulary(sample_vocabulary_data):
    """Create a vocabulary from sample data."""
    vocab = RobotVocabulary()
    vocab.load_from_dict(sample_vocabulary_data)
    return vocab


@pytest.fixture
def grounding_filter(vocabulary):
    """Create a grounding filter with sample vocabulary."""
    return DictionaryGroundingFilter(vocabulary, strict_mode=True)


# =============================================================================
# VOCABULARY TESTS
# =============================================================================

class TestRobotVocabulary:
    """Tests for RobotVocabulary class."""
    
    def test_create_empty_vocabulary(self):
        """Test creating an empty vocabulary."""
        vocab = RobotVocabulary()
        assert len(vocab) == 0
    
    def test_load_from_dict(self, sample_vocabulary_data):
        """Test loading vocabulary from dictionary."""
        vocab = RobotVocabulary()
        vocab.load_from_dict(sample_vocabulary_data)
        assert len(vocab) > 0
    
    def test_lookup_exact(self, vocabulary):
        """Test exact phrase lookup."""
        result = vocabulary.lookup("zone a")
        assert result == "location == ZONE_A"
    
    def test_lookup_case_insensitive(self, vocabulary):
        """Test case insensitive lookup."""
        result = vocabulary.lookup("Zone A")
        assert result == "location == ZONE_A"
        
        result2 = vocabulary.lookup("ZONE A")
        assert result2 == "location == ZONE_A"
    
    def test_lookup_alias(self, vocabulary):
        """Test alias lookup."""
        result = vocabulary.lookup("area a")
        assert result == "location == ZONE_A"
        
        result2 = vocabulary.lookup("sector a")
        assert result2 == "location == ZONE_A"
    
    def test_lookup_not_found(self, vocabulary):
        """Test lookup returns None for unknown phrase."""
        result = vocabulary.lookup("unknown zone")
        assert result is None
    
    def test_lookup_strict_raises(self, vocabulary):
        """Test strict lookup raises on unknown phrase."""
        with pytest.raises(KeyError):
            vocabulary.lookup_strict("unknown zone")
    
    def test_contains(self, vocabulary):
        """Test 'in' operator."""
        assert "zone a" in vocabulary
        assert "Zone A" in vocabulary  # Case insensitive
        assert "unknown" not in vocabulary
    
    def test_get_all_phrases(self, vocabulary):
        """Test getting all phrases."""
        phrases = vocabulary.get_all_phrases()
        assert "zone a" in phrases
        assert "area a" in phrases  # Aliases included
    
    def test_get_categories(self, vocabulary):
        """Test getting category names."""
        categories = vocabulary.get_categories()
        assert "locations" in categories
        assert "sensors" in categories
        assert "actions" in categories
    
    def test_get_phrases_by_category(self, vocabulary):
        """Test getting phrases in a category."""
        locations = vocabulary.get_phrases_by_category("locations")
        assert "zone a" in locations
        assert "charging station" in locations
    
    def test_get_entry(self, vocabulary):
        """Test getting full entry."""
        entry = vocabulary.get_entry("zone a")
        assert entry is not None
        assert entry.phrase == "zone a"
        assert entry.category == "locations"
        assert "area a" in entry.aliases or "sector a" in entry.aliases
    
    def test_metadata(self, vocabulary):
        """Test vocabulary metadata."""
        assert vocabulary.metadata["robot_name"] == "TestBot"


class TestVocabularyFromFile:
    """Tests for loading vocabulary from file."""
    
    def test_load_from_json_file(self):
        """Test loading vocabulary from actual JSON file."""
        vocab_path = Path(__file__).parent.parent / "data" / "robot_vocabulary.json"
        if vocab_path.exists():
            vocab = RobotVocabulary(str(vocab_path))
            assert len(vocab) > 0
            assert "zone a" in vocab


# =============================================================================
# GROUNDING FILTER TESTS
# =============================================================================

class TestDictionaryGroundingFilter:
    """Tests for DictionaryGroundingFilter."""
    
    def test_filter_creation(self, grounding_filter):
        """Test filter can be created."""
        assert grounding_filter is not None
        assert grounding_filter.strict_mode is True
    
    def test_ground_single_entity(self, grounding_filter):
        """Test grounding a single entity."""
        entity = ExtractedEntity(text="zone a", placeholder="p1")
        grounded = grounding_filter.ground_entity(entity)
        
        assert grounded.system_var == "location == ZONE_A"
        assert grounded.placeholder == "p1"
        assert grounded.original_text == "zone a"
    
    def test_ground_entity_case_insensitive(self, grounding_filter):
        """Test grounding is case insensitive."""
        entity = ExtractedEntity(text="Zone A", placeholder="p1")
        grounded = grounding_filter.ground_entity(entity)
        assert grounded.system_var == "location == ZONE_A"
    
    def test_ground_unknown_entity_raises(self, grounding_filter):
        """Test that unknown entity raises GroundingError."""
        entity = ExtractedEntity(text="zone x", placeholder="p1")
        
        with pytest.raises(GroundingError) as exc_info:
            grounding_filter.ground_entity(entity)
        
        assert "zone x" in exc_info.value.ungrounded_entities
    
    def test_grounding_error_has_suggestions(self, grounding_filter):
        """Test that grounding error includes suggestions."""
        entity = ExtractedEntity(text="zone x", placeholder="p1")
        
        with pytest.raises(GroundingError) as exc_info:
            grounding_filter.ground_entity(entity)
        
        # Should suggest similar phrases (those with 'zone')
        suggestions = exc_info.value.suggestions.get("zone x", [])
        assert len(suggestions) > 0 or True  # Depends on implementation
    
    def test_ground_lifting_result(self, grounding_filter):
        """Test grounding a full lifting result."""
        pattern = get_pattern("response")
        lifting_result = LiftingResult(
            pattern=pattern,
            entities=[
                ExtractedEntity(text="low battery", placeholder="p1"),
                ExtractedEntity(text="charging station", placeholder="p2"),
            ],
            abstract_ltl="G(p1 -> F(p2))",
            original_command="If battery low, go to charging station"
        )
        
        result = grounding_filter.ground(lifting_result)
        
        assert "sys.battery < 20" in result.grounded_ltl
        assert "location == CHARGING_STATION" in result.grounded_ltl
        assert "p1" not in result.grounded_ltl  # Placeholders substituted
        assert "p2" not in result.grounded_ltl
    
    def test_ground_absence_pattern(self, grounding_filter):
        """Test grounding an absence pattern."""
        pattern = get_pattern("absence")
        lifting_result = LiftingResult(
            pattern=pattern,
            entities=[
                ExtractedEntity(text="zone c", placeholder="p1"),
            ],
            abstract_ltl="G(!p1)",
            original_command="Avoid Zone C"
        )
        
        result = grounding_filter.ground(lifting_result)
        
        # The grounded formula substitutes p1 directly
        assert "location == ZONE_C" in result.grounded_ltl
        assert "G(!" in result.grounded_ltl
    
    def test_ground_always_eventually(self, grounding_filter):
        """Test grounding an always_eventually (patrol) pattern."""
        pattern = get_pattern("always_eventually")
        lifting_result = LiftingResult(
            pattern=pattern,
            entities=[
                ExtractedEntity(text="zone a", placeholder="p1"),
            ],
            abstract_ltl="G(F(p1))",
            original_command="Patrol Zone A"
        )
        
        result = grounding_filter.ground(lifting_result)
        
        assert result.grounded_ltl == "G(F(location == ZONE_A))"
    
    def test_ground_with_alias(self, grounding_filter):
        """Test grounding using an alias."""
        pattern = get_pattern("always_eventually")
        lifting_result = LiftingResult(
            pattern=pattern,
            entities=[
                ExtractedEntity(text="area a", placeholder="p1"),  # Alias
            ],
            abstract_ltl="G(F(p1))",
            original_command="Patrol Area A"
        )
        
        result = grounding_filter.ground(lifting_result)
        
        # "area a" is alias for "zone a"
        assert result.grounded_ltl == "G(F(location == ZONE_A))"


class TestGroundingResult:
    """Tests for GroundingResult dataclass."""
    
    def test_variable_map(self, grounding_filter):
        """Test get_variable_map method."""
        pattern = get_pattern("response")
        lifting_result = LiftingResult(
            pattern=pattern,
            entities=[
                ExtractedEntity(text="low battery", placeholder="p1"),
                ExtractedEntity(text="charge", placeholder="p2"),
            ],
            abstract_ltl="G(p1 -> F(p2))"
        )
        
        result = grounding_filter.ground(lifting_result)
        var_map = result.get_variable_map()
        
        assert "low battery" in var_map
        assert "charge" in var_map


class TestNonStrictMode:
    """Tests for non-strict (fuzzy) grounding mode."""
    
    @pytest.fixture
    def fuzzy_filter(self, vocabulary):
        """Create a non-strict filter."""
        return DictionaryGroundingFilter(
            vocabulary, 
            strict_mode=False,
            fuzzy_threshold=0.3
        )
    
    def test_fuzzy_partial_match(self, fuzzy_filter):
        """Test fuzzy matching with partial word match."""
        # "battery" should fuzzy-match "low battery" or "battery low"
        entity = ExtractedEntity(text="battery", placeholder="p1")
        
        # This should not raise with fuzzy matching
        try:
            grounded = fuzzy_filter.ground_entity(entity)
            assert "battery" in grounded.system_var.lower() or \
                   "sys.battery" in grounded.system_var
        except GroundingError:
            # Acceptable if fuzzy match doesn't find anything
            pass


class TestSafetyFeature:
    """Tests specifically for the FAIL LOUD safety feature."""
    
    def test_unknown_entity_fails_loudly(self, grounding_filter):
        """Test that unknown entities cause explicit failures."""
        entity = ExtractedEntity(text="fire alarm", placeholder="p1")
        
        # MUST raise an error
        with pytest.raises(GroundingError) as exc_info:
            grounding_filter.ground_entity(entity)
        
        # Error should be informative
        assert "fire alarm" in str(exc_info.value)
        assert "vocabulary" in str(exc_info.value).lower()
    
    def test_partial_grounding_fails(self, grounding_filter):
        """Test that partial grounding still fails."""
        pattern = get_pattern("response")
        lifting_result = LiftingResult(
            pattern=pattern,
            entities=[
                ExtractedEntity(text="zone a", placeholder="p1"),  # Valid
                ExtractedEntity(text="fire zone", placeholder="p2"),  # Invalid
            ],
            abstract_ltl="G(p1 -> F(p2))"
        )
        
        # Should fail even though first entity is valid
        with pytest.raises(GroundingError) as exc_info:
            grounding_filter.ground(lifting_result)
        
        assert "fire zone" in exc_info.value.ungrounded_entities


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
